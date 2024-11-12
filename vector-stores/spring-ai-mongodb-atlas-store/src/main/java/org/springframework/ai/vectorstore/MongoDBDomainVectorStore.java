/*
 * Copyright 2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.springframework.ai.vectorstore;

import io.micrometer.observation.ObservationRegistry;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import org.springframework.ai.data.Content;
import org.springframework.ai.data.DomainDocumentAccessor;
import org.springframework.ai.data.Embedding;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.BatchingStrategy;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingOptionsBuilder;
import org.springframework.ai.embedding.TokenCountBatchingStrategy;
import org.springframework.ai.model.EmbeddingUtils;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationConvention;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.data.mapping.MappingException;
import org.springframework.data.mapping.context.MappingContext;
import org.springframework.data.mongodb.UncategorizedMongoDbException;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.aggregation.Aggregation;
import org.springframework.data.mongodb.core.mapping.MongoPersistentEntity;
import org.springframework.data.mongodb.core.mapping.MongoPersistentProperty;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.util.Assert;

import com.mongodb.MongoCommandException;

/**
 * @author Mark Paluch
 */
public class MongoDBDomainVectorStore<T> implements DomainVectorStore<T>, InitializingBean {

	public static final String SCORE_FIELD_NAME = "score";

	private static final int INDEX_ALREADY_EXISTS_ERROR_CODE = 68;

	private static final String INDEX_ALREADY_EXISTS_ERROR_CODE_NAME = "IndexAlreadyExists";

	private static final int DEFAULT_NUM_CANDIDATES = 200;

	private static final String DEFAULT_VECTOR_INDEX_NAME = "vector_index";

	private final MongoTemplate mongoTemplate;

	private final MappingContext<? extends MongoPersistentEntity<?>, MongoPersistentProperty> context;

	private final MongoPersistentEntity<T> entity;

	private final MongoPersistentProperty embedding;

	private final EmbeddingModel embeddingModel;

	private final MongoDBDomainVectorStoreConfig config;

	private final MongoDBAtlasFilterExpressionConverter filterExpressionConverter = new MongoDBAtlasFilterExpressionConverter(
			"");

	private final BatchingStrategy batchingStrategy;

	private final boolean initializeSchema;

	public MongoDBDomainVectorStore(Class<T> domainType, MongoTemplate mongoTemplate, EmbeddingModel embeddingModel,
			boolean initializeSchema) {
		this(domainType, mongoTemplate, embeddingModel, MongoDBDomainVectorStoreConfig.defaultConfig(), initializeSchema);
	}

	public MongoDBDomainVectorStore(Class<T> domainType, MongoTemplate mongoTemplate, EmbeddingModel embeddingModel,
			MongoDBDomainVectorStoreConfig config, boolean initializeSchema) {
		this(domainType, mongoTemplate, embeddingModel, config, initializeSchema, ObservationRegistry.NOOP, null,
				new TokenCountBatchingStrategy());
	}

	@SuppressWarnings("unchecked")
	public MongoDBDomainVectorStore(Class<T> domainType, MongoTemplate mongoTemplate, EmbeddingModel embeddingModel,
			MongoDBDomainVectorStoreConfig config, boolean initializeSchema, ObservationRegistry observationRegistry,
			VectorStoreObservationConvention customObservationConvention, BatchingStrategy batchingStrategy) {

		this.mongoTemplate = mongoTemplate;
		this.context = mongoTemplate.getConverter().getMappingContext();
		this.entity = (MongoPersistentEntity<T>) context.getRequiredPersistentEntity(domainType);

		MongoPersistentProperty embedding = this.entity.getPersistentProperty(Embedding.class);

		if (embedding == null) {
			throw new MappingException(("Domain type [%s] does not have an embedding property. "
					+ "Annotate the embedding property holding the vector data with @Embedding.")
					.formatted(this.entity.getType().getName()));
		}

		this.embedding = embedding;

		MongoPersistentProperty content = this.entity.getPersistentProperty(Content.class);

		if (content == null) {
			throw new MappingException(("Domain type [%s] does not have an content property. "
					+ "Annotate the content property holding the vector data with @Content.")
					.formatted(this.entity.getType().getName()));
		}

		config.metadataFieldsToFilter.stream()
				.forEach(it -> context.getPersistentPropertyPath(it, this.entity.getTypeInformation()));

		this.embeddingModel = embeddingModel;
		this.config = config;
		this.batchingStrategy = batchingStrategy;
		this.initializeSchema = initializeSchema;
	}

	@Override
	public void afterPropertiesSet() throws Exception {
		if (!this.initializeSchema) {
			return;
		}

		// Create the collection if it does not exist
		createCollection();
		// Create search index
		createSearchIndex();
	}

	protected void createCollection() {
		if (!this.mongoTemplate.collectionExists(this.entity.getType())) {
			this.mongoTemplate.createCollection(this.entity.getType());
		}
	}

	protected void createSearchIndex() {
		try {
			this.mongoTemplate.executeCommand(createSearchIndexDefinition());
		} catch (UncategorizedMongoDbException e) {
			Throwable cause = e.getCause();
			if (cause instanceof MongoCommandException commandException) {
				// Ignore any IndexAlreadyExists errors
				if (INDEX_ALREADY_EXISTS_ERROR_CODE == commandException.getCode()
						|| INDEX_ALREADY_EXISTS_ERROR_CODE_NAME.equals(commandException.getErrorCodeName())) {
					return;
				}
			}
			throw e;
		}
	}

	/**
	 * Provides the Definition for the search index
	 */
	protected org.bson.Document createSearchIndexDefinition() {
		List<org.bson.Document> vectorFields = new ArrayList<>();

		vectorFields.add(new org.bson.Document().append("type", "vector").append("path", this.embedding.getFieldName())
				.append("numDimensions", this.embeddingModel.dimensions()).append("similarity", "cosine"));

		// TODO: Map property names to mongo paths
		vectorFields.addAll(this.config.metadataFieldsToFilter.stream()
				.map(fieldName -> new org.bson.Document().append("type", "filter").append("path", fieldName)).toList());

		return new org.bson.Document().append("createSearchIndexes", this.entity.getCollection()).append("indexes",
				List.of(new org.bson.Document().append("name", this.config.vectorIndexName).append("type", "vectorSearch")
						.append("definition", new org.bson.Document("fields", vectorFields))));
	}

	@Override
	public List<T> add(List<T> documents) {

		List<T> updatedDocuments = embed(documents);

		return (List<T>) this.mongoTemplate.insertAll(updatedDocuments);
	}

	private List<T> embed(List<T> documents) {

		List<Document> aiDocuments = new ArrayList<>(documents.size());
		List<T> updatedDocuments = new ArrayList<>(documents.size());

		DomainDocumentAccessor accessor = new DomainDocumentAccessor(this.context);
		for (T document : documents) {
			aiDocuments.add(accessor.getDocument(document));
		}
		this.embeddingModel.embed(aiDocuments, EmbeddingOptionsBuilder.builder().build(), this.batchingStrategy);

		for (int i = 0; i < documents.size(); i++) {

			T document = documents.get(i);
			Document aiDocument = aiDocuments.get(i);
			updatedDocuments.add(accessor.setEmbedding(document, aiDocument.getEmbedding()));
		}

		return updatedDocuments;
	}

	@Override
	public List<T> update(List<T> documents) {

		List<T> updatedDocuments = embed(documents);
		List<T> saved = new ArrayList<>(updatedDocuments.size());

		for (T updatedDocument : updatedDocuments) {
			saved.add(this.mongoTemplate.save(updatedDocument));
		}

		return saved;
	}

	@Override
	public Optional<Boolean> delete(List<Object> idList) {
		Query query = new Query(Criteria.where(this.entity.getRequiredIdProperty().getName()).in(idList));

		var deleteRes = this.mongoTemplate.remove(query, this.entity.getType());
		long deleteCount = deleteRes.getDeletedCount();

		return Optional.of(deleteCount == idList.size());
	}

	@Override
	public List<T> search(SearchRequest request) {

		String nativeFilterExpressions = (request.getFilterExpression() != null)
				? this.filterExpressionConverter.convertExpression(request.getFilterExpression()) : "";

		float[] queryEmbedding = this.embeddingModel.embed(request.getQuery());
		return doSimilaritySearch(request, queryEmbedding, nativeFilterExpressions);
	}

	protected List<T> doSimilaritySearch(SearchRequest request, float[] queryEmbedding,
			String nativeFilterExpressions) {
		var vectorSearch = new VectorSearchAggregation(EmbeddingUtils.toList(queryEmbedding), this.embedding.getFieldName(),
				this.config.numCandidates, this.config.vectorIndexName, request.getTopK(), nativeFilterExpressions);

		Aggregation aggregation = Aggregation.newAggregation(
				vectorSearch, Aggregation.addFields().addField(SCORE_FIELD_NAME)
						.withValueOfExpression("{\"$meta\":\"vectorSearchScore\"}").build(),
				Aggregation.match(new Criteria(SCORE_FIELD_NAME).gte(request.getSimilarityThreshold())));

		return this.mongoTemplate.aggregate(aggregation, this.entity.getType(), this.entity.getType()).getMappedResults();
	}

	public static final class MongoDBDomainVectorStoreConfig {

		private final String vectorIndexName;

		private final List<String> metadataFieldsToFilter;

		private final int numCandidates;

		private MongoDBDomainVectorStoreConfig(MongoDBDomainVectorStoreConfig.Builder builder) {
			this.vectorIndexName = builder.vectorIndexName;
			this.numCandidates = builder.numCandidates;
			this.metadataFieldsToFilter = builder.metadataFieldsToFilter;
		}

		public static MongoDBDomainVectorStoreConfig.Builder builder() {
			return new MongoDBDomainVectorStoreConfig.Builder();
		}

		public static MongoDBDomainVectorStoreConfig defaultConfig() {
			return builder().build();
		}

		public static final class Builder {

			private String vectorIndexName = DEFAULT_VECTOR_INDEX_NAME;

			private int numCandidates = DEFAULT_NUM_CANDIDATES;

			private List<String> metadataFieldsToFilter = Collections.emptyList();

			private Builder() {}

			public static MongoDBDomainVectorStoreConfig defaultConfig() {
				return builder().build();
			}

			/**
			 * Configures the vector index name. This must match the name of the Vector Search Index Name in Atlas
			 *
			 * @param vectorIndexName
			 * @return this builder
			 */
			public MongoDBDomainVectorStoreConfig.Builder withVectorIndexName(String vectorIndexName) {
				Assert.notNull(vectorIndexName, "Vector Index Name must not be null");
				Assert.notNull(vectorIndexName, "Vector Index Name must not be empty");
				this.vectorIndexName = vectorIndexName;
				return this;
			}

			public MongoDBDomainVectorStoreConfig.Builder withMetadataFieldsToFilter(List<String> metadataFieldsToFilter) {
				Assert.notEmpty(metadataFieldsToFilter, "Fields list must not be empty");
				this.metadataFieldsToFilter = metadataFieldsToFilter;
				return this;
			}

			public MongoDBDomainVectorStoreConfig build() {
				return new MongoDBDomainVectorStoreConfig(this);
			}

		}

	}

}
