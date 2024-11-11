/*
 * Copyright 2023-2024 the original author or authors.
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

import com.mongodb.client.result.DeleteResult;
import io.micrometer.observation.ObservationRegistry;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import org.springframework.ai.data.Content;
import org.springframework.ai.data.Embedding;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.BatchingStrategy;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.TokenCountBatchingStrategy;
import org.springframework.ai.model.EmbeddingUtils;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationConvention;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.data.mapping.MappingException;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.aggregation.Aggregation;
import org.springframework.data.mongodb.core.mapping.FieldName;
import org.springframework.data.mongodb.core.mapping.MongoPersistentEntity;
import org.springframework.data.mongodb.core.mapping.MongoPersistentProperty;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.util.Assert;

/**
 * @author Chris Smith
 * @author Soby Chacko
 * @author Christian Tzolov
 * @author Thomas Vitale
 * @author Mark Paluch
 * @since 1.0.0
 */
public class MongoDBAtlasDomainVectorStore<T> extends MongoDBAtlasVectorStoreSupport implements InitializingBean {

	public static final String SCORE_FIELD_NAME = "score";

	private static final int DEFAULT_NUM_CANDIDATES = 200;

	private static final String DEFAULT_VECTOR_INDEX_NAME = "vector_index";

	private final MongoTemplate mongoTemplate;

	private final MongoPersistentEntity<T> entity;

	private final MongoPersistentProperty embedding;

	private final MongoPersistentProperty content;

	private final EmbeddingModel embeddingModel;

	private final MongoDBVectorStoreConfig config;

	private final MongoDBAtlasFilterExpressionConverter filterExpressionConverter = new MongoDBAtlasFilterExpressionConverter("");

	public MongoDBAtlasDomainVectorStore(Class<T> domainType, MongoTemplate mongoTemplate, EmbeddingModel embeddingModel,
			boolean initializeSchema) {
		this(domainType, mongoTemplate, embeddingModel, MongoDBVectorStoreConfig.defaultConfig(), initializeSchema);
	}

	public MongoDBAtlasDomainVectorStore(Class<T> domainType, MongoTemplate mongoTemplate, EmbeddingModel embeddingModel,
			MongoDBVectorStoreConfig config, boolean initializeSchema) {
		this(domainType, mongoTemplate, embeddingModel, config, initializeSchema, ObservationRegistry.NOOP, null,
				new TokenCountBatchingStrategy());
	}

	@SuppressWarnings("unchecked")
	public MongoDBAtlasDomainVectorStore(Class<T> domainType, MongoTemplate mongoTemplate, EmbeddingModel embeddingModel,
			MongoDBVectorStoreConfig config, boolean initializeSchema, ObservationRegistry observationRegistry,
			VectorStoreObservationConvention customObservationConvention, BatchingStrategy batchingStrategy) {

		super(mongoTemplate, embeddingModel, initializeSchema, observationRegistry, customObservationConvention, batchingStrategy);

		this.mongoTemplate = mongoTemplate;
		this.entity = (MongoPersistentEntity<T>) mongoTemplate.getConverter().getMappingContext().getRequiredPersistentEntity(domainType);

		MongoPersistentProperty embedding = this.entity.getPersistentProperty(Embedding.class);

		if (embedding == null) {
			throw new MappingException(("Domain type [%s] does not have an embedding property. "
										+ "Annotate the embedding property holding the vector data with @Embedding.").formatted(this.entity.getType().getName()));
		}

		this.embedding = embedding;

		MongoPersistentProperty content = this.entity.getPersistentProperty(Content.class);

		if (content == null) {
			throw new MappingException(("Domain type [%s] does not have an content property. "
										+ "Annotate the content property holding the vector data with @Content.").formatted(this.entity.getType().getName()));
		}

		this.content = content;
		this.embeddingModel = embeddingModel;
		this.config = config;
	}

	@Override
	protected void createCollection() {
		if (!this.mongoTemplate.collectionExists(this.entity.getType())) {
			this.mongoTemplate.createCollection(this.entity.getType());
		}
	}

	/**
	 * Provides the Definition for the search index
	 */
	@Override
	protected org.bson.Document createSearchIndexDefinition() {
		List<org.bson.Document> vectorFields = new ArrayList<>();

		vectorFields.add(new org.bson.Document().append("type", "vector")
			.append("path", getEmbeddingFieldName())
			.append("numDimensions", this.embeddingModel.dimensions())
			.append("similarity", "cosine"));

		vectorFields.addAll(this.config.metadataFieldsToFilter.stream()
			.map(fieldName -> new org.bson.Document().append("type", "filter").append("path", fieldName))
			.toList());

		return new org.bson.Document().append("createSearchIndexes", getCollectionName())
			.append("indexes",
					List.of(new org.bson.Document().append("name", this.config.vectorIndexName)
						.append("type", "vectorSearch")
						.append("definition", new org.bson.Document("fields", vectorFields))));
	}

	/**
	 * Maps a Bson Document to a Spring AI Document
	 * @param mongoDocument the mongoDocument to map to a Spring AI Document
	 * @return the Spring AI Document
	 */
	protected Document mapMongoDocument(org.bson.Document mongoDocument, float[] queryEmbedding) {
		String id = mongoDocument.getString(getIdFieldName());
		String content = mongoDocument.getString(this.content.getFieldName());

		Map<String, Object> metadata = new LinkedHashMap<>(mongoDocument);
		metadata.remove(getIdFieldName());
		metadata.remove(this.content.getFieldName());
		metadata.remove(this.embedding.getFieldName());

		Document document = new Document(id, content, metadata);
		document.setEmbedding(queryEmbedding);

		return document;
	}

	@Override
	protected void doSave(List<Document> documents) {

		for (Document document : documents) {

			org.bson.Document bson = new org.bson.Document();

			Set<String> metadataKeys = document.getMetadata().keySet();

			// validate metadata keys
			for (String metadataKey : metadataKeys) {
				this.mongoTemplate.getConverter().getMappingContext()
						.getPersistentPropertyPath(metadataKey, this.entity.getTypeInformation());
			}

			bson.putAll(document.getMetadata());
			bson.put(getEmbeddingFieldName(), document.getEmbedding());
			bson.put(this.content.getFieldName(), document.getContent());

			if(document.getId() != null){
				bson.put(getIdFieldName(), document.getId());
			}

			T object = this.mongoTemplate.getConverter().read(this.entity.getType(), bson);

			this.mongoTemplate.save(object);
		}
	}

	@Override
	public Optional<Boolean> doDelete(List<String> idList) {
		Query query = new Query(Criteria.where(this.entity.hasIdProperty() ? this.entity.getRequiredIdProperty().getName() : getIdFieldName()).in(idList));

		var deleteRes = doRemove(query);
		long deleteCount = deleteRes.getDeletedCount();

		return Optional.of(deleteCount == idList.size());
	}

	@Override
	protected DeleteResult doRemove(Query query) {
		return this.mongoTemplate.remove(query, this.entity.getType());
	}

	@Override
	protected String convertFilter(SearchRequest request) {
		return this.filterExpressionConverter.convertExpression(request.getFilterExpression());
	}

	@Override
	protected List<Document> doSimilaritySearch(SearchRequest request, float[] queryEmbedding, String nativeFilterExpressions) {
		var vectorSearch = new VectorSearchAggregation(EmbeddingUtils.toList(queryEmbedding), getEmbeddingFieldName(),
				this.config.numCandidates, this.config.vectorIndexName, request.getTopK(), nativeFilterExpressions);

		Aggregation aggregation = Aggregation.newAggregation(vectorSearch,
				Aggregation.addFields()
					.addField(SCORE_FIELD_NAME)
					.withValueOfExpression("{\"$meta\":\"vectorSearchScore\"}")
					.build(),
				Aggregation.match(new Criteria(SCORE_FIELD_NAME).gte(request.getSimilarityThreshold())));

		return this.mongoTemplate.aggregate(aggregation, this.entity.getType(), org.bson.Document.class)
			.getMappedResults()
			.stream()
			.map(d -> mapMongoDocument(d, queryEmbedding))
			.toList();
	}

	@Override
	protected String getCollectionName() {
		return this.entity.getCollection();
	}

	@Override
	protected String getIdFieldName() {
		return entity.hasIdProperty() ? entity.getRequiredIdProperty()
				.getFieldName() : FieldName.ID.name();
	}

	@Override
	protected String getEmbeddingFieldName() {
		return this.embedding.getFieldName();
	}

	public static final class MongoDBVectorStoreConfig {

		private final String vectorIndexName;

		private final List<String> metadataFieldsToFilter;

		private final int numCandidates;

		private MongoDBVectorStoreConfig(Builder builder) {
			this.vectorIndexName = builder.vectorIndexName;
			this.numCandidates = builder.numCandidates;
			this.metadataFieldsToFilter = builder.metadataFieldsToFilter;
		}

		public static Builder builder() {
			return new Builder();
		}

		public static MongoDBVectorStoreConfig defaultConfig() {
			return builder().build();
		}

		public static final class Builder {

			private String vectorIndexName = DEFAULT_VECTOR_INDEX_NAME;

			private int numCandidates = DEFAULT_NUM_CANDIDATES;

			private List<String> metadataFieldsToFilter = Collections.emptyList();

			private Builder() {
			}

			public static MongoDBVectorStoreConfig defaultConfig() {
				return builder().build();
			}

			/**
			 * Configures the vector index name. This must match the name of the Vector
			 * Search Index Name in Atlas
			 * @param vectorIndexName
			 * @return this builder
			 */
			public Builder withVectorIndexName(String vectorIndexName) {
				Assert.notNull(vectorIndexName, "Vector Index Name must not be null");
				Assert.notNull(vectorIndexName, "Vector Index Name must not be empty");
				this.vectorIndexName = vectorIndexName;
				return this;
			}

			public Builder withMetadataFieldsToFilter(List<String> metadataFieldsToFilter) {
				Assert.notEmpty(metadataFieldsToFilter, "Fields list must not be empty");
				this.metadataFieldsToFilter = metadataFieldsToFilter;
				return this;
			}

			public MongoDBVectorStoreConfig build() {
				return new MongoDBVectorStoreConfig(this);
			}

		}

	}

}
