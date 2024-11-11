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

import java.util.List;
import java.util.Optional;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.BatchingStrategy;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingOptionsBuilder;
import org.springframework.ai.embedding.TokenCountBatchingStrategy;
import org.springframework.ai.observation.conventions.VectorStoreProvider;
import org.springframework.ai.vectorstore.observation.AbstractObservationVectorStore;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationContext;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationConvention;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.data.mongodb.UncategorizedMongoDbException;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;

import com.mongodb.MongoCommandException;

/**
 * @author Chris Smith
 * @author Soby Chacko
 * @author Christian Tzolov
 * @author Thomas Vitale
 * @since 1.0.0
 */
public abstract class MongoDBAtlasVectorStoreSupport extends AbstractObservationVectorStore implements InitializingBean {

	public static final String SCORE_FIELD_NAME = "score";

	private static final int DEFAULT_NUM_CANDIDATES = 200;

	private static final int INDEX_ALREADY_EXISTS_ERROR_CODE = 68;

	private static final String INDEX_ALREADY_EXISTS_ERROR_CODE_NAME = "IndexAlreadyExists";

	private final MongoTemplate mongoTemplate;

	private final EmbeddingModel embeddingModel;

	private final MongoDBAtlasFilterExpressionConverter filterExpressionConverter = new MongoDBAtlasFilterExpressionConverter();

	private final boolean initializeSchema;

	private final BatchingStrategy batchingStrategy;

	public MongoDBAtlasVectorStoreSupport(MongoTemplate mongoTemplate, EmbeddingModel embeddingModel,
			boolean initializeSchema) {
		this(mongoTemplate, embeddingModel, initializeSchema, ObservationRegistry.NOOP, null,
				new TokenCountBatchingStrategy());
	}

	public MongoDBAtlasVectorStoreSupport(MongoTemplate mongoTemplate, EmbeddingModel embeddingModel,
			 boolean initializeSchema, ObservationRegistry observationRegistry,
			VectorStoreObservationConvention customObservationConvention, BatchingStrategy batchingStrategy) {

		super(observationRegistry, customObservationConvention);

		this.mongoTemplate = mongoTemplate;
		this.embeddingModel = embeddingModel;

		this.initializeSchema = initializeSchema;
		this.batchingStrategy = batchingStrategy;
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
		if (!this.mongoTemplate.collectionExists(getCollectionName())) {
			this.mongoTemplate.createCollection(getCollectionName());
		}
	}

	protected void createSearchIndex() {
		try {
			this.mongoTemplate.executeCommand(createSearchIndexDefinition());
		}
		catch (UncategorizedMongoDbException e) {
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
	protected abstract org.bson.Document createSearchIndexDefinition();

	/**
	 * Maps a Bson Document to a Spring AI Document
	 * @param mongoDocument the mongoDocument to map to a Spring AI Document
	 * @return the Spring AI Document
	 */
	protected abstract Document mapMongoDocument(org.bson.Document mongoDocument, float[] queryEmbedding);

	@Override
	public void doAdd(List<Document> documents) {
		this.embeddingModel.embed(documents, EmbeddingOptionsBuilder.builder().build(), this.batchingStrategy);
		doSave(documents);
	}

	protected abstract void doSave(List<Document> documents);

	@Override
	public Optional<Boolean> doDelete(List<String> idList) {
		Query query = new Query(Criteria.where(getIdFieldName()).in(idList));

		var deleteRes = doRemove(query);
		long deleteCount = deleteRes.getDeletedCount();

		return Optional.of(deleteCount == idList.size());
	}

	protected abstract DeleteResult doRemove(Query query);

	@Override
	public List<Document> similaritySearch(String query) {
		return similaritySearch(SearchRequest.query(query));
	}

	@Override
	public List<Document> doSimilaritySearch(SearchRequest request) {

		String nativeFilterExpressions = (request.getFilterExpression() != null)
				? convertFilter(request) : "";

		float[] queryEmbedding = this.embeddingModel.embed(request.getQuery());
		return doSimilaritySearch(request, queryEmbedding, nativeFilterExpressions);
	}

	protected String convertFilter(SearchRequest request) {
		return this.filterExpressionConverter.convertExpression(request.getFilterExpression());
	}

	protected abstract List<Document> doSimilaritySearch(SearchRequest request, float[] queryEmbedding, String nativeFilterExpressions);

	@Override
	public VectorStoreObservationContext.Builder createObservationContextBuilder(String operationName) {

		return VectorStoreObservationContext.builder(VectorStoreProvider.MONGODB.value(), operationName)
			.withCollectionName(getCollectionName())
			.withDimensions(this.embeddingModel.dimensions())
			.withFieldName(getEmbeddingFieldName());
	}

	protected abstract String getCollectionName();

	protected abstract String getIdFieldName();

	protected abstract String getEmbeddingFieldName();

}
