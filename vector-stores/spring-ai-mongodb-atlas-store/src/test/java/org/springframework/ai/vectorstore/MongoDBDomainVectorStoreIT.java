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

import static org.assertj.core.api.Assertions.*;

import io.micrometer.observation.ObservationRegistry;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import org.springframework.ai.data.Content;
import org.springframework.ai.data.Embedding;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.ollama.OllamaEmbeddingModel;
import org.springframework.ai.ollama.api.OllamaApi;
import org.springframework.ai.ollama.api.OllamaModel;
import org.springframework.ai.ollama.api.OllamaOptions;
import org.springframework.ai.ollama.management.ModelManagementOptions;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.data.mongodb.core.MongoTemplate;

import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.mongodb.MongoDBAtlasLocalContainer;

/**
 * @author Mark Paluch
 */
@Testcontainers
@EnabledIfEnvironmentVariable(named = "OPENAI_API_KEY", matches = ".+")
class MongoDBDomainVectorStoreIT {

	@Container
	private static MongoDBAtlasLocalContainer container = new MongoDBAtlasLocalContainer(MongoDbImage.DEFAULT_IMAGE).withReuse(true);

	private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
		.withUserConfiguration(TestApplication.class)
		.withPropertyValues("spring.data.mongodb.database=springaisample",
				String.format("spring.data.mongodb.uri=" + container.getConnectionString()));

	@BeforeEach
	public void beforeEach() {
		this.contextRunner.run(context -> {
			MongoTemplate mongoTemplate = context.getBean(MongoTemplate.class);
			mongoTemplate.getCollection(mongoTemplate.getCollectionName(MyDocument.class)).deleteMany(new org.bson.Document());
		});
	}

	@Test
	void vectorStoreTest() {
		this.contextRunner.run(context -> {
			DomainVectorStore<MyDocument> vectorStore = context.getBean(DomainVectorStore.class);

			List<MyDocument> documents = List.of(
					new MyDocument(
							"Spring AI rocks!! Spring AI rocks!! Spring AI rocks!! Spring AI rocks!! Spring AI rocks!!"),
					new MyDocument("Hello World Hello World Hello World Hello World Hello World Hello World Hello World"),
					new MyDocument(
							"Great Depression Great Depression Great Depression Great Depression Great Depression Great Depression"));

			documents.get(0).setMeta1("meta1");
			documents.get(2).setMeta2("meta2");

			vectorStore.add(documents);
			Thread.sleep(5000); // Await a second for the document to be indexed

			List<MyDocument> results = vectorStore.search(SearchRequest.query("Great").withTopK(1));

			assertThat(results).hasSize(1);
			MyDocument resultDoc = results.get(0);
			assertThat(resultDoc.getId()).isEqualTo(documents.get(2).getId());
			assertThat(resultDoc.getContent()).isEqualTo(
					"Great Depression Great Depression Great Depression Great Depression Great Depression Great Depression");
			assertThat(resultDoc.getMeta2()).isEqualTo("meta2");

			// Remove all documents from the store
			vectorStore.delete(documents.stream().map(MyDocument::getId).collect(Collectors.toList()));

			List<MyDocument> results2 = vectorStore.search(SearchRequest.query("Great").withTopK(1));
			assertThat(results2).isEmpty();
		});
	}

	@Test
	void documentUpdateTest() {
		this.contextRunner.run(context -> {
			DomainVectorStore<MyDocument> vectorStore = context.getBean(DomainVectorStore.class);

			MyDocument document = new MyDocument("Spring AI rocks!!");
			document.setMeta1("meta1");

			vectorStore.add(List.of(document));
			Thread.sleep(5000); // Await a second for the document to be indexed

			List<MyDocument> results = vectorStore.search(SearchRequest.query("Spring").withTopK(5));

			assertThat(results).hasSize(1);
			MyDocument resultDoc = results.get(0);
			assertThat(resultDoc.getId()).isEqualTo(document.getId());
			assertThat(resultDoc.getContent()).isEqualTo("Spring AI rocks!!");
			assertThat(resultDoc.getMeta1()).isEqualTo("meta1");

			MyDocument sameIdDocument = new MyDocument(
					"The World is Big and Salvation Lurks Around the Corner");
			sameIdDocument.setId(document.getId());
			sameIdDocument.setMeta2("meta2");

			vectorStore.update(List.of(sameIdDocument));

			results = vectorStore.search(SearchRequest.query("FooBar").withTopK(5));

			assertThat(results).hasSize(1);
			resultDoc = results.get(0);
			assertThat(resultDoc.getId()).isEqualTo(document.getId());
			assertThat(resultDoc.getContent()).isEqualTo("The World is Big and Salvation Lurks Around the Corner");
			assertThat(resultDoc.getMeta2()).isEqualTo("meta2");
		});
	}

	@Test
	void searchWithFilters() {
		this.contextRunner.run(context -> {
			DomainVectorStore<MyDocument> vectorStore = context.getBean(DomainVectorStore.class);

			var bgDocument = new MyDocument("The World is Big and Salvation Lurks Around the Corner",
					"BG",  2020);
			var nlDocument = new MyDocument("The World is Big and Salvation Lurks Around the Corner", "NL", 0);
			var bgDocument2 = new MyDocument("The World is Big and Salvation Lurks Around the Corner",
					"BG", 2023);

			vectorStore.add(List.of(bgDocument, nlDocument, bgDocument2));
			Thread.sleep(5000); // Await a second for the document to be indexed

			List<MyDocument> results = vectorStore.search(SearchRequest.query("The World").withTopK(5));
			assertThat(results).hasSize(3);

			results = vectorStore.search(SearchRequest.query("The World")
				.withTopK(5)
				.withSimilarityThresholdAll()
				.withFilterExpression("country == 'NL'"));
			assertThat(results).hasSize(1);
			assertThat(results.get(0).getId()).isEqualTo(nlDocument.getId());

			results = vectorStore.search(SearchRequest.query("The World")
				.withTopK(5)
				.withSimilarityThresholdAll()
				.withFilterExpression("country == 'BG'"));

			assertThat(results).hasSize(2);
			assertThat(results.get(0).getId()).isIn(bgDocument.getId(), bgDocument2.getId());
			assertThat(results.get(1).getId()).isIn(bgDocument.getId(), bgDocument2.getId());

			results = vectorStore.search(SearchRequest.query("The World")
				.withTopK(5)
				.withSimilarityThresholdAll()
				.withFilterExpression("country == 'BG' && year == 2020"));

			assertThat(results).hasSize(1);
			assertThat(results.get(0).getId()).isEqualTo(bgDocument.getId());

			results = vectorStore.search(SearchRequest.query("The World")
				.withTopK(5)
				.withSimilarityThresholdAll()
				.withFilterExpression("NOT(country == 'BG' && year == 2020)"));

			assertThat(results).hasSize(2);
			assertThat(results.get(0).getId()).isIn(nlDocument.getId(), bgDocument2.getId());
			assertThat(results.get(1).getId()).isIn(nlDocument.getId(), bgDocument2.getId());

		});
	}

	@SpringBootConfiguration
	@EnableAutoConfiguration
	public static class TestApplication {

		@Bean
		public DomainVectorStore<MyDocument> vectorStore(MongoTemplate mongoTemplate, EmbeddingModel embeddingModel) {
			return new MongoDBDomainVectorStore<>(MyDocument.class, mongoTemplate, embeddingModel,
					MongoDBDomainVectorStore.MongoDBDomainVectorStoreConfig.builder()
						.withMetadataFieldsToFilter(List.of("country", "year"))
						.build(),
					true);
		}

		@Bean
		public OllamaApi ollamaApi() {
			return new OllamaApi();
		}

		@Bean
		public EmbeddingModel embeddingModel(OllamaApi api) {
			return new OllamaEmbeddingModel(api, OllamaOptions.builder().withModel(OllamaModel.LLAMA3_2).build(), ObservationRegistry.NOOP, ModelManagementOptions.builder().build());
		}

	}

	@org.springframework.data.mongodb.core.mapping.Document("my_document")
	static class MyDocument{

		String id;

		@Content
		String content;

		String meta1;

		String meta2;

		@Embedding
		float[] embedding;

		String country;

		int year;

		public MyDocument() {
		}

		public MyDocument(String content) {
			this.content = content;
		}

		public MyDocument(String content, String country, int year) {
			this.content = content;
			this.country = country;
			this.year = year;
		}

		public String getId() {
			return id;
		}

		public void setId(String id) {
			this.id = id;
		}

		public String getContent() {
			return content;
		}

		public void setContent(String content) {
			this.content = content;
		}

		public float[] getEmbedding() {
			return embedding;
		}

		public void setEmbedding(float[] embedding) {
			this.embedding = embedding;
		}

		public String getCountry() {
			return country;
		}

		public void setCountry(String country) {
			this.country = country;
		}

		public int getYear() {
			return year;
		}

		public void setYear(int year) {
			this.year = year;
		}

		public String getMeta1() {
			return meta1;
		}

		public void setMeta1(String meta1) {
			this.meta1 = meta1;
		}

		public String getMeta2() {
			return meta2;
		}

		public void setMeta2(String meta2) {
			this.meta2 = meta2;
		}
	}

}
