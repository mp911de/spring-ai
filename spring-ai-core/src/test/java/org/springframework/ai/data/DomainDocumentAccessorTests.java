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
package org.springframework.ai.data;

import static org.assertj.core.api.Assertions.*;

import java.util.List;

import org.junit.jupiter.api.Test;

import org.springframework.ai.document.Document;
import org.springframework.data.mongodb.core.mapping.MongoMappingContext;

/**
 * Unit tests for {@link DomainDocumentAccessor}.
 *
 * @author Mark Paluch
 */
class DomainDocumentAccessorTests {

	private final MongoMappingContext context = new MongoMappingContext();

	@Test
	void shouldExtractContent() {

		CustomerReport report = new CustomerReport("Spring", "Best framework ever!");
		DomainDocumentAccessor accessor = new DomainDocumentAccessor(context);

		String content = accessor.getContent(report);

		assertThat(content).isEqualTo("Best framework ever!");
	}

	@Test
	void shouldExtractContentUsingContentFunction() {

		WithContentSource report = new WithContentSource("Spring", "Best framework ever!");
		DomainDocumentAccessor accessor = new DomainDocumentAccessor(context);

		String content = accessor.getContent(report);

		assertThat(content).isEqualTo("Spring Best framework ever!");
	}

	@Test
	void shouldExtractDocument() {

		CustomerReport report = new CustomerReport("Spring", "Best framework ever!");
		DomainDocumentAccessor accessor = new DomainDocumentAccessor(context);

		Document document = accessor.getDocument(report);

		assertThat(document.getContent()).isEqualTo("Best framework ever!");
		assertThat(document.getMetadata()).hasSize(1).containsEntry("customerName", "Spring");
	}

	@Test
	void shouldApplyEmbedding() {

		CustomerReport report = new CustomerReport("Spring", "Best framework ever!");
		DomainDocumentAccessor accessor = new DomainDocumentAccessor(context);

		accessor.setEmbedding(report, List.of(1.1f, 2.2f));

		assertThat(report.embedding).hasSize(2).contains((double) 1.1f, (double) 2.2f);
	}

	@Test
	void shouldApplyEmbeddingToImmutable() {

		ImmutableCustomerReport report = new ImmutableCustomerReport(new double[0]);
		DomainDocumentAccessor accessor = new DomainDocumentAccessor(context);

		ImmutableCustomerReport result = accessor.setEmbedding(report, List.of(1.1f, 2.2f));

		assertThat(report.embedding).isEmpty();
		assertThat(result.embedding).hasSize(2).contains(1.1f, 2.2f);
	}

	static class CustomerReport {

		String customerName;

		@Content String notes;

		@Embedding List<Double> embedding;

		CustomerReport(String customerName, String notes) {
			this.customerName = customerName;
			this.notes = notes;
		}
	}

	@ContentSource(ConvertedReportContentConverter.class)
	static class WithContentSource {

		String customerName;

		String notes;

		WithContentSource(String customerName, String notes) {
			this.customerName = customerName;
			this.notes = notes;
		}
	}

	static class ConvertedReportContentConverter implements ContentFunction<WithContentSource> {
		@Override
		public String apply(WithContentSource source) {
			return source.customerName + " " + source.notes;
		}
	}

	static class ImmutableCustomerReport {

		@Embedding final double[] embedding;

		ImmutableCustomerReport(double[] embedding) {
			this.embedding = embedding;
		}
	}
}
