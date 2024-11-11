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

import java.util.Collection;
import java.util.List;

import org.springframework.ai.document.Document;
import org.springframework.beans.BeanUtils;
import org.springframework.core.CollectionFactory;
import org.springframework.core.convert.ConversionService;
import org.springframework.core.convert.support.DefaultConversionService;
import org.springframework.data.mapping.PersistentEntity;
import org.springframework.data.mapping.PersistentProperty;
import org.springframework.data.mapping.PersistentPropertyAccessor;
import org.springframework.data.mapping.context.MappingContext;
import org.springframework.data.mapping.context.PersistentEntities;
import org.springframework.data.util.ProxyUtils;
import org.springframework.util.Assert;
import org.springframework.util.ClassUtils;
import org.springframework.util.ConcurrentLruCache;

/**
 * @author Mark Paluch
 */
public class DomainDocumentAccessor {

	private final PersistentEntities entities;
	private final ConversionService conversionService;

	@SuppressWarnings("unchecked") private final ConcurrentLruCache<PersistentEntity<?, ?>, ContentFunction<Object>> converterCache = new ConcurrentLruCache<>(
			256, entity -> {

				ContentSource converterAnnotation = entity.getRequiredAnnotation(ContentSource.class);
				return (ContentFunction<Object>) BeanUtils.instantiateClass(converterAnnotation.value());
			});

	public DomainDocumentAccessor(PersistentEntities entities, ConversionService conversionService) {
		this.entities = entities;
		this.conversionService = conversionService;
	}

	public DomainDocumentAccessor(MappingContext<?, ?> mappingContext) {
		this(new PersistentEntities(List.of(mappingContext)), DefaultConversionService.getSharedInstance());
	}

	/**
	 * Extract a {@link Document} from the given entity {@code object}. The entity class must either declare a content
	 * property annotated with {@code @Content} or define a {@code @ContentConverter}. All non-content/embedding
	 * properties are returned as document metadata.
	 *
	 * @param object
	 * @return
	 */
	public Document getDocument(Object object) {

		Assert.notNull(object, "Entity must not be null");

		Class<?> userClass = ProxyUtils.getUserClass(object);
		PersistentEntity<?, ? extends PersistentProperty<?>> entity = entities.getRequiredPersistentEntity(userClass);

		PersistentProperty<? extends PersistentProperty<?>> contentProperty = entity.getPersistentProperty(Content.class);
		PersistentProperty<? extends PersistentProperty<?>> embeddingProperty = entity
				.getPersistentProperty(Embedding.class);
		PersistentPropertyAccessor<Object> accessor = entity.getPropertyAccessor(object);

		Document document = new Document(doGetContent(object, entity));

		for (PersistentProperty<? extends PersistentProperty<?>> persistentProperty : entity) {

			if (persistentProperty == contentProperty || persistentProperty == embeddingProperty) {
				continue;
			}

			document.getMetadata().put(persistentProperty.getName(), accessor.getProperty(persistentProperty));
		}

		return document;
	}

	/**
	 * Retrieve the content for an entity {@code object}. The entity class must either declare a content property
	 * annotated with {@code @Content} or define a {@code @ContentConverter}.
	 *
	 * @param object
	 * @return
	 */
	public String getContent(Object object) {

		Assert.notNull(object, "Entity must not be null");

		Class<?> userClass = ProxyUtils.getUserClass(object);

		PersistentEntity<?, ? extends PersistentProperty<?>> entity = entities.getRequiredPersistentEntity(userClass);
		return doGetContent(object, entity);
	}

	private String doGetContent(Object object, PersistentEntity<?, ?> entity) {

		if (entity.isAnnotationPresent(ContentSource.class)) {

			ContentFunction<Object> contentFunction = converterCache.get(entity);
			return contentFunction.apply(object);
		}

		PersistentProperty<? extends PersistentProperty<?>> contentProperty = entity.getPersistentProperty(Content.class);

		if (contentProperty == null) {
			throw new IllegalStateException("Cannot find @Content property for " + entity.getName());
		}

		PersistentPropertyAccessor<Object> accessor = entity.getPropertyAccessor(object);

		Object content = accessor.getProperty(contentProperty);

		if (content == null) {
			throw new IllegalArgumentException("Content is null");
		}

		return content instanceof String s ? s : conversionService.convert(content, String.class);
	}

	/**
	 * Associate the {@code embedding} vector with the given entity {@code object}. The entity class must contain a
	 * embedding property annotated with {@code @Embedding}.
	 *
	 * @param object the entity object to hold the embedding vector.
	 * @param embedding embedding vector.
	 * @return
	 * @param <T>
	 */
	@SuppressWarnings("unchecked")
	public <T> T setEmbedding(T object, Collection<? extends Number> embedding) {

		Assert.notNull(object, "Entity must not be null");

		Class<?> userClass = ProxyUtils.getUserClass(object);
		PersistentEntity<?, ? extends PersistentProperty<?>> entity = entities.getRequiredPersistentEntity(userClass);
		PersistentProperty<? extends PersistentProperty<?>> embeddingProperty = entity
				.getPersistentProperty(Embedding.class);

		if (embeddingProperty == null) {
			throw new IllegalStateException("Cannot find @Embedding property for " + userClass.getName());
		}

		PersistentPropertyAccessor<Object> accessor = entity.getPropertyAccessor(object);

		if (embeddingProperty.isArray()) {
			accessor.setProperty(embeddingProperty, conversionService.convert(embedding, embeddingProperty.getType()));
		} else {

			Collection<Object> collection = CollectionFactory.createCollection(embeddingProperty.getType(), embedding.size());
			Class<?> actualType = embeddingProperty.getActualType();

			for (Number number : embedding) {
				if (ClassUtils.isAssignable(actualType, number.getClass())) {
					collection.add(number);
				} else {
					collection.add(conversionService.convert(number, actualType));
				}
			}

			accessor.setProperty(embeddingProperty, collection);
		}

		return (T) accessor.getBean();
	}
}
