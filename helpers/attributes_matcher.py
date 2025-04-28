import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class AttributesMatcher:
    def __init__(self, category_name, driver, similarity_mode="description", threshold=0.65):
        self.similarity_mode = similarity_mode
        self.threshold = threshold
        self.driver = driver
        self.category_name = category_name
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

        # Load existing attributes
        self.existing_attributes = self._load_existing_attributes(category_name)
        self._compute_attribute_embeddings()

    def _load_existing_attributes(self, category_name):
        """
        Load existing  attributes from the Neo4j graph database.
        This method extracts all attributes connected to at least one product in the given category.
        """
        query = """
        MATCH (c:Category {name: $category_name})<-[:BELONGS_TO]-(p:Product)-[:HAS_ATTRIBUTE]->(a:Attribute)
        RETURN DISTINCT a.name AS id, a.description AS description, a.embedding as stored_embedding
        """
        with self.driver.session() as session:
            result = session.run(query, category_name=category_name)
            return [record.data() for record in result]

    def _compute_attribute_embeddings(self):
        """
        Compute and add embeddings for all attributes that don't already have them.
        This handles both attributes with existing stored embeddings and those without.
        """
        for param in self.existing_attributes:
            if param.get('stored_embedding') is not None and len(param.get('stored_embedding', [])) > 0:
                # Use the stored embedding
                param['description_embedding'] = np.array(param['stored_embedding'])
            else:
                # Compute embedding for description
                if param.get('description'):
                    try:
                        param['description_embedding'] = self.embedding_model.encode(param['description'])
                    except Exception as e:
                        param['description_embedding'] = None
                else:
                    param['description_embedding'] = None

            try:
                param['id_embedding'] = self.embedding_model.encode(param['id'])
            except Exception as e:
                param['id_embedding'] = None

    def refresh_attributes(self):
        """
        Refresh the list of known attributes from the database.
        """
        self.existing_attributes = self._load_existing_attributes(self.category_name)
        self._compute_attribute_embeddings()

    def validate_extracted_attributes(self, extracted_graph, logger):
        """
        Validate extracted attributes against existing knowledge
        """
        validated_graph = {
            "nodes": [],
            "relationships": []
        }

        id_mapping = {}
        new_attributes_count = 0

        # Validate attributes and create ID mapping
        for node in extracted_graph.get("nodes", []):
            if node["type"] == "Attribute":
                original_id = node["id"]
                validated_node = self._validate_attribute(node, logger)
                if validated_node:
                    validated_graph["nodes"].append(validated_node)
                    if original_id != validated_node["id"]:
                        id_mapping[original_id] = validated_node["id"]
                        logger.info(
                            f"Mapped attribute '{original_id}' to existing attribute '{validated_node['id']}' with confidence {validated_node.get('validation_confidence', 0):.2f}")
                    elif validated_node.get('is_new_attribute', False):
                        new_attributes_count += 1
                        logger.info(
                            f"Added new attribute '{validated_node['id']}' with confidence {validated_node.get('validation_confidence', 0):.2f}")
            else:
                validated_graph["nodes"].append(node)

        for relationship in extracted_graph.get("relationships", []):
            updated_relationship = relationship.copy()
            if relationship["source"] in id_mapping:
                updated_relationship["source"] = id_mapping[relationship["source"]]
            if relationship["target"] in id_mapping:
                updated_relationship["target"] = id_mapping[relationship["target"]]
            validated_graph["relationships"].append(updated_relationship)

        return validated_graph

    def _validate_attribute(self, attribute, logger):
        """
        Validate a single attribute using semantic similarity.
        """
        # Calculate similarity scores for this attribute against all existing attributes
        similarity_scores = self._compute_semantic_similarity(attribute, logger)

        # Find the most similar existing attribute if it meets the threshold
        matched_attribute = self._find_existing_attribute(attribute, similarity_scores, self.threshold, logger)

        if matched_attribute:
            updated_attribute = attribute.copy()
            updated_attribute['id'] = matched_attribute['id']
            updated_attribute['validation_confidence'] = max(similarity_scores) if similarity_scores else 0
            return updated_attribute

        new_attribute = attribute.copy()

        # Calculate and store embeddings for the new attribute
        description = new_attribute['properties'].get('description', '')
        if description:
            description_embedding = self.embedding_model.encode(description).tolist()
            if 'properties' not in new_attribute:
                new_attribute['properties'] = {}
            new_attribute['properties']['embedding'] = description_embedding

        new_attribute['properties']['during_extraction'] = True
        new_attribute['is_new_attribute'] = True

        new_attribute['validation_confidence'] = max(similarity_scores) if similarity_scores else 0

        return new_attribute

    def _compute_semantic_similarity(self, attribute, logger):
        """
        Compute semantic similarity between the extracted attribute and each existing attribute.
        The similarity can be computed based on description, id, or both.
        """
        scores = []

        # Prepare query embeddings based on the selected similarity mode
        query_desc_embedding = None
        query_id_embedding = None

        if self.similarity_mode in ["description", "both"]:
            query_description = attribute['properties'].get('description', '')
            if query_description:
                try:
                    query_desc_embedding = self.embedding_model.encode(query_description)
                except Exception as e:
                    logger.warning(f"Failed to encode description for attribute '{attribute['id']}': {e}")

        if self.similarity_mode in ["id", "both"]:
            query_id = attribute.get('id', '')
            if query_id:
                try:
                    query_id_embedding = self.embedding_model.encode(query_id)
                except Exception as e:
                    logger.warning(f"Failed to encode ID for attribute '{attribute['id']}': {e}")

        for param in self.existing_attributes:
            total_score = 0
            components = 0

            # Compare descriptions if that mode is selected
            if self.similarity_mode in ["description", "both"]:
                if query_desc_embedding is not None and param.get('description_embedding') is not None:
                    try:
                        desc_embedding = param['description_embedding']
                        if not isinstance(desc_embedding, np.ndarray):
                            if isinstance(desc_embedding, list):
                                desc_embedding = np.array(desc_embedding)
                            else:
                                logger.warning(f"Invalid description embedding format for attribute '{param['id']}'")
                                desc_embedding = None

                        if desc_embedding is not None:
                            score_desc = cosine_similarity([query_desc_embedding], [desc_embedding])[0][0]
                            total_score += score_desc*2
                            components += 2
                    except Exception as e:
                        logger.warning(f"Error calculating description similarity for attribute '{param['id']}': {e}")

            # Compare ids if that mode is selected
            if self.similarity_mode in ["id", "both"]:
                if query_id_embedding is not None and param.get('id_embedding') is not None:
                    try:
                        id_embedding = param['id_embedding']
                        score_id = cosine_similarity([query_id_embedding], [id_embedding])[0][0]
                        total_score += score_id
                        components += 1
                    except Exception as e:
                        logger.warning(f"Error calculating ID similarity for attribute '{param['id']}': {e}")

            # Calculate average similarity score
            avg_score = total_score / components if components > 0 else 0
            scores.append(avg_score)

        return scores

    def _find_existing_attribute(self, attribute, similarities, threshold, logger):
        """
        Find the most similar existing attribute based on a semantic similarity threshold.
        """
        if not similarities or max(similarities) < threshold:
            return None

        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)

        logger.debug(
            f"Attribute '{attribute['id']}' matched to '{self.existing_attributes[max_index]['id']}' with similarity {max_similarity:.4f}")

        return self.existing_attributes[max_index]
