from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_vector_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up start
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password1234"
# Set up end

class FeatureVectorComputer:
    def __init__(self, uri, username, password, model_name='all-mpnet-base-v2'):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.model = SentenceTransformer(model_name)

        # Dictionary of prefixes and their conversion factors to base unit
        self.prefix_factors = {
            'p': 1e-12,
            'n': 1e-9,
            'µ': 1e-6,
            'u': 1e-6,
            'm': 1e-3,
            'k': 1e3,
            'K': 1e3,
            'M': 1e6,
            'G': 1e9
        }

        self.prioritized_attributes = dict()

    def close(self):
        self.driver.close()

    def get_top_attributes(self, category_name, limit=18):
        with self.driver.session() as session:
            result = session.run(
                "MATCH (c:Category {name: $category_name}) "
                "MATCH (p:Product)-[:BELONGS_TO]->(c) "
                "MATCH (p)-[r:HAS_ATTRIBUTE]->(a:Attribute) "
                "WITH a.name AS attr_name, COUNT(DISTINCT p) AS product_count "
                "ORDER BY product_count DESC "
                "LIMIT $limit "
                "RETURN COLLECT(attr_name) AS top_attributes",
                category_name=category_name, limit=limit
            )
            record = result.single()
            return record["top_attributes"] if record else []

    def get_unprocessed_products_count(self, category_name):
        """Get count of products that need processing"""
        with self.driver.session() as session:
            count_result = session.run(
                "MATCH (p:Product)-[:BELONGS_TO]->(c:Category {name: $category_name}) "
                "WHERE p.feature_vector_1 IS NULL "
                "RETURN count(p) AS total",
                category_name=category_name
            )
            total_unprocessed = count_result.single()["total"]

            total_result = session.run(
                "MATCH (p:Product)-[:BELONGS_TO]->(c:Category {name: $category_name}) "
                "RETURN count(p) AS total",
                category_name=category_name
            )
            total_products = total_result.single()["total"]

            return total_unprocessed, total_products

    def get_unprocessed_products_with_attributes_batch(self, category_name, top_attributes, batch_size=100):
        """Get batch of products that need processing (no feature vectors)"""
        with self.driver.session() as session:
            total_unprocessed, total_products = self.get_unprocessed_products_count(category_name)
            logger.info(f"Products in {category_name}: {total_unprocessed} unprocessed out of {total_products} total")

            if total_unprocessed == 0:
                logger.info(f"All products in category {category_name} already processed")
                return

            # Process in batches
            skip = 0
            while skip < total_unprocessed:
                start_time = time.time()
                result = session.run(
                    "MATCH (p:Product)-[:BELONGS_TO]->(c:Category {name: $category_name}) "
                    "WHERE p.feature_vector_1 IS NULL "
                    "LIMIT $limit "
                    "OPTIONAL MATCH (p)-[:HAS_MEASUREMENT]->(m:Measurement)-[:FOR_ATTRIBUTE]->(a:Attribute)"
                    "WHERE a.name IN $top_attributes "
                    "RETURN p.name AS product_name, "
                    "COLLECT({attr: a.name, value: m.value, unit: m.unit}) AS attributes",
                    category_name=category_name,
                    top_attributes=top_attributes,
                    limit=batch_size
                )

                products_data = []
                for record in result:
                    product_name = record["product_name"]
                    attributes = {}
                    for attr_data in record["attributes"]:
                        if attr_data["attr"] is not None:
                            attributes[attr_data["attr"]] = (attr_data["value"], attr_data["unit"])

                    products_data.append({
                        "product_name": product_name,
                        "attributes": attributes,
                        "category_name": category_name
                    })

                batch_time = time.time() - start_time
                logger.info(
                    f"Fetched batch {skip // batch_size + 1} ({len(products_data)} products) in {batch_time:.2f} seconds")

                print(skip, len(products_data))

                skip += batch_size
                yield products_data

    def standardize_unit(self, value, unit):
        if not unit or not value:
            return value, unit

        if len(unit) > 1 and unit[0] in self.prefix_factors:
            prefix = unit[0]
            base_unit = unit[1:]
            try:
                return str(float(value) * self.prefix_factors[prefix]), base_unit
            except (ValueError, TypeError):
                return value, unit

        return value, unit

    def compose_attribute_string(self, product_attributes, top_attributes, standardize=False, add_attribute_name=False):
        sorted_attrs = sorted(top_attributes)
        attr_strings = []
        for attr in sorted_attrs:
            if attr in product_attributes:
                value, unit = product_attributes[attr]
                if standardize and value and unit:
                    value, unit = self.standardize_unit(value, unit)

                if add_attribute_name:
                    attr_str = f"{attr}: {value}"
                    if unit:
                        attr_str += f" {unit}"
                else:
                    attr_str = f"{value}"
                    if unit:
                        attr_str += f" {unit}"
                attr_strings.append(attr_str)
            else:
                attr_strings.append(" N/A")
                continue

        return ", ".join(attr_strings)

    def compute_embeddings_batch(self, products_data, top_attributes):
        results = []

        if not products_data:
            return results

        attribute_strings = []
        for product in products_data:
            product_attributes = product["attributes"]
            category_name = product.get("category_name")

            product_strings = []

            # Generate various attribute strings for top attributes
            attr_string_1 = self.compose_attribute_string(
                product_attributes, top_attributes, standardize=False, add_attribute_name=True)
            product_strings.append(attr_string_1)

            attr_string_2 = self.compose_attribute_string(
                product_attributes, top_attributes, standardize=False, add_attribute_name=False)
            product_strings.append(attr_string_2)

            attr_string_3 = self.compose_attribute_string(
                product_attributes, top_attributes, standardize=True, add_attribute_name=False)
            product_strings.append(attr_string_3)

            has_prioritized = False

            # Generate prioritized attribute strings if category has prioritized attributes
            if category_name and category_name in self.prioritized_attributes:
                prioritized_attrs = self.prioritized_attributes[category_name]

                attr_string_prioritized_1 = self.compose_attribute_string(
                    product_attributes, prioritized_attrs, standardize=False, add_attribute_name=False)
                product_strings.append(attr_string_prioritized_1)

                attr_string_prioritized_2 = self.compose_attribute_string(
                    product_attributes, prioritized_attrs, standardize=True, add_attribute_name=False)
                product_strings.append(attr_string_prioritized_2)

                has_prioritized = True

            # Store the strings and a flag indicating if there are prioritized attributes
            attribute_strings.append((product_strings, has_prioritized))

        all_strings = []
        for product_strings, _ in attribute_strings:
            all_strings.extend(product_strings)

        if not all_strings:
            return []

        embeddings = self.model.encode(all_strings, show_progress_bar=False)

        embedding_index = 0
        results = []

        for i, (product_strings, has_prioritized) in enumerate(attribute_strings):
            product = products_data[i]

            embedding_1 = embeddings[embedding_index].tolist()
            embedding_index += 1

            embedding_2 = embeddings[embedding_index].tolist()
            embedding_index += 1

            embedding_3 = embeddings[embedding_index].tolist()
            embedding_index += 1

            result = {
                "product_name": product["product_name"],
                "embeddings": (embedding_1, embedding_2, embedding_3)
            }

            if has_prioritized:
                embedding_prioritized_1 = embeddings[embedding_index].tolist()
                embedding_index += 1

                embedding_prioritized_2 = embeddings[embedding_index].tolist()
                embedding_index += 1

                result["prioritized_embeddings"] = (embedding_prioritized_1, embedding_prioritized_2)

            results.append(result)

        return results

    def store_embeddings_batch(self, embeddings_data):
        if not embeddings_data:
            return

        with self.driver.session() as session:
            for chunk in self._chunk_list(embeddings_data, 50):  # Smaller chunks for reliability
                # Prepare batch parameters
                parameters = []
                for item in chunk:
                    product_name = item["product_name"]
                    embedding_1, embedding_2, embedding_3 = item["embeddings"]
                    prioritized_embeddings = item.get("prioritized_embeddings")

                    params = {
                        "product_name": product_name,
                        "embedding_1": embedding_1,
                        "embedding_2": embedding_2,
                        "embedding_3": embedding_3
                    }

                    if prioritized_embeddings:
                        embedding_prioritized_1, embedding_prioritized_2 = prioritized_embeddings
                        if embedding_prioritized_1:
                            params["prioritized_embedding_1"] = embedding_prioritized_1
                        if embedding_prioritized_2:
                            params["prioritized_embedding_2"] = embedding_prioritized_2

                    parameters.append(params)

                try:
                    cypher_query = (
                        "UNWIND $batch AS param "
                        "MATCH (p:Product {name: param.product_name}) "
                        "SET p.feature_vector_1 = param.embedding_1, "
                        "    p.feature_vector_2 = param.embedding_2, "
                        "    p.feature_vector_3 = param.embedding_3"
                    )

                    # Add prioritized embedding properties
                    if any("prioritized_embedding_1" in param for param in parameters):
                        cypher_query += ", p.feature_vector_prioritized_1 = param.prioritized_embedding_1"
                    if any("prioritized_embedding_2" in param for param in parameters):
                        cypher_query += ", p.feature_vector_prioritized_2 = param.prioritized_embedding_2"

                    session.run(cypher_query, batch=parameters)

                    logger.info(f"Stored {len(chunk)} embeddings successfully")
                except Exception as e:
                    logger.error(f"Error storing embeddings: {str(e)}")

    def _chunk_list(self, lst, chunk_size):
        """Split a list into chunks of specified size"""
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    def compute_feature_vectors_for_category(self, category_name, batch_size=100, prioritized_attributes=None):
        """Main function to compute feature vectors for all products in a category"""
        logger.info(f"Processing category: {category_name}")
        start_time = time.time()

        # If prioritized attributes are provided, update the dictionary
        if prioritized_attributes:
            self.prioritized_attributes[category_name] = prioritized_attributes
            logger.info(f"Using prioritized attributes for {category_name}: {prioritized_attributes}")

        # Get top attributes
        top_attributes = self.get_top_attributes(category_name)
        if not top_attributes:
            logger.warning(f"No attributes found for category: {category_name}")
            return

        logger.info(f"Top attributes: {top_attributes}")

        # Check if all products are already processed
        total_unprocessed, total_products = self.get_unprocessed_products_count(category_name)
        if total_unprocessed == 0:
            logger.info(f"All products in {category_name} category already have feature vectors computed")
            return 0

        # Process products in batches
        products_processed = 0
        for batch_idx, products_batch in enumerate(self.get_unprocessed_products_with_attributes_batch(
                category_name, top_attributes, batch_size)):

            if not products_batch:
                continue

            # Compute embeddings for the batch
            embeddings_data = self.compute_embeddings_batch(products_batch, top_attributes)

            # Store the embeddings
            self.store_embeddings_batch(embeddings_data)

            products_processed += len(products_batch)
            logger.info(
                f"Processed batch {batch_idx + 1} - {products_processed} out of {total_unprocessed} unprocessed products")

        total_time = time.time() - start_time
        logger.info(f"Completed processing {products_processed} products in {total_time:.2f} seconds")
        logger.info(
            f"Overall status: {total_products - total_unprocessed + products_processed}/{total_products} products processed")
        return products_processed



def main():
    processor = FeatureVectorComputer(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

    try:
        # Process two categories with prioritized attributes
        categories = [
            {
                "name": "Memory",
                "prioritized_attributes": ["memory_size", "memory_interface", "technology"]
            },
            {
                "name": "Drivers, Receivers, Transceivers",
                "prioritized_attributes": ["protocol", "number_of_drivers_receivers", "receiver_hysteresis"]
            }
        ]

        for category_info in categories:
            processor.compute_feature_vectors_for_category(
                category_info["name"],
                batch_size=200,
                prioritized_attributes=category_info.get("prioritized_attributes")
            )
    finally:
        processor.close()


main()
