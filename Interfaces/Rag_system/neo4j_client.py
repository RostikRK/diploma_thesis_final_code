import logging
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity

import config

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Client for interacting with Neo4j database."""

    def __init__(self, uri=config.NEO4J_URI, username=config.NEO4J_USERNAME, password=config.NEO4J_PASSWORD):
        """Initialize the Neo4j client."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info(f"Connected to Neo4j at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def get_package_embedding(self, product_name: str, manufacturer_name: str) -> Optional[List[float]]:
        """Get package embedding for a product."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (p:Product {name: $product_name})-[:PACKAGED_IN]->(pkg:Package) "
                "WHERE p.manufacturer = $manufacturer_name "
                "RETURN pkg.embedding AS embedding",
                product_name=product_name,
                manufacturer_name=manufacturer_name
            )
            record = result.single()
            return record["embedding"] if record else None

    def get_feature_vector(self, product_name: str, manufacturer_name: str) -> Optional[List[float]]:
        """Get feature vector for a product."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (p:Product {name: $product_name}) "
                "WHERE p.manufacturer = $manufacturer_name "
                "RETURN p.feature_vector_2 AS feature_vector",
                product_name=product_name,
                manufacturer_name=manufacturer_name
            )
            record = result.single()
            return record["feature_vector"] if record else None

    def get_feature_vector_prioritized(self, product_name: str, manufacturer_name: str) -> Optional[List[float]]:
        """Get prioritized feature vector for a product."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (p:Product {name: $product_name, manufacturer: $manufacturer_name}) "
                "RETURN p.feature_vector_prioritized_1 AS feature_vector_prioritized",
                product_name=product_name,
                manufacturer_name=manufacturer_name
            )
            record = result.single()
            return record["feature_vector_prioritized"] if record else None

    def get_product_category(self, part_number: str, manufacturer: str) -> Optional[str]:
        """Get the category for a specific part number."""
        with self.driver.session() as session:
            category_query = """
            MATCH (p:Product {name: $part_number})-[:BELONGS_TO]->(c:Category)
            WHERE p.manufacturer = $manufacturer
            RETURN c.name AS category
            LIMIT 1
            """
            result = session.run(category_query, part_number=part_number, manufacturer=manufacturer)
            record = result.single()
            return record["category"] if record else None

    def find_category_based_cross_references(self, part_number: str, manufacturer: str,
                                             category_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fallback to category-based search if vector similarity isn't possible."""
        with self.driver.session() as session:
            query = """
            MATCH (p:Product {name: $part_number})-[:BELONGS_TO]->(c:Category {name: $category_name})
            WHERE p.manufacturer = $manufacturer
            MATCH (similar:Product)-[:BELONGS_TO]->(c)
            WHERE similar.manufacturer <> $manufacturer
            RETURN similar.name AS cross_ref_part, 
                   similar.manufacturer AS cross_ref_manufacturer,
                   c.name AS category,
                   0.5 AS total_similarity
            LIMIT $limit
            """

            results = session.run(query,
                                  part_number=part_number,
                                  manufacturer=manufacturer,
                                  category_name=category_name,
                                  limit=limit)

            return [dict(record) for record in results]

    def find_cross_references(self, part_number: str, manufacturer: str, category: Optional[str] = None,
                              target_manufacturer: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find cross-reference components for a given part number and manufacturer.
        """
        category_name = category

        if not category_name:
            category_name = self.get_product_category(part_number, manufacturer)

        if not category_name:
            logger.warning(f"No category found for part {part_number} from {manufacturer}")
            return []

        # Get embeddings for the source product
        package_embedding = self.get_package_embedding(part_number, manufacturer)
        feature_vector = self.get_feature_vector(part_number, manufacturer)
        feature_vector_prioritized = self.get_feature_vector_prioritized(part_number, manufacturer)

        if not package_embedding or not feature_vector or not feature_vector_prioritized:
            logger.info(f"Missing embeddings for {part_number}, using basic category-based search")
            return self.find_category_based_cross_references(part_number, manufacturer, category_name, limit)

        query = """
        MATCH (p:Product {name: $part_number})-[:BELONGS_TO]->(c:Category {name: $category_name})
        WHERE p.manufacturer = $manufacturer
        MATCH (similar:Product)-[:BELONGS_TO]->(c)
        WHERE similar <> p
          AND ($target_manufacturer IS NULL OR similar.manufacturer = $target_manufacturer)
        MATCH (similar)-[:PACKAGED_IN]->(s_pkg:Package)
        RETURN similar.name AS name, 
               similar.manufacturer AS manufacturer, 
               s_pkg.embedding AS pkg_embedding, 
               similar.feature_vector_2 AS feature_vector,
               similar.feature_vector_prioritized_1 AS feature_vector_prioritized
        """

        with self.driver.session() as session:
            try:
                result = session.run(
                    query,
                    part_number=part_number,
                    manufacturer=manufacturer,
                    category_name=category_name,
                    target_manufacturer=target_manufacturer
                ).data()
            except Exception as e:
                logger.error(f"Error querying database for similar products: {str(e)}")
                return self.find_category_based_cross_references(part_number, manufacturer, category_name, limit)

        # Calculate similarity scores
        similarities = []

        for record in result:
            similar_name = record["name"]
            similar_manufacturer = record["manufacturer"]
            similar_pkg_embedding = record["pkg_embedding"]
            similar_feature_vector = record["feature_vector"]
            similar_feature_vector_prioritized = record["feature_vector_prioritized"]

            # Calculate cosine similarities
            pkg_similarity = cosine_similarity([package_embedding], [similar_pkg_embedding])[0][0]
            attr_similarity = cosine_similarity([feature_vector], [similar_feature_vector])[0][0]
            prior_similarity = cosine_similarity([feature_vector_prioritized], [similar_feature_vector_prioritized])[0][
                0]

            # Weighted combination
            total_score = 0.3 * pkg_similarity + 0.3 * prior_similarity + 0.4 * attr_similarity

            similarities.append({
                "cross_ref_part": similar_name,
                "cross_ref_manufacturer": similar_manufacturer,
                "category": category_name,
                "total_similarity": float(total_score)  # Convert numpy float to Python float for JSON serialization
            })

        # Sort by similarity score and return top n
        similarities.sort(key=lambda x: x["total_similarity"], reverse=True)
        return similarities[:limit]

    def execute_cypher(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return the results.
        """
        if params is None:
            params = {}

        with self.driver.session() as session:
            try:
                results = session.run(query, **params)
                return [dict(record) for record in results]
            except Exception as e:
                logger.error(f"Error executing Cypher query: {e}")
                raise

    def get_product_details(self, part_number: str, manufacturer: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a product.
        """
        with self.driver.session() as session:
            query = """
            MATCH (p:Product {name: $part_number})
            """

            if manufacturer:
                query += "WHERE p.manufacturer = $manufacturer "

            query += """
            OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
            OPTIONAL MATCH (p)-[:PACKAGED_IN]->(pkg:Package)
            OPTIONAL MATCH (p)-[:HAS_DATASHEET]->(d:Datasheet)
            RETURN p.name AS part_number, 
                   p.manufacturer AS manufacturer,
                   p.description AS description,
                   c.name AS category,
                   pkg.name AS package,
                   d.name AS datasheet_url
            LIMIT 1
            """

            params = {"part_number": part_number}
            if manufacturer:
                params["manufacturer"] = manufacturer

            result = session.run(query, **params)
            record = result.single()

            if record:
                return dict(record)
            return None

    def get_product_attributes(self, part_number: str, manufacturer: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get attributes for a specific product.
        """
        with self.driver.session() as session:
            query = """
            MATCH (p:Product {name: $part_number})
            """

            if manufacturer:
                query += "WHERE p.manufacturer = $manufacturer "

            query += """
            MATCH (p)-[:HAS_ATTRIBUTE]->(a:Attribute)
            OPTIONAL MATCH (p)-[:HAS_MEASUREMENT]->(m:Measurement)-[:FOR_ATTRIBUTE]->(a)
            RETURN a.name AS attribute_name, 
                   a.description AS attribute_description,
                   collect({value: m.value, unit: m.unit, conditions: m.test_conditions}) AS measurements
            """

            params = {"part_number": part_number}
            if manufacturer:
                params["manufacturer"] = manufacturer

            results = session.run(query, **params)
            return [dict(record) for record in results]
