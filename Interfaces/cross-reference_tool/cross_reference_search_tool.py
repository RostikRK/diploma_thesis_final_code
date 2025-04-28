from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity

# Set up start
uri = "bolt://localhost:7687"
username = "neo4j"
password = "password1234"
# Set up end

driver = GraphDatabase.driver(uri, auth=(username, password))


def get_package_embedding(driver, product_name, manufacturer_name):
    with driver.session() as session:
        result = session.run(
            "MATCH (p:Product {name: $product_name})-[:PACKAGED_IN]->(pkg:Package) "
            "WHERE p.manufacturer = $manufacturer_name "
            "RETURN pkg.embedding AS embedding",
            product_name=product_name,
            manufacturer_name=manufacturer_name
        )
        record = result.single()
        return record["embedding"] if record else None


def get_feature_vector(driver, product_name, manufacturer_name):
    with driver.session() as session:
        result = session.run(
            "MATCH (p:Product {name: $product_name}) "
            "WHERE p.manufacturer = $manufacturer_name "
            "RETURN p.feature_vector_2 AS feature_vector",
            product_name=product_name,
            manufacturer_name=manufacturer_name
        )
        record = result.single()
        return record["feature_vector"] if record else None


def get_feature_vector_prioritized(driver, product_name, manufacturer_name):
    with driver.session() as session:
        result = session.run(
            "MATCH (p:Product {name: $product_name, manufacturer: $manufacturer_name}) "
            "RETURN p.feature_vector_prioritized_1 AS feature_vector_prioritized",
            product_name=product_name,
            manufacturer_name=manufacturer_name
        )
        record = result.single()
        return record["feature_vector_prioritized"] if record else None


def get_similar_products(driver, part_number, manufacturer_name, category_name, target_manufacturer=None, n=10):
    """
    Find similar products to the given part number based on feature vectors and package embeddings.
    """
    # Get embeddings for the source product
    package_embedding = get_package_embedding(driver, part_number, manufacturer_name)
    feature_vector = get_feature_vector(driver, part_number, manufacturer_name)
    feature_vector_prioritized = get_feature_vector_prioritized(driver, part_number, manufacturer_name)

    if not package_embedding or not feature_vector:
        print(f"Warning: Missing embeddings for {part_number}")
        return []

    # Query for potential similar products
    query = """
    MATCH (p:Product {name: $product_name})-[:BELONGS_TO]->(c:Category {name: $category_name})
    WHERE p.manufacturer = $manufacturer_name
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

    with driver.session() as session:
        try:
            result = session.run(
                query,
                product_name=part_number,
                manufacturer_name=manufacturer_name,
                category_name=category_name,
                target_manufacturer=target_manufacturer
            ).data()
        except Exception as e:
            print(f"Error querying database: {str(e)}")
            return []

    # Calculate similarity scores
    similarities = []
    for record in result:
        similar_name = record["name"]
        similar_manufacturer = record["manufacturer"]
        similar_pkg_embedding = record["pkg_embedding"]
        similar_feature_vector = record["feature_vector"]
        similar_feature_vector_prioritized = record["feature_vector_prioritized"]

        # Calculate cosine similarities for package and feature vectors
        pkg_similarity = cosine_similarity([package_embedding], [similar_pkg_embedding])[0][0]
        attr_similarity = cosine_similarity([feature_vector], [similar_feature_vector])[0][0]
        prior_similarity = cosine_similarity([feature_vector_prioritized], [similar_feature_vector_prioritized])[0][0]

        # Weighted combination
        # total_score = 0.5 * pkg_similarity + 0.5 * attr_similarity
        total_score = 0.3 * pkg_similarity + 0.3 * prior_similarity + 0.4 * attr_similarity
        similarities.append((similar_name, similar_manufacturer, total_score))

    # Sort by similarity score and return top n
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:n]


similar_products = get_similar_products(
    driver,
    part_number="GD25D10CKIGR",
    manufacturer_name="GigaDevice Semiconductor (HK) Limited",
    category_name="Memory",
    target_manufacturer="Altera",
    n=2
)

for part, manufacturer, score in similar_products:
    print(f"{part} ({manufacturer}): {score:.4f}")
