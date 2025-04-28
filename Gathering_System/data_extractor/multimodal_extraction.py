import os
import json
import logging
import time
import torch
import io
import requests
from PIL import Image

from attributes_matcher import AttributesMatcher

from neo4j import GraphDatabase
from unsloth import FastVisionModel
from transformers import TextStreamer

from extraction_utils import (
    Graph, detect_loop, create_prompt_image, parse_graph, validate_graph
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multimodal_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up start
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password1234"

MODEL_NAME = "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 10096
MAX_RESPONCE_LENGTH = 5100
# Set up end


def get_datasheet_manufacturer_dict(driver):
    """Get products with chunked datasheets that need processing. Returns a list of dictionaries."""
    query = """
    MATCH (c:Category)<-[:BELONGS_TO]-(p:Product)
    MATCH (p:Product)-[:MANUFACTURED_BY]->(m:Manufacturer)
    OPTIONAL MATCH (p)-[:PACKAGED_IN]->(pkg:Package)
    MATCH (p)-[r:HAS_DATASHEET]->(d:Datasheet)
    WHERE r.state = "Chunked"
    RETURN d.name AS datasheet, m.name AS manufacturer, p.name AS product,
           pkg.name AS package, c.name as category
    """
    datasheet_list_of_dicts = []
    with driver.session() as session:
        result = session.execute_read(lambda tx: tx.run(query).data())
        for record in result:
            datasheet_url = record["datasheet"]
            manufacturer_name = record["manufacturer"]
            product_name = record["product"]
            package_name = record.get("package", "")
            category_name = record["category"]
            datasheet_list_of_dicts.append({
                "datasheet_url": datasheet_url,
                "manufacturer": manufacturer_name,
                "product_name": product_name,
                "package": package_name,
                "category": category_name
            })
    return datasheet_list_of_dicts


def get_images_for_datasheet(datasheet_url, start_page=0, max_pages=150):
    """
    Retrieves all images for a given datasheet URL by making requests to the API.
    """
    images = []
    for page_num in range(start_page, max_pages + 1):
        params = {
            "datasheet_url": datasheet_url,
            "page_num": str(page_num)
        }
        response = requests.get("http://localhost:8082/image", params=params)

        if response.status_code == 200:
            try:
                img_data = io.BytesIO(response.content)
                img = Image.open(img_data).convert('RGB')
                images.append((img, page_num))
            except Exception as e:
                logger.error(f"Error opening image for {datasheet_url} page {page_num}: {e}")
                break
        elif response.status_code == 404:
            break
        else:
            logger.error(f"Error fetching image for {datasheet_url} page {page_num}: HTTP {response.status_code}")
            break
    return images


def mark_datasheet_product_as_processed(driver, product_key):
    """Mark a product-datasheet relationship as processed"""
    query = """
    MATCH (p:Product {name: $product_name, manufacturer: $manufacturer_name})-[r:HAS_DATASHEET]->(d:Datasheet)
    SET r.state='Processed'
    """
    with driver.session() as session:
        session.execute_write(lambda tx: tx.run(query,
                                                product_name=product_key['pn'],
                                                manufacturer_name=product_key['manufacturer']))


def load_model():
    """Load the vision language model"""
    logger.info(f"Loading model {MODEL_NAME}")

    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True
    )

    FastVisionModel.for_inference(model)

    return model, tokenizer



def generate_extraction(model, tokenizer, prompt, image, use_loop_prevention=True):
    """Generate parameter extraction from the prompt and image"""
    try:
        if isinstance(prompt, list) and isinstance(prompt[0], dict) and 'role' in prompt[0]:
            input_text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
            inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
        else:
            input_text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
            inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

        loop_detected = False

        if use_loop_prevention:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_RESPONCE_LENGTH,
                use_cache=True,
                temperature=0.6,
                min_p=0.1,
                top_p=0.7
            )
            prompt_length = inputs['input_ids'].shape[1]
            generated_text = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True).strip()

            if detect_loop(generated_text):
                loop_detected = True
                logger.warning("Loop detected, trying with different parameters")
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_RESPONCE_LENGTH,
                    use_cache=True,
                    temperature=0.25,
                    min_p=0.1,
                    top_p=0.85,
                )
                generated_text = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:],
                                                  skip_special_tokens=True).strip()
                if detect_loop(generated_text):
                    logger.warning("Loop detected again, returning empty graph")
                    return "{}", True
            return generated_text, loop_detected
        else:
            text_streamer = TextStreamer(tokenizer, skip_prompt=True)
            output_ids = model.generate(
                **inputs,
                streamer=text_streamer,
                max_new_tokens=MAX_RESPONCE_LENGTH,
                use_cache=True,
                temperature=0.2,
                min_p=0.1,
                top_p=0.8
            )
            prompt_length = inputs['input_ids'].shape[1]
            generated_text = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
            return generated_text.strip(), loop_detected
    except Exception as e:
        logger.error(f"Error generating extraction: {e}")
        return "{}", False


def create_or_update_attribute(tx, attr_data):
    """
    Create or update an attribute in Neo4j with embedding and extraction info.
    """
    # Check if we have an embedding to store
    has_embedding = "embedding" in attr_data and attr_data["embedding"] and len(attr_data["embedding"]) > 0

    if has_embedding:
        query = """
        MERGE (a:Attribute {name: $name})
        SET a.description = $description,
            a.possible_symbols = $possible_symbols,
            a.embedding = $embedding,
            a.during_extraction = $during_extraction
        RETURN a
        """
        tx.run(query,
               name=attr_data["name"],
               description=attr_data["description"],
               possible_symbols=attr_data.get("possible_symbols", []),
               embedding=attr_data.get("embedding", []),
               during_extraction=attr_data.get("during_extraction", False))
    else:
        query = """
        MERGE (a:Attribute {name: $name})
        SET a.description = $description,
            a.possible_symbols = $possible_symbols,
            a.during_extraction = $during_extraction
        RETURN a
        """
        tx.run(query,
               name=attr_data["name"],
               description=attr_data["description"],
               possible_symbols=attr_data.get("possible_symbols", []),
               during_extraction=attr_data.get("during_extraction", False))


def create_measurement(tx, product_key, attribute):
    """
    Create a measurement and link it to a product and attribute.
    """
    query = """
    MATCH (p:Product {name: $product_name, manufacturer: $manufacturer_name})
    MATCH (a:Attribute {name: $attribute_name})
    MERGE (p)-[:HAS_ATTRIBUTE]->(a)
    MERGE (m:Measurement {
        value: $value, 
        unit: $unit, 
        test_conditions: $test_conditions
    })<-[:HAS_MEASUREMENT]-(p)
    MERGE (m)-[:FOR_ATTRIBUTE]->(a)
    """
    tx.run(query,
           product_name=product_key["pn"],
           manufacturer_name=product_key["manufacturer"],
           attribute_name=attribute['name'],
           value=attribute['value'],
           unit=attribute['unit'],
           test_conditions=attribute['test_condition'])


def import_data_to_neo4j(driver, graph_data, product_key):
    """
    Import extracted data to Neo4j, handling embeddings and during_extraction flag.
    Filters out attributes that don't have any associated measurements.
    """
    try:
        # First identify which attributes have relationships with measurements
        attributes_with_measurements = set()
        for rel in graph_data.get("relationships", []):
            if rel["type"] == "FOR_ATTRIBUTE":
                attributes_with_measurements.add(rel["target"])

        logger.info(
            f"Found {len(attributes_with_measurements)} attributes with measurements out of {sum(1 for node in graph_data.get('nodes', []) if node['type'] == 'Attribute')} total attributes")
        # Write attributes
        with driver.session() as session:
            for node in graph_data.get("nodes", []):
                if node["type"] == "Attribute" and node["id"] in attributes_with_measurements:
                    properties = node.get("properties", {})

                    attr_data = {
                        "name": node["id"],
                        "description": properties.get("description", ""),
                        "possible_symbols": properties.get("possible_symbols", []),
                        "embedding": properties.get("embedding", []),
                        "during_extraction": properties.get("during_extraction", False)
                    }

                    session.execute_write(create_or_update_attribute, attr_data)
        # Write measurements
        with driver.session() as session:
            measurement_map = {}
            for node in graph_data.get("nodes", []):
                if node["type"] == "Measurement":
                    properties = node.get("properties", {})
                    measurement_map[node["id"]] = {
                        "value": properties.get("value", ""),
                        "unit": properties.get("unit", ""),
                        "test_conditions": properties.get("test_conditions", "")
                    }

            for rel in graph_data.get("relationships", []):
                if rel["type"] == "FOR_ATTRIBUTE":
                    measurement_id = rel["source"]
                    attribute_id = rel["target"]
                    if measurement_id in measurement_map:
                        measurement_data = {
                            "name": attribute_id,
                            "value": measurement_map[measurement_id].get("value", ""),
                            "unit": measurement_map[measurement_id].get("unit", ""),
                            "test_condition": measurement_map[measurement_id].get("test_conditions", "")
                        }

                        session.execute_write(create_measurement, product_key, measurement_data)
        return True
    except Exception as e:
        logger.error(f"Error importing data to Neo4j: {e}")
        return False


def extract_graph_from_image(model, tokenizer, image, product_data, use_loop_prevention=True, max_feedback_attempts=2):
    """
    Extracts a graph from an image using two prompts, validates the output,
    and uses a feedback loop to correct errors if necessary.
    """
    all_raw_responses = []
    any_loop_detected = False
    # First prompt attempt
    prompt_2 = create_prompt_image(product_data, prompt_num=2)
    conversation_2 = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_2}]}]
    response_2, loop_detected_2 = generate_extraction(model, tokenizer, conversation_2, image, use_loop_prevention)
    graph_2 = parse_graph(response_2, logger)
    all_raw_responses.append(response_2)
    any_loop_detected = any_loop_detected or loop_detected_2

    # If graph_2 is non-empty and passes validation, return it
    errors_2 = validate_graph(graph_2) if graph_2.nodes else ["Empty graph"]
    if graph_2.nodes and not errors_2:
        return graph_2, response_2, any_loop_detected

    # Try second prompt
    prompt_1 = create_prompt_image(product_data, prompt_num=1)
    conversation_1 = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_1}]}]
    response_1, loop_detected_1 = generate_extraction(model, tokenizer, conversation_1, image, use_loop_prevention)
    graph_1 = parse_graph(response_1, logger)
    all_raw_responses.append(response_1)
    any_loop_detected = any_loop_detected or loop_detected_1

    # If graph_1 is non-empty and passes validation, return it
    errors_1 = validate_graph(graph_1) if graph_1.nodes else ["Empty graph"]
    if graph_1.nodes and not errors_1:
        return graph_1, response_1, any_loop_detected

    # If both graphs are empty, return empty graph
    if not graph_2.nodes and not graph_1.nodes:
        return Graph(nodes=[], relationships=[]), "\n\n---\n\n".join(all_raw_responses), any_loop_detected

    # Choose the graph with fewer errors or the non-empty one
    if graph_2.nodes and (not graph_1.nodes or len(errors_2) <= len(errors_1)):
        graph = graph_2
        errors = errors_2
        conversation = conversation_2.copy()
        conversation.append({"role": "assistant", "content": response_2})
        current_response = response_2
    else:
        graph = graph_1
        errors = errors_1
        conversation = conversation_1.copy()
        conversation.append({"role": "assistant", "content": response_1})
        current_response = response_1

    # Feedback loop to correct errors
    for attempt in range(max_feedback_attempts):
        if not errors:
            return graph, current_response, any_loop_detected

        # Provide feedback to the model
        feedback = (
                "There are some errors in the provided graph:\n" +
                "\n".join(errors) +
                "\nPlease correct these errors and provide the updated graph."
        )

        # Add feedback to conversation
        conversation.append({"role": "user", "content": feedback})

        # Generate corrected response
        corrected_response, loop_detected = generate_extraction(model, tokenizer, conversation, image, use_loop_prevention)
        all_raw_responses.append(corrected_response)
        any_loop_detected = any_loop_detected or loop_detected
        conversation.append({"role": "assistant", "content": corrected_response})
        current_response = corrected_response

        # Parse and validate the corrected graph
        corrected_graph = parse_graph(corrected_response, logger)
        errors = validate_graph(corrected_graph)

        # Update the graph
        if not errors or (corrected_graph.nodes and len(errors) < len(validate_graph(graph))):
            graph = corrected_graph

    # After max attempts, return the best graph we have
    return graph, "\n\n---\n\n".join(all_raw_responses), any_loop_detected



def main():
    logger.info("Starting multimodal parameter extraction process")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        model, tokenizer = load_model()

        # Get products with chunked datasheets
        products = get_datasheet_manufacturer_dict(driver)
        logger.info(f"Found {len(products)} products with chunked datasheets to process")

        # Process each product
        for product_data in products:
            datasheet_url = product_data["datasheet_url"]
            manufacturer = product_data["manufacturer"]
            product_name = product_data["product_name"]
            category = product_data["category"]
            logger.info(f"Processing product: {product_name} from {manufacturer}, datasheet: {datasheet_url}")

            # Create a product key for database operations
            product_key = {
                "pn": product_name,
                "manufacturer": manufacturer
            }

            # Get images for this datasheet
            image_pages = get_images_for_datasheet(datasheet_url)
            logger.info(f"Retrieved {len(image_pages)} image pages for datasheet {datasheet_url}")

            matcher = AttributesMatcher(
                category,
                driver,
                similarity_mode="both",
                threshold=0.75
            )

            # Process each image
            successful_pages = 0
            for img, page_num in image_pages:
                try:
                    logger.info(f"Processing image for page {page_num}")

                    graph, raw_response_text, loop_detected = extract_graph_from_image(model, tokenizer, img, product_data)

                    if not graph.nodes:
                        logger.info(f"No parameters extracted from image on page {page_num}")
                        continue

                    final_graph = matcher.validate_extracted_attributes(graph.model_dump(), logger)

                    if import_data_to_neo4j(driver, final_graph, product_key):
                        successful_pages += 1
                        matcher.refresh_parameters()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error processing image for page {page_num}: {e}")

            # Mark datasheet as processed if at least one page was processed successfully
            if successful_pages > 0:
                mark_datasheet_product_as_processed(driver, product_key)
                logger.info(f"Successfully processed {successful_pages} pages for product {product_name}")
            else:
                logger.warning(f"No pages were successfully processed for product {product_name}")

    except Exception as e:
        logger.error(f"Error in main process: {e}")
    finally:
        driver.close()
        logger.info("Parameter extraction process completed")


if __name__ == "__main__":
    main()
