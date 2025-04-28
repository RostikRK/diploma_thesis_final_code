import os
import io
import json
import logging
import time
import requests
from PIL import Image
from tqdm import tqdm
from neo4j import GraphDatabase
from unsloth import FastVisionModel
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("datasheet_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up start
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password1234"

API_BASE_URL = "http://localhost:8082"

MODEL_NAME = "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 10096
# Set up end


def get_unvalidated_datasheets(driver):
    """Get datasheets that haven't been validated yet"""
    query = """
    MATCH (c:Category)<-[:BELONGS_TO]-(p:Product)
    MATCH (p:Product)-[:MANUFACTURED_BY]->(m:Manufacturer)
    OPTIONAL MATCH (p)-[:PACKAGED_IN]->(pkg:Package)
    MATCH (p)-[r:HAS_DATASHEET]->(d:Datasheet)
    WHERE r.image_validated IS NULL AND r.state="Chunked"
    RETURN DISTINCT d.name AS datasheet, m.name AS manufacturer, p.name AS product
    """
    datasheet_list = []
    with driver.session() as session:
        result = session.execute_read(lambda tx: tx.run(query).data())
        for record in result:
            datasheet_url = record["datasheet"]
            manufacturer_name = record["manufacturer"]
            product_name = record["product"]
            datasheet_list.append((datasheet_url, manufacturer_name, product_name))
    return datasheet_list


def get_images_for_datasheet(datasheet_url, start_page=0, max_pages=190):
    """
    Retrieves all images for a given datasheet URL by making GET requests to the API.
    """
    images = []
    for page_num in range(start_page, max_pages + 1):
        params = {
            "datasheet_url": datasheet_url,
            "page_num": str(page_num)
        }
        response = requests.get(f"{API_BASE_URL}/image", params=params)

        if response.status_code == 200:
            try:
                # Convert the response content (bytes) into a PIL Image
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


def mark_datasheet_as_validated(driver, datasheet_url):
    """Mark a datasheet as validated"""
    query = """
    MATCH (p:Product)-[r:HAS_DATASHEET]->(d:Datasheet {name: $datasheet_url})
    SET r.image_validated = true
    """
    with driver.session() as session:
        session.execute_write(lambda tx: tx.run(query, datasheet_url=datasheet_url))


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


def create_content_prompt(product_info):
    """Create a prompt for content validation"""
    manufacturer = product_info[1]
    product_name = product_info[2]
    package = product_info[3]
    category = product_info[4]

    prompt = (
        f"# Datasheet Content Analyzer\n\n"
        f"## Task\n"
        f"Analyze this datasheet page and determine if it contains any useful technical data (numeric parameters, tables with specifications, or electrical characteristics) that can be extracted for the product {product_name} manufactured by {manufacturer} in the {package} package. The product is a {category} component.\n\n"
        f"## Content Classification\n"
        f"Classify this page as either:\n"
        f"1. USEFUL:\n"
        f"    - Features page, characteristics page\n"
        f"    - Page contains parametric tables, part number code information\n"
        f"    - Page contains sections of text with specific numeric parameters (voltage, current, timing, temperature etc.)\n"
        f"2. NOT_USEFUL:\n"
        f"    - Page contains ONLY general information, package diagrams, marketing content\n"
        f"    - Page shows only pin configurations without measurable parameters\n"
        f"    - Table of contents page, revision history page, legal notices page\n\n"

        f"## Output Format\n"
        f"Return ONLY a JSON object with:\n"
        f"- 'classification': 'USEFUL' or 'NOT_USEFUL'\n"
        f"- 'reason': Brief explanation (max 50 words)\n"
        f"- 'parameters_present': List of parameter types detected (only if 'USEFUL')\n\n"

        f"Example output:\n"
        f"```json\n"
        f"{{\n"
        f"  \"classification\": \"USEFUL\",\n"
        f"  \"reason\": \"Contains electrical characteristics table with voltage and current specifications\",\n"
        f"  \"parameters_present\": [\"operating voltage\", \"output current\", \"switching frequency\"]\n"
        f"}}\n"
        f"```\n\n"

        f"OR\n\n"

        f"```json\n"
        f"{{\n"
        f"  \"classification\": \"NOT_USEFUL\",\n"
        f"  \"reason\": \"Page shows only package dimensions without measurable parameters\",\n"
        f"  \"parameters_present\": []\n"
        f"}}\n"
        f"```\n\n"

        f"Analyze the image and return your JSON classification:"
    )
    return prompt


def create_layout_prompt(product_info):
    """Create a prompt for layout validation"""
    manufacturer = product_info[1]
    product_name = product_info[2]

    prompt = (
        f"# Datasheet Layout Analyzer\n\n"
        f"## Task\n"
        f"Analyze the layout complexity of this datasheet page for the product {product_name} manufactured by {manufacturer}. Determine if the page needs advanced processing due to complex layout.\n\n"

        f"## Layout Classification\n"
        f"Classify the page layout as either:\n"
        f"1. COMPLEX_LAYOUT (needs better processing):\n"
        f"    - Page contains data tables with multiple rows and columns\n"
        f"    - Page is divided into multiple columns (text flows in columns)\n"
        f"    - Page has intricate diagrams with embedded text and parameters\n"
        f"2. SIMPLE_LAYOUT (standard processing is sufficient):\n"
        f"    - Page has primarily single-column text\n"
        f"    - Page has simple diagrams without complex text layout\n"
        f"    - Page has minimal or no tables\n\n"

        f"## Output Format\n"
        f"Return ONLY a JSON object with:\n"
        f"- 'classification': 'COMPLEX_LAYOUT' or 'SIMPLE_LAYOUT'\n"
        f"- 'reason': Brief explanation for the layout classification (max 50 words)\n\n"

        f"Example output:\n"
        f"```json\n"
        f"{{\n"
        f"  \"classification\": \"COMPLEX_LAYOUT\",\n"
        f"  \"reason\": \"Page contains a multi-column specification table with numerous parameters\"\n"
        f"}}\n"
        f"```\n\n"

        f"OR\n\n"

        f"```json\n"
        f"{{\n"
        f"  \"classification\": \"SIMPLE_LAYOUT\",\n"
        f"  \"reason\": \"Single column text with simple diagram\"\n"
        f"}}\n"
        f"```\n\n"

        f"Analyze the image and return your JSON classification:"
    )
    return prompt


def validate_content(model, tokenizer, image, product_info):
    """Validate the content of an image"""
    try:
        # Create prompt and conversation
        prompt = create_content_prompt(product_info)
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]

        input_text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

        # Generate the response
        output_ids = model.generate(
            **inputs,
            max_new_tokens=812,
            use_cache=True,
            temperature=0.2,
            top_p=0.9
        )

        # Decode the response
        prompt_length = inputs['input_ids'].shape[1]
        response = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True).strip()

        # Extract JSON from response
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            else:
                return {
                    "classification": "ERROR",
                    "reason": "Could not extract JSON from response",
                    "parameters_present": []
                }
        except json.JSONDecodeError:
            return {
                "classification": "ERROR",
                "reason": "Invalid JSON in response",
                "parameters_present": []
            }
    except Exception as e:
        return {
            "classification": "ERROR",
            "reason": f"Processing error: {str(e)}",
            "parameters_present": []
        }


def validate_layout(model, tokenizer, image, product_info):
    """Validate the layout of an image"""
    try:
        # Create prompt and conversation
        prompt = create_layout_prompt(product_info)
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]

        input_text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

        # Generate the response
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.2,
            top_p=0.9
        )

        # Decode the response
        prompt_length = inputs['input_ids'].shape[1]
        response = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True).strip()

        # Extract JSON from response
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            else:
                return {
                    "classification": "ERROR",
                    "reason": "Could not extract JSON from response"
                }
        except json.JSONDecodeError:
            return {
                "classification": "ERROR",
                "reason": "Invalid JSON in response"
            }
    except Exception as e:
        return {
            "classification": "ERROR",
            "reason": f"Processing error: {str(e)}"
        }


def save_info_json_via_api(datasheet_url, validation_results):
    """
    Save validation results to info.json via the API.
    """
    try:
        # Convert validation results to JSON string
        info_json = json.dumps(validation_results)

        # Send the info.json to the API
        response = requests.post(
            f"{API_BASE_URL}/info",
            data={
                "datasheet_url": datasheet_url,
                "info_json": info_json
            }
        )

        response.raise_for_status()

        logger.info(f"Successfully saved info.json for {datasheet_url} via API")
        return True
    except requests.RequestException as e:
        logger.error(f"Error saving info.json via API: {e}")
        return False


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        model, tokenizer = load_model()

        # Get unvalidated datasheets
        datasheets = get_unvalidated_datasheets(driver)
        logger.info(f"Found {len(datasheets)} datasheets to validate")

        # Process each datasheet
        for datasheet_info in tqdm(datasheets, desc="Processing datasheets"):
            datasheet_url = datasheet_info[0]
            logger.info(f"Processing datasheet: {datasheet_url}")

            # Get images for this datasheet
            images = get_images_for_datasheet(datasheet_url)
            logger.info(f"Retrieved {len(images)} images for datasheet: {datasheet_url}")

            if not images:
                logger.warning(f"No images found for datasheet: {datasheet_url}")
                continue

            validation_results = {
                "total_pages": len(images),
                "simple_layout_pages": [],
                "complex_layout_pages": [],
                "useful_pages": [],
                "not_useful_pages": [],
                "error_pages": [],
                "page_details": []
            }

            # Process each image and save the results
            for img, page_num in tqdm(images, desc="Validating pages"):
                content_result = validate_content(model, tokenizer, img, datasheet_info)
                layout_result = validate_layout(model, tokenizer, img, datasheet_info)

                page_result = {
                    "page_number": page_num,
                    "content_classification": content_result.get("classification", "ERROR"),
                    "content_reason": content_result.get("reason", "No reason provided"),
                    "parameters_present": content_result.get("parameters_present", []),
                    "layout_classification": layout_result.get("classification", "ERROR"),
                    "layout_reason": layout_result.get("reason", "No reason provided")
                }

                validation_results["page_details"].append(page_result)

                # Update the specific classification lists
                if page_result["content_classification"] == "USEFUL":
                    validation_results["useful_pages"].append(page_num)
                elif page_result["content_classification"] == "NOT_USEFUL":
                    validation_results["not_useful_pages"].append(page_num)

                if page_result["layout_classification"] == "SIMPLE_LAYOUT":
                    validation_results["simple_layout_pages"].append(page_num)
                elif page_result["layout_classification"] == "COMPLEX_LAYOUT":
                    validation_results["complex_layout_pages"].append(page_num)

                if (page_result["content_classification"] == "ERROR" or
                        page_result["layout_classification"] == "ERROR"):
                    validation_results["error_pages"].append(page_num)

                print(
                    f"Page {page_num}: Content: {page_result['content_classification']} | Layout: {page_result['layout_classification']}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                time.sleep(0.5)

            # Save the validation results via API
            if save_info_json_via_api(datasheet_url, validation_results):
                # Mark the datasheet as validated
                mark_datasheet_as_validated(driver, datasheet_url)
                logger.info(f"Datasheet validated and marked as such: {datasheet_url}")

                print(f"\n=== Validation Summary for {datasheet_url} ===")
                print(f"Total pages: {validation_results['total_pages']}")
                print(f"Useful pages: {len(validation_results['useful_pages'])}")
                print(f"Not useful pages: {len(validation_results['not_useful_pages'])}")
                print(f"Simple layout pages: {len(validation_results['simple_layout_pages'])}")
                print(f"Complex layout pages: {len(validation_results['complex_layout_pages'])}")
                print(f"Error pages: {len(validation_results['error_pages'])}")
                print("=====================================\n")

    except Exception as e:
        logger.error(f"Error in main process: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
