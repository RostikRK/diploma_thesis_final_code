import neo4j
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer
import os
import torch
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up start
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password1234"

MODEL_DIR = "saved_model"
# MODEL_NAME = "unsloth/phi-4-unsloth-bnb-4bit"
MODEL_NAME = "unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 6096
# Set up end


def load_or_create_model():
    """Load a saved model if it exists, otherwise create and save a new one"""
    model_path = os.path.join(MODEL_DIR, MODEL_NAME.split("/")[-1])

    if os.path.exists(model_path):
        logger.info(f"Loading pre-existing model from {model_path}")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=True
            )

            tokenizer = get_chat_template(
                tokenizer,
                # chat_template="phi-4", # change if switching the model
                chat_template="mistral",
            )

            FastLanguageModel.for_inference(model)
            logger.info("Successfully loaded saved model")
            return model, tokenizer
        except Exception as e:
            logger.warning(f"Error loading saved model: {str(e)}. Creating new model.")
    else:
        logger.info(f"No saved model found at {model_path}. Creating new model.")

    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info(f"Loading model from {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True
    )

    FastLanguageModel.for_inference(model)

    tokenizer = get_chat_template(
        tokenizer,
        # chat_template="phi-4",
        chat_template="mistral",
    )

    logger.info(f"Saving model to {model_path}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    return model, tokenizer


def get_attributes_without_description(tx):
    query = """
    MATCH (a:Attribute)
    WHERE a.description IS NULL OR a.description = ""
    RETURN a.name as name
    """
    result = tx.run(query)
    return [record["name"] for record in result]


def get_sample_measurements(tx, attribute_name, limit=3):
    query = """
    MATCH (m:Measurement)-[:FOR_ATTRIBUTE]->(a:Attribute {name: $attribute_name})
    RETURN m.value as value, m.unit as unit, m.test_conditions as test_conditions
    LIMIT $limit
    """
    result = tx.run(query, attribute_name=attribute_name, limit=limit)
    return [{"value": record["value"], "unit": record["unit"], "test_conditions": record["test_conditions"]}
            for record in result]


def get_products_with_attribute(tx, attribute_name, limit=3):
    query = """
    MATCH (p:Product)-[:HAS_ATTRIBUTE]->(a:Attribute {name: $attribute_name})
    RETURN p.name as name, p.manufacturer as manufacturer
    LIMIT $limit
    """
    result = tx.run(query, attribute_name=attribute_name, limit=limit)
    return [{"name": record["name"], "manufacturer": record["manufacturer"]}
            for record in result]


def create_description_prompt(attribute_name, measurements, products):
    system_prompt = (
        "You are a technical assistant specializing in electronic components and semiconductors. "
        "Your task is to generate accurate, concise descriptions for electronic component attributes."
    )

    # Format measurements for better context
    measurements_text = ""
    if measurements:
        measurements_text = "Example measurements for this attribute:\n" + "\n".join(
            f"- Value: {m['value']} {m['unit']}{' (under conditions: ' + m['test_conditions'] + ')' if m['test_conditions'] else ''}"
            for m in measurements
        ) + "\n"

    # Format products for better context
    products_text = ""
    if products:
        products_text = "This attribute is used by products such as:\n" + "\n".join(
            f"- {p['name']} from {p['manufacturer']}" for p in products) + "\n"

    prompt = (
        "You are a technical assistant specializing in electronic components and semiconductors. "
        "Your task is to generate accurate, concise descriptions for electronic component attributes.\n\n"
        f"### Attribute Name: {attribute_name}\n\n"
        f"{measurements_text}"
        f"{products_text}"
        "Your task is to write a brief, technical description for this attribute as it relates to "
        "electronic components or semiconductors. The description should be:\n"
        "1. Concise (20-30 words)\n"
        "2. Technical and precise, using electronic engineering terminology\n"
        "3. Structured to start with the full attribute name, followed by what it measures or represents, "
        "typical units, and, if space allows, its significance in component selection\n\n"
        "Example: 'Operating Temperature Range measures thermal limits, in °C, critical for reliability.'\n\n"
        "DO NOT include the word 'attribute' in your description. Return ONLY the description text, nothing else."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return messages

def generate_description(model, tokenizer, prompt):
    inputs = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    output_ids = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=120,
        use_cache=True,
        temperature=0.25,
        min_p=0.05,
        top_p=0.9
    )

    prompt_length = inputs.shape[1]
    generated_text = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
    return generated_text.strip()

def update_attribute_description(tx, attribute_name, description):
    query = """
    MATCH (a:Attribute {name: $attribute_name})
    SET a.description = $description
    RETURN a.name as name, a.description as description
    """
    result = tx.run(query, attribute_name=attribute_name, description=description)
    records = list(result)
    if records:
        return records[0]
    return None


def clean_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA memory cache cleared")


def main():
    logger.info("Starting attribute description generation process")
    driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    model, tokenizer = load_or_create_model()

    try:
        with driver.session() as session:
            # Get attributes without description
            attributes = session.execute_read(get_attributes_without_description)
            logger.info(f"Found {len(attributes)} attributes without descriptions")

            for i, attribute_name in enumerate(attributes):
                logger.info(f"Processing attribute {i + 1}/{len(attributes)}: {attribute_name}")

                # Get sample measurements and products for context
                measurements = session.execute_read(get_sample_measurements, attribute_name)
                products = session.execute_read(get_products_with_attribute, attribute_name)

                # Create prompt and generate description
                prompt = create_description_prompt(attribute_name, measurements, products)
                description = generate_description(model, tokenizer, prompt)


                if description and len(description) > 0:
                    if not (description.startswith('{') or description.startswith('[') or
                            description.startswith('<') or description.startswith('#')):

                        result = session.execute_write(update_attribute_description, attribute_name, description)
                        if result:
                            logger.info(
                                f"Updated attribute: {result['name']} with description: {result['description']}")
                        else:
                            logger.error(f"Failed to update attribute: {attribute_name}")
                    else:
                        logger.warning(f"Invalid description format for {attribute_name}: {description}")
                else:
                    logger.warning(f"Empty description generated for {attribute_name}")

                # Periodically clean up memory
                if (i + 1) % 10 == 0:
                    clean_cuda_memory()

                time.sleep(0.5)

    finally:
        driver.close()
        clean_cuda_memory()
        logger.info("Process completed")


if __name__ == "__main__":
    main()
