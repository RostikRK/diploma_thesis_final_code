import neo4j
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer
import os
import torch
import logging
import time
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up start
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password1234"

MODEL_DIR = "saved_model"
MODEL_NAME = "unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 6096
# Set up end


embedding_model = SentenceTransformer('all-mpnet-base-v2')

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
                chat_template="mistral", #change if switch the model
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
        chat_template="mistral",
    )

    logger.info(f"Saving model to {model_path}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    return model, tokenizer


def get_packages_without_description(tx):
    query = """
    MATCH (p:Package)
    WHERE p.description IS NULL OR p.description = ""
    RETURN p.name as name, p.standardized_package_names as standardized_names
    """
    result = tx.run(query)
    return [{"name": record["name"], "standardized_names": record["standardized_names"]} for record in result]


def get_products_with_package(tx, package_name, limit=3):
    query = """
    MATCH (p:Product)-[:PACKAGED_IN]->(pkg:Package {name: $package_name})
    RETURN p.name as name, p.manufacturer as manufacturer
    LIMIT $limit
    """
    result = tx.run(query, package_name=package_name, limit=limit)
    return [{"name": record["name"], "manufacturer": record["manufacturer"]} for record in result]


def create_package_description_prompt(package_name, standardized_names, products):
    system_prompt = (
        "You are a technical assistant specializing in electronic components and packaging. "
        "Your task is to generate an accurate, concise description for an electronic component package."
    )

    # Format standardized package names for context
    standardized_text = ""
    if standardized_names:
        standardized_text = "Standardized package names:\n" + "\n".join(
            f"- {name}" for name in standardized_names) + "\n"

    # Format products using this package for extra context
    products_text = ""
    if products:
        products_text = "This package is used by products such as:\n" + "\n".join(
            f"- {p['name']} from {p['manufacturer']}" for p in products) + "\n"

    prompt = (
        f"### Package Name: {package_name}\n\n"
        f"{standardized_text}"
        f"{products_text}"
        "Your task is to write a brief, technical description for this electronic component package. "
        "The description should be:\n"
        "1. Concise (20-30 words)\n"
        "2. Technical and precise, using industry-specific terminology\n"
        "3. Structured to start with the full package name, followed by its mounting type (e.g., surface-mount, through-hole), "
        " number of pins, material (e.g., plastic, ceramic), and, if space allows, some package features, common applications\n\n"
        "Example: 'SOT-23, a surface-mount package, 3 pins, plastic, used for transistors.'\n\n"
        "Return ONLY the description text, nothing else."
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

def update_package_description(tx, package_name, description, embedding):
    query = """
    MATCH (p:Package {name: $package_name})
    SET p.description = $description,
        p.embedding = $embedding
    RETURN p.name as name, p.description as description
    """
    result = tx.run(query, package_name=package_name, description=description, embedding=embedding)
    records = list(result)
    if records:
        return records[0]
    return None


def clean_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA memory cache cleared")


def main():
    logger.info("Starting package description generation process")

    # Load the language model and tokenizer
    model, tokenizer = load_or_create_model()

    driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            # Retrieve packages that need a description
            packages = session.execute_read(get_packages_without_description)
            logger.info(f"Found {len(packages)} packages without descriptions")

            for i, package in enumerate(packages):
                package_name = package["name"]
                standardized_names = package.get("standardized_names") or []
                logger.info(f"Processing package {i + 1}/{len(packages)}: {package_name}")

                # Get sample measurements and products for context
                products = session.execute_read(get_products_with_package, package_name)

                # Create the prompt and generate description
                prompt = create_package_description_prompt(package_name, standardized_names, products)
                description = generate_description(model, tokenizer, prompt)

                if description and len(description) > 0:
                    if not (description.startswith('{') or description.startswith('[') or
                            description.startswith('<') or description.startswith('#')):

                        # generate embedding for new description
                        embedding = embedding_model.encode(description).tolist()

                        result = session.execute_write(update_package_description, package_name, description, embedding)
                        if result:
                            logger.info(f"Updated package: {result['name']} with description: {result['description']}")
                        else:
                            logger.error(f"Failed to update package: {package_name}")
                    else:
                        logger.warning(f"Invalid description format for {package_name}: {description}")
                else:
                    logger.warning(f"Empty description generated for {package_name}")

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
