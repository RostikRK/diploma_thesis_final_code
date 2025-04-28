import os
import json
import logging
import time
import re
import glob
import io
import sys
import os
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch

from helpers.extraction_utils import (
    Graph, load_product_data, detect_loop, load_expected_result, create_prompt_image,
    parse_graph, validate_graph
)

from extraction_evaluation_utils import (
    calculate_json_accuracy, calculate_attribute_metrics_with_embeddings, calculate_measurement_metrics,
    calculate_measurement_metrics_with_combined_similarity, validate_json_structure
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multimodal_extraction_gemini.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# MODEL_NAME = "gemini-pro-vision"

# Set up start
gemini_api_key = ''
MODEL_NAME = "gemini-2.5-pro-preview-03-25"
MAX_SEQ_LENGTH = 25000
MAX_RESPONSE_LENGTH = 5100
# Set up end

def load_model(api_key: str):
    """Set up the Gemini client with the provided API key"""
    logger.info(f"Setting up Gemini client for {MODEL_NAME}")

    try:
        # Create the client with the API key
        client = genai.Client(api_key=api_key)
        logger.info(f"Gemini client configured successfully")
        return client
    except Exception as e:
        logger.error(f"Error setting up Gemini client: {e}")
        raise


def prepare_image_part(image: Image.Image) -> types.Part:
    """Convert PIL Image to the correct format for Gemini API"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


def generate_extraction(client, prompt, image, use_loop_prevention=True):
    """Generate parameter extraction from the prompt and image using Gemini"""
    try:
        image_part = prepare_image_part(image)

        # Set generation parameters
        config = types.GenerateContentConfig(
            temperature=0.6 if use_loop_prevention else 0.2,
            top_p=0.7 if use_loop_prevention else 0.8,
            top_k=40,
            max_output_tokens=MAX_RESPONSE_LENGTH,
        )

        if isinstance(prompt, list) and isinstance(prompt[0], dict) and 'role' in prompt[0]:
            chat = client.chats.create(model=MODEL_NAME)

            # Process all messages except the last one to build history
            for i, message in enumerate(prompt[:-1]):
                if message['role'] == 'user':
                    if isinstance(message['content'], list):
                        text_parts = [part['text'] for part in message['content'] if part['type'] == 'text']
                        chat.send_message(" ".join(text_parts))
                    else:
                        chat.send_message(message['content'])

            # Process the last message
            last_message = prompt[-1]
            if last_message['role'] == 'user':
                if isinstance(last_message['content'], list) and any(
                        p['type'] == 'image' for p in last_message['content']):
                    text_parts = [part['text'] for part in last_message['content'] if part['type'] == 'text']
                    text_content = " ".join(text_parts)
                    response = chat.send_message([text_content, image_part])
                else:
                    content = last_message['content']
                    if isinstance(content, list):
                        content = " ".join([part['text'] for part in content if part['type'] == 'text'])
                    response = chat.send_message(content)
            else:
                response = chat.send_message("Please continue with the extraction.")
        else:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, image_part],
                config=config
            )

        generated_text = response.text

        loop_detected = False
        if use_loop_prevention and detect_loop(generated_text):
            loop_detected = True
            logger.warning("Loop detected, trying with different parameters")

            # Adjust generation parameters for retry
            config = types.GenerateContentConfig(
                temperature=0.25,
                top_p=0.85,
                top_k=40,
                max_output_tokens=MAX_RESPONSE_LENGTH,
            )

            # Retry with the adjusted parameters
            if isinstance(prompt, list) and isinstance(prompt[0], dict) and 'role' in prompt[0]:
                chat = client.chats.create(model=MODEL_NAME)
                for i, message in enumerate(prompt[:-1]):
                    if message['role'] == 'user':
                        if isinstance(message['content'], list):
                            text_parts = [part['text'] for part in message['content'] if part['type'] == 'text']
                            chat.send_message(" ".join(text_parts))
                        else:
                            chat.send_message(message['content'])

                # Process the last message
                last_message = prompt[-1]
                if last_message['role'] == 'user':
                    if isinstance(last_message['content'], list) and any(
                            p['type'] == 'image' for p in last_message['content']):
                        text_parts = [part['text'] for part in last_message['content'] if part['type'] == 'text']
                        text_content = " ".join(text_parts)
                        response = chat.send_message([text_content, image_part])
                    else:
                        content = last_message['content']
                        if isinstance(content, list):
                            content = " ".join([part['text'] for part in content if part['type'] == 'text'])
                        response = chat.send_message([text_content, image_part])
            else:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[prompt, image_part],
                    config=config
                )

            generated_text = response.text

            if detect_loop(generated_text):
                logger.warning("Loop detected again, returning empty graph")
                return "{}", True

        return generated_text, loop_detected

    except Exception as e:
        logger.error(f"Error generating extraction: {e}")
        return "{}", False


def extract_graph_from_image(client, image, product_data, use_loop_prevention=True, max_feedback_attempts=2):
    """
    Extracts a graph from an image using two prompts, validates the output,
    and uses a feedback loop to correct errors if necessary.
    """
    all_raw_responses = []
    any_loop_detected = False

    # First prompt attempt
    prompt_2 = create_prompt_image(product_data, prompt_num=2)
    conversation_2 = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_2}]}]
    response_2, loop_detected_2 = generate_extraction(client, conversation_2, image, use_loop_prevention)
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
    response_1, loop_detected_1 = generate_extraction(client, conversation_1, image, use_loop_prevention)
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
        corrected_response, loop_detected = generate_extraction(client, conversation, image, use_loop_prevention)
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


def run_tests(test_dir: str, output_dir: str, api_key: str = None):
    """Run tests on all test files in the directory"""
    os.makedirs(output_dir, exist_ok=True)

    # Find all product data files
    product_files = glob.glob(os.path.join(test_dir, "product_data_*.txt"))
    product_files.sort()


    client = load_model(api_key)

    print("Loading sentence transformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model.to(device)
    print(f"Model loaded and moved to {device}")

    results = []
    total_start_time = time.time()

    for i, product_file in enumerate(product_files):
        try:
            test_num = re.search(r"product_data_(\d+)\.txt", product_file).group(1)
            expected_file = os.path.join(test_dir, f"result_{test_num}.json")

            if not os.path.exists(expected_file):
                logger.warning(f"Expected result file not found: {expected_file}")
                continue

            logger.info(f"Processing test case {test_num} ({i + 1}/{len(product_files)})")

            # Load product data and expected result
            product_data = load_product_data(product_file)
            expected_result = load_expected_result(expected_file)

            if not product_data or not expected_result:
                logger.warning(f"Skipping test case {test_num} due to missing data")
                continue

            # Get image for the specified page
            start_time = time.time()
            image_file = os.path.join(test_dir, f"image_{test_num}.png")
            image = Image.open(image_file).convert('RGB')

            if not image:
                logger.warning(f"Failed to get image for test case {test_num}")
                continue

            # Extract parameters from image using Gemini
            extracted_graph, raw_responses, loops_detected = extract_graph_from_image(
                client, image, product_data
            )
            extracted_json = extracted_graph.model_dump() if extracted_graph else {}


            json_accuracy = calculate_json_accuracy(extracted_json, expected_result)
            attr_metrics = calculate_attribute_metrics_with_embeddings(extracted_json, expected_result, embedding_model)
            meas_metrics = calculate_measurement_metrics(extracted_json, expected_result)
            meas_attr_sim_metrics = calculate_measurement_metrics_with_combined_similarity(
                extracted_json, expected_result, embedding_model
            )
            struct_metrics = validate_json_structure(extracted_json, expected_result)

            processing_time = time.time() - start_time

            individual_result = {
                "test_case": test_num,
                "product_name": product_data['product_name'],
                "extracted_json": extracted_json,
                "expected_json": expected_result,
                "raw_responses": raw_responses,
                "loops_detected": loops_detected,
                "json_accuracy": json_accuracy,
                "attr_accuracy": attr_metrics["accuracy"],
                "attr_precision": attr_metrics["precision"],
                "attr_recall": attr_metrics["recall"],
                "attr_f1": attr_metrics["f1"],
                "attr_combined": attr_metrics["combined_similarity"],
                "meas_accuracy": meas_metrics["accuracy"],
                "meas_precision": meas_metrics["precision"],
                "meas_recall": meas_metrics["recall"],
                "meas_f1": meas_metrics["f1"],
                "meas_jaro_winkler": meas_metrics["jaro_winkler"],
                "meas_attr_sim_accuracy": meas_attr_sim_metrics["accuracy"],
                "meas_attr_sim_precision": meas_attr_sim_metrics["precision"],
                "meas_attr_sim_recall": meas_attr_sim_metrics["recall"],
                "meas_attr_sim_f1": meas_attr_sim_metrics["f1"],
                "structure_score": struct_metrics["structure_score"],
                "empty_state_match": struct_metrics["empty_state_match"],
                "has_attribute_nodes": struct_metrics["has_attribute_nodes"],
                "has_measurement_nodes": struct_metrics["has_measurement_nodes"],
                "has_relationships": struct_metrics["has_relationships"],
                "false_positive_empty": struct_metrics["false_positive_empty"],
                "false_negative_empty": struct_metrics["false_negative_empty"],
                "processing_time": processing_time
            }

            # Save to results list (aggregation data)
            results.append({
                "test_case": test_num,
                "product_name": product_data['product_name'],
                "json_accuracy": json_accuracy,
                "loops_detected": loops_detected,
                "attr_accuracy": attr_metrics["accuracy"],
                "attr_precision": attr_metrics["precision"],
                "attr_recall": attr_metrics["recall"],
                "attr_f1": attr_metrics["f1"],
                "attr_combined": attr_metrics["combined_similarity"],
                "meas_accuracy": meas_metrics["accuracy"],
                "meas_precision": meas_metrics["precision"],
                "meas_recall": meas_metrics["recall"],
                "meas_f1": meas_metrics["f1"],
                "meas_jaro_winkler": meas_metrics["jaro_winkler"],
                "meas_attr_sim_accuracy": meas_attr_sim_metrics["accuracy"],
                "meas_attr_sim_precision": meas_attr_sim_metrics["precision"],
                "meas_attr_sim_recall": meas_attr_sim_metrics["recall"],
                "meas_attr_sim_f1": meas_attr_sim_metrics["f1"],
                "structure_score": struct_metrics["structure_score"],
                "processing_time": processing_time
            })

            with open(os.path.join(output_dir, f"test_result_{test_num}.json"), 'w') as f:
                json.dump(individual_result, f, indent=2)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing test case {product_file}: {e}")

    total_time = time.time() - total_start_time

    results_df = pd.DataFrame(results)

    # Calculate overall metrics
    overall_metrics = {
        "total_tests": len(results),
        "total_time": total_time,
        "avg_processing_time": results_df['processing_time'].mean() if len(results) > 0 else 0,
        "json_accuracy_avg": results_df['json_accuracy'].mean() if len(results) > 0 else 0,
        "loops_detected_count": sum(1 for r in results if r.get('loops_detected', False)),
        "attr_accuracy_avg": results_df['attr_accuracy'].mean() if len(results) > 0 else 0,
        "attr_precision_avg": results_df['attr_precision'].mean() if len(results) > 0 else 0,
        "attr_recall_avg": results_df['attr_recall'].mean() if len(results) > 0 else 0,
        "attr_f1_avg": results_df['attr_f1'].mean() if len(results) > 0 else 0,
        "attr_combined_avg": results_df['attr_combined'].mean() if len(results) > 0 else 0,
        "meas_accuracy_avg": results_df['meas_accuracy'].mean() if len(results) > 0 else 0,
        "meas_precision_avg": results_df['meas_precision'].mean() if len(results) > 0 else 0,
        "meas_recall_avg": results_df['meas_recall'].mean() if len(results) > 0 else 0,
        "meas_f1_avg": results_df['meas_f1'].mean() if len(results) > 0 else 0,
        "meas_jaro_winkler_avg": results_df['meas_jaro_winkler'].mean() if len(results) > 0 else 0,
        "meas_attr_sim_accuracy_avg": results_df['meas_attr_sim_accuracy'].mean() if len(results) > 0 else 0,
        "meas_attr_sim_precision_avg": results_df['meas_attr_sim_precision'].mean() if len(results) > 0 else 0,
        "meas_attr_sim_recall_avg": results_df['meas_attr_sim_recall'].mean() if len(results) > 0 else 0,
        "meas_attr_sim_f1_avg": results_df['meas_attr_sim_f1'].mean() if len(results) > 0 else 0,
        "structure_score_avg": results_df['structure_score'].mean() if len(results) > 0 else 0
    }

    with open(os.path.join(output_dir, 'overall_metrics.json'), 'w') as f:
        json.dump(overall_metrics, f, indent=2)

    results_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)

    logger.info(f"Testing completed. Processed {len(results)} test cases in {total_time:.2f} seconds")
    logger.info(f"Overall JSON accuracy: {overall_metrics['json_accuracy_avg']:.4f}")
    logger.info(f"Overall attribute F1 score: {overall_metrics['attr_f1_avg']:.4f}")
    logger.info(f"Overall attribute Jaro-Winkler: {overall_metrics['attr_combined_avg']:.4f}")
    logger.info(f"Overall measurement F1 score: {overall_metrics['meas_f1_avg']:.4f}")
    logger.info(f"Overall measurement Jaro-Winkler: {overall_metrics['meas_jaro_winkler_avg']:.4f}")
    logger.info(f"Overall structure score: {overall_metrics['structure_score_avg']:.4f}")

    return overall_metrics




if __name__ == "__main__":

    test_dir= '../../json_test_df'

    output_dir = 'results_gemini_new'

    run_tests(test_dir, output_dir, gemini_api_key)
