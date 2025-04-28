import json
import torch
from typing import Dict, List, Any, Optional, Tuple
from jellyfish import jaro_winkler_similarity


def calculate_json_accuracy(extracted_json: Dict, expected_json: Dict) -> float:
    """Calculate JSON accuracy - exact match"""
    if not extracted_json or not expected_json:
        return 0.0

    if json.dumps(extracted_json, sort_keys=True) == json.dumps(expected_json, sort_keys=True):
        return 1.0
    return 0.0

def jaro_winkler_similarity_threshold(s1: str, s2: str, threshold: float = 0.5) -> float:
    """
    Calculate Jaro-Winkler similarity between two strings with a threshold.
    Similarities below the threshold are rounded to 0.
    """
    # Handle edge cases
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    s1 = str(s1) if not isinstance(s1, str) else s1
    s2 = str(s2) if not isinstance(s2, str) else s2

    similarity = jaro_winkler_similarity(s1, s2)
    return similarity if similarity >= threshold else 0.0

def extract_attribute_ids(graph_data: Dict) -> set:
    """Extract attribute IDs from a graph"""
    attribute_ids = set()
    for node in graph_data.get("nodes", []):
        if node.get("type") == "Attribute":
            attribute_ids.add(node.get("id", "").lower())
    return attribute_ids

def extract_measurement_values(graph_data: Dict) -> Dict[str, List[Dict]]:
    """
    Extract measurement values from a graph
    Returns a dictionary mapping attribute IDs to lists of measurement data
    """
    measurements = {}

    # Create a mapping of measurement IDs to their data
    measurement_map = {}
    for node in graph_data.get("nodes", []):
        if node.get("type") == "Measurement":
            measurement_map[node.get("id")] = {
                "value": node.get("properties", {}).get("value"),
                "unit": node.get("properties", {}).get("unit"),
                "test_conditions": node.get("properties", {}).get("test_conditions", "")
            }

    # Map measurements to attributes using relationships
    for rel in graph_data.get("relationships", []):
        if rel.get("type") == "FOR_ATTRIBUTE":
            source = rel.get("source")
            target = rel.get("target")

            if source in measurement_map and target:
                if target not in measurements:
                    measurements[target] = []

                measurements[target].append(measurement_map[source])

    return measurements


def calculate_combined_similarity(s1: str, s2: str, s1_desc: Optional[str] = None,
                                  s2_desc: Optional[str] = None, jw_weight: float = 0.5,
                                  model=None, threshold: float = 0.5) -> float:
    """
    Calculate similarity using both Jaro-Winkler and
    embeddings.
    """
    # Handle edge cases
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    s1 = str(s1) if not isinstance(s1, str) else s1
    s2 = str(s2) if not isinstance(s2, str) else s2

    jw_similarity = jaro_winkler_similarity(s1, s2)

    # If we don't have descriptions or model, just return Jaro-Winkler
    if s1_desc is None or s2_desc is None or model is None:
        return jw_similarity if jw_similarity >= threshold else 0.0

    # Ensure we have string descriptions
    s1_desc = str(s1_desc) if not isinstance(s1_desc, str) else s1_desc
    s2_desc = str(s2_desc) if not isinstance(s2_desc, str) else s2_desc

    # Calculate descriptions embedding similarity
    embedding1 = model.encode(s1_desc, convert_to_tensor=True)
    embedding2 = model.encode(s2_desc, convert_to_tensor=True)
    embedding_similarity = torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0),
                                                                 dim=1).item()

    # Combine similarities
    combined_similarity = jw_weight * jw_similarity + (1 - jw_weight) * embedding_similarity

    return combined_similarity if combined_similarity >= threshold else 0.0

def calculate_attribute_metrics(extracted_json: Dict, expected_json: Dict) -> Dict[str, float]:
    """Calculate metrics for attributes with Jaro-Winkler similarity and threshold"""
    extracted_attrs = extract_attribute_ids(extracted_json)
    expected_attrs = extract_attribute_ids(expected_json)

    if not expected_attrs and not extracted_attrs:
        return {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "jaro_winkler": 1.0
        }

    if not expected_attrs:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "jaro_winkler": 0.0
        }


    true_positives = len(extracted_attrs.intersection(expected_attrs))
    false_positives = len(extracted_attrs - expected_attrs)
    false_negatives = len(expected_attrs - extracted_attrs)

    accuracy = true_positives / (true_positives + false_positives + false_negatives) if (
                                                                                                    true_positives + false_positives + false_negatives) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    jw_similarities = []

    for expected_attr in expected_attrs:
        best_similarity = 0.0
        for extracted_attr in extracted_attrs:
            similarity = jaro_winkler_similarity_threshold(expected_attr.lower(), extracted_attr.lower())
            best_similarity = max(best_similarity, similarity)
        jw_similarities.append(best_similarity)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaro_winkler": sum(jw_similarities) / len(jw_similarities) if jw_similarities else 0.0
    }

def calculate_attribute_metrics_with_embeddings(extracted_json: Dict, expected_json: Dict,
                                                embedding_model=None, jw_weight: float = 0.5,
                                                threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate metrics for attributes with combined similarity (Jaro-Winkler + embeddings)
    """
    # Create maps of attribute to descriptions
    extracted_attr_descriptions = {}
    for node in extracted_json.get("nodes", []):
        if node.get("type") == "Attribute":
            attr_id = node.get("id", "").lower()
            extracted_attr_descriptions[attr_id] = node.get("properties", {}).get("description", "")

    expected_attr_descriptions = {}
    for node in expected_json.get("nodes", []):
        if node.get("type") == "Attribute":
            attr_id = node.get("id", "").lower()
            expected_attr_descriptions[attr_id] = node.get("properties", {}).get("description", "")

    extracted_attrs = set(extracted_attr_descriptions.keys())
    expected_attrs = set(expected_attr_descriptions.keys())

    if not expected_attrs and not extracted_attrs:
        return {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "combined_similarity": 1.0
        }

    if not expected_attrs:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "combined_similarity": 0.0
        }


    true_positives = len(extracted_attrs.intersection(expected_attrs))
    false_positives = len(extracted_attrs - expected_attrs)
    false_negatives = len(expected_attrs - extracted_attrs)


    accuracy = true_positives / (true_positives + false_positives + false_negatives) if (
                                                                                                    true_positives + false_positives + false_negatives) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate combined similarity for each expected attribute
    combined_similarities = []

    for expected_attr in expected_attrs:
        expected_desc = expected_attr_descriptions.get(expected_attr, "")
        best_similarity = 0.0

        for extracted_attr in extracted_attrs:
            extracted_desc = extracted_attr_descriptions.get(extracted_attr, "")

            similarity = calculate_combined_similarity(
                expected_attr,
                extracted_attr,
                expected_desc,
                extracted_desc,
                jw_weight=jw_weight,
                model=embedding_model,
                threshold=threshold
            )

            best_similarity = max(best_similarity, similarity)

        combined_similarities.append(best_similarity)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "combined_similarity": sum(combined_similarities) / len(combined_similarities) if combined_similarities else 0.0
    }

def calculate_measurement_metrics_with_combined_similarity(
        extracted_json: Dict, expected_json: Dict,
        embedding_model=None, jw_weight: float = 0.5,
        threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate metrics for measurements using combined attribute similarity.
    This approach first matches attributes using combined similarity,
    then compares measurements within matched attributes.
    Uses a greedy approach to ensure one-to-one attribute matching.
    """
    extracted_measurements = extract_measurement_values(extracted_json)
    expected_measurements = extract_measurement_values(expected_json)

    expected_empty = not expected_json.get("nodes", []) and not expected_json.get("relationships", [])

    if expected_empty and extracted_measurements:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    # Create maps of attribute IDs to descriptions
    extracted_attr_descriptions = {}
    for node in extracted_json.get("nodes", []):
        if node.get("type") == "Attribute":
            attr_id = node.get("id", "").lower()
            extracted_attr_descriptions[attr_id] = node.get("properties", {}).get("description", "")

    expected_attr_descriptions = {}
    for node in expected_json.get("nodes", []):
        if node.get("type") == "Attribute":
            attr_id = node.get("id", "").lower()
            expected_attr_descriptions[attr_id] = node.get("properties", {}).get("description", "")

    # Get all attribute IDs
    extracted_attrs = set(extracted_attr_descriptions.keys())
    expected_attrs = set(expected_attr_descriptions.keys())

    if not expected_measurements and not extracted_measurements:
        return {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0
        }

    if expected_measurements and not extracted_measurements:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    # Calculate all pairwise similarities between expected and extracted attributes
    similarity_matrix = []
    for expected_attr in expected_attrs:
        expected_desc = expected_attr_descriptions.get(expected_attr, "")
        row = []

        for extracted_attr in extracted_attrs:
            extracted_desc = extracted_attr_descriptions.get(extracted_attr, "")

            similarity = calculate_combined_similarity(
                expected_attr,
                extracted_attr,
                expected_desc,
                extracted_desc,
                jw_weight=jw_weight,
                model=embedding_model,
                threshold=threshold
            )

            row.append((extracted_attr, similarity))

        similarity_matrix.append((expected_attr, row))

    # Use a greedy algorithm to find the best one-to-one matching
    attr_similarity_map = {}
    used_extracted_attrs = set()

    # Sort expected attributes by their highest similarity score
    sorted_expected = sorted(
        similarity_matrix,
        key=lambda x: max([sim for _, sim in x[1]]) if x[1] else 0,
        reverse=True
    )

    for expected_attr, similarities in sorted_expected:
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

        # Find the best unused extracted attribute
        for extracted_attr, similarity in sorted_similarities:
            if similarity >= threshold and extracted_attr not in used_extracted_attrs:
                attr_similarity_map[expected_attr] = (extracted_attr, similarity)
                used_extracted_attrs.add(extracted_attr)
                break

    true_positives = 0
    extracted_total = 0
    expected_total = 0

    # Count total measurements
    for _, measurements in expected_measurements.items():
        expected_total += len(measurements)

    for _, measurements in extracted_measurements.items():
        extracted_total += len(measurements)

    for expected_attr, expected_list in expected_measurements.items():
        if expected_attr in attr_similarity_map:
            extracted_attr, _ = attr_similarity_map[expected_attr]

            extracted_list = extracted_measurements.get(extracted_attr, [])

            # Mark which measurements have been matched (to avoid double counting)
            matched_extracted = [False] * len(extracted_list)

            for expected_m in expected_list:
                found_match = False

                # Try to find a match in the extracted measurements
                for i, extracted_m in enumerate(extracted_list):
                    if not matched_extracted[i] and measurement_match(expected_m, extracted_m):
                        matched_extracted[i] = True
                        found_match = True
                        true_positives += 1
                        break

    false_positives = extracted_total - true_positives
    false_negatives = expected_total - true_positives

    precision = true_positives / extracted_total if extracted_total > 0 else 0
    recall = true_positives / expected_total if expected_total > 0 else 0
    total = true_positives + false_positives + false_negatives

    accuracy = true_positives / total if total > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def measurement_match(m1: Dict, m2: Dict, just_values=True) -> bool:
    """
    Check if two measurements match
    """
    if just_values:
        if str(m1.get("value")) == str(m2.get("value")):
            return True
    else:
        if str(m1.get("value")) == str(m2.get("value")) and m1.get("unit") == m2.get("unit"):
            return True
    return False


def calculate_measurement_metrics_with_attribute_similarity(extracted_json: Dict, expected_json: Dict) -> Dict[
    str, float]:
    """
    Calculate metrics for measurements using attribute similarity as a bridge.
    """
    extracted_measurements = extract_measurement_values(extracted_json)
    expected_measurements = extract_measurement_values(expected_json)

    expected_empty = not expected_json.get("nodes", []) and not expected_json.get("relationships", [])

    if expected_empty and extracted_measurements:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    extracted_attrs = extract_attribute_ids(extracted_json)
    expected_attrs = extract_attribute_ids(expected_json)

    if not expected_measurements and not extracted_measurements:
        return {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0
        }

    if expected_measurements and not extracted_measurements:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    # Build similarity map between expected and extracted attributes
    attr_similarity_map = {}
    for expected_attr in expected_attrs:
        best_match = None
        best_similarity = 0.0
        for extracted_attr in extracted_attrs:
            similarity = jaro_winkler_similarity_threshold(expected_attr.lower(), extracted_attr.lower())
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = extracted_attr
        if best_similarity > 0:
            attr_similarity_map[expected_attr] = (best_match, best_similarity)

    true_positives = 0
    extracted_total = 0
    expected_total = 0

    for _, measurements in expected_measurements.items():
        expected_total += len(measurements)

    for _, measurements in extracted_measurements.items():
        extracted_total += len(measurements)

    for expected_attr, expected_list in expected_measurements.items():
        if expected_attr in attr_similarity_map:
            extracted_attr, _ = attr_similarity_map[expected_attr]

            extracted_list = extracted_measurements.get(extracted_attr, [])

            # Mark which measurements have been matched (to avoid double counting)
            matched_extracted = [False] * len(extracted_list)

            for expected_m in expected_list:
                found_match = False

                # Try to find a match in the extracted measurements
                for i, extracted_m in enumerate(extracted_list):
                    if not matched_extracted[i] and measurement_match(expected_m, extracted_m):
                        matched_extracted[i] = True
                        found_match = True
                        true_positives += 1
                        break

    false_positives = extracted_total - true_positives
    false_negatives = expected_total - true_positives

    precision = true_positives / extracted_total if extracted_total > 0 else 0
    recall = true_positives / expected_total if expected_total > 0 else 0

    total = true_positives + false_positives + false_negatives

    accuracy = true_positives / total if total > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def calculate_measurement_metrics(extracted_json: Dict, expected_json: Dict) -> Dict[str, float]:
    """Calculate metrics for measurements with Jaro-Winkler similarity and threshold"""
    extracted_measurements = extract_measurement_values(extracted_json)
    expected_measurements = extract_measurement_values(expected_json)

    expected_empty = not expected_json.get("nodes", []) and not expected_json.get("relationships", [])

    # If expected is empty but we extracted measurements, scores should be 0
    if expected_empty and extracted_measurements:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "jaro_winkler": 0.0
        }

    # If both expected and extracted are empty, perfect score
    if not expected_measurements and not extracted_measurements:
        return {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "jaro_winkler": 1.0
        }

    # If expected has measurements but extracted is empty, zero score
    if expected_measurements and not extracted_measurements:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "jaro_winkler": 0.0
        }

    true_positives = 0
    false_positives = 0

    for attr_id, extracted_list in extracted_measurements.items():
        if attr_id in expected_measurements:
            expected_list = expected_measurements[attr_id]

            for extracted_m in extracted_list:
                found_match = False
                for expected_m in expected_list:
                    if measurement_match(extracted_m, expected_m):
                        found_match = True
                        break

                if found_match:
                    true_positives += 1
                else:
                    false_positives += 1
        else:
            false_positives += len(extracted_list)

    false_negatives = 0
    for attr_id, expected_list in expected_measurements.items():
        if attr_id in extracted_measurements:
            extracted_list = extracted_measurements[attr_id]

            for expected_m in expected_list:
                found_match = False
                for extracted_m in extracted_list:
                    if measurement_match(expected_m, extracted_m):
                        found_match = True
                        break

                if not found_match:
                    false_negatives += 1
        else:
            false_negatives += len(expected_list)

    # Calculate traditional metrics
    accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    jw_similarities = []

    for attr_id, expected_list in expected_measurements.items():
        for expected_m in expected_list:
            best_similarity = 0.0

            if attr_id in extracted_measurements:
                for extracted_m in extracted_measurements[attr_id]:
                    expected_str = f"{expected_m.get('value')}_{expected_m.get('unit')}"
                    extracted_str = f"{extracted_m.get('value')}_{extracted_m.get('unit')}"
                    similarity = jaro_winkler_similarity_threshold(expected_str, extracted_str)
                    best_similarity = max(best_similarity, similarity)

            jw_similarities.append(best_similarity)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaro_winkler": sum(jw_similarities) / len(jw_similarities) if jw_similarities else 0.0
    }


def validate_json_structure(extracted_json: Dict, expected_json: Dict) -> Dict[str, float]:
    """Compare structural aspects of the extracted and expected JSON"""
    results = {}

    expected_empty = not expected_json.get("nodes", []) and not expected_json.get("relationships", [])
    extracted_empty = not extracted_json.get("nodes", []) and not extracted_json.get("relationships", [])

    results["empty_state_match"] = 1.0 if expected_empty == extracted_empty else 0.0

    # False positive check
    results["false_positive_empty"] = 1.0 if expected_empty and not extracted_empty else 0.0

    # False negative check
    results["false_negative_empty"] = 1.0 if not expected_empty and extracted_empty else 0.0

    expected_node_types = {node.get("type") for node in expected_json.get("nodes", [])}
    extracted_node_types = {node.get("type") for node in extracted_json.get("nodes", [])}

    results["has_attribute_nodes"] = 1.0 if ("Attribute" in expected_node_types) == (
                "Attribute" in extracted_node_types) else 0.0
    results["has_measurement_nodes"] = 1.0 if ("Measurement" in expected_node_types) == (
                "Measurement" in extracted_node_types) else 0.0

    expected_has_relations = bool(expected_json.get("relationships", []))
    extracted_has_relations = bool(extracted_json.get("relationships", []))

    results["has_relationships"] = 1.0 if expected_has_relations == extracted_has_relations else 0.0

    results["structure_score"] = (results["empty_state_match"] + results["has_attribute_nodes"] + results["has_measurement_nodes"] + results["has_relationships"]) / 4.0

    return results
