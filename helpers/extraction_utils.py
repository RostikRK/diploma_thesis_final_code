import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from pydantic import BaseModel
from jellyfish import jaro_winkler_similarity

class Node(BaseModel):
    id: str
    type: str
    properties: Dict[str, Any] = {}


class Relationship(BaseModel):
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = {}


class Graph(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]


def load_product_data(file_path: str) -> Dict[str, Any]:
    """Load product data from a text file"""
    with open(file_path, 'r') as f:
        line = f.read().strip()
        parts = [part.strip() for part in line.split('|')]

        if len(parts) >= 6:
            return {
                'datasheet_url': parts[0],
                'manufacturer': parts[1],
                'product_name': parts[2],
                'package': parts[3],
                'category': parts[4],
                'page': int(parts[5])
            }
        else:
            return {}


def detect_loop(text: str) -> bool:
    """
    JSON-specific loop detection.
    """
    # Get rid of think block
    if '</think>' in text:
        text = re.sub(r'^.*?</think>', '', text, flags=re.DOTALL)

    lines = text.strip().split('\n')
    if len(lines) >= 3:
        for i in range(len(lines) - 2):
            if lines[i] == lines[i + 1] == lines[i + 2]:
                return True

    # JSON structure-specific checks
    try:
        if text.count('{"nodes":') > 1:
            return True

        # Check for duplicate node id
        node_ids = re.findall(r'"id"\s*:\s*"([^"]+)"', text)
        if len(node_ids) != len(set(node_ids)):
            return True

        measurement_pairs = []
        for m in re.finditer(r'"type"\s*:\s*"Measurement".*?"value"\s*:\s*([0-9.]+).*?"unit"\s*:\s*"([^"]+)"', text,
                             re.DOTALL):
            value = m.group(1)
            unit = m.group(2)
            pair = (value, unit)
            measurement_pairs.append(pair)
            if measurement_pairs.count(pair) >= 8:
                return True
    except:
        pass

    return False


def load_expected_result(file_path: str) -> Dict[str, Any]:
    """Load expected extraction result from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {}


def create_prompt_text(product_data, text, prompt_num=2):
    """Create a prompt for parameter extraction"""
    product_name = product_data['product_name']
    manufacturer = product_data['manufacturer']
    package = product_data['package']

    system_prompt = (
        "# Knowledge Graph Instructions for Qwen.\n"
        "## 1. Overview\n"
        "You are a technical information extraction system designed for processing electronics component datasheets and extracting information in structured "
        "formats to build a knowledge graph.\n"
        "Try to capture as much information from the text as possible without "
        "sacrificing accuracy. Do not add any information that is not explicitly "
        "mentioned in the text.\n"
        "- **Nodes** represent entities and concepts.\n"
        "- The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.\n"
        "## 2. Labeling Nodes\n"
        "- **Consistency**: Use only basic types for node labels. For example, always label a person as 'Person', not 'Mathematician' or 'Scientist'.\n"
        "- **Node IDs**: Do not use integers, exception the Product IDs as they are represented by Part Number. Use names or human-readable identifiers found in the text.\n"
        "## 3. Coreference Resolution\n"
        "- **Maintain Entity Consistency**: If an entity (e.g., \"John Doe\") is mentioned multiple times under different forms (like \"Joe\" or \"he\"), always use the most complete identifier (\"John Doe\").\n"
        "## 4. Strict Compliance\n"
        "Adhere strictly to these rules. Do not add any extra text, commentary, or echo the input. When you are asked to return the JSON object return just it, do not include any additional formating like '```json'\n"
    )


    extraction_rules = (
        'Rules:\n'
        '1. Product and Package nodes are already in the graph so there is no need to define them again, do not include nodes with these types in the final answer, you can use them just for relationships.\n'
        '2. Technical datasheets are usually extremely complex, have many tables, and may contain information about a family of products rather than a specific product. You have to focus on the one that this task is about. Usually in the heading or in a column it is indicated which products this section or column is about, so be careful and check whether the product fits this section or column criteria.\n'
        '3. Extract all numeric technical parameters as Attributes and Measurements. Do not forget to make sure that they are exactly for the asked product.\n'
        '4. Attributes id should be in snake case, without digits.\n'
        '5. If attributes represent min, typical or max value, then in the end of the Attribute id add "min", "_typ" or "_max". If there are test_conditions then extract them respectively to min, typ or max values. \n'
        '6. Do not include anything except the id and description when defining the Attribute.\n'
        '7. The all retrieved information for specific attribute should be written as a Measurement for this Attribute, in Measurement can be just value (just numeric), unit (in which the value is measured) and test_conditions if they are specified (as a string) .\n '
        '8. If there are more than 5 measurements for a specific attribute related to the product we are searching, then skip this attribute and measurement and do not include them in the final answer. .\n'
        '9. There is no need to define new Relationships, use just Product-HAS_ATTRIBUTE->Attribute if at least one measurement for the attribute exists and Measurement-FOR_ATTRIBUTE->Attribute .\n'
    )

    if prompt_num == 1:
        prompt = (
                "### Your task will be to extract from the provided text all unique numeric product attributes and measurement for this attribute and connect them to the product. \n\n"
                "### The Graph is already populated with different data, for this specific task you should know this part of graph schema:\n\n"
                "Nodes :\n"
                "- Product (id, description, datasheet, status)\n"
                "- Attribute (id, description, )\n"
                "- Measurement (value, unit, test_conditions)\n"
                "- Package (id, standardized_package_names)\n"
                "\n"
                "Relationships:\n"
                "- Product-HAS_ATTRIBUTE->Attribute\n"
                "- Measurement-FOR_ATTRIBUTE->Attribute\n"
                "- Product-PACKAGED_IN->Package\n"
                "\n\n"
                "### When extracting the attributes you should consider all of this rules:\n\n"
                + extraction_rules + "\n\n"
                "### Return your answer strictly as a single JSON object with exactly two keys: \"nodes\" and \"relationships\". "
                "### The JSON should conform exactly to this format :\n\n"
                "{\n"
                "  \"nodes\": [\n"
                "    {\"id\": \"current_output_max\", \"type\": \"Attribute\", \"properties\": {\"description\": \"Maximum current output\"}},\n"
                "    {\"id\": \"1\", \"type\": \"Measurement\", \"properties\": {\"value\": 32, \"unit\": \"mA\", \"test_conditions\": \"VCC = 4.5 V\"}}\n"
                "  ],\n"
                "  \"relationships\": [\n"
                "    {\"source\": \"1\", \"target\": \"current_output_max\", \"type\": \"FOR_ATTRIBUTE\"}\n"
                "  ]\n"
                "}\n\n"
                '### If the provided text does not contain attributes with measurements for the product you are looking for, then return "{}"! \n'
                '### Before answering, make sure that the final output will comply with all the rules and follow the given pattern.'
                '### When reasoning, be sure to not get stuck in the repetitive loop.'
                "### Make sure you output the final JSON object immediately, do not explain anything, do not provide multiple JSON objects, do not include any additional text, commentary, or echo the input.\n\n"
                f"So you should process the provided text from the datasheet for Product with full id = '{product_name}'. It is manufactured by {manufacturer}. "
                f"The product package is {package}.\n"
                f"\nText is: \n{text}\n\n Output is:"
        )
    else:
        prompt = (
            "# Knowledge Graph Numeric Extraction for Qwen.\n"
            "## Overview\n"
            "You are a technical information extraction system focused solely on extracting numeric attributes and their measurements from electronics component datasheets. Your task is to extract all unique numeric parameters related to the product and output them in a strictly defined JSON format without any extra text or commentary.\n\n"
            "## Extraction Rules\n"
            "1. **Focus:** Extract numeric technical parameters (values, units, and optional test conditions) that are explicitly stated for the product or its package.\n"
            "2. **Attribute Node:** for each numerical parameter that has a numerical value explicitly indicated in the text, create an Attribute node that uniquely represents a single parameter. The Attribute node must have:\n"
            "   - `id`: a snake_case string using only letters. Append the id with '_min', '_typ', or '_max' if the parameter specifies minimum, typical, or maximum values. If the attribute represents a range, split it into separate Attribute nodes for min and max.\n"
            "   - `description`: a brief (20-30 words), technical description for this attribute. Do not include test conditions into the attribute description. \n"
            "3. **Measurement Node:** For each numeric values related to the product and specific extracted attribute node, create Measurement nodes with:\n"
            "   - `id`: enumeration 1,2,3... \n"
            "   - `value`: the numeric value (number only).\n"
            "   - `unit`: the measurement unit.\n"
            "   - `test_conditions`: include only if specified.\n"
            "4. **Limits:** If more than 5 measurements exist for a specific attribute, skip that attribute entirely.  \n"
            "5. **No Extra Information:** Never define the Node with id that have been already defined in output. Do not include any commentary or additional formatting.\n\n"
            "6. **Do not stuck in the loop:** .\n\n"
            "## Graph Schema\n"
            "Nodes:\n"
            "- Attribute (id, description)\n"
            "- Measurement (value, unit, test_conditions)\n"
            "Relationships:\n"
            "- Measurement-FOR_ATTRIBUTE->Attribute\n"
            "## Product Context\n"
            f"Process the provided text from the datasheet for the product with id '{product_name}' manufactured by {manufacturer}, in the {package} package. Focus solely on numeric attributes that relate directly to this product.\n\n"
            "## Output Format\n"
            "Return a single JSON object with exactly two keys: \"nodes\" and \"relationships\". The output must follow this format exactly, as shown in the example below:\n\n"
            "Example:\n"
            "{\n"
            "  \"nodes\": [\n"
            "    {\"id\": \"current_output_max\", \"type\": \"Attribute\", \"properties\": {\"description\": \"Maximum current output refers to the highest continuous current an electronic component can safely supply without exceeding thermal or electrical limits.\"}},\n"
            "    {\"id\": \"1\", \"type\": \"Measurement\", \"properties\": {\"value\": 32, \"unit\": \"mA\", \"test_conditions\": \"VCC = 4.5 V\"}}\n"
            "  ],\n"
            "  \"relationships\": [\n"
            "    {\"source\": \"1\", \"target\": \"current_output_max\", \"type\": \"FOR_ATTRIBUTE\"}\n"
            "  ]\n"
            "}\n\n"
            "## Final Instruction\n"
            "Before providing the output, ensure that the final JSON object fully complies with all the rules and exactly follows the given output pattern. Do not include into the final JSON Attributes that do not have a Measurement with numeric value for it. If no numeric attributes with measurements for the specified product are found, return \"{}\".\n"
            "Return only the JSON object as defined above.\n"
            f"\n ### Text is: \n{text}\n\n ### Output is:"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return messages

def create_prompt_image(product_data, prompt_num=2):
    """Create a prompt for parameter extraction from image"""
    product_name = product_data['product_name']
    manufacturer = product_data['manufacturer']
    package = product_data['package']

    if prompt_num == 1:
        prompt = (
            "### Your task will be to extract from the attached image all unique numeric product attributes and measurement for this attribute and connect them to the product. \n\n"
            "### The Graph is already populated with different data, for this specific task you should know this part of graph schema:\n\n"
            "Nodes :\n"
            "- Product (id, description, datasheet, status)\n"
            "- Attribute (id, description, )\n"
            "- Measurement (value, unit, test_conditions)\n"
            "- Package (id, standardized_package_names)\n"
            "\n"
            "Relationships:\n"
            "- Product-HAS_ATTRIBUTE->Attribute\n"
            "- Measurement-FOR_ATTRIBUTE->Attribute\n"
            "- Product-PACKAGED_IN->Package\n"
            "\n\n"
            "### When extracting the attributes you should consider all of this rules:\n\n"
            "Rules:\n"
            "1. Product and Package nodes are already in the graph so there is no need to define them again, do not include nodes with these types in the final answer, you can use them just for relationships.\n"
            "2. Technical datasheets are usually extremely complex, have many tables, and may contain information about a family of products rather than a specific product. You have to focus on the one that this task is about. Usually in the heading or in a column it is indicated which products this section or column is about, so be careful and check whether the product fits this section or column criteria.\n"
            "3. Extract all numeric technical parameters as Attributes and Measurements. Do not forget to make sure that they are exactly for the asked product.\n"
            "4. Attributes id should be in snake case, without digits.\n"
            "5. If attributes represent min, typical or max value, then in the end of the Attribute id add \"_min\", \"_typ\" or \"_max\". If there are test_conditions then extract them respectively to min, typ or max values. \n"
            "6. Do not include anything except the id and description when defining the Attribute.\n"
            "7. The all retrieved information for specific attribute should be written as a Measurement for this Attribute, in Measurement can be just value (just numeric), unit (in which the value is measured) and test_conditions if they are specified (as a string) .\n "
            "8. If there are more than 5 measurements for a specific attribute related to the product we are searching, then skip this attribute and measurement and do not include them in the final answer. .\n"
            "9. There is no need to define new Relationships, use just Product-HAS_ATTRIBUTE->Attribute if at least one measurement for the attribute exists and Measurement-FOR_ATTRIBUTE->Attribute .\n"
            "\n\n"
            "### Return your answer strictly as a single JSON object with exactly two keys: \"nodes\" and \"relationships\". "
            "### The JSON should conform exactly to this format :\n\n"
            "{\n"
            "  \"nodes\": [\n"
            "    {\"id\": \"current_output_max\", \"type\": \"Attribute\", \"properties\": {\"description\": \"Maximum current output\"}},\n"
            "    {\"id\": \"1\", \"type\": \"Measurement\", \"properties\": {\"value\": 32, \"unit\": \"mA\", \"test_conditions\": \"VCC = 4.5 V\"}}\n"
            "  ],\n"
            "  \"relationships\": [\n"
            "    {\"source\": \"1\", \"target\": \"current_output_max\", \"type\": \"FOR_ATTRIBUTE\"}\n"
            "  ]\n"
            "}\n\n"
            '### If attached image does not contain attributes with measurements for the product you are looking for, then return "{}"! \n'
            '### Before answering, make sure that the final output will comply with all the rules and follow the given pattern.'
            '### When reasoning, be sure to not get stuck in the repetitive loop.'
            "### Make sure you output the final JSON object immediately, do not explain anything, do not provide multiple JSON objects, do not include any additional text, commentary, or echo the input.\n\n"
            f"So you should process the attached image of page from the datasheet for Product with full id = '{product_name}'. It is manufactured by {manufacturer}. "
            f"The product package is {package}.\n"
            "\n\n Output is:"
        )
    else:  # prompt_num == 2
        prompt = (
            "# Knowledge Graph Numeric Extraction for LLM.\n"
            "## Overview\n"
            "You are a technical information extraction system focused solely on extracting numeric attributes and their measurements from electronics component datasheets. Your task is to extract all unique numeric parameters related to the product and output them in a strictly defined JSON format without any extra text or commentary.\n\n"
            "## Extraction Rules\n"
            "1. **Focus:** Extract numeric technical parameters (values, units, and optional test conditions) that are explicitly stated for the product or its package.\n"
            "2. **Attribute Node:** for each numerical parameter that has a numerical value explicitly indicated on the image, create an Attribute node that uniquely represents a single parameter. The Attribute node must have:\n"
            "   - `id`: a snake_case string using only letters. Append the id with '_min', '_typ', or '_max' if the parameter specifies minimum, typical, or maximum values. If the attribute represents a range, split it into separate Attribute nodes for min and max.\n"
            "   - `description`: a brief (20-30 words), technical description for this attribute. Do not include test conditions into the attribute description. \n"
            "3. **Measurement Node:** For each numeric values related to the product and specific extracted attribute node, create Measurement nodes with:\n"
            "   - `id`: enumeration 1,2,3... \n"
            "   - `value`: the numeric value (number only).\n"
            "   - `unit`: the measurement unit.\n"
            "   - `test_conditions`: include only if specified.\n"
            "4. **Limits:** If more than 5 measurements exist for a specific attribute, skip that attribute entirely.  \n"
            "5. **No Extra Information:** Never define the Node with id that have been already defined in output. Do not include any commentary or additional formatting.\n\n"
            "6. **Do not stuck in the loop:** .\n\n"
            "## Graph Schema\n"
            "Nodes:\n"
            "- Attribute (id, description)\n"
            "- Measurement (value, unit, test_conditions)\n"
            "Relationships:\n"
            "- Measurement-FOR_ATTRIBUTE->Attribute\n"
            "## Product Context\n"
            f"Process the attached image of page from the datasheet for the product with id '{product_name}' manufactured by {manufacturer}, in the {package} package. Focus solely on numeric attributes that relate directly to this product. Do not process the information on charts and diagrams.\n\n"
            "## Output Format\n"
            "Return a single JSON object with exactly two keys: \"nodes\" and \"relationships\". The output must follow this format exactly, as shown in the example below:\n\n"
            "Example:\n"
            "{\n"
            "  \"nodes\": [\n"
            "    {\"id\": \"current_output_max\", \"type\": \"Attribute\", \"properties\": {\"description\": \"Maximum current output refers to the highest continuous current an electronic component can safely supply without exceeding thermal or electrical limits.\"}},\n"
            "    {\"id\": \"1\", \"type\": \"Measurement\", \"properties\": {\"value\": 32, \"unit\": \"mA\", \"test_conditions\": \"VCC = 4.5 V\"}}\n"
            "  ],\n"
            "  \"relationships\": [\n"
            "    {\"source\": \"1\", \"target\": \"current_output_max\", \"type\": \"FOR_ATTRIBUTE\"}\n"
            "  ]\n"
            "}\n\n"
            "## Final Instruction\n"
            "Before providing the output, ensure that the final JSON object fully complies with all the rules and exactly follows the given output pattern. Do not include into the final JSON Attributes that do not have a Measurement with numeric value for it. If no numeric attributes with measurements for the specified product are found, return \"{}\".\n"
            "Return only the JSON object as defined above.\n"
            "\n\n ### Output is:"
        )

    return prompt



def parse_graph(raw_text: str, logger) -> Graph:
    """Parse the generated text into a Graph object"""
    try:
        raw_text = raw_text.replace("<|endoftext|>", "").strip()

        # Remove think blocks
        if '</think>' in raw_text:
            raw_text = re.sub(r'^.*?</think>', '', raw_text, flags=re.DOTALL)

        pattern1 = r'({[\s\S]*"nodes"[\s\S]*"relationships"[\s\S]*})'
        m = re.search(pattern1, raw_text, re.DOTALL)
        if m:
            candidate = m.group(1)
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "nodes" in data and "relationships" in data:
                    return Graph.model_validate(data)
            except Exception as e:
                logger.info(f"Pattern 1 match failed to parse: {e}")

        m = re.search(r'({[\s\S]*})', raw_text, re.DOTALL)
        if m:
            candidate = m.group(1)
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and not data:
                    return Graph(nodes=[], relationships=[])
                if isinstance(data, dict) and "nodes" in data and "relationships" in data:
                    return Graph.model_validate(data)
            except Exception as e:
                logger.info(f"Pattern 2 match failed to parse: {e}")

        # Fallback: Look for markdown code blocks which might contain JSON
        code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_text)
        for block in code_blocks:
            try:
                data = json.loads(block)
                if isinstance(data, dict) and "nodes" in data and "relationships" in data:
                    return Graph.model_validate(data)
            except Exception:
                continue

        # Last fallback: split into candidate fragments
        candidates = re.findall(r'({.*?})', raw_text, re.DOTALL)
        merged = {"nodes": [], "relationships": []}
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
                if not obj or (isinstance(obj, dict) and len(obj) == 0):
                    continue
                if "nodes" in obj and "relationships" in obj:
                    merged["nodes"].extend(obj["nodes"])
                    merged["relationships"].extend(obj["relationships"])
            except Exception:
                continue

        if merged["nodes"] or merged["relationships"]:
            deduped_nodes = {}
            for node in merged["nodes"]:
                key = node.get("id")
                if key and key not in deduped_nodes:
                    deduped_nodes[key] = node
            merged["nodes"] = list(deduped_nodes.values())

            # Deduplicate relationships
            deduped_rels = {}
            for rel in merged["relationships"]:
                source = rel.get("source") or rel.get("start_node_id")
                target = rel.get("target") or rel.get("end_node_id")
                rtype = rel.get("type")
                key = (source, target, rtype)
                if key not in deduped_rels:
                    deduped_rels[key] = rel
            merged["relationships"] = list(deduped_rels.values())

            return Graph.model_validate(merged)

        return Graph(nodes=[], relationships=[])
    except Exception as e:
        logger.error(f"Error parsing graph: {e}")
        return Graph(nodes=[], relationships=[])



def validate_graph(graph: Graph) -> list[str]:
    """
    Validates the graph against specified rules.
    Returns a list of error messages; empty list if valid.
    """
    errors = []

    node_ids = {node.id for node in graph.nodes}

    attribute_ids = {node.id for node in graph.nodes if node.type == "Attribute"}

    # Check for missing descriptions
    for node in graph.nodes:
        if node.type == "Attribute":
            if "description" not in node.properties or not node.properties["description"]:
                errors.append(f"Attribute '{node.id}' does not have a description.")

    # Check for attributes without measurements
    measurement_targets = {rel.target for rel in graph.relationships if rel.type == "FOR_ATTRIBUTE"}
    for attr_id in attribute_ids:
        if attr_id not in measurement_targets:
            errors.append(f"Attribute '{attr_id}' does not have any measurements.")

    # Check for undefined nodes in relationships
    for rel in graph.relationships:
        if rel.source not in node_ids:
            errors.append(f"Relationship references undefined source node '{rel.source}'.")
        if rel.target not in node_ids:
            errors.append(f"Relationship references undefined target node '{rel.target}'.")

    return errors
