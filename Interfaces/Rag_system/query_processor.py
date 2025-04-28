import logging
import json
from typing import Dict, List, Any, Optional
from llm_service import llm_service
from neo4j_client import Neo4jClient
from weaviate_http_client import WeaviateClient
import config
from models import RouterResponse

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Processor for user queries."""

    def __init__(self):
        self.neo4j_client = Neo4jClient()
        self.weaviate_client = WeaviateClient()
        self.image_api_url = config.IMAGE_API_URL

    def process_query(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        """
        try:
            # Route the query to determine purpose
            route_result = self.route_query(query, chat_history)
            query_type = route_result.query_type
            params = route_result.parameters

            logger.info(f"Query routed as {query_type} with params: {params}")

            # Process based on query type
            if query_type == "CROSS_REFERENCE":
                return self.handle_cross_reference_query(params, query)
            elif query_type == "ENTITY_SEARCH":
                return self.handle_entity_graph_query(params, query)
            else:  # GENERAL_SEARCH
                return self.handle_vector_search_query(query, params)

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)

            # Generate error response
            error_prompt = f"""
            # Task: Generate Error Response

            ## Context
            The user asked: {query}

            We encountered an error while processing this request: {str(e)}

            ## Instructions
            Create a helpful response that:
            1. Apologizes for the error
            2. Explains that we encountered a technical issue (without technical details)
            3. Suggests the user try reformulating their question
            """

            response_content = llm_service.generate(
                error_prompt,
                system_prompt="You are a helpful technical assistant."
            )

            return {
                "content": response_content,
                "error": str(e)
            }

    def route_query(self, query: str, chat_history: Optional[List] = None) -> RouterResponse:
        """
        Determine the intent of the query and extract relevant parameters.
        """
        router_prompt = f"""
        # Task: Query Classification and Parameter Extraction

        ## Input Query
        {query}

        ## Instructions
        Analyze the query and determine which of the following types it belongs to:

        1. CROSS_REFERENCE - User wants to find equivalent or alternative components
        2. ENTITY_SEARCH - User wants specific information about products, manufacturers, packages, parameters of the product, etc.
        3. GENERAL_SEARCH - User wants information from datasheets or general knowledge

        Then extract all relevant parameters that would help with searching.

        ## Output Format
        Return a JSON object with the following structure:
        ```json
        {{
            "query_type": "CROSS_REFERENCE|ENTITY_SEARCH|GENERAL_SEARCH",
            "parameters": {{
                "part_number": "extracted part number if present",
                "manufacturer": "extracted manufacturer if present",
                "category": "extracted category if present",
                "attribute": "extracted attribute if present",
                "other_relevant_params": "any other relevant parameters"
            }}
        }}
        ```

        Include only parameters that are explicitly mentioned or strongly implied in the query.
        """

        response = llm_service.generate(
            router_prompt,
            system_prompt=config.SYSTEM_PROMPTS["router"]
        )

        result = llm_service.extract_json(response)

        query_type = result.get("query_type", "GENERAL_SEARCH")
        parameters = result.get("parameters", {})

        return RouterResponse(query_type=query_type, parameters=parameters)

    def handle_cross_reference_query(self, params: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Process cross-reference queries by finding alternative components.
        """
        # Extract key parameters
        part_number = params.get('part_number')
        manufacturer = params.get('manufacturer')
        category = params.get('category')
        target_manufacturer = params.get('target_manufacturer')

        # Validate parameters and re-extract if necessary
        if not part_number or not manufacturer:
            extraction_prompt = f"""
            # Task: Extract Electronic Component Parameters

            ## Input
            {original_query}

            ## Instructions
            Extract the part number, manufacturer name, and category (if mentioned) from the query about electronic components.
            If there's any ambiguity, make your best guess based on context.
            Also extract target manufacturer if specified (the manufacturer whose products to search for alternatives).

            ## Output Format
            Return ONLY a JSON object with this structure:
            {{
                "part_number": "extracted part number",
                "manufacturer": "extracted manufacturer",
                "category": "extracted category (if mentioned)",
                "target_manufacturer": "extracted target manufacturer (if specified)"
            }}
            """

            # Generate and parse response
            extraction_result = llm_service.generate(
                extraction_prompt,
                system_prompt="You are a technical parameter extraction assistant."
            )

            extracted = llm_service.extract_json(extraction_result)
            part_number = extracted.get('part_number', part_number)
            manufacturer = extracted.get('manufacturer', manufacturer)
            category = extracted.get('category', category)
            target_manufacturer = extracted.get('target_manufacturer', target_manufacturer)

        # If still no valid parameters, return error with guidance
        if not part_number or not manufacturer:
            missing_params_prompt = f"""
            # Task: Generate Missing Parameters Response

            ## Context
            The user asked: {original_query}

            We are trying to perform a cross-reference search but couldn't identify the required parameters.

            ## Instructions
            Create a helpful response that:
            1. Explains we need both a part number and manufacturer to find cross-references
            2. Provides an example of how to ask for cross-references correctly
            3. Asks the user to provide the missing information
            """

            response_content = llm_service.generate(
                missing_params_prompt,
                system_prompt="You are a helpful technical assistant."
            )

            return {
                "content": response_content,
                "error": "Missing required parameters"
            }

        # Find cross-references
        cross_references = self.neo4j_client.find_cross_references(
            part_number,
            manufacturer,
            category,
            target_manufacturer
        )

        # Format response using LLM
        if cross_references:
            context = {
                "original_part": part_number,
                "manufacturer": manufacturer,
                "category": category if category else "Unknown",
                "cross_references": cross_references
            }

            response_prompt = f"""
            # Task: Format Cross-Reference Results

            ## Context
            Original part: {part_number}
            Manufacturer: {manufacturer}
            Category: {category if category else "Not specified"}

            Cross-reference results:
            {json.dumps(cross_references, indent=2)}

            ## Instructions
            Create a well-formatted response that:
            1. Clearly identifies the original part
            2. Presents the cross-references in a logical order
            3. Includes similarity scores where available
            4. Mentions manufacturers of alternative parts

            Use markdown formatting for readability.

            ## Original Query
            {original_query}
            """

            response_content = llm_service.generate(
                response_prompt,
                system_prompt=config.SYSTEM_PROMPTS["cross_reference"]
            )

            return {
                "content": response_content,
                "references": cross_references
            }
        else:
            part_exists = self.neo4j_client.get_product_details(part_number, manufacturer) is not None

            if part_exists:
                reason = "no similar components found in our database"
            else:
                reason = "the specified part number may not exist in our database"

            no_results_prompt = f"""
            # Task: No Cross-References Found Response

            ## Context
            Original part: {part_number}
            Manufacturer: {manufacturer}
            Category: {category if category else "Not specified"}

            No cross-references were found because {reason}.

            ## Instructions
            Create a helpful response that:
            1. Clearly explains that no cross-references were found and why
            2. Suggests possible alternatives (check spelling, try another part, etc.)
            3. Offers alternative actions the user might take

            ## Original Query
            {original_query}
            """

            response_content = llm_service.generate(
                no_results_prompt,
                system_prompt=config.SYSTEM_PROMPTS["cross_reference"]
            )

            return {
                "content": response_content,
                "references": []
            }

    def handle_entity_graph_query(self, params: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Process entity-focused queries by searching the knowledge graph.
        """
        part_number = params.get("part_number")
        manufacturer = params.get("manufacturer")
        category = params.get("category")
        attribute = params.get("attribute")
        results = []

        if part_number:
            # Get product details
            product_details = self.neo4j_client.get_product_details(part_number, manufacturer)
            if product_details:
                results.append({"product_details": product_details})

                # Get product attributes
                attributes = self.neo4j_client.get_product_attributes(part_number, manufacturer)
                if attributes:
                    results.append({"product_attributes": attributes})
        else:
            cypher_generation_prompt = f"""
            # Task: Generate Neo4j Cypher Query

            ## Database Schema
            - (Product) nodes with properties: name, manufacturer, description
            - (Manufacturer) nodes with properties: name
            - (Category) nodes with properties: name
            - (Package) nodes with properties: name, standardized_package_names
            - (Datasheet) nodes with properties: name (URL)
            - (Attribute) nodes with properties: id, description
            - (Measurement) nodes with properties: value, unit, test_conditions

            ## Relationships
            - (Product)-[:MANUFACTURED_BY]->(Manufacturer)
            - (Product)-[:BELONGS_TO]->(Category)
            - (Product)-[:PACKAGED_IN]->(Package)
            - (Product)-[:HAS_DATASHEET]->(Datasheet)
            - (Product)-[:HAS_ATTRIBUTE]->(Attribute)
            - (Measurement)-[:FOR_ATTRIBUTE]->(Attribute)

            ## Query Information
            User Query: {original_query}
            Extracted Parameters: {json.dumps(params)}

            ## Instructions
            Generate a Cypher query that will retrieve the requested information from Neo4j.
            Make the query as specific as possible based on the extracted parameters.
            Ensure the query returns only the information requested.
            Limit results to 10 items maximum.

            ## Output Format
            Return ONLY the Cypher query with no explanation or commentary.
            """

            # Generate Cypher query
            cypher_query = llm_service.generate(
                cypher_generation_prompt,
                system_prompt="You are a Cypher query generator."
            ).strip()

            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()

            # Execute query if it appears valid
            if cypher_query.lower().startswith("match") or cypher_query.lower().startswith("call"):
                try:
                    query_results = self.neo4j_client.execute_cypher(cypher_query)
                    if query_results:
                        results.append({"cypher_results": query_results})
                except Exception as e:
                    logger.error(f"Error executing generated Cypher query: {e}")

            if not results:
                if manufacturer:
                    query = """
                    MATCH (m:Manufacturer {name: $manufacturer})<-[:MANUFACTURED_BY]-(p:Product)
                    RETURN p.name as part_number, p.description as description
                    LIMIT 10
                    """
                    query_results = self.neo4j_client.execute_cypher(query, {"manufacturer": manufacturer})
                    if query_results:
                        results.append({"manufacturer_products": query_results})

                elif category:
                    query = """
                    MATCH (c:Category {name: $category})<-[:BELONGS_TO]-(p:Product)
                    RETURN p.name as part_number, p.manufacturer as manufacturer, p.description as description
                    LIMIT 10
                    """
                    query_results = self.neo4j_client.execute_cypher(query, {"category": category})
                    if query_results:
                        results.append({"category_products": query_results})

        # Format response using LLM
        if results:
            response_prompt = f"""
            # Task: Format Entity Search Results

            ## Context
            Original query: {original_query}
            Retrieved information: {json.dumps(results, indent=2)}

            ## Instructions
            Create a well-formatted response that:
            1. Directly answers the user's question based on the data provided
            2. Organizes the information logically
            3. Uses markdown formatting for readability
            4. Highlights the most relevant information first

            Focus on clarity and conciseness. If technical parameters are present,
            present them in a structured way.
            """

            response_content = llm_service.generate(
                response_prompt,
                system_prompt=config.SYSTEM_PROMPTS["entity_search"]
            )

            return {
                "content": response_content,
                "references": results
            }
        else:
            # No results found - fallback to vector search
            return self.handle_vector_search_query(original_query, params, entities_not_found=True)

    def handle_vector_search_query(self, query: str, params: Optional[Dict[str, Any]] = None,
                                   entities_not_found: bool = False) -> Dict[str, Any]:
        """
        Process general queries or fallbacks using vector search.
        """
        filter_params = {}
        limit = 5

        # If we have specific entities, use them to narrow search
        if params:
            part_number = params.get("part_number")
            if part_number:
                filter_params["datasheet_url"] = part_number
                limit = 8

        # Perform hybrid search
        chunks = self.weaviate_client.hybrid_search(query, filter_params, limit)

        # If no results, try semantic search
        if not chunks:
            chunks = self.weaviate_client.semantic_search(query, filter_params, limit)

        # Generate response
        if chunks:
            context_prompt = f"""
            # Task: Generate Response from Datasheet Information

            ## Context
            The user asked: {query}

            {'We could not find the exact information in our knowledge graph, but' if entities_not_found else ''} 
            we found these relevant datasheet chunks:
            {json.dumps(chunks, indent=2)}

            ## Instructions
            Create a comprehensive answer that:
            1. Directly addresses the user's question
            2. Synthesizes information from multiple chunks when relevant
            3. Uses markdown formatting for readability
            4. Includes specific technical details when available
            5. References the sources of information

            If the chunks contain images or diagrams, mention their existence.
            """

            response_content = llm_service.generate(
                context_prompt,
                system_prompt=config.SYSTEM_PROMPTS["general_search"]
            )

            return {
                "content": response_content,
                "references": chunks,
            }
        else:
            # No results found in either database
            no_results_prompt = f"""
            # Task: No Information Found Response

            ## Context
            The user asked: {query}

            We couldn't find any relevant information in our database.

            ## Instructions
            Create a helpful response that:
            1. Clearly explains that no information was found
            2. Suggests how the user might reformulate their query
            3. Offers alternative approaches they could try

            """

            response_content = llm_service.generate(
                no_results_prompt,
                system_prompt=config.SYSTEM_PROMPTS["general_search"]
            )

            return {
                "content": response_content,
                "references": []
            }


query_processor = QueryProcessor()
