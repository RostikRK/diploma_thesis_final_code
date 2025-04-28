import logging
import json
import requests
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
import time

import config

logger = logging.getLogger(__name__)


class WeaviateClient:
    """Client for interacting with Weaviate vector database using direct HTTP requests."""

    def __init__(self, url=config.WEAVIATE_URL, embedding_model=config.EMBEDDING_MODEL):
        self.url = url.rstrip("/")

        try:
            response = requests.get(f"{self.url}/v1/meta")
            if response.status_code == 200:
                meta = response.json()
                logger.info(f"Connected to Weaviate at {url}, version: {meta.get('version', 'unknown')}")
            else:
                logger.warning(f"Connected to Weaviate, but got status code: {response.status_code}")

            # Initialize the embedding model
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")

            # Check if the schema contains expected collection
            self._check_schema()

        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise

    def _check_schema(self):
        """Check if the DatasheetChunk collection exists, create if not."""
        try:
            response = requests.get(f"{self.url}/v1/schema")
            if response.status_code != 200:
                logger.warning(f"Error checking schema: {response.status_code}")
                return

            schema = response.json()
            class_names = [cls["class"] for cls in schema.get("classes", [])]

            if "DatasheetChunk" in class_names:
                logger.info("Found DatasheetChunk collection in Weaviate")
            else:
                logger.warning("DatasheetChunk collection not found in Weaviate - creating schema")
                self._create_schema()
        except Exception as e:
            logger.error(f"Error checking schema: {e}")

    def _create_schema(self):
        """Create the initial schema if needed."""
        try:
            class_def = {
                "class": "DatasheetChunk",
                "description": "A chunk of a datasheet document",
                "vectorizer": "text2vec-transformers",
                "vectorIndexType": "hnsw",
                "vectorIndexConfig": {
                    "distance": "cosine",
                    "ef": 200,
                    "efConstruction": 256,
                    "maxConnections": 64
                },
                "properties": [
                    {
                        "name": "text",
                        "description": "The text content of the chunk",
                        "dataType": ["text"],
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "header",
                        "description": "The header or title of the section",
                        "dataType": ["text"],
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "start_page",
                        "description": "The starting page number",
                        "dataType": ["int"],
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            }
                        }
                    },
                    {
                        "name": "end_page",
                        "description": "The ending page number",
                        "dataType": ["int"],
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            }
                        }
                    },
                    {
                        "name": "datasheet_url",
                        "description": "URL or filename of the datasheet",
                        "dataType": ["text"],
                        "indexFilterable": True,
                        "indexSearchable": True,
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            }
                        }
                    },
                    {
                        "name": "image_url",
                        "description": "URL of related image if available",
                        "dataType": ["text"],
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            }
                        }
                    },
                    {
                        "name": "manufacturers",
                        "description": "List of manufacturer names associated with this datasheet",
                        "dataType": ["text[]"],
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            }
                        }
                    },
                    {
                        "name": "products",
                        "description": "List of product names/part numbers associated with this datasheet",
                        "dataType": ["text[]"],
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            }
                        }
                    }
                ]
            }

            # Create the class
            response = requests.post(
                f"{self.url}/v1/schema",
                json=class_def
            )

            if response.status_code == 200:
                logger.info("Created DatasheetChunk collection in Weaviate")
            else:
                logger.error(f"Failed to create schema: {response.status_code} - {response.text}")
                raise Exception(f"Failed to create schema: {response.text}")
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for text.
        """
        return self.embedding_model.encode(text).tolist()

    def semantic_search(self, query: str, filter_params: Optional[Dict[str, Any]] = None, limit: int = 5) -> List[
        Dict[str, Any]]:
        """
        Perform semantic search on datasheet chunks using direct GraphQL API.
        """
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query)

        graphql_query = {
            "query": self._build_vector_search_query(query_embedding, filter_params, limit)
        }

        try:
            response = requests.post(
                f"{self.url}/v1/graphql",
                json=graphql_query
            )

            if response.status_code != 200:
                logger.error(f"Error in GraphQL query: {response.status_code} - {response.text}")
                return []

            result = response.json()

            if "errors" in result:
                logger.error(f"GraphQL errors: {json.dumps(result['errors'], indent=2)}")
                return []

            # Extract results
            if "data" in result and "Get" in result["data"] and "DatasheetChunk" in result["data"]["Get"]:
                chunks = result["data"]["Get"]["DatasheetChunk"]

                # Format results to match the expected format
                formatted_chunks = []
                for chunk in chunks:
                    additional = chunk.pop("_additional", {})
                    score = additional.get("certainty", 0)

                    formatted_chunks.append({
                        **chunk,
                        "score": score
                    })

                return formatted_chunks
            else:
                logger.warning(f"Unexpected response format: {json.dumps(result, indent=2)}")
                return []
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []

    def hybrid_search(self, query: str, filter_params: Optional[Dict[str, Any]] = None, limit: int = 5) -> List[
        Dict[str, Any]]:
        """
        Perform hybrid search (vector + keyword) on datasheet chunks.
        """
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query)

        graphql_query = {
            "query": self._build_hybrid_search_query(query, query_embedding, filter_params, limit)
        }

        try:
            # Execute the query
            response = requests.post(
                f"{self.url}/v1/graphql",
                json=graphql_query
            )

            if response.status_code != 200:
                logger.error(f"Error in GraphQL query: {response.status_code} - {response.text}")
                return []

            result = response.json()

            # Check for GraphQL errors
            if "errors" in result:
                logger.error(f"GraphQL errors: {json.dumps(result['errors'], indent=2)}")
                # Try fallback to standard vector search if hybrid search fails
                logger.info("Falling back to standard vector search")
                return self.semantic_search(query, filter_params, limit)

            # Extract results
            if "data" in result and "Get" in result["data"] and "DatasheetChunk" in result["data"]["Get"]:
                chunks = result["data"]["Get"]["DatasheetChunk"]
                formatted_chunks = []
                for chunk in chunks:
                    # Extract _additional data
                    additional = chunk.pop("_additional", {})
                    score = additional.get("score", 0)

                    formatted_chunks.append({
                        **chunk,
                        "score": score
                    })

                return formatted_chunks
            else:
                logger.warning(f"Unexpected response format: {json.dumps(result, indent=2)}")
                return []
        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            # Try fallback to standard vector search
            logger.info("Falling back to standard vector search after error")
            return self.semantic_search(query, filter_params, limit)

    def _build_vector_search_query(self, query_embedding: List[float], filter_params: Optional[Dict[str, Any]] = None,
                                   limit: int = 5) -> str:
        """Build GraphQL query for vector search."""
        # Start building the where filter if needed
        where_filter = ""
        if filter_params:
            conditions = []

            # Handle datasheet_url filter
            if "datasheet_url" in filter_params:
                conditions.append(f"""
                {{
                    path: ["datasheet_url"],
                    operator: Like,
                    valueText: "*{filter_params['datasheet_url']}*"
                }}
                """)

            # Handle page number filter
            if "page" in filter_params:
                conditions.append(f"""
                {{
                    path: ["start_page"],
                    operator: Equal,
                    valueInt: {filter_params['page']}
                }}
                """)

            # Combine where conditions if any
            if conditions:
                where_clauses = ", ".join(conditions)
                where_filter = f"where: {{ operator: And, operands: [{where_clauses}] }},"

        # Build full query
        query = f"""
        {{
          Get {{
            DatasheetChunk(
              nearVector: {{
                vector: {json.dumps(query_embedding)},
                certainty: 0.7
              }},
              {where_filter}
              limit: {limit}
            ) {{
              text
              header
              start_page
              end_page
              datasheet_url
              image_url
              manufacturers
              products
              _additional {{
                certainty
                id
              }}
            }}
          }}
        }}
        """
        return query

    def _build_hybrid_search_query(self, query_text: str, query_embedding: List[float],
                                   filter_params: Optional[Dict[str, Any]] = None, limit: int = 5) -> str:
        """Build GraphQL query for hybrid search."""

        # First try to build a hybrid query with the bm25 module
        try:
            where_filter = ""
            if filter_params:
                conditions = []

                # Handle datasheet_url filter
                if "datasheet_url" in filter_params:
                    conditions.append(f"""
                    {{
                        path: ["datasheet_url"],
                        operator: Like,
                        valueText: "*{filter_params['datasheet_url']}*"
                    }}
                    """)

                # Handle page number filter
                if "page" in filter_params:
                    conditions.append(f"""
                    {{
                        path: ["start_page"],
                        operator: Equal,
                        valueInt: {filter_params['page']}
                    }}
                    """)

                if conditions:
                    where_clauses = ", ".join(conditions)
                    where_filter = f"where: {{ operator: And, operands: [{where_clauses}] }},"

            query = f"""
            {{
              Get {{
                DatasheetChunk(
                  hybrid: {{
                    query: "{query_text}",
                    alpha: 0.5,
                    vector: {json.dumps(query_embedding)}
                  }},
                  {where_filter}
                  limit: {limit}
                ) {{
                  text
                  header
                  start_page
                  end_page
                  datasheet_url
                  image_url
                  manufacturers
                  products
                  _additional {{
                    score
                    id
                  }}
                }}
              }}
            }}
            """
            return query
        except Exception as e:
            logger.error(f"Error building hybrid query: {e}")
            # Fallback to standard vector search query
            return self._build_vector_search_query(query_embedding, filter_params, limit)

    def add_chunk(self, chunk_data: Dict[str, Any], embedding: Optional[List[float]] = None) -> Optional[str]:
        """
        Add a datasheet chunk to Weaviate.
        """
        try:
            # Create embedding
            if embedding is None and "text" in chunk_data:
                embedding = self.generate_embedding(chunk_data["text"])

            # Prepare object data
            object_data = {
                "class": "DatasheetChunk",
                "properties": chunk_data,
                "vector": embedding
            }

            # Send to Weaviate
            response = requests.post(
                f"{self.url}/v1/objects",
                json=object_data
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Added chunk with ID: {result.get('id')}")
                return result.get('id')
            else:
                logger.error(f"Failed to add chunk: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error adding chunk: {e}")
            return None

    def batch_add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add multiple chunks to Weaviate.
        """
        successful = 0

        for chunk in chunks:
            try:
                text = chunk.get("text", "")
                embedding = self.generate_embedding(text)

                # Add chunk
                if self.add_chunk(chunk, embedding):
                    successful += 1
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing chunk in batch: {e}")

        return successful
