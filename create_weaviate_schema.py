import requests
import json
import time
import argparse

# Set up start
WEAVIATE_URL = 'http://localhost:8080'
# Set up end

def setup_weaviate_schema(weaviate_url="http://localhost:8080", collection_name="DatasheetChunk", verbose=True):
    """
   Set up the Weaviate schema for datasheet processing using direct HTTP requests.
   """
    weaviate_url = weaviate_url.rstrip("/")

    # For further use to redefine schemas
    try:
        response = requests.get(f"{weaviate_url}/v1/schema")
        if response.status_code != 200:
            print(f"Error getting schema: {response.status_code}")
            return False

        schema = response.json()
        classes = [cls["class"] for cls in schema.get("classes", [])]

        if collection_name in classes:
            if verbose:
                print(f"Collection {collection_name} already exists.")
                print("Deleting existing collection...")

            # Delete existing collection
            delete_response = requests.delete(f"{weaviate_url}/v1/schema/{collection_name}")
            if delete_response.status_code != 200:
                print(f"Error deleting collection: {delete_response.status_code}")
                print(delete_response.text)
                return False

            # Wait for deletion to complete
            if verbose:
                print("Waiting for deletion to complete...")
            time.sleep(2)
    except Exception as e:
        print(f"Error checking schema: {e}")
        return False

    # Define the collection
    class_def = {
        "class": collection_name,
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
                "name": "header_line",
                "description": "The original header line with markdown formatting",
                "dataType": ["text"],
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": True
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
                "name": "level",
                "description": "The heading level (1 for h1, 2 for h2, etc.)",
                "dataType": ["int"],
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

    # Create the collection
    try:
        if verbose:
            print(f"Creating collection {collection_name}...")

        response = requests.post(
            f"{weaviate_url}/v1/schema",
            json=class_def
        )

        if response.status_code == 200:
            if verbose:
                print(f"Successfully created collection {collection_name}")
            return True
        else:
            print(f"Error creating collection: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error creating collection: {e}")
        return False


def main():
    collection = 'DatasheetChunk'

    success = setup_weaviate_schema(
        weaviate_url=WEAVIATE_URL,
        collection_name=collection,
        verbose=False
    )

    if success:
        print("Schema setup completed successfully")
    else:
        print("Schema setup failed")
        exit(1)


if __name__ == "__main__":
    main()
