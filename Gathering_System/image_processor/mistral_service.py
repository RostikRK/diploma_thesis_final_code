import logging
import base64
import hashlib
import requests
import time
from mistralai import Mistral
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import re
import weaviate
import random
from neo4j import GraphDatabase


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mistral_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up start
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password1234"

API_BASE_URL = "http://localhost:8082"
WEAVIATE_URL = "http://localhost:8080"
MISTRAL_API_KEY = ""
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Set up end


def get_datasheets_with_validated_images(driver):
    """Get datasheets that have validated images"""
    query = """
    MATCH (p:Product)-[r:HAS_DATASHEET]->(d:Datasheet)
    WHERE r.image_validated = true AND r.state="Chunked"
    RETURN DISTINCT d.name AS datasheet_url
    """
    datasheet_urls = []
    with driver.session() as session:
        result = session.execute_read(lambda tx: tx.run(query).data())
        for record in result:
            datasheet_urls.append(record["datasheet_url"])
    return datasheet_urls

class MistralOCREnhancer:
    def __init__(self, datasheet_url):
        self.datasheet_url = datasheet_url
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.weaviate_client = weaviate.connect_to_local(skip_init_checks=True)
        self.random = random
        logger.info(f"Initialized MistralOCREnhancer for datasheet: {datasheet_url}")

    def get_datasheet_info(self):
        """Retrieve the information JSON for the datasheet."""
        try:
            response = requests.get(
                f"{API_BASE_URL}/info",
                params={"datasheet_url": self.datasheet_url}
            )

            if response.status_code == 200:
                info = response.json()
                logger.info(f"Retrieved info for datasheet with {info.get('total_pages', 0)} pages")
                return info
            else:
                logger.error(f"Failed to retrieve info: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving datasheet info: {e}")
            return None

    def identify_target_pages(self, info):
        """Identify pages that are useful and have complex layout."""
        if not info:
            return []

        # Get the intersection of useful pages and complex layout pages
        useful_pages = set(info.get('useful_pages', []))
        complex_pages = set(info.get('complex_layout_pages', []))
        target_pages = sorted(useful_pages.intersection(complex_pages))

        logger.info(f"Found {len(target_pages)} pages that are both useful and have complex layout")
        return target_pages

    def get_page_image(self, page_num):
        """Retrieve an image for a specific page."""
        try:
            response = requests.get(
                f"{API_BASE_URL}/image",
                params={
                    "datasheet_url": self.datasheet_url,
                    "page_num": str(page_num)
                }
            )

            if response.status_code == 200:
                # Convert the image to base64 for Mistral OCR
                encoded_image = base64.b64encode(response.content).decode('utf-8')
                return encoded_image
            else:
                logger.error(f"Failed to retrieve image for page {page_num}: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving image for page {page_num}: {e}")
            return None

    def process_image_with_mistral(self, base64_image, max_retries=3, initial_delay=1.0):
        """Process the image with Mistral OCR and return the markdown with retry logic."""
        retry_count = 0
        delay = initial_delay

        while retry_count <= max_retries:
            try:
                if retry_count > 0:
                    logger.info(f"Retry attempt {retry_count}/{max_retries} after {delay:.1f}s delay")
                    time.sleep(delay)

                ocr_response = self.client.ocr.process(
                    model="mistral-ocr-latest",
                    document={
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                    include_image_base64=False
                )

                # Extract the markdown text
                if ocr_response.pages and len(ocr_response.pages) > 0:
                    markdown_text = ocr_response.pages[0].markdown
                    logger.info(f"Successfully extracted {len(markdown_text)} characters with Mistral OCR")
                    return markdown_text
                else:
                    logger.warning("No text extracted from the image")
                    return ""

            except Exception as e:
                error_message = str(e)
                if "429" in error_message or "rate limit" in error_message.lower():
                    retry_count += 1
                    delay = initial_delay * (2 ** retry_count) + (random.random() * initial_delay)
                    logger.warning(f"Rate limit exceeded, will retry in {delay:.1f}s ({retry_count}/{max_retries})")

                    if retry_count > max_retries:
                        logger.error(f"Max retries exceeded for Mistral OCR: {e}")
                        return ""
                else:
                    logger.error(f"Error processing image with Mistral OCR: {e}")
                    return ""

        return ""

    def group_consecutive_pages(self, target_pages):
        """Group consecutive page numbers into intervals."""
        if not target_pages:
            return []

        intervals = []
        start = target_pages[0]
        end = start

        for page in target_pages[1:]:
            if page == end + 1:
                end = page
            else:
                intervals.append((start, end))
                start = page
                end = page

        intervals.append((start, end))
        logger.info(f"Grouped pages into {len(intervals)} intervals: {intervals}")
        return intervals

    def get_existing_chunks(self):
        """Retrieve existing chunks for this datasheet from Weaviate."""
        try:
            # Query using GraphQL to get all chunks for this datasheet
            query = f"""
            {{
              Get {{
                DatasheetChunk(
                  where: {{
                    path: ["datasheet_url"],
                    operator: Equal,
                    valueText: "{self.datasheet_url}"
                  }}
                ) {{
                  text
                  header
                  start_page
                  end_page
                  datasheet_url
                  manufacturers
                  products
                  _additional {{
                    id
                  }}
                }}
              }}
            }}
            """

            response = requests.post(
                f"{WEAVIATE_URL}/v1/graphql",
                json={"query": query}
            )

            if response.status_code != 200:
                logger.error(f"Error querying Weaviate: {response.status_code} - {response.text}")
                return []

            result = response.json()

            if "data" in result and "Get" in result["data"] and "DatasheetChunk" in result["data"]["Get"]:
                chunks = result["data"]["Get"]["DatasheetChunk"]
                logger.info(f"Retrieved {len(chunks)} existing chunks from Weaviate")
                return chunks
            else:
                logger.warning("No chunks found or unexpected response format")
                return []
        except Exception as e:
            logger.error(f"Error retrieving existing chunks: {e}")
            return []

    def get_manufacturer_product_info(self, existing_chunks):
        """Extract manufacturer and product info from existing chunks."""
        manufacturers = []
        products = []

        for chunk in existing_chunks:
            if chunk.get("manufacturers"):
                manufacturers.extend(chunk["manufacturers"])
            if chunk.get("products"):
                products.extend(chunk["products"])

        # Remove duplicates
        manufacturers = list(set(manufacturers))
        products = list(set(products))

        return manufacturers, products

    def delete_chunk(self, chunk_id):
        """Delete a chunk from Weaviate by ID."""
        try:
            response = requests.delete(f"{WEAVIATE_URL}/v1/objects/DatasheetChunk/{chunk_id}")

            if response.status_code == 204:
                logger.info(f"Successfully deleted chunk {chunk_id}")
                return True
            else:
                logger.error(f"Failed to delete chunk {chunk_id}: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error deleting chunk {chunk_id}: {e}")
            return False

    def chunk_markdown_by_headers(self, markdown_text, start_page, end_page=None):
        """Split markdown by headers to create meaningful chunks."""
        if end_page is None:
            end_page = start_page

        header_pattern = re.compile(r'^(#+\s+.*)$', re.MULTILINE)

        # Find all headers
        headers = header_pattern.findall(markdown_text)

        if not headers:
            # If no headers found, return the whole text as one chunk
            return [{
                "header": f"Page {start_page}" if start_page == end_page else f"Pages {start_page}-{end_page}",
                "text": markdown_text,
                "start_page": start_page,
                "end_page": end_page
            }]

        # Split the markdown by headers
        chunks = []
        sections = header_pattern.split(markdown_text)

        # First element is any text before the first header
        preamble = sections[0].strip()
        if preamble:
            chunks.append({
                "header": f"Page {start_page} Preamble" if start_page == end_page else f"Pages {start_page}-{end_page} Preamble",
                "text": preamble,
                "start_page": start_page,
                "end_page": end_page
            })

        # Process header-content pairs
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                header = sections[i].strip()
                content = sections[i + 1].strip()

                if content:
                    chunks.append({
                        "header": header,
                        "text": f"{header}\n\n{content}",
                        "start_page": start_page,
                        "end_page": end_page
                    })

        logger.info(f"Split pages {start_page}-{end_page} into {len(chunks)} chunks based on headers")
        return chunks

    def add_chunk_to_weaviate(self, chunk, manufacturers=None, products=None):
        """Add a new chunk to Weaviate."""
        try:
            if not chunk.get("text"):
                logger.warning("Skipping empty chunk")
                return False

            # Create embedding
            embedding = self.model.encode(chunk["text"]).tolist()

            # Ensure chunk has all required fields
            chunk_data = {
                "text": chunk["text"],
                "header": chunk.get("header", ""),
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
                "datasheet_url": self.datasheet_url,
                "image_url": "",
                "manufacturers": manufacturers or [],
                "products": products or []
            }

            # Create the object in Weaviate using HTTP request
            object_data = {
                "class": "DatasheetChunk",
                "properties": chunk_data,
                "vector": embedding
            }

            response = requests.post(
                f"{WEAVIATE_URL}/v1/objects",
                json=object_data
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Added new chunk with ID: {result.get('id')}")
                return True
            else:
                logger.error(f"Failed to add chunk: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error adding chunk to Weaviate: {e}")
            return False

    def run_enhancement(self):
        try:
            # Get datasheet info
            info = self.get_datasheet_info()
            if not info:
                logger.error("Failed to retrieve datasheet info, exiting")
                return False

            # Identify target pages (useful + complex layout)
            target_pages = self.identify_target_pages(info)
            if not target_pages:
                logger.warning("No pages found that are both useful and have complex layout")
                return False

            intervals = self.group_consecutive_pages(target_pages)

            # Get existing chunks from Weaviate
            existing_chunks = self.get_existing_chunks()

            manufacturers, products = self.get_manufacturer_product_info(existing_chunks)
            logger.info(f"Found manufacturers: {manufacturers}")
            logger.info(f"Found products: {products}")

            processed_chunks = []

            ocr_delay = 2.0

            for start_page, end_page in tqdm(intervals, desc="Processing intervals"):
                if start_page == end_page:
                    # Single page processing
                    logger.info(f"Processing single page {start_page}")

                    base64_image = self.get_page_image(start_page)
                    if not base64_image:
                        logger.warning(f"Skipping page {start_page} due to image retrieval failure")
                        continue
                    markdown_text = self.process_image_with_mistral(base64_image)
                    if not markdown_text:
                        logger.warning(f"Skipping page {start_page} due to OCR failure")
                        continue

                    page_chunks = self.chunk_markdown_by_headers(markdown_text, start_page)
                    processed_chunks.extend(page_chunks)

                    time.sleep(ocr_delay)
                else:
                    # Multi-page interval
                    logger.info(f"Processing multi-page interval {start_page}-{end_page}")
                    interval_text = ""

                    # Process each page in the interval
                    for page_num in range(start_page, end_page + 1):
                        logger.info(f"Processing page {page_num} in interval")

                        base64_image = self.get_page_image(page_num)
                        if not base64_image:
                            logger.warning(f"Skipping page {page_num} in interval due to image retrieval failure")
                            continue

                        markdown_text = self.process_image_with_mistral(base64_image)
                        if not markdown_text:
                            logger.warning(f"Skipping page {page_num} in interval due to OCR failure")
                            continue

                        time.sleep(ocr_delay)

                    if interval_text:
                        # Chunk the combined text
                        interval_chunks = self.chunk_markdown_by_headers(interval_text, start_page, end_page)
                        processed_chunks.extend(interval_chunks)

            chunks_to_delete = []

            for chunk in existing_chunks:
                chunk_start = chunk.get("start_page")
                chunk_end = chunk.get("end_page")
                chunk_id = chunk.get("_additional", {}).get("id")

                if chunk_id is None:
                    continue

                fully_contained = False
                for start_page, end_page in intervals:
                    if chunk_start is not None and chunk_end is not None and \
                            chunk_start >= start_page and chunk_end <= end_page:
                        chunks_to_delete.append(chunk_id)
                        fully_contained = True
                        break

            # Delete overlapping chunks
            logger.info(f"Deleting {len(chunks_to_delete)} overlapping chunks")
            for chunk_id in chunks_to_delete:
                self.delete_chunk(chunk_id)

            # Add new chunks to Weaviate
            success_count = 0
            logger.info(f"Adding {len(processed_chunks)} new chunks")
            for chunk in tqdm(processed_chunks, desc="Adding chunks to Weaviate"):
                if self.add_chunk_to_weaviate(chunk, manufacturers, products):
                    success_count += 1
                time.sleep(0.1)

            logger.info(f"Enhancement complete. Added {success_count}/{len(processed_chunks)} new chunks.")
            return True
        finally:
            try:
                if hasattr(self, 'weaviate_client') and self.weaviate_client is not None:
                    self.weaviate_client.close()
                    logger.info("Weaviate client connection closed properly")
            except Exception as e:
                logger.error(f"Error closing Weaviate client: {e}")


def main():
    print("Select mode:")
    print("1 - Process specific datasheet")
    print("2 - Process all datasheets with validated images")

    mode = input("Enter mode (1 or 2): ")

    if mode == "1":
        datasheet_url = input("Enter datasheet URL: ")
        enhancer = MistralOCREnhancer(datasheet_url)
        enhancer.run_enhancement()
    elif mode == "2":
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        try:
            datasheet_urls = get_datasheets_with_validated_images(driver)
            logger.info(f"Found {len(datasheet_urls)} datasheets with validated images")

            for i, datasheet_url in enumerate(datasheet_urls):
                logger.info(f"Processing datasheet {i + 1}/{len(datasheet_urls)}: {datasheet_url}")
                enhancer = MistralOCREnhancer(datasheet_url)
                enhancer.run_enhancement()
                logger.info(f"Completed processing datasheet: {datasheet_url}")

                if i < len(datasheet_urls) - 1:
                    time.sleep(5)

            logger.info("Completed processing all datasheets with validated images")
        except Exception as e:
            logger.error(f"Error processing datasheets: {e}")
        finally:
            driver.close()
    else:
        logger.error("Invalid mode selected")


if __name__ == "__main__":
    main()
