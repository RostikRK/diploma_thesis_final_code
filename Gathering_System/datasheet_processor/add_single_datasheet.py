import os
import sys
import pymupdf
import pymupdf4llm
import weaviate
from tqdm import tqdm
from neo4j import GraphDatabase
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from helpers.md_processing import create_directories, extract_header_info_wev, process_page_images
from helpers.download_utils import is_valid_pdf
from helpers.Neo4j_operator import Neo4jOperations

# Set up start
uri = "bolt://localhost:7687"
username = "neo4j"
password = "password1234"
# Set up end

def create_or_update_product_package(neo4j_ops, part_number, manufacturer_name, package_name, category_name):
    """
    Create or update a product, manufacturer, package and link to category.
    """
    if not neo4j_ops.check_category_exists(category_name):
        return False, f"Category '{category_name}' does not exist in the database."

    neo4j_ops.create_or_update_manufacturer(manufacturer_name)

    # Create or update package
    package_data = {
        "package_name": package_name,
        "standardized_package_names": [package_name]
    }
    neo4j_ops.create_or_update_package(package_data)

    # Create or update product and link to enteties
    product_data = {
        "ManufacturerProductNumber": part_number,
        "Manufacturer": {"Name": manufacturer_name},
        "Description": {"DetailedDescription": "Added manually"},
        "ProductStatus": {"Status": "Active"}
    }
    neo4j_ops.create_or_update_product(product_data, manufacturer_name, "Unprocessed")
    product_key = {'pn': part_number, 'manufacturer': manufacturer_name}
    neo4j_ops.link_product_package(product_key, package_name, {})
    neo4j_ops.link_product_category(product_key, category_name)

    return True, "Product, manufacturer, and package created/updated and linked successfully."


def create_and_process_datasheet(neo4j_ops, weaviate_client, part_number, manufacturer_name,
                                 pdf_path, datasheet_url, category_name):
    """
    Creates a datasheet entry if needed, links it to the product,
    and processes the PDF.
    """
    product_key = {'pn': part_number, 'manufacturer': manufacturer_name}

    neo4j_ops.create_or_update_datasheet(datasheet_url)
    neo4j_ops.link_product_to_datasheet(product_key, datasheet_url, "Unprocessed")

    items = []

    try:
        if not os.path.exists(pdf_path):
            return False, f"PDF file not found at {pdf_path}"

        if not is_valid_pdf(pdf_path):
            return False, f"The file at {pdf_path} is not a valid PDF"

        # Open and process the PDF
        doc = pymupdf.open(pdf_path)
        md_text = pymupdf4llm.to_markdown(pdf_path)
        num_pages = len(doc)

        base_dir = f"./data_{part_number.replace('/', '_')}"
        create_directories(base_dir)

        headers = extract_header_info_wev(
            md_text, base_dir, pdf_path, items, list(manufacturer_name), list(part_number),
            weaviate_client, datasheet_url=datasheet_url
        )

        for page_num in tqdm(range(num_pages), desc=f"Processing PDF pages"):
            page = doc[page_num]
            process_page_images(
                page, page_num, base_dir, items,
                save_to_storage=True,
                data=(datasheet_url, category_name)
            )

        neo4j_ops.mark_datasheet_as(datasheet_url, "Chunked")

        doc.close()

        return True, "Datasheet successfully processed and marked as 'Chunked'"

    except Exception as e:
        neo4j_ops.mark_datasheet_as(datasheet_url, "ProcessingError")

        return False, f"Error processing datasheet: {str(e)}"


def add_single_datasheet():
    """
    Main function to add a single datasheet based on user input.
    """
    driver = GraphDatabase.driver(uri, auth=(username, password))
    neo4j_ops = Neo4jOperations(driver)

    weaviate_client = weaviate.connect_to_local(skip_init_checks=True)

    try:
        # Get product information from user
        print("\n--- Product Information ---")
        part_number = input("Enter part number: ")
        manufacturer = input("Enter manufacturer name: ")
        package = input("Enter package name: ")
        category = input("Enter category name: ")

        # Get datasheet information
        print("\n--- Datasheet Information ---")
        pdf_path = input("Enter path to PDF file: ")
        datasheet_url = input("Enter URL where datasheet was downloaded from: ")

        print("\n--- Creating/updating product information ---")
        success, message = create_or_update_product_package(neo4j_ops, part_number, manufacturer, package, category)
        if not success:
            print(f"Error: {message}")
            return

        print("\n--- Processing datasheet ---")
        success, message = create_and_process_datasheet(neo4j_ops, weaviate_client, part_number, manufacturer,
                                                        pdf_path, datasheet_url, category)
        print(f"{'Success' if success else 'Error'}: {message}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        driver.close()


if __name__ == "__main__":
    add_single_datasheet()

