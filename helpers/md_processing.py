import os
import re

import pandas as pd
import requests
import pymupdf
import base64
from tqdm import tqdm
from io import StringIO
from sentence_transformers import SentenceTransformer

header_pattern = re.compile(r'^(#+\s*.*)$', re.MULTILINE)
page_separator = "\n\n-----\n\n"


def create_directories(base_dir):
    """Ensure subfolders for images, text, tables, and page snapshots exist."""
    directories = ["images", "text_mark", "tables_", "page_images"]
    for dir_name in directories:
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)


def extract_header_info_wev(markdown_text, base_dir, pdf_filename, items, manufacturers, products,
                            weaviate_client, datasheet_url=None, header_image_url=None,
                            verbose=False):
    """
    Parse markdown headers into hierarchical chunks, save each chunk to disk,
    compute embeddings, and push immediately to Weaviate using the client API.
    """
    if verbose:
        print(f"Processing markdown from {pdf_filename}...")

    if verbose:
        print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    lines = markdown_text.splitlines(keepends=True)
    if verbose:
        print(f"Processing {len(lines)} lines of markdown...")

    # Build page mapping
    pages_by_line = []
    current_page = 1
    for line in lines:
        if line.strip() == "-----":
            pages_by_line.append(current_page)
            current_page += 1
        else:
            pages_by_line.append(current_page)

    headers_info = []
    header_stack = []
    header_pattern = re.compile(r'^(#+)')

    if verbose:
        print("Identifying headers...")

    for i, line in enumerate(lines):
        if line.lstrip().startswith('#'):
            match = header_pattern.match(line.lstrip())
            level = len(match.group(1)) if match else 1
            header_line = line.strip()
            header_text = header_line.lstrip('#').strip()

            header_dict = {
                "header_line": header_line,
                "header_text": header_text,
                "level": level,
                "start_line": i,
                "page": pages_by_line[i]
            }

            while header_stack and header_stack[-1]["level"] >= level:
                header_stack.pop()

            hierarchy = [h["header_text"] for h in header_stack] + [header_text]
            header_dict["hierarchy"] = hierarchy

            headers_info.append(header_dict)
            header_stack.append(header_dict)

    if verbose:
        print(f"Found {len(headers_info)} headers.")

    text_mark_dir = os.path.join(base_dir, "text_mark")
    os.makedirs(text_mark_dir, exist_ok=True)

    inserted_count = 0

    if verbose:
        header_iter = tqdm(enumerate(headers_info), total=len(headers_info), desc="Processing headers")
    else:
        header_iter = enumerate(headers_info)

    for idx, header in header_iter:
        # Determine chunk bounds
        start_line = header["start_line"]
        end_line = len(lines)
        for next_header in headers_info[idx + 1:]:
            end_line = next_header["start_line"]
            break

        header["end_line"] = end_line
        header["section_text"] = "".join(lines[start_line:end_line])

        # Determine page range
        section_pages = set(pages_by_line[start_line:end_line])
        header["start_page"] = pages_by_line[start_line]
        header["end_page"] = max(section_pages) if section_pages else pages_by_line[start_line]

        # Write section to text file
        base_name = os.path.basename(pdf_filename)
        base_name_without_ext = os.path.splitext(base_name)[0]
        file_name = f"{base_name_without_ext}_header_{idx}_page_{header['start_page']}.txt"
        text_file_name = os.path.join(text_mark_dir, file_name)

        with open(text_file_name, "w", encoding="utf-8") as f:
            f.write(header["section_text"])

        items.append({
            "page": header["start_page"],
            "type": "header",
            "text": header["section_text"],
            "path": text_file_name,
            "metadata": header
        })

        # Build Weaviate object
        try:
            embedding = model.encode(header["section_text"]).tolist()

            properties = {
                "text": header["section_text"],
                "header": header["header_text"],
                "header_line": header["header_line"],
                "start_page": header["start_page"],
                "end_page": header["end_page"],
                "datasheet_url": datasheet_url if datasheet_url is not None else pdf_filename,
                "image_url": header_image_url if header_image_url is not None else "",
                "level": header["level"],
                "manufacturers": manufacturers,
                "products": products
            }

            collection = weaviate_client.collections.get("DatasheetChunk")
            collection.data.insert(
                properties,
                vector=embedding
            )
            inserted_count += 1
        except Exception as e:
            print(f"Error inserting header {idx}: {e}")

    if verbose:
        print(f"Completed processing {len(headers_info)} headers.")
        print(f"Inserted {inserted_count} items to Weaviate.")

    return headers_info



# def extract_header_info_wev(markdown_text, base_dir, pdf_filename, items, manufacturers, products,
#                             weaviate_url="http://localhost:8080", datasheet_url=None,
#                             header_image_url=None, batch_size=10, verbose=False):
#     """
#     Parse markdown headers into hierarchical chunks, save each chunk to disk,
#     compute embeddings, and push to Weaviate via HTTP.
#     """
#     if verbose:
#         print(f"Processing markdown from {pdf_filename}...")
#
#     if verbose:
#         print("Loading embedding model...")
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     weaviate_url = weaviate_url.rstrip("/")
#
#     # Check if collection exists
#     try:
#         response = requests.get(f"{weaviate_url}/v1/schema/DatasheetChunk")
#         if response.status_code != 200:
#             if verbose:
#                 print("DatasheetChunk collection not found")
#             return []
#     except Exception as e:
#         print(f"Error checking collection: {e}")
#         return []
#
#
#     lines = markdown_text.splitlines(keepends=True)
#     if verbose:
#         print(f"Processing {len(lines)} lines of markdown...")
#
#     # Build a mapping of line numbers to page numbers
#     pages_by_line = []
#     current_page = 1
#     for line in lines:
#         if line.strip() == "-----":
#             pages_by_line.append(current_page)
#             current_page += 1
#         else:
#             pages_by_line.append(current_page)
#
#     headers_info = []
#     header_stack = []
#     header_pattern = re.compile(r'^(#+)')
#
#     # Identify headers and collect basic information
#     if verbose:
#         print("Identifying headers...")
#
#     for i, line in enumerate(lines):
#         if line.lstrip().startswith('#'):
#             match = header_pattern.match(line.lstrip())
#             level = len(match.group(1)) if match else 1
#             header_line = line.strip()
#             header_text = header_line.lstrip('#').strip()
#
#             header_dict = {
#                 "header_line": header_line,
#                 "header_text": header_text,
#                 "level": level,
#                 "start_line": i,
#                 "page": pages_by_line[i]
#             }
#
#             # Maintain a stack for header hierarchy
#             while header_stack and header_stack[-1]["level"] >= level:
#                 header_stack.pop()
#
#             # Build hierarchy path
#             hierarchy = [h["header_text"] for h in header_stack] + [header_text]
#             header_dict["hierarchy"] = hierarchy
#
#             headers_info.append(header_dict)
#             header_stack.append(header_dict)
#
#     if verbose:
#         print(f"Found {len(headers_info)} headers")
#
#     text_mark_dir = os.path.join(base_dir, "text_mark")
#     os.makedirs(text_mark_dir, exist_ok=True)
#
#     batch_objects = []
#     inserted_count = 0
#
#     if verbose:
#         header_iter = tqdm(enumerate(headers_info), total=len(headers_info), desc="Processing headers")
#     else:
#         header_iter = enumerate(headers_info)
#
#     for idx, header in header_iter:
#         # Determine chunk bounds
#         start_line = header["start_line"]
#         end_line = len(lines)
#
#         for next_header in headers_info[idx + 1:]:
#             end_line = next_header["start_line"]
#             break
#
#         header["end_line"] = end_line
#         header["section_text"] = "".join(lines[start_line:end_line])
#
#         # Determine page range
#         section_pages = set(pages_by_line[start_line:end_line])
#         header["start_page"] = pages_by_line[start_line]
#         header["end_page"] = max(section_pages) if section_pages else pages_by_line[start_line]
#
#         # Build the file name and write out the header section to a text file
#         base_name = os.path.basename(pdf_filename)
#         base_name_without_ext = os.path.splitext(base_name)[0]
#         file_name = f"{base_name_without_ext}_header_{idx}_page_{header['start_page']}.txt"
#         text_file_name = os.path.join(text_mark_dir, file_name)
#
#         with open(text_file_name, "w", encoding="utf-8") as f:
#             f.write(header["section_text"])
#
#         items.append({
#             "page": header["start_page"],
#             "type": "header",
#             "text": header["section_text"],
#             "path": text_file_name,
#             "metadata": header
#         })
#
#         # Build Weaviate object
#         try:
#             embedding = model.encode(header["section_text"]).tolist()
#
#             object_data = {
#                 "class": "DatasheetChunk",
#                 "properties": {
#                     "text": header["section_text"],
#                     "header": header["header_text"],
#                     "header_line": header["header_line"],
#                     "start_page": header["start_page"],
#                     "end_page": header["end_page"],
#                     "datasheet_url": datasheet_url if datasheet_url is not None else pdf_filename,
#                     "image_url": header_image_url if header_image_url is not None else "",
#                     "level": header["level"],
#                     "manufacturers": manufacturers,
#                     "products": products
#                 },
#                 "vector": embedding
#             }
#
#             batch_objects.append(object_data)
#
#             if len(batch_objects) >= batch_size or idx == len(headers_info) - 1:
#                 if verbose:
#                     print(f"Inserting batch of {len(batch_objects)} items to Weaviate...")
#
#                 # Process batch one by one
#                 for obj in batch_objects:
#                     try:
#                         response = requests.post(
#                             f"{weaviate_url}/v1/objects",
#                             json=obj
#                         )
#
#                         if response.status_code == 200:
#                             inserted_count += 1
#                         else:
#                             if verbose:
#                                 print(f"Error inserting object: {response.status_code}")
#                                 print(response.text)
#                     except Exception as e:
#                         if verbose:
#                             print(f"Error inserting object: {e}")
#
#                 batch_objects = []  # Clear the batch
#
#                 if verbose:
#                     print(f"Successfully inserted batch. Total inserted: {inserted_count}")
#         except Exception as e:
#             print(f"Error processing header {idx}: {e}")
#
#     if verbose:
#         print(f"Completed processing {len(headers_info)} headers")
#         print(f"Inserted {inserted_count} items to Weaviate")
#
#     return headers_info



def format_table_text(csv_text):
    """
    Convert CSV snippet into a Markdown grid table; fallback to raw text on error.
    """
    try:
        df = pd.read_csv(StringIO(csv_text))
        df = df.fillna('')
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        md_table = df.to_markdown(index=False, tablefmt="grid")
        return md_table
    except Exception as e:
        print("Error formatting table text:", e)
        return csv_text


def process_page_images(page, page_num, base_dir, items, save_to_storage=False, data=None):
    """
    Convert a PDF page to PNG, base64-encode it, and store id in Image DB.
    """
    pix = page.get_pixmap()
    if pix.n > 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

    filename = f"page_{int(page_num):03d}.png"

    temp_dir = os.path.join(base_dir, "temp_page_images")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, filename)
    pix.save(temp_path)

    with open(temp_path, 'rb') as f:
        image_bytes = f.read()


    page_image = base64.b64encode(image_bytes).decode('utf8')

    # Build the entry for this page
    entry = {
        "page": page_num,
        "type": "page",
        "local_path": temp_path,
        "image": page_image
    }

    if save_to_storage:

        datasheet_url = data[0]
        api_url = "http://localhost:8082/image"

        files = {
            "image": (filename, image_bytes, "image/png")
        }
        form_data = {
            "datasheet_url": datasheet_url,
            "page_num": str(page_num)
        }

        try:
            response = requests.post(api_url, files=files, data=form_data)
            response.raise_for_status()
        except requests.RequestException as e:
            raise Exception(f"Failed to upload image via API: {e}")

    items.append(entry)

