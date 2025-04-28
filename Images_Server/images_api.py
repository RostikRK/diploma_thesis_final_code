import json
import os
import hashlib
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse

app = FastAPI()
STORAGE_DIR = "./images/"

def encode_datasheet_url(datasheet_url: str) -> str:
    """Encode the datasheet URL into a safe directory name using MD5 hash."""
    return hashlib.md5(datasheet_url.encode("utf-8")).hexdigest()

def get_paths(datasheet_url: str, page_num: str):
    """
    Generate the directory path and file path for a given datasheet_url and page number.
    """
    encoded = encode_datasheet_url(datasheet_url)
    dir_path = os.path.join(STORAGE_DIR, encoded)
    filename = f"page_{int(page_num):03d}.png"
    file_path = os.path.join(dir_path, filename)
    return dir_path, file_path

@app.post("/image")
async def write_image(
    datasheet_url: str = Form(...),
    page_num: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Save an uploaded PNG image under a directory named by the MD5 of datasheet_url.
    """
    dir_path, file_path = get_paths(datasheet_url, page_num)
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        content = await image.read()
        with open(file_path, "wb") as f:
            f.write(content)
        return {"message": "Image saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write image: {str(e)}")

@app.get("/image")
def read_image(
    datasheet_url: str = Query(...),
    page_num: str = Query(...)
):
    """
    Retrieve a previously saved image.
    """
    dir_path, file_path = get_paths(datasheet_url, page_num)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        return FileResponse(path=file_path, media_type="image/png", filename=f"page_{page_num}.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read image: {str(e)}")

@app.delete("/image")
def delete_image(
    datasheet_url: str = Query(...),
    page_num: str = Query(...)
):
    """
    Delete a stored image.
    """
    dir_path, file_path = get_paths(datasheet_url, page_num)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        os.remove(file_path)
        return {"message": "Image deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {str(e)}")


@app.post("/info")
async def write_info(
        datasheet_url: str = Form(...),
        info_json: str = Form(...)
):
    """
    Save validated JSON metadata alongside images for the same datasheet.
    """
    try:
        encoded = encode_datasheet_url(datasheet_url)
        dir_path = os.path.join(STORAGE_DIR, encoded)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        info_path = os.path.join(dir_path, "info.json")
        info_data = json.loads(info_json)
        with open(info_path, "w") as f:
            json.dump(info_data, f, indent=2)

        return {"message": "Info JSON saved successfully"}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write info JSON: {str(e)}")


@app.get("/info")
def read_info(
        datasheet_url: str = Query(...)
):
    """
    Retrieve the stored info.json for a given datasheet.
    """
    encoded = encode_datasheet_url(datasheet_url)
    dir_path = os.path.join(STORAGE_DIR, encoded)
    info_path = os.path.join(dir_path, "info.json")

    if not os.path.exists(info_path):
        raise HTTPException(status_code=404, detail="Info JSON not found")

    try:
        with open(info_path, "r") as f:
            info_data = json.load(f)
        return info_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read info JSON: {str(e)}")
