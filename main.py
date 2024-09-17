from fastapi import FastAPI, UploadFile, File, HTTPException
from google.cloud import storage
import os

app = FastAPI()

GCS_BUCKET_NAME = 'medsight-test'

storage_client = storage.Client()

bucket = storage_client.bucket(GCS_BUCKET_NAME)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image to server (Using GCS for demo only)
    """
    try:
        # Create a blob (GCS object) and upload the file to the bucket
        blob = bucket.blob(file.filename)
        blob.upload_from_file(file.file, content_type=file.content_type)

        file_url = f"gs://{GCS_BUCKET_NAME}/{file.filename}"

        return {"message": "Image uploaded successfully", "filename": file.filename, "url": file_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image to GCS: {str(e)}")
    
@app.post("/upload_batch/")
async def upload_batch(files: list[UploadFile] = File(...)):
    """
    Upload a batch of files to Google Cloud Storage.
    """
    try:
        uploaded_files = []

        # Iterate through each file in the request
        for file in files:
            blob = bucket.blob(file.filename)
            blob.upload_from_file(file.file, content_type=file.content_type)

            uploaded_files.append({"filename": file.filename, "url": f"gs://{GCS_BUCKET_NAME}/{file.filename}"})

        return {"message": "Batch upload successful", "files": uploaded_files}

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/get_image_list/")
async def get_image_list():
    """
    Retrieve a list of all images (files).
    """
    try:
        blobs = bucket.list_blobs()

        image_list = [blob.name for blob in blobs]

        return {"images": image_list}

    except Exception as e:
        return {"error": str(e)}

