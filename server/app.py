from fastapi import FastAPI, File, UploadFile
from tempfile import TemporaryDirectory
import os
import shutil

from docling.document_converter import DocumentConverter
from docling.backend.idp_backend import IDPChunker
from pathlib import Path

def set_aksk():
    os.environ.setdefault("ALIBABA_CLOUD_ACCESS_KEY_ID", "your_ak")
    os.environ.setdefault("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "your_sk")
    
set_aksk()
app = FastAPI()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        with TemporaryDirectory(prefix="upload_") as temp_dir_path:
            file_path = os.path.join(temp_dir_path, file.filename)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            source =Path(file_path)
            converter = DocumentConverter(use_idp=True)
            result = converter.convert(source)
            chunks = list(IDPChunker().chunk(result.document))
            return [{"text":chunk.text, "images":chunk.image} for chunk in chunks]
            
    except Exception as e:
        return {"error": str(e)}
    finally:
        file.file.close()