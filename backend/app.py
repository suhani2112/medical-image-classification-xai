import os
import uuid
import shutil

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from preprocess import load_image, preprocess_image
from predict import predict_image
from gradcam import generate_gradcam


app = FastAPI(
    title="Medical Image Classification XAI Backend",
    description="Chest X-ray Pneumonia Detection using CNN and Grad-CAM",
    version="1.0.0"
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_DIR = os.path.join(BASE_DIR, "results", "uploads")
GRADCAM_DIR = os.path.join(BASE_DIR, "results", "gradcam")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "Medical Image Classification XAI backend is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]

    upload_path = os.path.join(
        UPLOAD_DIR,
        f"{file_id}{file_extension}"
    )

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = load_image(upload_path)
    input_image = preprocess_image(image)

    prediction_result = predict_image(input_image)

    gradcam_path = os.path.join(
        GRADCAM_DIR,
        f"gradcam_{file_id}.jpg"
    )

    generate_gradcam(
        image=image,
        input_image=input_image,
        save_path=gradcam_path
    )

    return JSONResponse(
        content={
            "prediction": prediction_result["label"],
            "confidence": prediction_result["confidence"],
            "raw_probability": prediction_result["raw_probability"],
            "gradcam_path": gradcam_path
        }
    )