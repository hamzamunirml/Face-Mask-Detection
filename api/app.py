"""
Phase 9 - REST API Development
FastAPI service exposing endpoints for image upload and mask prediction.

Endpoints:
    GET  /              -> health check
    POST /predict        -> upload an image, get mask/no-mask + confidence
    GET  /model-info      -> info about the loaded model

Run:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import io
import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model

MODEL_PATH = os.getenv("MODEL_PATH", "../models/mobilenetv2_best.h5")
IMG_SIZE = (128, 128)
LABELS = {0: "Mask", 1: "No Mask"}

app = FastAPI(
    title="Face Mask Detection API",
    description="Upload an image to detect whether a face mask is being worn.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_ml_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}. "
              f"Train a model first (see src/train.py).")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.array(image).astype("float32") / 255.0
    array = np.expand_dims(array, axis=0)
    return array


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Face Mask Detection API is running",
        "model_loaded": model is not None,
    }


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": model.name,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_params": model.count_params(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train and place a model file first.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
        processed = preprocess_image(image_bytes)
        pred_prob = float(model.predict(processed, verbose=0)[0][0])
        label_idx = int(pred_prob >= 0.5)
        confidence = pred_prob if label_idx == 1 else 1 - pred_prob

        return JSONResponse({
            "prediction": LABELS[label_idx],
            "confidence": round(confidence * 100, 2),
            "raw_score": round(pred_prob, 4),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
