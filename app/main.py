from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from ultralytics import YOLO


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Real-Time Industrial Defect Detection API",
    version="1.0.0",
    description="YOLO-based industrial surface defect detection backend"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# PROMETHEUS METRICS
# =========================================================

REQUESTS = Counter(
    "defect_api_requests_total",
    "Total number of API requests",
    ["endpoint"]
)

INFERENCES = Counter(
    "defect_inferences_total",
    "Total number of inference calls"
)

INFERENCE_TIME = Histogram(
    "defect_inference_seconds",
    "Inference execution time"
)

DETECTIONS = Counter(
    "defect_detections_total",
    "Total detected defects",
    ["class_name"]
)


# =========================================================
# MODEL
# =========================================================

model = None


@app.on_event("startup")
def load_model():

    global model

    if MODEL_PATH.exists():

        model = YOLO(str(MODEL_PATH))

        print("YOLO model loaded successfully.")
        print(f"Model path: {MODEL_PATH}")

    else:

        model = None

        print("WARNING: best.pt not found.")
        print("Place your trained model at:")
        print(MODEL_PATH)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "project": "Real-Time Industrial Defect Detection System",
        "status": "running",
        "model_loaded": model is not None,
        "documentation": "/docs"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    REQUESTS.labels("/health").inc()

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH)
    }


# =========================================================
# MODEL CLASSES
# =========================================================

@app.get("/classes")
def get_classes():

    REQUESTS.labels("/classes").inc()

    if model is None:

        return {
            "classes": [],
            "message": "Model is not loaded."
        }

    names = model.names

    return {
        "classes": {
            int(class_id): class_name
            for class_id, class_name in names.items()
        }
    }


# =========================================================
# INFERENCE FUNCTION
# =========================================================

def run_inference(
    image: np.ndarray,
    confidence: float = 0.25
):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "YOLO model is not loaded. "
                "Place trained best.pt inside models/"
            )
        )

    INFERENCES.inc()

    start_time = time.perf_counter()

    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    elapsed_time = time.perf_counter() - start_time

    INFERENCE_TIME.observe(elapsed_time)

    result = results[0]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls.item())

            score = float(box.conf.item())

            x1, y1, x2, y2 = [
                float(value)
                for value in box.xyxy[0].tolist()
            ]

            class_name = result.names.get(
                class_id,
                str(class_id)
            )

            DETECTIONS.labels(class_name).inc()

            detections.append({

                "class_id": class_id,

                "class_name": class_name,

                "confidence": round(
                    score,
                    4
                ),

                "bbox": {

                    "x1": round(x1, 2),

                    "y1": round(y1, 2),

                    "x2": round(x2, 2),

                    "y2": round(y2, 2)
                }
            })

    return detections, elapsed_time


# =========================================================
# PREDICT API
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = 0.25
):

    REQUESTS.labels("/predict").inc()

    # Check file type

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    # Read uploaded file

    file_data = await file.read()

    # Convert to OpenCV image

    image = cv2.imdecode(
        np.frombuffer(
            file_data,
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # Run YOLO

    detections, elapsed_time = run_inference(
        image,
        confidence
    )

    # Return JSON

    return {

        "request_id": str(uuid.uuid4()),

        "filename": file.filename,

        "inference_time_ms": round(
            elapsed_time * 1000,
            2
        ),

        "detection_count": len(
            detections
        ),

        "detections": detections
    }


# =========================================================
# PREDICT + ANNOTATED IMAGE
# =========================================================

@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    confidence: float = 0.25
):

    REQUESTS.labels(
        "/predict/image"
    ).inc()

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    file_data = await file.read()

    image = cv2.imdecode(
        np.frombuffer(
            file_data,
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    detections, _ = run_inference(
        image,
        confidence
    )

    # Draw bounding boxes

    for detection in detections:

        bbox = detection["bbox"]

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        class_name = detection[
            "class_name"
        ]

        score = detection[
            "confidence"
        ]

        # Bounding box

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 180, 255),
            2
        )

        # Label

        label = (
            f"{class_name} "
            f"{score:.2f}"
        )

        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 255),
            2
        )

    # Encode image

    success, encoded_image = cv2.imencode(
        ".jpg",
        image
    )

    if not success:

        raise HTTPException(
            status_code=500,
            detail="Unable to encode output image."
        )

    return Response(
        content=encoded_image.tobytes(),
        media_type="image/jpeg"
    )


# =========================================================
# PROMETHEUS METRICS
# =========================================================

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
