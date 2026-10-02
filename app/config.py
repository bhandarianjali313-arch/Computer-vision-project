import os


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_NAME = "AI-Based Real-Time Industrial Defect Detection"

VERSION = "1.0.0"

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/best.pt"
)

CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "CONFIDENCE_THRESHOLD",
        "0.25"
    )
)

IOU_THRESHOLD = float(
    os.getenv(
        "IOU_THRESHOLD",
        "0.45"
    )
)

IMAGE_SIZE = int(
    os.getenv(
        "IMAGE_SIZE",
        "640"
    )
)

MAX_IMAGE_SIZE_MB = 10
