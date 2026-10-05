import os
from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


def resolve_project_path(
    environment_name: str,
    default_relative_path: str,
) -> Path:
    """
    Resolve a project path.

    Environment variables may provide either:
    - an absolute path
    - a path relative to the project root
    """

    configured = os.getenv(
        environment_name
    )

    if configured:
        path = Path(
            configured
        ).expanduser()

        if not path.is_absolute():
            path = (
                BASE_DIR
                / path
            )

        return path.resolve()

    return (
        BASE_DIR
        / default_relative_path
    ).resolve()


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

PROJECT_NAME = (
    "AI-Based Real-Time Industrial "
    "Defect Detection"
)

VERSION = "1.1.0"


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_PATH = resolve_project_path(
    "MODEL_PATH",
    (
        "models/optimized/"
        "yolo_neu_optimized_best.pt"
    ),
)

QUALITY_POLICY_PATH = (
    resolve_project_path(
        "QUALITY_POLICY_PATH",
        "ml/configs/quality_policy.yaml",
    )
)

CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "CONFIDENCE_THRESHOLD",
        "0.25",
    )
)

IOU_THRESHOLD = float(
    os.getenv(
        "IOU_THRESHOLD",
        "0.45",
    )
)

IMAGE_SIZE = int(
    os.getenv(
        "IMAGE_SIZE",
        "416",
    )
)

INFERENCE_DEVICE = os.getenv(
    "INFERENCE_DEVICE",
    "cpu",
)

MAX_IMAGE_SIZE_MB = int(
    os.getenv(
        "MAX_IMAGE_SIZE_MB",
        "10",
    )
)