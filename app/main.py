import uuid

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.responses import (
    Response,
)

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from app.config import (
    IMAGE_SIZE,
    INFERENCE_DEVICE,
    MAX_IMAGE_SIZE_MB,
    MODEL_PATH,
    PROJECT_NAME,
    VERSION,
)

from app.detector import (
    DefectDetector,
)

from app.utils import (
    decode_image,
    encode_jpeg,
)


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(
    title=PROJECT_NAME,
    version=VERSION,
    description=(
        "Real-time industrial surface "
        "defect detection and operational "
        "quality triage API."
    ),
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=False,
    allow_methods=[
        "*"
    ],
    allow_headers=[
        "*"
    ],
)


# =========================================================
# PROMETHEUS METRICS
# =========================================================

REQUESTS = Counter(
    "defect_api_requests_total",
    "Total number of API requests",
    [
        "endpoint"
    ],
)

INFERENCES = Counter(
    "defect_inferences_total",
    "Total inference operations",
)

INFERENCE_TIME = Histogram(
    "defect_inference_seconds",
    "YOLO inference execution time",
)

DETECTIONS = Counter(
    "defect_detections_total",
    "Detected defects after quality processing",
    [
        "class_name"
    ],
)

QUALITY_DECISIONS = Counter(
    "defect_quality_decisions_total",
    "Operational quality decisions",
    [
        "decision"
    ],
)


# =========================================================
# DETECTOR
# =========================================================

detector = None


@app.on_event(
    "startup"
)
def startup_event():
    """
    Load one shared model instance when
    the FastAPI service starts.
    """

    global detector

    detector = (
        DefectDetector()
    )


def get_detector(
) -> DefectDetector:
    """
    Return the shared detector or produce
    HTTP 503 when model weights are missing.
    """

    if (
        detector is None
        or not detector.is_ready()
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Optimized defect detector "
                "is not available. "
                f"Expected model: "
                f"{MODEL_PATH}"
            ),
        )

    return detector


# =========================================================
# UPLOAD VALIDATION
# =========================================================

async def read_uploaded_image(
    file: UploadFile,
):
    """
    Validate and decode an uploaded image.
    """

    if (
        not file.content_type
        or not file.content_type.startswith(
            "image/"
        )
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload an "
                "image file."
            ),
        )

    image_bytes = (
        await file.read()
    )

    maximum_bytes = (
        MAX_IMAGE_SIZE_MB
        * 1024
        * 1024
    )

    if len(
        image_bytes
    ) > maximum_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                "Image exceeds the "
                f"{MAX_IMAGE_SIZE_MB} MB "
                "upload limit."
            ),
        )

    try:
        image = decode_image(
            image_bytes
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(
                error
            ),
        ) from error

    return image


def validate_confidence(
    confidence: float,
) -> float:
    try:
        return (
            DefectDetector
            .validate_confidence(
                confidence
            )
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


# =========================================================
# RESPONSE BUILDER
# =========================================================

def build_prediction_payload(
    filename: str | None,
    image,
    result: dict,
) -> dict:
    """
    Convert detector output into the
    public API response format.
    """

    height, width = (
        image.shape[:2]
    )

    quality = result[
        "quality"
    ]

    return {
        "request_id":
            str(
                uuid.uuid4()
            ),

        "filename":
            filename,

        "model": {
            "image_size":
                IMAGE_SIZE,

            "device":
                INFERENCE_DEVICE,
        },

        "image": {
            "width":
                int(
                    width
                ),

            "height":
                int(
                    height
                ),
        },

        "inference_time_ms":
            result[
                "inference_time_ms"
            ],

        "raw_detection_count":
            result[
                "detection_count"
            ],

        "detection_count":
            quality[
                "final_detection_count"
            ],

        "quality_decision":
            quality[
                "decision"
            ],

        "decision_reasons":
            quality[
                "decision_reasons"
            ],

        "triage_counts":
            quality[
                "triage_counts"
            ],

        "detections":
            quality[
                "detections"
            ],

        "policy_note":
            quality[
                "policy_note"
            ],
    }


def record_inference_metrics(
    result: dict,
) -> None:
    """
    Update Prometheus metrics from a
    completed inference result.
    """

    INFERENCES.inc()

    INFERENCE_TIME.observe(
        result[
            "inference_time_ms"
        ]
        / 1000.0
    )

    quality = result[
        "quality"
    ]

    QUALITY_DECISIONS.labels(
        quality[
            "decision"
        ]
    ).inc()

    for detection in quality[
        "detections"
    ]:
        DETECTIONS.labels(
            detection[
                "class_name"
            ]
        ).inc()


# =========================================================
# ROOT
# =========================================================

@app.get(
    "/"
)
def root():

    ready = (
        detector is not None
        and detector.is_ready()
    )

    return {
        "project":
            PROJECT_NAME,

        "version":
            VERSION,

        "status":
            "running",

        "model_loaded":
            ready,

        "documentation":
            "/docs",
    }


# =========================================================
# HEALTH
# =========================================================

@app.get(
    "/health"
)
def health():

    REQUESTS.labels(
        "/health"
    ).inc()

    ready = (
        detector is not None
        and detector.is_ready()
    )

    return {
        "status":
            (
                "ok"
                if ready
                else "degraded"
            ),

        "model_loaded":
            ready,

        "model_path":
            str(
                MODEL_PATH
            ),

        "image_size":
            IMAGE_SIZE,

        "device":
            INFERENCE_DEVICE,
    }


# =========================================================
# CLASSES
# =========================================================

@app.get(
    "/classes"
)
def get_classes():

    REQUESTS.labels(
        "/classes"
    ).inc()

    service = (
        get_detector()
    )

    names = (
        service.get_classes()
    )

    if isinstance(
        names,
        dict,
    ):
        classes = {
            int(
                class_id
            ):
                class_name

            for (
                class_id,
                class_name,
            ) in names.items()
        }

    else:
        classes = {
            index:
                class_name

            for (
                index,
                class_name,
            ) in enumerate(
                names
            )
        }

    return {
        "classes":
            classes
    }


# =========================================================
# PREDICT JSON
# =========================================================

@app.post(
    "/predict"
)
async def predict(
    file: UploadFile = File(
        ...
    ),
    confidence: float = 0.25,
):

    REQUESTS.labels(
        "/predict"
    ).inc()

    confidence = (
        validate_confidence(
            confidence
        )
    )

    image = (
        await read_uploaded_image(
            file
        )
    )

    service = (
        get_detector()
    )

    try:
        result = (
            service
            .detect_with_quality(
                image=image,
                confidence=confidence,
            )
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(
                error
            ),
        ) from error

    record_inference_metrics(
        result
    )

    return (
        build_prediction_payload(
            filename=file.filename,
            image=image,
            result=result,
        )
    )


# =========================================================
# PREDICT ANNOTATED IMAGE
# =========================================================

@app.post(
    "/predict/image"
)
async def predict_image(
    file: UploadFile = File(
        ...
    ),
    confidence: float = 0.25,
):

    REQUESTS.labels(
        "/predict/image"
    ).inc()

    confidence = (
        validate_confidence(
            confidence
        )
    )

    image = (
        await read_uploaded_image(
            file
        )
    )

    service = (
        get_detector()
    )

    try:
        (
            annotated_image,
            result,
        ) = (
            service
            .detect_and_draw(
                image=image,
                confidence=confidence,
            )
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(
                error
            ),
        ) from error

    record_inference_metrics(
        result
    )

    try:
        image_bytes = (
            encode_jpeg(
                annotated_image
            )
        )

    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(
                error
            ),
        ) from error

    return Response(
        content=image_bytes,
        media_type="image/jpeg",
        headers={
            "X-Quality-Decision":
                result[
                    "quality"
                ][
                    "decision"
                ]
        },
    )


# =========================================================
# METRICS
# =========================================================

@app.get(
    "/metrics"
)
def metrics():

    REQUESTS.labels(
        "/metrics"
    ).inc()

    return Response(
        content=generate_latest(),
        media_type=(
            CONTENT_TYPE_LATEST
        ),
    )