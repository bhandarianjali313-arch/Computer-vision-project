import numpy as np
import pytest

from app.config import (
    IMAGE_SIZE,
    MODEL_PATH,
    QUALITY_POLICY_PATH,
)

from app.detector import (
    DefectDetector,
)

from app.main import (
    build_prediction_payload,
)

from ml.src.inference.quality_decision import (
    load_quality_policy,
)


def make_detector_without_model():
    """
    Create an instance without calling
    __init__, so unit tests do not need
    the large YOLO weights.
    """

    detector = (
        DefectDetector
        .__new__(
            DefectDetector
        )
    )

    detector.model = None

    detector.quality_policy = (
        load_quality_policy(
            QUALITY_POLICY_PATH
        )
    )

    return detector


def test_backend_uses_optimized_image_size():

    assert (
        IMAGE_SIZE
        == 416
    )


def test_backend_uses_optimized_model_path():

    normalized = (
        str(
            MODEL_PATH
        )
        .replace(
            "\\",
            "/",
        )
    )

    assert (
        normalized.endswith(
            "models/optimized/"
            "yolo_neu_optimized_best.pt"
        )
    )


def test_quality_policy_exists():

    assert (
        QUALITY_POLICY_PATH
        .exists()
    )


def test_invalid_confidence_rejected():

    with pytest.raises(
        ValueError
    ):
        (
            DefectDetector
            .validate_confidence(
                1.5
            )
        )


def test_valid_confidence():

    result = (
        DefectDetector
        .validate_confidence(
            0.60
        )
    )

    assert (
        result
        == pytest.approx(
            0.60
        )
    )


def test_backend_detection_format_works_with_quality():

    detector = (
        make_detector_without_model()
    )

    image = np.zeros(
        (
            200,
            200,
            3,
        ),
        dtype=np.uint8,
    )

    detections = [
        {
            "class_id":
                1,

            "class_name":
                "inclusion",

            "confidence":
                0.60,

            "bbox": {
                "x1":
                    20,

                "y1":
                    20,

                "x2":
                    60,

                "y2":
                    60,
            },
        }
    ]

    quality = (
        detector.apply_quality(
            image=image,
            detections=detections,
        )
    )

    assert (
        quality[
            "final_detection_count"
        ]
        == 1
    )

    assert (
        quality[
            "decision"
        ]
        == "REVIEW"
    )


def test_prediction_payload():

    image = np.zeros(
        (
            200,
            200,
            3,
        ),
        dtype=np.uint8,
    )

    result = {
        "detection_count":
            1,

        "inference_time_ms":
            25.0,

        "detections":
            [],

        "quality": {
            "decision":
                "REVIEW",

            "decision_reasons":
                [
                    "Test reason"
                ],

            "final_detection_count":
                1,

            "triage_counts": {
                "LOW":
                    0,

                "MEDIUM":
                    1,

                "HIGH":
                    0,
            },

            "detections": [
                {
                    "class_name":
                        "scratches",

                    "confidence":
                        0.70,

                    "triage_level":
                        "MEDIUM",
                }
            ],

            "policy_note":
                "Test policy note",
        },
    }

    payload = (
        build_prediction_payload(
            filename="sample.jpg",
            image=image,
            result=result,
        )
    )

    assert (
        payload[
            "filename"
        ]
        == "sample.jpg"
    )

    assert (
        payload[
            "quality_decision"
        ]
        == "REVIEW"
    )

    assert (
        payload[
            "raw_detection_count"
        ]
        == 1
    )

    assert (
        payload[
            "detection_count"
        ]
        == 1
    )

    assert (
        payload[
            "image"
        ][
            "width"
        ]
        == 200
    )