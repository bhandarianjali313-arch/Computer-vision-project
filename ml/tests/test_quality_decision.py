from pathlib import Path

import pytest
import yaml

from ml.src.inference.quality_decision import (
    apply_quality_policy,
    bbox_iou,
    calculate_area_ratio,
    deduplicate_detections,
    load_quality_policy,
)


def build_policy():
    return {
        "postprocessing": {
            "minimum_confidence":
                0.25,

            "duplicate_iou_threshold":
                0.80,
        },

        "triage": {
            "medium_confidence":
                0.50,

            "high_confidence":
                0.75,

            "medium_area_ratio":
                0.03,

            "high_area_ratio":
                0.08,

            "reject_detection_count":
                3,
        },

        "decision": {
            "pass_label":
                "PASS",

            "review_label":
                "REVIEW",

            "reject_label":
                "REJECT",
        },
    }


def test_load_quality_policy(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "policy.yaml"
    )

    path.write_text(
        yaml.safe_dump(
            build_policy()
        ),
        encoding="utf-8",
    )

    loaded = (
        load_quality_policy(
            path
        )
    )

    assert (
        loaded[
            "decision"
        ][
            "reject_label"
        ]
        == "REJECT"
    )


def test_bbox_iou_identical():
    box = {
        "x1": 10.0,
        "y1": 10.0,
        "x2": 100.0,
        "y2": 100.0,
    }

    assert bbox_iou(
        box,
        box,
    ) == pytest.approx(
        1.0
    )


def test_duplicate_removal():

    detections = [
        {
            "class_name":
                "scratches",

            "confidence":
                0.90,

            "bbox_xyxy": {
                "x1": 10,
                "y1": 10,
                "x2": 100,
                "y2": 100,
            },
        },

        {
            "class_name":
                "scratches",

            "confidence":
                0.70,

            "bbox_xyxy": {
                "x1": 11,
                "y1": 11,
                "x2": 99,
                "y2": 99,
            },
        },
    ]

    result = (
        deduplicate_detections(
            detections,
            iou_threshold=0.80,
        )
    )

    assert len(result) == 1

    assert (
        result[0]["confidence"]
        == 0.90
    )


def test_area_ratio():

    detection = {
        "bbox_xyxy": {
            "x1": 0,
            "y1": 0,
            "x2": 100,
            "y2": 100,
        }
    }

    ratio = calculate_area_ratio(
        detection,
        image_width=200,
        image_height=200,
    )

    assert ratio == pytest.approx(
        0.25
    )


def test_pass_decision():

    result = apply_quality_policy(
        detections=[],
        image_width=200,
        image_height=200,
        policy=build_policy(),
    )

    assert (
        result["decision"]
        == "PASS"
    )


def test_review_decision():

    detections = [
        {
            "class_name":
                "inclusion",

            "confidence":
                0.60,

            "bbox_xyxy": {
                "x1": 20,
                "y1": 20,
                "x2": 60,
                "y2": 60,
            },
        }
    ]

    result = apply_quality_policy(
        detections=detections,
        image_width=200,
        image_height=200,
        policy=build_policy(),
    )

    assert (
        result["decision"]
        == "REVIEW"
    )


def test_reject_high_triage():

    detections = [
        {
            "class_name":
                "crazing",

            "confidence":
                0.90,

            "bbox_xyxy": {
                "x1": 20,
                "y1": 20,
                "x2": 100,
                "y2": 100,
            },
        }
    ]

    result = apply_quality_policy(
        detections=detections,
        image_width=200,
        image_height=200,
        policy=build_policy(),
    )

    assert (
        result["decision"]
        == "REJECT"
    )

    assert (
        result[
            "triage_counts"
        ]["HIGH"]
        == 1
    )


def test_reject_multiple_detections():

    detections = [
        {
            "class_name": "crazing",
            "confidence": 0.60,
            "bbox_xyxy": {
                "x1": 0,
                "y1": 0,
                "x2": 30,
                "y2": 30,
            },
        },

        {
            "class_name": "inclusion",
            "confidence": 0.60,
            "bbox_xyxy": {
                "x1": 50,
                "y1": 50,
                "x2": 80,
                "y2": 80,
            },
        },

        {
            "class_name": "scratches",
            "confidence": 0.60,
            "bbox_xyxy": {
                "x1": 100,
                "y1": 100,
                "x2": 130,
                "y2": 130,
            },
        },
    ]

    result = apply_quality_policy(
        detections=detections,
        image_width=200,
        image_height=200,
        policy=build_policy(),
    )

    assert (
        result["decision"]
        == "REJECT"
    )


def test_backend_bbox_format():

    detections = [
        {
            "class_name":
                "patches",

            "confidence":
                0.60,

            "bbox": {
                "x1": 10,
                "y1": 10,
                "x2": 50,
                "y2": 50,
            },
        }
    ]

    result = apply_quality_policy(
        detections=detections,
        image_width=200,
        image_height=200,
        policy=build_policy(),
    )

    assert (
        result[
            "final_detection_count"
        ]
        == 1
    )