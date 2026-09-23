from pathlib import Path

import yaml

from ml.src.evaluation.error_analysis import (
    calculate_iou,
    load_error_analysis_config,
    match_detections,
)


def test_calculate_iou_identical():
    box = [
        10,
        10,
        50,
        50,
    ]

    assert (
        calculate_iou(
            box,
            box,
        )
        == 1.0
    )


def test_calculate_iou_no_overlap():
    first = [
        0,
        0,
        10,
        10,
    ]

    second = [
        20,
        20,
        30,
        30,
    ]

    assert (
        calculate_iou(
            first,
            second,
        )
        == 0.0
    )


def test_correct_detection_is_true_positive():
    ground_truth = [
        {
            "class_id": 0,
            "class_name": "crazing",
            "bbox": [
                10,
                10,
                50,
                50,
            ],
        }
    ]

    predictions = [
        {
            "class_id": 0,
            "class_name": "crazing",
            "confidence": 0.90,
            "bbox": [
                10,
                10,
                50,
                50,
            ],
        }
    ]

    result = match_detections(
        ground_truth,
        predictions,
        iou_threshold=0.5,
    )

    assert len(
        result["true_positives"]
    ) == 1

    assert len(
        result["false_positives"]
    ) == 0

    assert len(
        result["false_negatives"]
    ) == 0

    assert len(
        result["class_confusions"]
    ) == 0


def test_wrong_class_is_confusion():
    ground_truth = [
        {
            "class_id": 0,
            "class_name": "crazing",
            "bbox": [
                10,
                10,
                50,
                50,
            ],
        }
    ]

    predictions = [
        {
            "class_id": 5,
            "class_name": "scratches",
            "confidence": 0.80,
            "bbox": [
                10,
                10,
                50,
                50,
            ],
        }
    ]

    result = match_detections(
        ground_truth,
        predictions,
        iou_threshold=0.5,
    )

    assert len(
        result["class_confusions"]
    ) == 1

    assert len(
        result["true_positives"]
    ) == 0


def test_unmatched_boxes_become_fp_and_fn():
    ground_truth = [
        {
            "class_id": 0,
            "class_name": "crazing",
            "bbox": [
                0,
                0,
                20,
                20,
            ],
        }
    ]

    predictions = [
        {
            "class_id": 0,
            "class_name": "crazing",
            "confidence": 0.75,
            "bbox": [
                100,
                100,
                150,
                150,
            ],
        }
    ]

    result = match_detections(
        ground_truth,
        predictions,
        iou_threshold=0.5,
    )

    assert len(
        result["false_positives"]
    ) == 1

    assert len(
        result["false_negatives"]
    ) == 1


def test_load_error_analysis_config(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "error_analysis.yaml"
    )

    path.write_text(
        yaml.safe_dump(
            {
                "analysis": {
                    "confidence_threshold":
                        0.25,

                    "match_iou_threshold":
                        0.50,

                    "imgsz":
                        320,

                    "max_error_previews":
                        24,
                }
            }
        ),
        encoding="utf-8",
    )

    config = (
        load_error_analysis_config(
            path
        )
    )

    assert (
        config[
            "confidence_threshold"
        ]
        == 0.25
    )

    assert (
        config[
            "match_iou_threshold"
        ]
        == 0.50
    )