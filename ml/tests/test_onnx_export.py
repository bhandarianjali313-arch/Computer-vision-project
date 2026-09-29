from pathlib import Path

import pytest
import yaml

from ml.src.optimization.onnx_export import (
    bbox_iou,
    calculate_sha256,
    compare_detections,
    load_onnx_export_config,
)


def test_load_onnx_export_config(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "onnx.yaml"
    )

    path.write_text(
        yaml.safe_dump(
            {
                "export": {
                    "imgsz": 416,
                    "batch": 1,
                    "opset": 12,
                    "dynamic": False,
                    "simplify": True,
                    "half": False,
                },

                "parity": {
                    "confidence_threshold":
                        0.25,

                    "iou_threshold":
                        0.45,

                    "sample_count":
                        10,

                    "match_iou_threshold":
                        0.90,

                    "confidence_tolerance":
                        0.05,

                    "minimum_match_rate":
                        0.95,
                },
            }
        ),
        encoding="utf-8",
    )

    config = (
        load_onnx_export_config(
            path
        )
    )

    assert (
        config["export"]["imgsz"]
        == 416
    )

    assert (
        config["export"]["batch"]
        == 1
    )

    assert (
        config["parity"][
            "minimum_match_rate"
        ]
        == 0.95
    )


def test_bbox_iou_identical():
    box = {
        "x1": 10.0,
        "y1": 20.0,
        "x2": 100.0,
        "y2": 120.0,
    }

    assert bbox_iou(
        box,
        box,
    ) == pytest.approx(
        1.0
    )


def test_bbox_iou_no_overlap():
    first = {
        "x1": 0.0,
        "y1": 0.0,
        "x2": 10.0,
        "y2": 10.0,
    }

    second = {
        "x1": 20.0,
        "y1": 20.0,
        "x2": 30.0,
        "y2": 30.0,
    }

    assert bbox_iou(
        first,
        second,
    ) == pytest.approx(
        0.0
    )


def test_compare_detections():
    pytorch = [
        {
            "class_id":
                0,

            "class_name":
                "crazing",

            "confidence":
                0.90,

            "bbox_xyxy": {
                "x1": 10.0,
                "y1": 10.0,
                "x2": 100.0,
                "y2": 100.0,
            },
        }
    ]

    onnx = [
        {
            "class_id":
                0,

            "class_name":
                "crazing",

            "confidence":
                0.89,

            "bbox_xyxy": {
                "x1": 10.5,
                "y1": 10.5,
                "x2": 100.5,
                "y2": 100.5,
            },
        }
    ]

    result = (
        compare_detections(
            pytorch_detections=(
                pytorch
            ),
            onnx_detections=(
                onnx
            ),
            match_iou_threshold=(
                0.90
            ),
            confidence_tolerance=(
                0.05
            ),
        )
    )

    assert (
        result["matched_count"]
        == 1
    )

    assert (
        result["match_rate"]
        == pytest.approx(
            1.0
        )
    )


def test_sha256(
    tmp_path: Path,
):
    file_path = (
        tmp_path
        / "sample.bin"
    )

    file_path.write_bytes(
        b"industrial-defect"
    )

    first = calculate_sha256(
        file_path
    )

    second = calculate_sha256(
        file_path
    )

    assert first == second

    assert len(
        first
    ) == 64