from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from ml.src.inference.image_inference import (
    annotate_image,
    build_prediction_report,
    extract_detections,
    load_inference_config,
)


def test_load_inference_config(
    tmp_path: Path,
):
    path = (
        tmp_path / "inference.yaml"
    )

    path.write_text(
        yaml.safe_dump(
            {
                "inference": {
                    "imgsz": 416,
                    "confidence_threshold":
                        0.25,
                    "iou_threshold":
                        0.45,
                    "max_detections":
                        100,
                }
            }
        ),
        encoding="utf-8",
    )

    config = (
        load_inference_config(
            path
        )
    )

    assert (
        config["imgsz"]
        == 416
    )

    assert (
        config[
            "confidence_threshold"
        ]
        == 0.25
    )


def test_extract_detections():
    boxes = SimpleNamespace(
        xyxy=torch.tensor(
            [
                [
                    10.0,
                    20.0,
                    100.0,
                    120.0,
                ]
            ]
        ),

        conf=torch.tensor(
            [0.90]
        ),

        cls=torch.tensor(
            [1.0]
        ),
    )

    result = SimpleNamespace(
        boxes=boxes
    )

    detections = (
        extract_detections(
            result
        )
    )

    assert len(
        detections
    ) == 1

    assert (
        detections[0][
            "class_name"
        ]
        == "inclusion"
    )

    assert (
        detections[0][
            "confidence"
        ]
        > 0.89
    )


def test_annotate_image():
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    detections = [
        {
            "class_id": 0,
            "class_name":
                "crazing",
            "confidence":
                0.90,
            "bbox_xyxy": {
                "x1": 20.0,
                "y1": 20.0,
                "x2": 100.0,
                "y2": 100.0,
            },
            "width": 80.0,
            "height": 80.0,
        }
    ]

    output = annotate_image(
        image,
        detections,
    )

    assert (
        output.shape
        == image.shape
    )

    assert (
        output.sum()
        > 0
    )


def test_build_prediction_report(
    tmp_path: Path,
):
    image_path = (
        tmp_path / "sample.jpg"
    )

    weights_path = (
        tmp_path / "model.pt"
    )

    report = (
        build_prediction_report(
            source_path=image_path,
            weights_path=weights_path,
            image_shape=(
                200,
                200,
                3,
            ),
            detections=[
                {
                    "class_id": 5,
                    "class_name":
                        "scratches",
                    "confidence":
                        0.8,
                    "bbox_xyxy": {},
                    "width": 10.0,
                    "height": 30.0,
                }
            ],
            inference_config={
                "imgsz": 416,
                "confidence_threshold":
                    0.25,
                "iou_threshold":
                    0.45,
            },
        )
    )

    assert (
        report[
            "detection_count"
        ]
        == 1
    )

    assert (
        report[
            "defect_counts"
        ]["scratches"]
        == 1
    )

    assert (
        report["image"]["width"]
        == 200
    )