from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from ml.src.data.voc_to_yolo import CLASS_NAMES


def load_inference_config(
    config_path: Path,
) -> dict[str, Any]:
    """
    Load image inference configuration.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Inference config not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Inference config must be a YAML mapping."
        )

    inference = config.get(
        "inference"
    )

    if not isinstance(
        inference,
        dict,
    ):
        raise ValueError(
            "Configuration must contain "
            "an 'inference' section."
        )

    required = {
        "imgsz",
        "confidence_threshold",
        "iou_threshold",
        "max_detections",
    }

    missing = required - set(
        inference
    )

    if missing:
        raise ValueError(
            "Missing inference settings: "
            f"{sorted(missing)}"
        )

    return inference


def extract_detections(
    result,
) -> list[dict[str, Any]]:
    """
    Convert Ultralytics prediction boxes into
    a JSON-serializable representation.
    """

    detections: list[
        dict[str, Any]
    ] = []

    boxes = getattr(
        result,
        "boxes",
        None,
    )

    if boxes is None:
        return detections

    xyxy_tensor = getattr(
        boxes,
        "xyxy",
        None,
    )

    confidence_tensor = getattr(
        boxes,
        "conf",
        None,
    )

    class_tensor = getattr(
        boxes,
        "cls",
        None,
    )

    if (
        xyxy_tensor is None
        or confidence_tensor is None
        or class_tensor is None
    ):
        return detections

    xyxy_values = (
        xyxy_tensor
        .detach()
        .cpu()
        .numpy()
    )

    confidence_values = (
        confidence_tensor
        .detach()
        .cpu()
        .numpy()
    )

    class_values = (
        class_tensor
        .detach()
        .cpu()
        .numpy()
    )

    # No detections.
    if len(xyxy_values) == 0:
        return detections

    if not (
        len(xyxy_values)
        == len(confidence_values)
        == len(class_values)
    ):
        raise ValueError(
            "Prediction box, confidence, and class "
            "counts do not match."
        )

    for (
        xyxy,
        confidence,
        class_id,
    ) in zip(
        xyxy_values,
        confidence_values,
        class_values,
    ):
        class_id = int(
            class_id
        )

        if not (
            0 <= class_id
            < len(CLASS_NAMES)
        ):
            raise ValueError(
                f"Unexpected class ID: "
                f"{class_id}"
            )

        if len(xyxy) != 4:
            raise ValueError(
                "Expected bounding box in "
                "XYXY format with four values."
            )

        x1, y1, x2, y2 = [
            float(value)
            for value
            in xyxy.tolist()
        ]

        detections.append(
            {
                "class_id":
                    class_id,

                "class_name":
                    CLASS_NAMES[
                        class_id
                    ],

                "confidence":
                    float(
                        confidence
                    ),

                "bbox_xyxy": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },

                "width":
                    float(
                        x2 - x1
                    ),

                "height":
                    float(
                        y2 - y1
                    ),
            }
        )

    return detections


def annotate_image(
    image: np.ndarray,
    detections: list[
        dict[str, Any]
    ],
) -> np.ndarray:
    """
    Draw model predictions on an image.
    """

    output = image.copy()

    for detection in detections:
        bbox = detection[
            "bbox_xyxy"
        ]

        x1 = int(
            round(
                bbox["x1"]
            )
        )

        y1 = int(
            round(
                bbox["y1"]
            )
        )

        x2 = int(
            round(
                bbox["x2"]
            )
        )

        y2 = int(
            round(
                bbox["y2"]
            )
        )

        class_id = detection[
            "class_id"
        ]

        color = (
            int(
                (
                    37
                    * (
                        class_id
                        + 1
                    )
                )
                % 255
            ),
            int(
                (
                    97
                    * (
                        class_id
                        + 1
                    )
                )
                % 255
            ),
            int(
                (
                    157
                    * (
                        class_id
                        + 1
                    )
                )
                % 255
            ),
        )

        cv2.rectangle(
            output,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            color,
            2,
        )

        label = (
            f"{detection['class_name']} "
            f"{detection['confidence']:.2f}"
        )

        cv2.putText(
            output,
            label,
            (
                x1,
                max(
                    y1 - 6,
                    12,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )

    return output


def build_prediction_report(
    source_path: Path,
    weights_path: Path,
    image_shape: tuple[int, ...],
    detections: list[
        dict[str, Any]
    ],
    inference_config: dict[
        str,
        Any,
    ],
) -> dict[str, Any]:
    """
    Build structured inference output.
    """

    height = int(
        image_shape[0]
    )

    width = int(
        image_shape[1]
    )

    defect_counts = {
        class_name: 0
        for class_name
        in CLASS_NAMES
    }

    for detection in detections:
        class_name = detection[
            "class_name"
        ]

        if class_name not in defect_counts:
            raise ValueError(
                f"Unexpected class name: "
                f"{class_name}"
            )

        defect_counts[
            class_name
        ] += 1

    return {
        "source_image":
            str(
                source_path.resolve()
            ),

        "model_weights":
            str(
                weights_path.resolve()
            ),

        "image": {
            "width":
                width,

            "height":
                height,
        },

        "inference": {
            "imgsz":
                inference_config[
                    "imgsz"
                ],

            "confidence_threshold":
                inference_config[
                    "confidence_threshold"
                ],

            "iou_threshold":
                inference_config[
                    "iou_threshold"
                ],
        },

        "detection_count":
            len(detections),

        "defect_counts":
            defect_counts,

        "detections":
            detections,
    }


def save_prediction_json(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save prediction results.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )