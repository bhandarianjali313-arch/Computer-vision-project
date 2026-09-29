from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import onnx
import yaml


def load_onnx_export_config(
    config_path: Path,
) -> dict[str, Any]:
    """
    Load ONNX export configuration.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"ONNX export config not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "ONNX export configuration must be a YAML mapping."
        )

    export_config = config.get(
        "export"
    )

    parity_config = config.get(
        "parity"
    )

    if not isinstance(
        export_config,
        dict,
    ):
        raise ValueError(
            "Configuration must contain an 'export' section."
        )

    if not isinstance(
        parity_config,
        dict,
    ):
        raise ValueError(
            "Configuration must contain a 'parity' section."
        )

    required_export = {
        "imgsz",
        "batch",
        "opset",
        "dynamic",
        "simplify",
        "half",
    }

    missing_export = (
        required_export
        - set(export_config)
    )

    if missing_export:
        raise ValueError(
            "Missing export settings: "
            f"{sorted(missing_export)}"
        )

    required_parity = {
        "confidence_threshold",
        "iou_threshold",
        "sample_count",
        "match_iou_threshold",
        "confidence_tolerance",
        "minimum_match_rate",
    }

    missing_parity = (
        required_parity
        - set(parity_config)
    )

    if missing_parity:
        raise ValueError(
            "Missing parity settings: "
            f"{sorted(missing_parity)}"
        )

    return config


def calculate_sha256(
    file_path: Path,
) -> str:
    """
    Calculate SHA-256 for a model artifact.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open(
        "rb",
    ) as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def tensor_shape(
    value_info,
) -> list[int | str | None]:
    """
    Extract a readable ONNX tensor shape.
    """

    tensor_type = (
        value_info
        .type
        .tensor_type
    )

    shape = []

    for dimension in (
        tensor_type.shape.dim
    ):
        if dimension.HasField(
            "dim_value"
        ):
            shape.append(
                int(
                    dimension.dim_value
                )
            )

        elif dimension.HasField(
            "dim_param"
        ):
            shape.append(
                str(
                    dimension.dim_param
                )
            )

        else:
            shape.append(
                None
            )

    return shape


def inspect_onnx_model(
    model_path: Path,
) -> dict[str, Any]:
    """
    Validate and inspect an ONNX model.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}"
        )

    model = onnx.load(
        str(model_path)
    )

    onnx.checker.check_model(
        model
    )

    inputs = []

    for item in model.graph.input:
        inputs.append(
            {
                "name":
                    item.name,

                "shape":
                    tensor_shape(
                        item
                    ),
            }
        )

    outputs = []

    for item in model.graph.output:
        outputs.append(
            {
                "name":
                    item.name,

                "shape":
                    tensor_shape(
                        item
                    ),
            }
        )

    return {
        "valid":
            True,

        "ir_version":
            int(
                model.ir_version
            ),

        "opset_versions": [
            int(
                opset.version
            )
            for opset
            in model.opset_import
        ],

        "graph_name":
            model.graph.name,

        "node_count":
            len(
                model.graph.node
            ),

        "input_count":
            len(inputs),

        "output_count":
            len(outputs),

        "inputs":
            inputs,

        "outputs":
            outputs,

        "file_size_mb":
            (
                model_path
                .stat()
                .st_size
                / (1024 * 1024)
            ),

        "sha256":
            calculate_sha256(
                model_path
            ),
    }


def bbox_iou(
    first: dict[str, float],
    second: dict[str, float],
) -> float:
    """
    Calculate IoU for two bbox_xyxy dictionaries.
    """

    x1 = max(
        first["x1"],
        second["x1"],
    )

    y1 = max(
        first["y1"],
        second["y1"],
    )

    x2 = min(
        first["x2"],
        second["x2"],
    )

    y2 = min(
        first["y2"],
        second["y2"],
    )

    intersection_width = max(
        0.0,
        x2 - x1,
    )

    intersection_height = max(
        0.0,
        y2 - y1,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    first_area = max(
        0.0,
        first["x2"]
        - first["x1"],
    ) * max(
        0.0,
        first["y2"]
        - first["y1"],
    )

    second_area = max(
        0.0,
        second["x2"]
        - second["x1"],
    ) * max(
        0.0,
        second["y2"]
        - second["y1"],
    )

    union = (
        first_area
        + second_area
        - intersection
    )

    if union <= 0:
        return 0.0

    return float(
        intersection / union
    )


def compare_detections(
    pytorch_detections: list[
        dict[str, Any]
    ],
    onnx_detections: list[
        dict[str, Any]
    ],
    match_iou_threshold: float,
    confidence_tolerance: float,
) -> dict[str, Any]:
    """
    Match PyTorch detections against ONNX detections.

    Matching requires:
      - same class
      - sufficient bounding-box IoU
      - similar confidence
    """

    matched_onnx = set()

    matches = []

    unmatched_pytorch = []

    for pytorch_detection in (
        pytorch_detections
    ):
        best_index = None
        best_iou = 0.0

        for (
            index,
            onnx_detection,
        ) in enumerate(
            onnx_detections
        ):
            if index in matched_onnx:
                continue

            if (
                pytorch_detection[
                    "class_id"
                ]
                != onnx_detection[
                    "class_id"
                ]
            ):
                continue

            iou = bbox_iou(
                pytorch_detection[
                    "bbox_xyxy"
                ],
                onnx_detection[
                    "bbox_xyxy"
                ],
            )

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is None:
            unmatched_pytorch.append(
                pytorch_detection
            )
            continue

        onnx_detection = (
            onnx_detections[
                best_index
            ]
        )

        confidence_difference = abs(
            pytorch_detection[
                "confidence"
            ]
            - onnx_detection[
                "confidence"
            ]
        )

        if (
            best_iou
            >= match_iou_threshold
            and confidence_difference
            <= confidence_tolerance
        ):
            matched_onnx.add(
                best_index
            )

            matches.append(
                {
                    "class_name":
                        pytorch_detection[
                            "class_name"
                        ],

                    "iou":
                        float(
                            best_iou
                        ),

                    "confidence_difference":
                        float(
                            confidence_difference
                        ),
                }
            )

        else:
            unmatched_pytorch.append(
                pytorch_detection
            )

    unmatched_onnx = [
        detection
        for index, detection
        in enumerate(
            onnx_detections
        )
        if index not in matched_onnx
    ]

    denominator = max(
        len(
            pytorch_detections
        ),
        len(
            onnx_detections
        ),
    )

    if denominator == 0:
        match_rate = 1.0
    else:
        match_rate = (
            len(matches)
            / denominator
        )

    return {
        "pytorch_count":
            len(
                pytorch_detections
            ),

        "onnx_count":
            len(
                onnx_detections
            ),

        "matched_count":
            len(matches),

        "match_rate":
            float(
                match_rate
            ),

        "matches":
            matches,

        "unmatched_pytorch_count":
            len(
                unmatched_pytorch
            ),

        "unmatched_onnx_count":
            len(
                unmatched_onnx
            ),
    }


def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save JSON artifact.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )