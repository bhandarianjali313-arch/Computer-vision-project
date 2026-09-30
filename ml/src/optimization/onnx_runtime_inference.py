from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from ml.src.data.voc_to_yolo import CLASS_NAMES


@dataclass
class LetterboxMetadata:
    """
    Information required to map predictions from
    model-input coordinates back to the original image.
    """

    original_width: int
    original_height: int

    input_width: int
    input_height: int

    scale: float

    pad_left: int
    pad_top: int


def load_runtime_config(
    config_path: Path,
) -> dict[str, Any]:
    """
    Load Day 18 ONNX Runtime configuration.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"ONNX Runtime config not found: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "ONNX Runtime config must be "
            "a YAML mapping."
        )

    runtime = config.get(
        "runtime"
    )

    benchmark = config.get(
        "benchmark"
    )

    parity = config.get(
        "parity"
    )

    if not isinstance(
        runtime,
        dict,
    ):
        raise ValueError(
            "Configuration must contain "
            "a 'runtime' section."
        )

    if not isinstance(
        benchmark,
        dict,
    ):
        raise ValueError(
            "Configuration must contain "
            "a 'benchmark' section."
        )

    if not isinstance(
        parity,
        dict,
    ):
        raise ValueError(
            "Configuration must contain "
            "a 'parity' section."
        )

    runtime_required = {
        "imgsz",
        "confidence_threshold",
        "iou_threshold",
        "max_detections",
        "provider",
    }

    benchmark_required = {
        "sample_count",
        "warmup_count",
    }

    parity_required = {
        "sample_count",
        "match_iou_threshold",
        "confidence_tolerance",
        "minimum_match_rate",
    }

    missing_runtime = (
        runtime_required
        - set(runtime)
    )

    missing_benchmark = (
        benchmark_required
        - set(benchmark)
    )

    missing_parity = (
        parity_required
        - set(parity)
    )

    if missing_runtime:
        raise ValueError(
            "Missing runtime settings: "
            f"{sorted(missing_runtime)}"
        )

    if missing_benchmark:
        raise ValueError(
            "Missing benchmark settings: "
            f"{sorted(missing_benchmark)}"
        )

    if missing_parity:
        raise ValueError(
            "Missing parity settings: "
            f"{sorted(missing_parity)}"
        )

    return config


def letterbox_image(
    image: np.ndarray,
    target_size: int,
) -> tuple[
    np.ndarray,
    LetterboxMetadata,
]:
    """
    Resize an image while preserving aspect ratio.

    The remaining area is padded with the standard
    YOLO value 114.
    """

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    if image.ndim != 3:
        raise ValueError(
            "Expected an HWC color image."
        )

    original_height, original_width = (
        image.shape[:2]
    )

    if (
        original_width <= 0
        or original_height <= 0
    ):
        raise ValueError(
            "Image dimensions must be positive."
        )

    scale = min(
        target_size / original_width,
        target_size / original_height,
    )

    resized_width = int(
        round(
            original_width
            * scale
        )
    )

    resized_height = int(
        round(
            original_height
            * scale
        )
    )

    resized = cv2.resize(
        image,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    horizontal_padding = (
        target_size
        - resized_width
    )

    vertical_padding = (
        target_size
        - resized_height
    )

    pad_left = (
        horizontal_padding // 2
    )

    pad_right = (
        horizontal_padding
        - pad_left
    )

    pad_top = (
        vertical_padding // 2
    )

    pad_bottom = (
        vertical_padding
        - pad_top
    )

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(
            114,
            114,
            114,
        ),
    )

    metadata = (
        LetterboxMetadata(
            original_width=(
                original_width
            ),
            original_height=(
                original_height
            ),
            input_width=(
                target_size
            ),
            input_height=(
                target_size
            ),
            scale=float(
                scale
            ),
            pad_left=int(
                pad_left
            ),
            pad_top=int(
                pad_top
            ),
        )
    )

    return (
        padded,
        metadata,
    )


def preprocess_image(
    image: np.ndarray,
    target_size: int,
) -> tuple[
    np.ndarray,
    LetterboxMetadata,
]:
    """
    Convert OpenCV BGR HWC image into the float32
    RGB NCHW tensor expected by YOLO ONNX.
    """

    letterboxed, metadata = (
        letterbox_image(
            image,
            target_size,
        )
    )

    rgb = cv2.cvtColor(
        letterboxed,
        cv2.COLOR_BGR2RGB,
    )

    tensor = (
        rgb
        .astype(
            np.float32
        )
        / 255.0
    )

    tensor = np.transpose(
        tensor,
        (
            2,
            0,
            1,
        ),
    )

    tensor = np.expand_dims(
        tensor,
        axis=0,
    )

    tensor = np.ascontiguousarray(
        tensor,
        dtype=np.float32,
    )

    return (
        tensor,
        metadata,
    )


def normalize_yolo_output(
    output: np.ndarray,
    number_of_classes: int,
) -> np.ndarray:
    """
    Normalize common YOLOv8 ONNX output layouts.

    Typical Ultralytics output:
        [1, 4 + classes, predictions]

    We convert it to:
        [predictions, 4 + classes]
    """

    array = np.asarray(
        output
    )

    if (
        array.ndim == 3
        and array.shape[0] == 1
    ):
        array = array[0]

    if array.ndim != 2:
        raise ValueError(
            "Unsupported YOLO ONNX output "
            f"shape: {array.shape}"
        )

    expected_features = (
        4
        + number_of_classes
    )

    if (
        array.shape[0]
        == expected_features
    ):
        array = array.T

    elif (
        array.shape[1]
        == expected_features
    ):
        pass

    else:
        raise ValueError(
            "Could not determine YOLO output "
            "layout. Expected one dimension "
            f"to equal {expected_features}, "
            f"but received {array.shape}."
        )

    return np.asarray(
        array,
        dtype=np.float32,
    )


def xywh_to_xyxy(
    boxes: np.ndarray,
) -> np.ndarray:
    """
    Convert center-x, center-y, width, height
    into x1, y1, x2, y2.
    """

    boxes = np.asarray(
        boxes,
        dtype=np.float32,
    )

    result = np.empty_like(
        boxes,
        dtype=np.float32,
    )

    result[:, 0] = (
        boxes[:, 0]
        - boxes[:, 2] / 2.0
    )

    result[:, 1] = (
        boxes[:, 1]
        - boxes[:, 3] / 2.0
    )

    result[:, 2] = (
        boxes[:, 0]
        + boxes[:, 2] / 2.0
    )

    result[:, 3] = (
        boxes[:, 1]
        + boxes[:, 3] / 2.0
    )

    return result


def restore_boxes_to_original(
    boxes: np.ndarray,
    metadata: LetterboxMetadata,
) -> np.ndarray:
    """
    Undo letterbox padding/scaling.
    """

    restored = np.asarray(
        boxes,
        dtype=np.float32,
    ).copy()

    restored[:, [0, 2]] -= (
        metadata.pad_left
    )

    restored[:, [1, 3]] -= (
        metadata.pad_top
    )

    restored /= (
        metadata.scale
    )

    restored[:, 0] = np.clip(
        restored[:, 0],
        0,
        metadata.original_width,
    )

    restored[:, 2] = np.clip(
        restored[:, 2],
        0,
        metadata.original_width,
    )

    restored[:, 1] = np.clip(
        restored[:, 1],
        0,
        metadata.original_height,
    )

    restored[:, 3] = np.clip(
        restored[:, 3],
        0,
        metadata.original_height,
    )

    return restored


def class_aware_nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    confidence_threshold: float,
    iou_threshold: float,
) -> list[int]:
    """
    Apply OpenCV NMS independently to each class.

    This is class-aware NMS, matching the intended
    behavior of standard object-detection pipelines.
    """

    selected_indices = []

    unique_classes = np.unique(
        class_ids
    )

    for class_id in unique_classes:
        class_mask = (
            class_ids
            == class_id
        )

        original_indices = np.where(
            class_mask
        )[0]

        class_boxes = (
            boxes_xyxy[
                class_mask
            ]
        )

        class_scores = (
            scores[
                class_mask
            ]
        )

        opencv_boxes = []

        for box in class_boxes:
            x1, y1, x2, y2 = (
                box.tolist()
            )

            opencv_boxes.append(
                [
                    float(x1),
                    float(y1),
                    float(
                        max(
                            0.0,
                            x2 - x1,
                        )
                    ),
                    float(
                        max(
                            0.0,
                            y2 - y1,
                        )
                    ),
                ]
            )

        nms_indices = (
            cv2.dnn.NMSBoxes(
                bboxes=opencv_boxes,
                scores=(
                    class_scores
                    .astype(float)
                    .tolist()
                ),
                score_threshold=float(
                    confidence_threshold
                ),
                nms_threshold=float(
                    iou_threshold
                ),
            )
        )

        if (
            nms_indices is None
            or len(nms_indices) == 0
        ):
            continue

        flattened = np.asarray(
            nms_indices
        ).reshape(-1)

        for local_index in flattened:
            selected_indices.append(
                int(
                    original_indices[
                        int(
                            local_index
                        )
                    ]
                )
            )

    selected_indices.sort(
        key=lambda index: float(
            scores[index]
        ),
        reverse=True,
    )

    return selected_indices


def decode_yolo_output(
    raw_output: np.ndarray,
    metadata: LetterboxMetadata,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    """
    Decode raw YOLOv8 ONNX output into the same
    detection representation used elsewhere
    in the project.
    """

    predictions = (
        normalize_yolo_output(
            raw_output,
            number_of_classes=(
                len(
                    CLASS_NAMES
                )
            ),
        )
    )

    if predictions.size == 0:
        return []

    box_xywh = predictions[
        :,
        :4
    ]

    class_scores = predictions[
        :,
        4:
    ]

    class_ids = np.argmax(
        class_scores,
        axis=1,
    ).astype(
        np.int32
    )

    scores = class_scores[
        np.arange(
            len(
                predictions
            )
        ),
        class_ids,
    ]

    keep = (
        scores
        >= confidence_threshold
    )

    if not np.any(
        keep
    ):
        return []

    box_xywh = box_xywh[
        keep
    ]

    scores = scores[
        keep
    ]

    class_ids = class_ids[
        keep
    ]

    boxes_xyxy = xywh_to_xyxy(
        box_xywh
    )

    boxes_xyxy = (
        restore_boxes_to_original(
            boxes_xyxy,
            metadata,
        )
    )

    selected = class_aware_nms(
        boxes_xyxy=boxes_xyxy,
        scores=scores,
        class_ids=class_ids,
        confidence_threshold=(
            confidence_threshold
        ),
        iou_threshold=(
            iou_threshold
        ),
    )

    selected = selected[
        :max_detections
    ]

    detections = []

    for index in selected:
        class_id = int(
            class_ids[
                index
            ]
        )

        x1, y1, x2, y2 = [
            float(value)
            for value
            in boxes_xyxy[
                index
            ].tolist()
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
                        scores[
                            index
                        ]
                    ),

                "bbox_xyxy": {
                    "x1":
                        x1,

                    "y1":
                        y1,

                    "x2":
                        x2,

                    "y2":
                        y2,
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


def summarize_latency(
    values_ms: list[float],
) -> dict[str, float]:
    """
    Calculate latency and throughput statistics.
    """

    if not values_ms:
        raise ValueError(
            "Latency values cannot be empty."
        )

    mean_ms = float(
        statistics.mean(
            values_ms
        )
    )

    median_ms = float(
        statistics.median(
            values_ms
        )
    )

    p95_ms = float(
        np.percentile(
            values_ms,
            95,
        )
    )

    fps = (
        1000.0 / mean_ms
        if mean_ms > 0
        else 0.0
    )

    return {
        "mean_ms":
            mean_ms,

        "median_ms":
            median_ms,

        "p95_ms":
            p95_ms,

        "approx_fps":
            float(
                fps
            ),
    }


class ONNXRuntimeDetector:
    """
    Direct ONNX Runtime YOLO detector.

    Ultralytics is NOT used to execute
    the ONNX model in this class.
    """

    def __init__(
        self,
        model_path: Path,
        imgsz: int,
        confidence_threshold: float,
        iou_threshold: float,
        max_detections: int,
        provider: str = (
            "CPUExecutionProvider"
        ),
    ):
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: "
                f"{model_path}"
            )

        available_providers = (
            ort.get_available_providers()
        )

        if (
            provider
            not in available_providers
        ):
            raise ValueError(
                f"Requested provider "
                f"'{provider}' is unavailable. "
                f"Available providers: "
                f"{available_providers}"
            )

        self.model_path = (
            model_path.resolve()
        )

        self.imgsz = int(
            imgsz
        )

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.iou_threshold = float(
            iou_threshold
        )

        self.max_detections = int(
            max_detections
        )

        self.provider = (
            provider
        )

        session_options = (
            ort.SessionOptions()
        )

        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel
            .ORT_ENABLE_ALL
        )

        self.session = (
            ort.InferenceSession(
                str(
                    self.model_path
                ),
                sess_options=(
                    session_options
                ),
                providers=[
                    provider
                ],
            )
        )

        inputs = (
            self.session
            .get_inputs()
        )

        outputs = (
            self.session
            .get_outputs()
        )

        if len(inputs) != 1:
            raise ValueError(
                "Expected exactly one "
                f"ONNX input, found "
                f"{len(inputs)}."
            )

        if not outputs:
            raise ValueError(
                "ONNX model exposes "
                "no outputs."
            )

        self.input_name = (
            inputs[0].name
        )

        self.output_names = [
            output.name
            for output
            in outputs
        ]

    def predict(
        self,
        image: np.ndarray,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, float],
    ]:
        """
        Run preprocessing, ONNX execution,
        and custom postprocessing.
        """

        preprocess_start = (
            time.perf_counter()
        )

        tensor, metadata = (
            preprocess_image(
                image,
                self.imgsz,
            )
        )

        preprocess_ms = (
            (
                time.perf_counter()
                - preprocess_start
            )
            * 1000.0
        )

        inference_start = (
            time.perf_counter()
        )

        outputs = (
            self.session.run(
                self.output_names,
                {
                    self.input_name:
                        tensor
                },
            )
        )

        inference_ms = (
            (
                time.perf_counter()
                - inference_start
            )
            * 1000.0
        )

        if not outputs:
            raise RuntimeError(
                "ONNX Runtime returned "
                "no outputs."
            )

        postprocess_start = (
            time.perf_counter()
        )

        detections = (
            decode_yolo_output(
                raw_output=(
                    outputs[0]
                ),
                metadata=metadata,
                confidence_threshold=(
                    self
                    .confidence_threshold
                ),
                iou_threshold=(
                    self
                    .iou_threshold
                ),
                max_detections=(
                    self
                    .max_detections
                ),
            )
        )

        postprocess_ms = (
            (
                time.perf_counter()
                - postprocess_start
            )
            * 1000.0
        )

        total_ms = (
            preprocess_ms
            + inference_ms
            + postprocess_ms
        )

        timing = {
            "preprocess_ms":
                float(
                    preprocess_ms
                ),

            "inference_ms":
                float(
                    inference_ms
                ),

            "postprocess_ms":
                float(
                    postprocess_ms
                ),

            "total_ms":
                float(
                    total_ms
                ),
        }

        return (
            detections,
            timing,
        )