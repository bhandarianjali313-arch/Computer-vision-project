from pathlib import Path

import numpy as np
import pytest
import yaml

from ml.src.optimization.onnx_runtime_inference import (
    LetterboxMetadata,
    class_aware_nms,
    decode_yolo_output,
    letterbox_image,
    load_runtime_config,
    normalize_yolo_output,
    preprocess_image,
    restore_boxes_to_original,
    summarize_latency,
)


def test_load_runtime_config(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "runtime.yaml"
    )

    path.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "imgsz":
                        416,

                    "confidence_threshold":
                        0.25,

                    "iou_threshold":
                        0.45,

                    "max_detections":
                        100,

                    "provider":
                        "CPUExecutionProvider",
                },

                "benchmark": {
                    "sample_count":
                        50,

                    "warmup_count":
                        5,
                },

                "parity": {
                    "sample_count":
                        10,

                    "match_iou_threshold":
                        0.85,

                    "confidence_tolerance":
                        0.05,

                    "minimum_match_rate":
                        0.90,
                },
            }
        ),
        encoding="utf-8",
    )

    config = (
        load_runtime_config(
            path
        )
    )

    assert (
        config["runtime"]["imgsz"]
        == 416
    )

    assert (
        config["benchmark"][
            "sample_count"
        ]
        == 50
    )


def test_letterbox_square_image():
    image = np.zeros(
        (
            200,
            200,
            3,
        ),
        dtype=np.uint8,
    )

    output, metadata = (
        letterbox_image(
            image,
            416,
        )
    )

    assert (
        output.shape
        == (
            416,
            416,
            3,
        )
    )

    assert (
        metadata.pad_left
        == 0
    )

    assert (
        metadata.pad_top
        == 0
    )

    assert (
        metadata.scale
        == pytest.approx(
            2.08
        )
    )


def test_preprocess_image():
    image = np.full(
        (
            200,
            200,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    tensor, _ = (
        preprocess_image(
            image,
            416,
        )
    )

    assert (
        tensor.shape
        == (
            1,
            3,
            416,
            416,
        )
    )

    assert (
        tensor.dtype
        == np.float32
    )

    assert (
        tensor.min()
        >= 0.0
    )

    assert (
        tensor.max()
        <= 1.0
    )


def test_normalize_channel_first_output():
    output = np.zeros(
        (
            1,
            10,
            100,
        ),
        dtype=np.float32,
    )

    normalized = (
        normalize_yolo_output(
            output,
            number_of_classes=6,
        )
    )

    assert (
        normalized.shape
        == (
            100,
            10,
        )
    )


def test_normalize_prediction_first_output():
    output = np.zeros(
        (
            1,
            100,
            10,
        ),
        dtype=np.float32,
    )

    normalized = (
        normalize_yolo_output(
            output,
            number_of_classes=6,
        )
    )

    assert (
        normalized.shape
        == (
            100,
            10,
        )
    )


def test_restore_boxes():
    metadata = (
        LetterboxMetadata(
            original_width=200,
            original_height=200,
            input_width=416,
            input_height=416,
            scale=2.08,
            pad_left=0,
            pad_top=0,
        )
    )

    boxes = np.array(
        [
            [
                20.8,
                41.6,
                208.0,
                249.6,
            ]
        ],
        dtype=np.float32,
    )

    restored = (
        restore_boxes_to_original(
            boxes,
            metadata,
        )
    )

    assert (
        restored[0][0]
        == pytest.approx(
            10.0,
            abs=0.01,
        )
    )

    assert (
        restored[0][1]
        == pytest.approx(
            20.0,
            abs=0.01,
        )
    )

    assert (
        restored[0][2]
        == pytest.approx(
            100.0,
            abs=0.01,
        )
    )

    assert (
        restored[0][3]
        == pytest.approx(
            120.0,
            abs=0.01,
        )
    )


def test_class_aware_nms():
    boxes = np.array(
        [
            [
                10,
                10,
                100,
                100,
            ],
            [
                12,
                12,
                98,
                98,
            ],
            [
                10,
                10,
                100,
                100,
            ],
        ],
        dtype=np.float32,
    )

    scores = np.array(
        [
            0.95,
            0.80,
            0.90,
        ],
        dtype=np.float32,
    )

    class_ids = np.array(
        [
            0,
            0,
            1,
        ],
        dtype=np.int32,
    )

    selected = (
        class_aware_nms(
            boxes_xyxy=boxes,
            scores=scores,
            class_ids=class_ids,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )
    )

    # The lower-confidence overlapping class-0
    # box should be suppressed.
    # The class-1 box remains because NMS
    # is performed independently per class.
    assert len(
        selected
    ) == 2

    assert 0 in selected
    assert 2 in selected


def test_decode_synthetic_prediction():
    metadata = (
        LetterboxMetadata(
            original_width=200,
            original_height=200,
            input_width=416,
            input_height=416,
            scale=2.08,
            pad_left=0,
            pad_top=0,
        )
    )

    # Shape:
    # 1 x (4 + 6 classes) x 1 prediction
    output = np.zeros(
        (
            1,
            10,
            1,
        ),
        dtype=np.float32,
    )

    # xywh in model-input coordinates.
    output[
        0,
        0,
        0,
    ] = 208.0

    output[
        0,
        1,
        0,
    ] = 208.0

    output[
        0,
        2,
        0,
    ] = 104.0

    output[
        0,
        3,
        0,
    ] = 104.0

    # Class 2 = patches
    output[
        0,
        4 + 2,
        0,
    ] = 0.90

    detections = (
        decode_yolo_output(
            raw_output=output,
            metadata=metadata,
            confidence_threshold=0.25,
            iou_threshold=0.45,
            max_detections=100,
        )
    )

    assert len(
        detections
    ) == 1

    assert (
        detections[0][
            "class_name"
        ]
        == "patches"
    )

    assert (
        detections[0][
            "confidence"
        ]
        == pytest.approx(
            0.90,
            abs=0.001,
        )
    )


def test_latency_summary():
    summary = summarize_latency(
        [
            40.0,
            50.0,
            60.0,
        ]
    )

    assert (
        summary["mean_ms"]
        == pytest.approx(
            50.0
        )
    )

    assert (
        summary["median_ms"]
        == pytest.approx(
            50.0
        )
    )

    assert (
        summary["approx_fps"]
        == pytest.approx(
            20.0
        )
    )