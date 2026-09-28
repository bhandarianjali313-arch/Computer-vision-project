from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from ml.src.inference.video_inference import (
    calculate_live_fps,
    load_video_inference_config,
    overlay_runtime_information,
    parse_video_source,
    summarize_video_run,
    update_class_counts,
)


def test_parse_webcam_source():
    assert (
        parse_video_source(
            "0"
        )
        == 0
    )

    assert (
        parse_video_source(
            "1"
        )
        == 1
    )


def test_parse_video_file_source():
    source = (
        "videos/sample.mp4"
    )

    assert (
        parse_video_source(
            source
        )
        == source
    )


def test_load_video_config(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "video.yaml"
    )

    path.write_text(
        yaml.safe_dump(
            {
                "video_inference": {
                    "imgsz":
                        416,

                    "confidence_threshold":
                        0.25,

                    "iou_threshold":
                        0.45,

                    "max_detections":
                        100,

                    "fps_window":
                        30,

                    "fallback_output_fps":
                        20.0,
                }
            }
        ),
        encoding="utf-8",
    )

    config = (
        load_video_inference_config(
            path
        )
    )

    assert (
        config["imgsz"]
        == 416
    )

    assert (
        config[
            "fps_window"
        ]
        == 30
    )


def test_calculate_live_fps():
    fps = calculate_live_fps(
        [
            40.0,
            50.0,
            60.0,
        ]
    )

    assert fps == pytest.approx(
        20.0
    )


def test_update_class_counts():
    counts = Counter()

    detections = [
        {
            "class_name":
                "crazing"
        },
        {
            "class_name":
                "crazing"
        },
        {
            "class_name":
                "scratches"
        },
    ]

    update_class_counts(
        counts,
        detections,
    )

    assert (
        counts["crazing"]
        == 2
    )

    assert (
        counts["scratches"]
        == 1
    )


def test_runtime_overlay():
    frame = np.zeros(
        (
            200,
            300,
            3,
        ),
        dtype=np.uint8,
    )

    result = (
        overlay_runtime_information(
            frame=frame,
            frame_number=10,
            detection_count=2,
            fps=20.5,
        )
    )

    assert (
        result.shape
        == frame.shape
    )

    assert (
        result.sum()
        > 0
    )


def test_summarize_video_run():
    class_counts = Counter(
        {
            "crazing": 3,
            "scratches": 1,
        }
    )

    summary = (
        summarize_video_run(
            frame_processing_ms=[
                40.0,
                50.0,
                60.0,
            ],
            frames_processed=3,
            total_detections=4,
            class_counts=class_counts,
            wall_elapsed_seconds=0.18,
            source_fps=30.0,
        )
    )

    assert (
        summary[
            "frames_processed"
        ]
        == 3
    )

    assert (
        summary[
            "total_detections"
        ]
        == 4
    )

    assert (
        summary[
            "mean_processing_ms"
        ]
        == pytest.approx(
            50.0
        )
    )

    assert (
        summary[
            "processing_fps"
        ]
        == pytest.approx(
            20.0
        )
    )

    assert (
        summary[
            "detections_by_class"
        ]["crazing"]
        == 3
    )