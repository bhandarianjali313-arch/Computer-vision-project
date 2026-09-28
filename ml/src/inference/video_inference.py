from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from ml.src.data.voc_to_yolo import CLASS_NAMES


def load_video_inference_config(
    config_path: Path,
) -> dict[str, Any]:
    """
    Load real-time video inference configuration.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Video inference config not found: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Video inference config must "
            "be a YAML mapping."
        )

    video_config = config.get(
        "video_inference"
    )

    if not isinstance(
        video_config,
        dict,
    ):
        raise ValueError(
            "Configuration must contain "
            "'video_inference'."
        )

    required = {
        "imgsz",
        "confidence_threshold",
        "iou_threshold",
        "max_detections",
        "fps_window",
        "fallback_output_fps",
    }

    missing = (
        required
        - set(video_config)
    )

    if missing:
        raise ValueError(
            "Missing video inference settings: "
            f"{sorted(missing)}"
        )

    return video_config


def parse_video_source(
    source: str,
) -> int | str:
    """
    Convert webcam numbers such as '0' or '1'
    into integers.

    File paths remain strings.
    """

    source = source.strip()

    try:
        camera_index = int(
            source
        )

        if camera_index >= 0:
            return camera_index

    except ValueError:
        pass

    return source


def get_capture_fps(
    capture,
    fallback: float = 20.0,
) -> float:
    """
    Read FPS from OpenCV capture.

    Some webcams return 0 or NaN, so a fallback
    value is required for output-video encoding.
    """

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    if (
        not math.isfinite(fps)
        or fps <= 0
    ):
        return float(
            fallback
        )

    return fps


def create_video_writer(
    output_path: Path,
    frame_width: int,
    frame_height: int,
    fps: float,
):
    """
    Create MP4 writer.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        float(fps),
        (
            int(frame_width),
            int(frame_height),
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Unable to create output video: "
            f"{output_path}"
        )

    return writer


def calculate_live_fps(
    recent_processing_ms: list[float],
) -> float:
    """
    Calculate rolling processing FPS.
    """

    if not recent_processing_ms:
        return 0.0

    mean_ms = statistics.mean(
        recent_processing_ms
    )

    if mean_ms <= 0:
        return 0.0

    return float(
        1000.0 / mean_ms
    )


def update_class_counts(
    class_counts: Counter,
    detections: list[
        dict[str, Any]
    ],
) -> None:
    """
    Add detections from one frame to cumulative
    class counts.
    """

    for detection in detections:
        class_name = detection[
            "class_name"
        ]

        if class_name not in CLASS_NAMES:
            raise ValueError(
                f"Unexpected defect class: "
                f"{class_name}"
            )

        class_counts[
            class_name
        ] += 1


def overlay_runtime_information(
    frame: np.ndarray,
    frame_number: int,
    detection_count: int,
    fps: float,
    show_frame_number: bool = True,
    show_detection_count: bool = True,
    show_fps: bool = True,
) -> np.ndarray:
    """
    Overlay live runtime information on a frame.
    """

    output = frame.copy()

    lines = []

    if show_frame_number:
        lines.append(
            f"Frame: {frame_number}"
        )

    if show_detection_count:
        lines.append(
            f"Detections: {detection_count}"
        )

    if show_fps:
        lines.append(
            f"Processing FPS: {fps:.1f}"
        )

    x = 8
    y = 18

    for line in lines:
        cv2.putText(
            output,
            line,
            (
                x,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                255,
                255,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            line,
            (
                x,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                0,
                0,
                0,
            ),
            1,
            cv2.LINE_AA,
        )

        y += 18

    return output


def summarize_video_run(
    frame_processing_ms: list[float],
    frames_processed: int,
    total_detections: int,
    class_counts: Counter,
    wall_elapsed_seconds: float,
    source_fps: float,
) -> dict[str, Any]:
    """
    Produce final video-inference statistics.
    """

    if frames_processed < 0:
        raise ValueError(
            "frames_processed cannot be negative."
        )

    if frame_processing_ms:
        mean_ms = statistics.mean(
            frame_processing_ms
        )

        median_ms = statistics.median(
            frame_processing_ms
        )

        p95_ms = float(
            np.percentile(
                frame_processing_ms,
                95,
            )
        )

        processing_fps = (
            1000.0 / mean_ms
            if mean_ms > 0
            else 0.0
        )

    else:
        mean_ms = 0.0
        median_ms = 0.0
        p95_ms = 0.0
        processing_fps = 0.0

    wall_fps = (
        frames_processed
        / wall_elapsed_seconds
        if wall_elapsed_seconds > 0
        else 0.0
    )

    average_detections = (
        total_detections
        / frames_processed
        if frames_processed > 0
        else 0.0
    )

    normalized_class_counts = {
        class_name:
            int(
                class_counts[
                    class_name
                ]
            )
        for class_name
        in CLASS_NAMES
    }

    return {
        "frames_processed":
            int(
                frames_processed
            ),

        "total_detections":
            int(
                total_detections
            ),

        "average_detections_per_frame":
            float(
                average_detections
            ),

        "source_fps":
            float(
                source_fps
            ),

        "wall_elapsed_seconds":
            float(
                wall_elapsed_seconds
            ),

        "wall_stream_fps":
            float(
                wall_fps
            ),

        "mean_processing_ms":
            float(
                mean_ms
            ),

        "median_processing_ms":
            float(
                median_ms
            ),

        "p95_processing_ms":
            float(
                p95_ms
            ),

        "processing_fps":
            float(
                processing_fps
            ),

        "detections_by_class":
            normalized_class_counts,
    }


def save_video_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save real-time inference statistics.
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