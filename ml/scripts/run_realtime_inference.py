import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

from ml.src.inference.image_inference import (
    annotate_image,
    extract_detections,
)
from ml.src.inference.video_inference import (
    calculate_live_fps,
    create_video_writer,
    get_capture_fps,
    load_video_inference_config,
    overlay_runtime_information,
    parse_video_source,
    save_video_report,
    summarize_video_run,
    update_class_counts,
)
from ml.src.training.yolo_trainer import (
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run real-time industrial defect "
            "detection from webcam or video."
        )
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help=(
            "Webcam number such as 0, "
            "or path to a video file."
        ),
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(
            "models/optimized/"
            "yolo_neu_optimized_best.pt"
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "ml/configs/"
            "video_inference.yaml"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help=(
            "Optional path for saving "
            "annotated output video."
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "outputs/inference/day16/"
            "realtime_report.json"
        ),
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help=(
            "Disable the OpenCV preview window."
        ),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Optional maximum number of "
            "frames to process."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: "
            f"{args.weights}"
        )

    if (
        args.max_frames is not None
        and args.max_frames <= 0
    ):
        raise ValueError(
            "--max-frames must be greater "
            "than zero."
        )

    config = (
        load_video_inference_config(
            args.config
        )
    )

    device = resolve_device(
        args.device
    )

    source = parse_video_source(
        args.source
    )

    print("=" * 78)
    print("REAL-TIME INDUSTRIAL DEFECT DETECTION")
    print("=" * 78)

    print(
        f"Source      : "
        f"{args.source}"
    )

    print(
        f"Model       : "
        f"{args.weights}"
    )

    print(
        f"Device      : "
        f"{device}"
    )

    print(
        f"Image size  : "
        f"{config['imgsz']}"
    )

    print(
        f"Confidence  : "
        f"{config['confidence_threshold']}"
    )

    print(
        f"Display     : "
        f"{not args.no_display}"
    )

    print(
        f"Save video  : "
        f"{args.output_video is not None}"
    )

    print("=" * 78)

    capture = cv2.VideoCapture(
        source
    )

    if not capture.isOpened():
        raise RuntimeError(
            "Unable to open video source: "
            f"{args.source}"
        )

    source_fps = get_capture_fps(
        capture,
        fallback=float(
            config[
                "fallback_output_fps"
            ]
        ),
    )

    model = YOLO(
        str(args.weights)
    )

    writer = None

    frames_processed = 0
    total_detections = 0

    class_counts = Counter()

    all_processing_ms = []

    fps_window = int(
        config["fps_window"]
    )

    recent_processing_ms = deque(
        maxlen=fps_window
    )

    run_start = (
        time.perf_counter()
    )

    stop_reason = (
        "end_of_stream"
    )

    try:
        while True:
            success, frame = (
                capture.read()
            )

            if not success:
                stop_reason = (
                    "end_of_stream"
                )
                break

            frame_start = (
                time.perf_counter()
            )

            results = model.predict(
                source=frame,
                imgsz=int(
                    config[
                        "imgsz"
                    ]
                ),
                conf=float(
                    config[
                        "confidence_threshold"
                    ]
                ),
                iou=float(
                    config[
                        "iou_threshold"
                    ]
                ),
                max_det=int(
                    config[
                        "max_detections"
                    ]
                ),
                device=device,
                verbose=False,
            )

            if len(results) != 1:
                raise RuntimeError(
                    "Expected one result "
                    "for one video frame."
                )

            detections = (
                extract_detections(
                    results[0]
                )
            )

            annotated = (
                annotate_image(
                    frame,
                    detections,
                )
            )

            processing_ms = (
                (
                    time.perf_counter()
                    - frame_start
                )
                * 1000.0
            )

            all_processing_ms.append(
                processing_ms
            )

            recent_processing_ms.append(
                processing_ms
            )

            live_fps = (
                calculate_live_fps(
                    list(
                        recent_processing_ms
                    )
                )
            )

            frames_processed += 1

            total_detections += len(
                detections
            )

            update_class_counts(
                class_counts,
                detections,
            )

            display_frame = (
                overlay_runtime_information(
                    frame=annotated,
                    frame_number=(
                        frames_processed
                    ),
                    detection_count=(
                        len(
                            detections
                        )
                    ),
                    fps=live_fps,
                    show_frame_number=bool(
                        config.get(
                            "show_frame_number",
                            True,
                        )
                    ),
                    show_detection_count=bool(
                        config.get(
                            "show_detection_count",
                            True,
                        )
                    ),
                    show_fps=bool(
                        config.get(
                            "show_fps",
                            True,
                        )
                    ),
                )
            )

            if (
                args.output_video
                is not None
                and writer is None
            ):
                height, width = (
                    display_frame.shape[
                        :2
                    ]
                )

                writer = (
                    create_video_writer(
                        output_path=(
                            args.output_video
                        ),
                        frame_width=width,
                        frame_height=height,
                        fps=source_fps,
                    )
                )

            if writer is not None:
                writer.write(
                    display_frame
                )

            if not args.no_display:
                cv2.imshow(
                    "Industrial Defect Detection",
                    display_frame,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    27,
                ):
                    stop_reason = (
                        "user_exit"
                    )
                    break

            if (
                args.max_frames
                is not None
                and frames_processed
                >= args.max_frames
            ):
                stop_reason = (
                    "max_frames"
                )
                break

    finally:
        capture.release()

        if writer is not None:
            writer.release()

        if not args.no_display:
            cv2.destroyAllWindows()

    wall_elapsed_seconds = (
        time.perf_counter()
        - run_start
    )

    statistics = (
        summarize_video_run(
            frame_processing_ms=(
                all_processing_ms
            ),
            frames_processed=(
                frames_processed
            ),
            total_detections=(
                total_detections
            ),
            class_counts=(
                class_counts
            ),
            wall_elapsed_seconds=(
                wall_elapsed_seconds
            ),
            source_fps=(
                source_fps
            ),
        )
    )

    report = {
        "source":
            args.source,

        "weights":
            str(
                args.weights.resolve()
            ),

        "device":
            device,

        "configuration": {
            "imgsz":
                config["imgsz"],

            "confidence_threshold":
                config[
                    "confidence_threshold"
                ],

            "iou_threshold":
                config[
                    "iou_threshold"
                ],

            "max_detections":
                config[
                    "max_detections"
                ],
        },

        "stop_reason":
            stop_reason,

        "output_video":
            (
                str(
                    args.output_video.resolve()
                )
                if args.output_video
                is not None
                else None
            ),

        "statistics":
            statistics,
    }

    save_video_report(
        report,
        args.report,
    )

    print("\n" + "=" * 78)
    print("REAL-TIME INFERENCE SUMMARY")
    print("=" * 78)

    print(
        f"Frames processed       : "
        f"{statistics['frames_processed']}"
    )

    print(
        f"Total detections       : "
        f"{statistics['total_detections']}"
    )

    print(
        f"Average detections     : "
        f"{statistics['average_detections_per_frame']:.2f}"
    )

    print(
        f"Mean processing time   : "
        f"{statistics['mean_processing_ms']:.2f} ms"
    )

    print(
        f"P95 processing time    : "
        f"{statistics['p95_processing_ms']:.2f} ms"
    )

    print(
        f"Processing FPS         : "
        f"{statistics['processing_fps']:.2f}"
    )

    print(
        f"Wall-stream FPS        : "
        f"{statistics['wall_stream_fps']:.2f}"
    )

    print("\nDetections by class:")

    for (
        class_name,
        count,
    ) in statistics[
        "detections_by_class"
    ].items():
        print(
            f"  {class_name:<20} "
            f"{count}"
        )

    print(
        f"\nStop reason : "
        f"{stop_reason}"
    )

    print(
        f"Report      : "
        f"{args.report}"
    )

    if args.output_video:
        print(
            f"Video       : "
            f"{args.output_video}"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()