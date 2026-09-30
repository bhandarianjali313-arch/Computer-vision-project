import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from ml.src.optimization.onnx_export import (
    compare_detections,
)
from ml.src.optimization.onnx_runtime_inference import (
    ONNXRuntimeDetector,
    load_runtime_config,
    summarize_latency,
)
from ml.src.inference.image_inference import (
    extract_detections,
)
from ml.src.training.yolo_trainer import (
    resolve_device,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark direct ONNX Runtime "
            "against the PyTorch YOLO model."
        )
    )

    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path(
            "models/onnx/"
            "yolo_neu_optimized.onnx"
        ),
    )

    parser.add_argument(
        "--pytorch",
        type=Path,
        default=Path(
            "models/optimized/"
            "yolo_neu_optimized_best.pt"
        ),
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_yolo/images/val"
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "ml/configs/"
            "onnx_runtime.yaml"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "outputs/optimization/"
            "day18_runtime_benchmark.json"
        ),
    )

    return parser.parse_args()


def load_images(
    directory: Path,
    count: int,
):
    if not directory.exists():
        raise FileNotFoundError(
            f"Image directory not found: "
            f"{directory}"
        )

    paths = sorted(
        [
            path
            for path
            in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    paths = paths[
        :min(
            count,
            len(paths),
        )
    ]

    if not paths:
        raise FileNotFoundError(
            "No benchmark images found."
        )

    samples = []

    for path in paths:
        image = cv2.imread(
            str(path)
        )

        if image is None:
            raise ValueError(
                f"Could not read: "
                f"{path}"
            )

        samples.append(
            (
                path,
                image,
            )
        )

    return samples


def save_report(
    report,
    output_path: Path,
):
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


def main():
    args = parse_args()

    if not args.onnx.exists():
        raise FileNotFoundError(
            f"ONNX model not found: "
            f"{args.onnx}"
        )

    if not args.pytorch.exists():
        raise FileNotFoundError(
            f"PyTorch model not found: "
            f"{args.pytorch}"
        )

    config = (
        load_runtime_config(
            args.config
        )
    )

    runtime = config[
        "runtime"
    ]

    benchmark = config[
        "benchmark"
    ]

    parity = config[
        "parity"
    ]

    sample_count = int(
        benchmark[
            "sample_count"
        ]
    )

    warmup_count = int(
        benchmark[
            "warmup_count"
        ]
    )

    parity_count = int(
        parity[
            "sample_count"
        ]
    )

    imgsz = int(
        runtime["imgsz"]
    )

    confidence = float(
        runtime[
            "confidence_threshold"
        ]
    )

    iou_threshold = float(
        runtime[
            "iou_threshold"
        ]
    )

    device = resolve_device(
        args.device
    )

    samples = load_images(
        directory=args.images,
        count=sample_count,
    )

    print("=" * 82)
    print("DAY 18 - PYTORCH VS DIRECT ONNX RUNTIME")
    print("=" * 82)

    print(
        f"Images             : "
        f"{len(samples)}"
    )

    print(
        f"Image size         : "
        f"{imgsz}"
    )

    print(
        f"Confidence         : "
        f"{confidence}"
    )

    print(
        f"PyTorch device     : "
        f"{device}"
    )

    print(
        f"ONNX provider      : "
        f"{runtime['provider']}"
    )

    print(
        f"Warmup iterations  : "
        f"{warmup_count}"
    )

    print("=" * 82)

    print(
        "\nLoading models..."
    )

    pytorch_model = YOLO(
        str(args.pytorch)
    )

    onnx_detector = (
        ONNXRuntimeDetector(
            model_path=args.onnx,
            imgsz=imgsz,
            confidence_threshold=(
                confidence
            ),
            iou_threshold=(
                iou_threshold
            ),
            max_detections=int(
                runtime[
                    "max_detections"
                ]
            ),
            provider=str(
                runtime[
                    "provider"
                ]
            ),
        )
    )

    warmup_samples = samples[
        :min(
            warmup_count,
            len(samples),
        )
    ]

    print(
        "Running warm-up..."
    )

    for _, image in warmup_samples:
        pytorch_model.predict(
            source=image,
            imgsz=imgsz,
            conf=confidence,
            iou=iou_threshold,
            device=device,
            verbose=False,
        )

        onnx_detector.predict(
            image
        )

    pytorch_latency = []

    onnx_preprocess = []
    onnx_inference = []
    onnx_postprocess = []
    onnx_total = []

    parity_reports = []

    print(
        "\nBenchmarking images..."
    )

    for index, (
        image_path,
        image,
    ) in enumerate(
        samples,
        start=1,
    ):
        pytorch_start = (
            time.perf_counter()
        )

        pytorch_result = (
            pytorch_model.predict(
                source=image,
                imgsz=imgsz,
                conf=confidence,
                iou=iou_threshold,
                device=device,
                verbose=False,
            )[0]
        )

        pytorch_ms = (
            (
                time.perf_counter()
                - pytorch_start
            )
            * 1000.0
        )

        pytorch_latency.append(
            pytorch_ms
        )

        (
            onnx_detections,
            onnx_timing,
        ) = onnx_detector.predict(
            image
        )

        onnx_preprocess.append(
            onnx_timing[
                "preprocess_ms"
            ]
        )

        onnx_inference.append(
            onnx_timing[
                "inference_ms"
            ]
        )

        onnx_postprocess.append(
            onnx_timing[
                "postprocess_ms"
            ]
        )

        onnx_total.append(
            onnx_timing[
                "total_ms"
            ]
        )

        if index <= parity_count:
            pytorch_detections = (
                extract_detections(
                    pytorch_result
                )
            )

            comparison = (
                compare_detections(
                    pytorch_detections=(
                        pytorch_detections
                    ),
                    onnx_detections=(
                        onnx_detections
                    ),
                    match_iou_threshold=float(
                        parity[
                            "match_iou_threshold"
                        ]
                    ),
                    confidence_tolerance=float(
                        parity[
                            "confidence_tolerance"
                        ]
                    ),
                )
            )

            parity_reports.append(
                {
                    "image":
                        image_path.name,

                    **comparison,
                }
            )

        if (
            index % 10 == 0
            or index == len(
                samples
            )
        ):
            print(
                f"Processed "
                f"{index}/"
                f"{len(samples)}"
            )

    pytorch_summary = (
        summarize_latency(
            pytorch_latency
        )
    )

    onnx_total_summary = (
        summarize_latency(
            onnx_total
        )
    )

    onnx_inference_summary = (
        summarize_latency(
            onnx_inference
        )
    )

    onnx_preprocess_summary = (
        summarize_latency(
            onnx_preprocess
        )
    )

    onnx_postprocess_summary = (
        summarize_latency(
            onnx_postprocess
        )
    )

    total_pytorch_detections = sum(
        item[
            "pytorch_count"
        ]
        for item in parity_reports
    )

    total_onnx_detections = sum(
        item[
            "onnx_count"
        ]
        for item in parity_reports
    )

    total_matches = sum(
        item[
            "matched_count"
        ]
        for item in parity_reports
    )

    parity_denominator = max(
        total_pytorch_detections,
        total_onnx_detections,
    )

    if parity_denominator == 0:
        overall_match_rate = 1.0

    else:
        overall_match_rate = (
            total_matches
            / parity_denominator
        )

    required_match_rate = float(
        parity[
            "minimum_match_rate"
        ]
    )

    parity_passed = (
        overall_match_rate
        >= required_match_rate
    )

    speedup = (
        pytorch_summary[
            "mean_ms"
        ]
        / onnx_total_summary[
            "mean_ms"
        ]
        if (
            onnx_total_summary[
                "mean_ms"
            ]
            > 0
        )
        else 0.0
    )

    latency_reduction_percent = (
        (
            pytorch_summary[
                "mean_ms"
            ]
            - onnx_total_summary[
                "mean_ms"
            ]
        )
        / pytorch_summary[
            "mean_ms"
        ]
        * 100.0
        if (
            pytorch_summary[
                "mean_ms"
            ]
            > 0
        )
        else 0.0
    )

    report = {
        "environment": {
            "pytorch_device":
                device,

            "onnx_provider":
                runtime[
                    "provider"
                ],

            "imgsz":
                imgsz,

            "benchmark_images":
                len(samples),

            "warmup_count":
                warmup_count,
        },

        "pytorch_end_to_end":
            pytorch_summary,

        "onnx_runtime": {
            "preprocess":
                onnx_preprocess_summary,

            "inference_only":
                onnx_inference_summary,

            "postprocess":
                onnx_postprocess_summary,

            "end_to_end":
                onnx_total_summary,
        },

        "comparison": {
            "speedup_factor":
                float(
                    speedup
                ),

            "latency_reduction_percent":
                float(
                    latency_reduction_percent
                ),
        },

        "parity": {
            "images_checked":
                len(
                    parity_reports
                ),

            "pytorch_detections":
                total_pytorch_detections,

            "onnx_detections":
                total_onnx_detections,

            "matched_detections":
                total_matches,

            "overall_match_rate":
                float(
                    overall_match_rate
                ),

            "required_match_rate":
                required_match_rate,

            "passed":
                parity_passed,

            "images":
                parity_reports,
        },
    }

    save_report(
        report,
        args.report,
    )

    print("\n" + "=" * 82)
    print("END-TO-END PERFORMANCE")
    print("=" * 82)

    print(
        f"{'Runtime':<24}"
        f"{'Mean ms':>12}"
        f"{'Median':>12}"
        f"{'P95':>12}"
        f"{'FPS':>12}"
    )

    print("-" * 72)

    print(
        f"{'PyTorch':<24}"
        f"{pytorch_summary['mean_ms']:>12.2f}"
        f"{pytorch_summary['median_ms']:>12.2f}"
        f"{pytorch_summary['p95_ms']:>12.2f}"
        f"{pytorch_summary['approx_fps']:>12.2f}"
    )

    print(
        f"{'ONNX Runtime':<24}"
        f"{onnx_total_summary['mean_ms']:>12.2f}"
        f"{onnx_total_summary['median_ms']:>12.2f}"
        f"{onnx_total_summary['p95_ms']:>12.2f}"
        f"{onnx_total_summary['approx_fps']:>12.2f}"
    )

    print("\nONNX Runtime breakdown:")

    print(
        f"  Preprocess mean   : "
        f"{onnx_preprocess_summary['mean_ms']:.2f} ms"
    )

    print(
        f"  Inference mean    : "
        f"{onnx_inference_summary['mean_ms']:.2f} ms"
    )

    print(
        f"  Postprocess mean  : "
        f"{onnx_postprocess_summary['mean_ms']:.2f} ms"
    )

    print("\nPerformance difference:")

    print(
        f"  Speedup factor    : "
        f"{speedup:.2f}x"
    )

    print(
        f"  Latency reduction : "
        f"{latency_reduction_percent:.2f}%"
    )

    print("\n" + "=" * 82)
    print("CUSTOM ONNX PARITY")
    print("=" * 82)

    print(
        f"Images checked       : "
        f"{len(parity_reports)}"
    )

    print(
        f"PyTorch detections   : "
        f"{total_pytorch_detections}"
    )

    print(
        f"ONNX detections      : "
        f"{total_onnx_detections}"
    )

    print(
        f"Matched detections   : "
        f"{total_matches}"
    )

    print(
        f"Overall match rate   : "
        f"{overall_match_rate:.2%}"
    )

    print(
        f"Required match rate  : "
        f"{required_match_rate:.2%}"
    )

    print(
        f"Parity status        : "
        f"{'PASSED' if parity_passed else 'FAILED'}"
    )

    print(
        f"\nReport: "
        f"{args.report}"
    )

    print("=" * 82)

    if not parity_passed:
        raise RuntimeError(
            "Direct ONNX Runtime parity "
            "check failed."
        )


if __name__ == "__main__":
    main()