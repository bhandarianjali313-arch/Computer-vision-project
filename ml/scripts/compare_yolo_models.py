import argparse
from pathlib import Path

from ultralytics import YOLO

from ml.src.data.voc_to_yolo import (
    CLASS_NAMES,
)
from ml.src.evaluation.model_comparison import (
    benchmark_model,
    calculate_deltas,
    collect_benchmark_images,
    get_model_size_mb,
    load_comparison_config,
    save_comparison,
)
from ml.src.evaluation.yolo_evaluator import (
    extract_detection_report,
)
from ml.src.training.optimized_training import (
    load_selected_candidate,
)
from ml.src.training.yolo_trainer import (
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline and optimized "
            "YOLO defect detectors."
        )
    )

    parser.add_argument(
        "--baseline-weights",
        type=Path,
        default=Path(
            "models/baseline/"
            "yolov8n_neu_baseline_best.pt"
        ),
    )

    parser.add_argument(
        "--optimized-weights",
        type=Path,
        default=Path(
            "models/optimized/"
            "yolo_neu_optimized_best.pt"
        ),
    )

    parser.add_argument(
        "--screening-results",
        type=Path,
        default=Path(
            "outputs/tuning/"
            "day12_screening_results.json"
        ),
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_training_dataset.yaml"
        ),
    )

    parser.add_argument(
        "--validation-images",
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
            "model_comparison.yaml"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/comparison/"
            "day14_model_comparison.json"
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/comparison/"
            "day14_model_comparison.csv"
        ),
    )

    return parser.parse_args()


def format_metric(
    value,
) -> str:
    if value is None:
        return "N/A"

    return f"{value:.4f}"


def evaluate_model(
    weights: Path,
    dataset_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
):
    model = YOLO(
        str(weights)
    )

    metrics = model.val(
        data=str(
            dataset_yaml.resolve()
        ),
        split="val",
        imgsz=imgsz,
        batch=batch,
        workers=0,
        device=device,
        plots=False,
        verbose=False,
    )

    report = extract_detection_report(
        metrics=metrics,
        class_names=CLASS_NAMES,
    )

    return (
        model,
        report,
    )


def main():
    args = parse_args()

    for path in (
        args.baseline_weights,
        args.optimized_weights,
        args.data,
        args.screening_results,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: "
                f"{path}"
            )

    config = load_comparison_config(
        args.config
    )

    selected = load_selected_candidate(
        args.screening_results
    )

    device = resolve_device(
        args.device
    )

    baseline_imgsz = int(
        config["baseline_imgsz"]
    )

    optimized_imgsz = int(
        selected["imgsz"]
    )

    validation = config[
        "validation"
    ]

    benchmark_config = config[
        "benchmark"
    ]

    batch = int(
        validation["batch"]
    )

    print("=" * 80)
    print("DAY 14 - BASELINE VS OPTIMIZED YOLO COMPARISON")
    print("=" * 80)

    print(
        f"Device             : "
        f"{device}"
    )

    print(
        f"Validation split   : "
        f"{args.validation_images}"
    )

    print(
        f"Baseline           : "
        f"YOLOv8n @ {baseline_imgsz}"
    )

    print(
        f"Optimized          : "
        f"{selected['model']} "
        f"@ {optimized_imgsz}"
    )

    print(
        f"Selected candidate : "
        f"{selected['name']}"
    )

    print("=" * 80)

    print("\nEvaluating baseline...")

    (
        baseline_model,
        baseline_report,
    ) = evaluate_model(
        weights=args.baseline_weights,
        dataset_yaml=args.data,
        imgsz=baseline_imgsz,
        batch=batch,
        device=device,
    )

    print("Evaluating optimized model...")

    (
        optimized_model,
        optimized_report,
    ) = evaluate_model(
        weights=args.optimized_weights,
        dataset_yaml=args.data,
        imgsz=optimized_imgsz,
        batch=batch,
        device=device,
    )

    benchmark_images = (
        collect_benchmark_images(
            images_dir=(
                args.validation_images
            ),
            sample_count=int(
                benchmark_config[
                    "sample_count"
                ]
            ),
        )
    )

    print(
        "\nBenchmarking baseline "
        "inference..."
    )

    baseline_latency = benchmark_model(
        model=baseline_model,
        images=benchmark_images,
        imgsz=baseline_imgsz,
        device=device,
        confidence=float(
            benchmark_config[
                "confidence"
            ]
        ),
        warmup_count=int(
            benchmark_config[
                "warmup_count"
            ]
        ),
    )

    print(
        "Benchmarking optimized "
        "inference..."
    )

    optimized_latency = benchmark_model(
        model=optimized_model,
        images=benchmark_images,
        imgsz=optimized_imgsz,
        device=device,
        confidence=float(
            benchmark_config[
                "confidence"
            ]
        ),
        warmup_count=int(
            benchmark_config[
                "warmup_count"
            ]
        ),
    )

    baseline_overall = (
        baseline_report[
            "overall"
        ]
    )

    optimized_overall = (
        optimized_report[
            "overall"
        ]
    )

    baseline = {
        "configuration":
            "baseline",

        "model":
            "yolov8n.pt",

        "imgsz":
            baseline_imgsz,

        "precision":
            baseline_overall[
                "precision"
            ],

        "recall":
            baseline_overall[
                "recall"
            ],

        "map50":
            baseline_overall[
                "map50"
            ],

        "map50_95":
            baseline_overall[
                "map50_95"
            ],

        "model_size_mb":
            get_model_size_mb(
                args.baseline_weights
            ),

        **baseline_latency,
    }

    optimized = {
        "configuration":
            "optimized",

        "model":
            selected["model"],

        "imgsz":
            optimized_imgsz,

        "precision":
            optimized_overall[
                "precision"
            ],

        "recall":
            optimized_overall[
                "recall"
            ],

        "map50":
            optimized_overall[
                "map50"
            ],

        "map50_95":
            optimized_overall[
                "map50_95"
            ],

        "model_size_mb":
            get_model_size_mb(
                args.optimized_weights
            ),

        **optimized_latency,
    }

    deltas = calculate_deltas(
        baseline=baseline,
        optimized=optimized,
    )

    metadata = {
        "validation_split":
            "val",

        "validation_image_count":
            360,

        "benchmark_image_count":
            len(benchmark_images),

        "benchmark_confidence":
            float(
                benchmark_config[
                    "confidence"
                ]
            ),

        "device":
            device,

        "optimized_selected_from":
            selected["name"],
    }

    save_comparison(
        baseline=baseline,
        optimized=optimized,
        deltas=deltas,
        metadata=metadata,
        json_path=args.output_json,
        csv_path=args.output_csv,
    )

    print("\n" + "=" * 80)
    print("VALIDATION COMPARISON")
    print("=" * 80)

    print(
        f"{'Metric':<22}"
        f"{'Baseline':>14}"
        f"{'Optimized':>14}"
        f"{'Delta':>14}"
    )

    print("-" * 64)

    for (
        display,
        key,
    ) in [
        (
            "Precision",
            "precision",
        ),
        (
            "Recall",
            "recall",
        ),
        (
            "mAP50",
            "map50",
        ),
        (
            "mAP50-95",
            "map50_95",
        ),
    ]:
        print(
            f"{display:<22}"
            f"{format_metric(baseline[key]):>14}"
            f"{format_metric(optimized[key]):>14}"
            f"{format_metric(deltas[key]):>14}"
        )

    print("\n" + "=" * 80)
    print("DEPLOYMENT CHARACTERISTICS")
    print("=" * 80)

    print(
        f"Baseline model size    : "
        f"{baseline['model_size_mb']:.2f} MB"
    )

    print(
        f"Optimized model size   : "
        f"{optimized['model_size_mb']:.2f} MB"
    )

    print(
        f"Baseline mean latency  : "
        f"{baseline['mean_latency_ms']:.2f} ms"
    )

    print(
        f"Optimized mean latency : "
        f"{optimized['mean_latency_ms']:.2f} ms"
    )

    print(
        f"Baseline approx FPS    : "
        f"{baseline['approx_fps']:.2f}"
    )

    print(
        f"Optimized approx FPS   : "
        f"{optimized['approx_fps']:.2f}"
    )

    print(
        f"\nJSON report : "
        f"{args.output_json}"
    )

    print(
        f"CSV report  : "
        f"{args.output_csv}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()