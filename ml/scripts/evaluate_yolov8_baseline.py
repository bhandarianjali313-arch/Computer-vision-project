import argparse
from pathlib import Path

from ultralytics import YOLO

from ml.src.data.voc_to_yolo import CLASS_NAMES
from ml.src.evaluation.yolo_evaluator import (
    extract_detection_report,
    load_evaluation_config,
    save_evaluation_json,
    save_per_class_csv,
)
from ml.src.training.yolo_trainer import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the YOLOv8 baseline "
            "on the NEU validation split."
        )
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(
            "models/baseline/"
            "yolov8n_neu_baseline_best.pt"
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
        "--config",
        type=Path,
        default=Path(
            "ml/configs/"
            "yolov8_evaluation.yaml"
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
            "outputs/evaluation/"
            "baseline_validation_metrics.json"
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/evaluation/"
            "baseline_validation_per_class.csv"
        ),
    )

    return parser.parse_args()


def format_metric(value):
    if value is None:
        return "N/A"

    return f"{value:.4f}"


def main():
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: {args.weights}"
        )

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data}"
        )

    evaluation_config = load_evaluation_config(
        args.config
    )

    device = resolve_device(
        args.device
    )

    run_root = Path(
        "outputs/evaluation"
    ).resolve()

    run_name = "baseline_validation"

    print("=" * 76)
    print("YOLOV8 BASELINE VALIDATION EVALUATION")
    print("=" * 76)

    print(f"Weights    : {args.weights}")
    print(f"Dataset    : {args.data}")
    print("Split      : validation")
    print(f"Device     : {device}")
    print(
        f"Image size : "
        f"{evaluation_config['imgsz']}"
    )

    print("=" * 76)

    model = YOLO(
        str(args.weights)
    )

    metrics = model.val(
        data=str(
            args.data.resolve()
        ),
        split="val",
        device=device,
        project=str(run_root),
        name=run_name,
        exist_ok=True,
        **evaluation_config,
    )

    report = extract_detection_report(
        metrics=metrics,
        class_names=CLASS_NAMES,
    )

    report["experiment"] = {
        "model": "YOLOv8n baseline",
        "weights": str(
            args.weights.resolve()
        ),
        "dataset": str(
            args.data.resolve()
        ),
        "split": "validation",
        "device": device,
        "image_size": evaluation_config[
            "imgsz"
        ],
        "validation_images": 360,
    }

    save_evaluation_json(
        report,
        args.output_json,
    )

    save_per_class_csv(
        report,
        args.output_csv,
    )

    overall = report["overall"]

    print("\n" + "=" * 76)
    print("OVERALL VALIDATION METRICS")
    print("=" * 76)

    print(
        "Precision   : "
        f"{format_metric(overall['precision'])}"
    )

    print(
        "Recall      : "
        f"{format_metric(overall['recall'])}"
    )

    print(
        "mAP@0.50    : "
        f"{format_metric(overall['map50'])}"
    )

    print(
        "mAP@0.50:95 : "
        f"{format_metric(overall['map50_95'])}"
    )

    print("\nPER-CLASS METRICS")
    print("-" * 76)

    for row in report["per_class"]:
        print(
            f"{row['class_name']:<20} "
            f"P={format_metric(row['precision'])}  "
            f"R={format_metric(row['recall'])}  "
            f"mAP50={format_metric(row['map50'])}  "
            f"mAP50-95="
            f"{format_metric(row['map50_95'])}"
        )

    print("\nINFERENCE SPEED")
    print("-" * 76)

    for stage, value in report[
        "speed_ms_per_image"
    ].items():
        print(
            f"{stage:<20}: "
            f"{format_metric(value)} ms"
        )

    print(
        f"\nJSON report : "
        f"{args.output_json}"
    )

    print(
        f"CSV report  : "
        f"{args.output_csv}"
    )

    print(
        f"Plots       : "
        f"{run_root / run_name}"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()