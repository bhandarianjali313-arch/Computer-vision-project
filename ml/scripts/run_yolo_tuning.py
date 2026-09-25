import argparse
import time
from pathlib import Path

from ultralytics import YOLO

from ml.src.evaluation.yolo_evaluator import (
    extract_detection_report,
)
from ml.src.training.hyperparameter_tuning import (
    build_screening_arguments,
    load_tuning_config,
    save_tuning_results,
    select_screening_candidate,
)
from ml.src.training.yolo_trainer import (
    resolve_device,
)
from ml.src.data.voc_to_yolo import (
    CLASS_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled YOLOv8 "
            "hyperparameter screening."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "ml/configs/yolov8_tuning.yaml"
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
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help=(
            "Run only one named candidate. "
            "Useful for debugging."
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run each selected candidate for "
            "1 epoch on 5 percent of training data."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/tuning/"
            "day12_screening_results.json"
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/tuning/"
            "day12_screening_results.csv"
        ),
    )

    return parser.parse_args()


def format_value(
    value,
):
    if value is None:
        return "N/A"

    return f"{value:.4f}"


def main():
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: "
            f"{args.data}"
        )

    config = load_tuning_config(
        args.config
    )

    candidates = config[
        "candidates"
    ]

    if args.candidate is not None:
        candidates = [
            candidate
            for candidate in candidates
            if candidate["name"]
            == args.candidate
        ]

        if not candidates:
            raise ValueError(
                f"Unknown candidate: "
                f"{args.candidate}"
            )

    device = resolve_device(
        args.device
    )

    project_dir = Path(
        "outputs/tuning/day12_runs"
    )

    print("=" * 78)
    print("YOLOV8 TARGETED HYPERPARAMETER SCREENING")
    print("=" * 78)

    print(
        f"Candidates : {len(candidates)}"
    )

    print(
        f"Device     : {device}"
    )

    print(
        f"Smoke mode : {args.smoke}"
    )

    print(
        "Selection  : validation mAP50-95"
    )

    print("=" * 78)

    results = []

    for (
        candidate_number,
        candidate,
    ) in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"\n[{candidate_number}/"
            f"{len(candidates)}] "
            f"{candidate['name']}"
        )

        print(
            f"Model : {candidate['model']}"
        )

        print(
            f"Image : {candidate['imgsz']}"
        )

        print(
            f"Batch : {candidate['batch']}"
        )

        training_args = (
            build_screening_arguments(
                screening=config[
                    "screening"
                ],
                augmentation=config[
                    "augmentation"
                ],
                candidate=candidate,
                dataset_yaml=args.data,
                device=device,
                project_dir=(
                    project_dir
                ),
                smoke=args.smoke,
            )
        )

        model = YOLO(
            candidate["model"]
        )

        start_time = (
            time.perf_counter()
        )

        model.train(
            **training_args
        )

        training_seconds = (
            time.perf_counter()
            - start_time
        )

        run_dir = (
            project_dir.resolve()
            / candidate["name"]
        )

        best_weights = (
            run_dir
            / "weights"
            / "best.pt"
        )

        if not best_weights.exists():
            raise FileNotFoundError(
                "Training completed but "
                f"best.pt was not found: "
                f"{best_weights}"
            )

        evaluation_model = YOLO(
            str(best_weights)
        )

        metrics = (
            evaluation_model.val(
                data=str(
                    args.data.resolve()
                ),
                split="val",
                imgsz=int(
                    candidate[
                        "imgsz"
                    ]
                ),
                batch=int(
                    candidate[
                        "batch"
                    ]
                ),
                workers=0,
                device=device,
                plots=False,
                verbose=False,
            )
        )

        report = (
            extract_detection_report(
                metrics=metrics,
                class_names=CLASS_NAMES,
            )
        )

        overall = report[
            "overall"
        ]

        result = {
            "name":
                candidate["name"],

            "model":
                candidate["model"],

            "imgsz":
                candidate["imgsz"],

            "batch":
                candidate["batch"],

            "precision":
                overall["precision"],

            "recall":
                overall["recall"],

            "map50":
                overall["map50"],

            "map50_95":
                overall["map50_95"],

            "training_seconds":
                training_seconds,

            "best_weights":
                str(best_weights),
        }

        results.append(
            result
        )

        print(
            "\nScreening result:"
        )

        print(
            "  Precision  : "
            f"{format_value(result['precision'])}"
        )

        print(
            "  Recall     : "
            f"{format_value(result['recall'])}"
        )

        print(
            "  mAP50      : "
            f"{format_value(result['map50'])}"
        )

        print(
            "  mAP50-95   : "
            f"{format_value(result['map50_95'])}"
        )

    selected = (
        select_screening_candidate(
            results
        )
    )

    save_tuning_results(
        results=results,
        selected=selected,
        json_path=args.output_json,
        csv_path=args.output_csv,
    )

    print("\n" + "=" * 78)
    print("SCREENING SUMMARY")
    print("=" * 78)

    for result in sorted(
        results,
        key=lambda row: (
            row["map50_95"]
            if row["map50_95"]
            is not None
            else -1
        ),
        reverse=True,
    ):
        print(
            f"{result['name']:<24} "
            f"mAP50="
            f"{format_value(result['map50'])}  "
            f"mAP50-95="
            f"{format_value(result['map50_95'])}  "
            f"R="
            f"{format_value(result['recall'])}"
        )

    print("\nSelected for Day 13:")

    print(
        f"  Candidate : "
        f"{selected['name']}"
    )

    print(
        f"  Model     : "
        f"{selected['model']}"
    )

    print(
        f"  imgsz     : "
        f"{selected['imgsz']}"
    )

    print(
        f"  batch     : "
        f"{selected['batch']}"
    )

    print(
        f"  mAP50-95  : "
        f"{format_value(selected['map50_95'])}"
    )

    print(
        f"\nJSON: {args.output_json}"
    )

    print(
        f"CSV : {args.output_csv}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()