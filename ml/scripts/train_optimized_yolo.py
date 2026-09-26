import argparse
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

from ml.src.training.optimized_training import (
    build_optimized_training_arguments,
    load_optimized_config,
    load_selected_candidate,
    save_training_manifest,
)
from ml.src.training.yolo_trainer import (
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the full YOLO detector selected "
            "by Day 12 hyperparameter screening."
        )
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
        "--config",
        type=Path,
        default=Path(
            "ml/configs/"
            "yolov8_optimized.yaml"
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
        "--overwrite",
        action="store_true",
        help=(
            "Delete an existing Day 13 "
            "training run before training."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: "
            f"{args.data}"
        )

    selected = (
        load_selected_candidate(
            args.screening_results
        )
    )

    config = (
        load_optimized_config(
            args.config
        )
    )

    device = resolve_device(
        args.device
    )

    project_dir = Path(
        "outputs/training"
    )

    run_name = (
        "yolov8_optimized"
    )

    run_dir = (
        project_dir
        / run_name
    )

    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Training run already exists: "
                f"{run_dir}\n"
                "Use --overwrite only if you "
                "want to replace it."
            )

        shutil.rmtree(
            run_dir
        )

    training_args = (
        build_optimized_training_arguments(
            config=config,
            selected_candidate=selected,
            dataset_yaml=args.data,
            device=device,
            project_dir=project_dir,
            run_name=run_name,
        )
    )

    print("=" * 78)
    print("DAY 13 - FULL SELECTED YOLO TRAINING")
    print("=" * 78)

    print(
        f"Screening candidate : "
        f"{selected['name']}"
    )

    print(
        f"Model               : "
        f"{selected['model']}"
    )

    print(
        f"Image size          : "
        f"{selected['imgsz']}"
    )

    print(
        f"Batch size          : "
        f"{selected['batch']}"
    )

    print(
        f"Epochs              : "
        f"{training_args['epochs']}"
    )

    print(
        f"Seed                : "
        f"{training_args['seed']}"
    )

    print(
        f"Device              : "
        f"{device}"
    )

    print(
        f"Dataset             : "
        f"{args.data}"
    )

    print("=" * 78)

    model = YOLO(
        selected["model"]
    )

    start_time = (
        time.perf_counter()
    )

    model.train(
        **training_args
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    best_weights = (
        run_dir
        / "weights"
        / "best.pt"
    )

    last_weights = (
        run_dir
        / "weights"
        / "last.pt"
    )

    if not best_weights.exists():
        raise FileNotFoundError(
            "Training finished but best.pt "
            f"was not found at {best_weights}"
        )

    if not last_weights.exists():
        raise FileNotFoundError(
            "Training finished but last.pt "
            f"was not found at {last_weights}"
        )

    stable_model_dir = Path(
        "models/optimized"
    )

    stable_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stable_best = (
        stable_model_dir
        / "yolo_neu_optimized_best.pt"
    )

    stable_last = (
        stable_model_dir
        / "yolo_neu_optimized_last.pt"
    )

    shutil.copy2(
        best_weights,
        stable_best,
    )

    shutil.copy2(
        last_weights,
        stable_last,
    )

    manifest_path = Path(
        "outputs/"
        "day13_optimized_training.json"
    )

    save_training_manifest(
        selected_candidate=selected,
        training_arguments=training_args,
        elapsed_seconds=elapsed_seconds,
        best_weights=stable_best,
        last_weights=stable_last,
        output_path=manifest_path,
    )

    print("\n" + "=" * 78)
    print("DAY 13 TRAINING COMPLETED")
    print("=" * 78)

    print(
        f"Training seconds : "
        f"{elapsed_seconds:.2f}"
    )

    print(
        f"Best checkpoint  : "
        f"{best_weights}"
    )

    print(
        f"Stable best      : "
        f"{stable_best}"
    )

    print(
        f"Stable last      : "
        f"{stable_last}"
    )

    print(
        f"Manifest         : "
        f"{manifest_path}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()