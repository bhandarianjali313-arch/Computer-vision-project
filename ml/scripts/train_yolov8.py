import argparse
from pathlib import Path

from ultralytics import YOLO

from ml.src.training.yolo_trainer import (
    build_training_arguments,
    load_training_config,
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train YOLOv8 for NEU industrial "
            "surface defect detection."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "ml/configs/yolov8_baseline.yaml"
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
        help=(
            "Training device. Examples: "
            "'cpu' or '0'. "
            "Defaults to CUDA GPU if available."
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a short 1-epoch training "
            "sanity check."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_training_config(
        args.config
    )

    if not args.data.exists():
        raise FileNotFoundError(
            "Training dataset YAML does not exist. "
            "Run:\n"
            "python -m "
            "ml.scripts.prepare_yolo_training"
        )

    device = resolve_device(
        args.device
    )

    training_args = (
        build_training_arguments(
            config=config,
            dataset_yaml=args.data,
            device=device,
            smoke=args.smoke,
        )
    )

    print("=" * 74)
    print("YOLOV8 INDUSTRIAL DEFECT TRAINING")
    print("=" * 74)

    print(
        f"Model       : "
        f"{config['model']}"
    )

    print(
        f"Dataset     : "
        f"{args.data}"
    )

    print(
        f"Device      : "
        f"{device}"
    )

    print(
        f"Smoke mode  : "
        f"{args.smoke}"
    )

    print(
        f"Epochs      : "
        f"{training_args['epochs']}"
    )

    print(
        f"Image size  : "
        f"{training_args['imgsz']}"
    )

    print(
        f"Batch size  : "
        f"{training_args['batch']}"
    )

    print("=" * 74)

    model = YOLO(
        config["model"]
    )

    results = model.train(
        **training_args
    )

    print("\nTraining completed.")

    save_dir = getattr(
        results,
        "save_dir",
        None,
    )

    if save_dir is not None:
        print(
            f"Results saved to: "
            f"{save_dir}"
        )


if __name__ == "__main__":
    main()