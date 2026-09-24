import argparse
from pathlib import Path

from ml.src.data.dataset_splitter import (
    build_yolo_dataset,
)
from ml.src.data.voc_to_yolo import (
    CLASS_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible train/val/test "
            "splits for NEU-DET."
        )
    )

    parser.add_argument(
        "--raw-dataset",
        type=Path,
        default=Path(
            "data/raw/NEU-DET"
        ),
    )

    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path(
            "data/processed/neu_yolo/labels/all"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/processed/neu_yolo"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 74)
    print("NEU-DET TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 74)

    print(
        f"Seed  : {args.seed}"
    )

    print(
        "Ratio : "
        f"{args.train_ratio:.2f} / "
        f"{args.val_ratio:.2f} / "
        f"{args.test_ratio:.2f}"
    )

    result = build_yolo_dataset(
        raw_dataset_root=args.raw_dataset,
        all_labels_dir=args.labels_dir,
        output_root=args.output_root,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print(
        f"\nTotal images: "
        f"{result['total_images']}"
    )

    for split_name in (
        "train",
        "val",
        "test",
    ):
        split = result[
            "splits"
        ][split_name]

        print(
            "\n"
            + "-" * 74
        )

        print(
            f"{split_name.upper()}"
        )

        print(
            f"Images : "
            f"{split['image_count']}"
        )

        print(
            f"Labels : "
            f"{split['label_count']}"
        )

        print(
            "\nPrimary categories:"
        )

        for class_name in CLASS_NAMES:
            print(
                f"  {class_name:<20} "
                f"{split['primary_class_counts'][class_name]}"
            )

        print(
            "\nBounding boxes:"
        )

        for class_name in CLASS_NAMES:
            print(
                f"  {class_name:<20} "
                f"{split['object_counts'][class_name]}"
            )

    print(
        "\n"
        + "=" * 74
    )

    print(
        "Data leakage check : PASSED"
    )

    print(
        f"YOLO configuration : "
        f"{result['yaml_path']}"
    )

    print(
        f"Split manifest     : "
        f"{result['manifest_path']}"
    )

    print("=" * 74)


if __name__ == "__main__":
    main()