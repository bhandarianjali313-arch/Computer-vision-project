import argparse
from pathlib import Path

from ml.src.augmentation.defect_augmentation import (
    generate_augmented_dataset,
    save_augmentation_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate augmented NEU-DET "
            "training samples."
        )
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(
            "data/processed/neu_yolo/images/train"
        ),
    )

    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path(
            "data/processed/neu_yolo/labels/train"
        ),
    )

    parser.add_argument(
        "--output-images-dir",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_yolo_augmented/images/train"
        ),
    )

    parser.add_argument(
        "--output-labels-dir",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_yolo_augmented/labels/train"
        ),
    )

    parser.add_argument(
        "--augmentations-per-image",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of source images "
            "for testing the pipeline."
        ),
    )

    parser.add_argument(
        "--clean-output",
        action="store_true",
        help=(
            "Delete previously generated "
            "augmentation output first."
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "outputs/"
            "day06_augmentation_report.json"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 72)
    print("NEU-DET TRAINING DATA AUGMENTATION")
    print("=" * 72)

    print(
        f"Training images : "
        f"{args.images_dir}"
    )

    print(
        f"Training labels : "
        f"{args.labels_dir}"
    )

    print(
        f"Augmentations   : "
        f"{args.augmentations_per_image} "
        f"per image"
    )

    if args.limit is not None:
        print(
            f"Source limit    : "
            f"{args.limit}"
        )

    print("-" * 72)

    report = generate_augmented_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_images_dir=(
            args.output_images_dir
        ),
        output_labels_dir=(
            args.output_labels_dir
        ),
        augmentations_per_image=(
            args.augmentations_per_image
        ),
        limit=args.limit,
        clean_output=args.clean_output,
    )

    print("\n" + "=" * 72)

    print(
        f"Source images     : "
        f"{report['source_image_count']}"
    )

    print(
        f"Original boxes    : "
        f"{report['original_box_count']}"
    )

    print(
        f"Generated images  : "
        f"{report['generated_image_count']}"
    )

    print(
        f"Generated boxes   : "
        f"{report['generated_box_count']}"
    )

    print(
        f"Skipped samples   : "
        f"{report['skipped_count']}"
    )

    save_augmentation_report(
        report,
        args.report,
    )

    print(
        f"\nReport saved to   : "
        f"{args.report}"
    )

    print(
        f"Images saved to   : "
        f"{args.output_images_dir}"
    )

    print(
        f"Labels saved to   : "
        f"{args.output_labels_dir}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()