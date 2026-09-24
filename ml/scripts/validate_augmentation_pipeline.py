import argparse
from pathlib import Path

from ml.src.augmentation.augmentation_quality import (
    generate_comparison_previews,
    save_quality_report,
    validate_augmented_dataset,
)
from ml.src.data.voc_to_yolo import CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate NEU-DET augmentation quality "
            "and generate visual comparisons."
        )
    )

    parser.add_argument(
        "--source-images",
        type=Path,
        default=Path(
            "data/processed/neu_yolo/images/train"
        ),
    )

    parser.add_argument(
        "--source-labels",
        type=Path,
        default=Path(
            "data/processed/neu_yolo/labels/train"
        ),
    )

    parser.add_argument(
        "--augmented-images",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_yolo_augmented/images/train"
        ),
    )

    parser.add_argument(
        "--augmented-labels",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_yolo_augmented/labels/train"
        ),
    )

    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path(
            "outputs/day07_augmentation_previews"
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "outputs/"
            "day07_augmentation_quality.json"
        ),
    )

    parser.add_argument(
        "--preview-count",
        type=int,
        default=18,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 74)
    print("NEU-DET AUGMENTATION QUALITY VALIDATION")
    print("=" * 74)

    report = validate_augmented_dataset(
        source_images_dir=args.source_images,
        source_labels_dir=args.source_labels,
        augmented_images_dir=(
            args.augmented_images
        ),
        augmented_labels_dir=(
            args.augmented_labels
        ),
    )

    print(
        f"Source images       : "
        f"{report['source_image_count']}"
    )

    print(
        f"Source labels       : "
        f"{report['source_label_count']}"
    )

    print(
        f"Augmented images    : "
        f"{report['augmented_image_count']}"
    )

    print(
        f"Augmented labels    : "
        f"{report['augmented_label_count']}"
    )

    print(
        f"Validated pairs     : "
        f"{report['validated_pair_count']}"
    )

    print(
        f"Original boxes      : "
        f"{report['original_box_count']}"
    )

    print(
        f"Augmented boxes     : "
        f"{report['augmented_box_count']}"
    )

    print(
        "Box retention       : "
        f"{report['box_retention_rate']:.2%}"
    )

    print("\nAugmented class distribution:")

    for class_name in CLASS_NAMES:
        print(
            f"  {class_name:<20} "
            f"{report['class_counts'][class_name]}"
        )

    print("\nStructural checks:")

    checks = {
        "Images without labels":
            report["images_without_labels"],
        "Labels without images":
            report["labels_without_images"],
        "Missing source images":
            report["missing_source_images"],
        "Missing source labels":
            report["missing_source_labels"],
        "Unreadable images":
            report["unreadable_images"],
        "Dimension mismatches":
            report["dimension_mismatches"],
        "Empty augmented labels":
            report["empty_augmented_labels"],
        "Class mismatches":
            report["class_mismatches"],
    }

    for name, problems in checks.items():
        print(
            f"  {name:<25}: "
            f"{len(problems)}"
        )

    previews = (
        generate_comparison_previews(
            source_images_dir=(
                args.source_images
            ),
            source_labels_dir=(
                args.source_labels
            ),
            augmented_images_dir=(
                args.augmented_images
            ),
            augmented_labels_dir=(
                args.augmented_labels
            ),
            output_dir=args.preview_dir,
            sample_count=args.preview_count,
            seed=42,
        )
    )

    report[
        "comparison_preview_count"
    ] = len(previews)

    save_quality_report(
        report,
        args.report,
    )

    print(
        f"\nComparison previews : "
        f"{len(previews)}"
    )

    print(
        f"Preview directory   : "
        f"{args.preview_dir}"
    )

    print(
        f"Quality report      : "
        f"{args.report}"
    )

    print("-" * 74)

    if report["passed"]:
        print(
            "Augmentation status : PASSED"
        )
    else:
        print(
            "Augmentation status : FAILED"
        )

        raise RuntimeError(
            "Augmentation quality validation failed."
        )

    print("=" * 74)


if __name__ == "__main__":
    main()