import argparse
import json
from pathlib import Path

from ml.src.data.neu_inspector import inspect_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect and validate the NEU-DET dataset."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/NEU-DET"),
        help="Path to the raw NEU-DET dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/day02_dataset_audit.json"
        ),
        help="Path where the audit report will be saved.",
    )

    return parser.parse_args()


def print_summary(report):
    print("=" * 70)
    print("NEU-DET DATASET AUDIT")
    print("=" * 70)

    print(f"Dataset root        : {report['dataset_root']}")
    print(f"Images              : {report['image_count']}")
    print(f"XML annotations     : {report['annotation_count']}")
    print(f"Annotated objects   : {report['total_objects']}")

    print("\nDetected classes:")

    for class_name in report["detected_classes"]:
        object_count = report["class_object_counts"].get(
            class_name,
            0,
        )

        image_count = report["images_per_class"].get(
            class_name,
            0,
        )

        print(
            f"  {class_name:<20} "
            f"objects={object_count:<5} "
            f"images={image_count}"
        )

    print("\nImage dimensions:")

    for size, count in report["image_dimensions"].items():
        print(f"  {size:<15} {count}")

    print("\nValidation:")

    print(
        "Images without XML   : "
        f"{len(report['images_without_annotations'])}"
    )

    print(
        "XML without images   : "
        f"{len(report['annotations_without_images'])}"
    )

    print(
        f"Invalid XML files    : "
        f"{report['invalid_xml_count']}"
    )

    print(
        f"Unreadable images    : "
        f"{report['image_read_error_count']}"
    )

    print(
        f"Dimension mismatches : "
        f"{report['dimension_mismatch_count']}"
    )

    print(
        f"Invalid boxes        : "
        f"{report['invalid_bbox_count']}"
    )

    print(
        f"Missing classes      : "
        f"{report['missing_expected_classes']}"
    )

    print(
        f"Unexpected classes   : "
        f"{report['unexpected_classes']}"
    )

    print("=" * 70)


def main():
    args = parse_args()

    report = inspect_dataset(args.dataset_root)

    print_summary(report)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print(f"\nAudit saved to: {args.output}")


if __name__ == "__main__":
    main()