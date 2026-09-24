import argparse
from pathlib import Path

from ml.src.data.voc_to_yolo import (
    CLASS_NAMES,
    convert_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert NEU-DET Pascal VOC annotations "
            "to YOLO format."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "data/raw/NEU-DET"
        ),
        help=(
            "Root directory of the raw "
            "NEU-DET dataset."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_yolo/labels/all"
        ),
        help=(
            "Directory where generated "
            "YOLO labels will be saved."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 72)
    print(
        "NEU-DET PASCAL VOC -> YOLO CONVERSION"
    )
    print("=" * 72)

    print("\nClass mapping:")

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"  {class_id} -> "
            f"{class_name}"
        )

    print()

    result = convert_dataset(
        dataset_root=(
            args.dataset_root
        ),
        output_dir=(
            args.output_dir
        ),
    )

    print(
        f"XML annotations    : "
        f"{result['annotation_count']}"
    )

    print(
        f"YOLO labels        : "
        f"{result['label_file_count']}"
    )

    print(
        f"Bounding boxes     : "
        f"{result['total_boxes']}"
    )

    print(
        f"Duplicates removed : "
        f"{result['duplicates_removed']}"
    )

    print(
        "\nObjects per class:"
    )

    for class_name, count in (
        result["class_counts"].items()
    ):
        print(
            f"  {class_name:<20} "
            f"{count}"
        )

    print(
        "\nLabels saved to:"
    )

    print(
        result["output_dir"]
    )

    print("=" * 72)


if __name__ == "__main__":
    main()