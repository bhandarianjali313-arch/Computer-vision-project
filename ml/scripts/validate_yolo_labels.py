import argparse
from collections import Counter
from pathlib import Path

from ml.src.data.voc_to_yolo import (
    CLASS_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate generated YOLO label files."
        )
    )

    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path(
            "data/processed/neu_yolo/labels/all"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    label_files = sorted(
        args.labels_dir.glob("*.txt")
    )

    if not label_files:
        raise FileNotFoundError(
            f"No YOLO labels found in "
            f"{args.labels_dir}"
        )

    invalid_lines = []
    class_counts = Counter()
    total_boxes = 0

    for label_file in label_files:
        lines = label_file.read_text(
            encoding="utf-8"
        ).splitlines()

        for line_number, line in enumerate(
            lines,
            start=1,
        ):
            parts = line.split()

            if len(parts) != 5:
                invalid_lines.append(
                    (
                        label_file.name,
                        line_number,
                        "Expected 5 values",
                    )
                )
                continue

            try:
                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:
                invalid_lines.append(
                    (
                        label_file.name,
                        line_number,
                        "Non-numeric value",
                    )
                )
                continue

            if not 0 <= class_id < len(
                CLASS_NAMES
            ):
                invalid_lines.append(
                    (
                        label_file.name,
                        line_number,
                        f"Invalid class ID {class_id}",
                    )
                )
                continue

            valid_coordinates = (
                0.0 <= x_center <= 1.0
                and 0.0 <= y_center <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
            )

            if not valid_coordinates:
                invalid_lines.append(
                    (
                        label_file.name,
                        line_number,
                        "Coordinates outside YOLO range",
                    )
                )
                continue

            total_boxes += 1
            class_counts[class_id] += 1

    print("=" * 70)
    print("YOLO LABEL VALIDATION")
    print("=" * 70)

    print(
        f"Label files       : "
        f"{len(label_files)}"
    )

    print(
        f"Valid boxes       : "
        f"{total_boxes}"
    )

    print(
        f"Invalid lines     : "
        f"{len(invalid_lines)}"
    )

    print("\nClass distribution:")

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"  {class_id} "
            f"{class_name:<20} "
            f"{class_counts[class_id]}"
        )

    if invalid_lines:
        print("\nFirst invalid entries:")

        for issue in invalid_lines[:10]:
            print(
                f"  {issue[0]}:"
                f"{issue[1]} - "
                f"{issue[2]}"
            )

        raise RuntimeError(
            "YOLO label validation failed."
        )

    print("\nValidation status : PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()