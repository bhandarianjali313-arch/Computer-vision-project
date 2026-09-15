import argparse
from collections import defaultdict
from pathlib import Path

from ml.src.data.voc_to_yolo import CLASS_NAMES
from ml.src.data.yolo_visualizer import (
    read_yolo_label,
    visualize_label_file,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize converted NEU-DET "
            "YOLO annotations."
        )
    )

    parser.add_argument(
        "--dataset-root",
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
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/day04_visualizations"
        ),
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=3,
        help=(
            "Number of sample images "
            "to visualize for each class."
        ),
    )

    return parser.parse_args()


def select_samples(
    labels_dir: Path,
    samples_per_class: int,
):
    """
    Select representative annotation files
    for all defect classes.
    """

    selected = defaultdict(list)

    label_files = sorted(
        labels_dir.glob("*.txt")
    )

    if not label_files:
        raise FileNotFoundError(
            f"No YOLO labels found in "
            f"{labels_dir}"
        )

    for label_file in label_files:
        annotations = read_yolo_label(
            label_file
        )

        classes_in_file = {
            annotation["class_id"]
            for annotation in annotations
        }

        for class_id in classes_in_file:
            if (
                len(selected[class_id])
                < samples_per_class
            ):
                selected[class_id].append(
                    label_file
                )

        if all(
            len(selected[class_id])
            >= samples_per_class
            for class_id in range(
                len(CLASS_NAMES)
            )
        ):
            break

    return selected


def main():
    args = parse_args()

    print("=" * 72)
    print("NEU-DET YOLO ANNOTATION VISUALIZATION")
    print("=" * 72)

    selected = select_samples(
        labels_dir=args.labels_dir,
        samples_per_class=args.samples_per_class,
    )

    total_generated = 0

    generated_stems = set()

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        samples = selected.get(
            class_id,
            [],
        )

        print(
            f"\nClass {class_id}: "
            f"{class_name}"
        )

        if not samples:
            print("  No samples found.")
            continue

        for label_path in samples:
            # Avoid saving the same image twice
            # when it contains multiple classes.
            if label_path.stem in generated_stems:
                print(
                    f"  reused: "
                    f"{label_path.stem}"
                )
                continue

            output_path = visualize_label_file(
                label_path=label_path,
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
            )

            generated_stems.add(
                label_path.stem
            )

            total_generated += 1

            print(
                f"  saved: "
                f"{output_path.name}"
            )

    print("\n" + "-" * 72)

    print(
        f"Visualization images generated: "
        f"{total_generated}"
    )

    print(
        f"Output directory: "
        f"{args.output_dir}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()