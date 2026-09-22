from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ml.src.data.neu_inspector import IMAGE_EXTENSIONS
from ml.src.data.voc_to_yolo import CLASS_NAMES


AUGMENTED_PATTERN = re.compile(
    r"^(?P<source>.+)_aug(?P<number>\d+)$"
)


def collect_images(
    directory: Path,
) -> dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Image directory not found: {directory}"
        )

    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    }


def collect_labels(
    directory: Path,
) -> dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Label directory not found: {directory}"
        )

    return {
        path.stem: path
        for path in sorted(
            directory.glob("*.txt")
        )
    }


def source_stem(
    augmented_stem: str,
) -> str:
    match = AUGMENTED_PATTERN.match(
        augmented_stem
    )

    if match is None:
        raise ValueError(
            f"Invalid augmented stem: {augmented_stem}"
        )

    return match.group("source")


def read_label_file(
    label_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Read one YOLO label file exactly once.

    Returns:
        parsed annotations
        normalized raw lines
    """

    try:
        text = label_path.read_text(
            encoding="utf-8"
        )
    except OSError as error:
        raise OSError(
            f"Could not read label file: {label_path}"
        ) from error

    raw_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    annotations: list[
        dict[str, Any]
    ] = []

    for line_number, line in enumerate(
        raw_lines,
        start=1,
    ):
        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO row in "
                f"{label_path}, "
                f"line {line_number}: "
                f"expected 5 values"
            )

        try:
            class_id = int(
                parts[0]
            )

            x_center = float(
                parts[1]
            )

            y_center = float(
                parts[2]
            )

            width = float(
                parts[3]
            )

            height = float(
                parts[4]
            )

        except ValueError as error:
            raise ValueError(
                f"Non-numeric YOLO row in "
                f"{label_path}, "
                f"line {line_number}"
            ) from error

        if not (
            0 <= class_id < len(
                CLASS_NAMES
            )
        ):
            raise ValueError(
                f"Invalid class ID "
                f"{class_id} in "
                f"{label_path}"
            )

        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise ValueError(
                f"Invalid YOLO coordinates "
                f"in {label_path}, "
                f"line {line_number}"
            )

        annotations.append(
            {
                "class_id":
                    class_id,
                "x_center":
                    x_center,
                "y_center":
                    y_center,
                "width":
                    width,
                "height":
                    height,
            }
        )

    return (
        annotations,
        raw_lines,
    )


def find_duplicate_lines(
    lines: list[str],
) -> list[str]:
    counts = Counter(
        lines
    )

    return sorted(
        line
        for line, count
        in counts.items()
        if count > 1
    )


def validate_training_data(
    train_images_dir: Path,
    train_labels_dir: Path,
    augmented_images_dir: Path,
    augmented_labels_dir: Path,
    val_images_dir: Path,
    test_images_dir: Path,
) -> dict[str, Any]:
    """
    Run strict integrity checks before model training.
    """

    print(
        "Collecting dataset files..."
    )

    train_images = collect_images(
        train_images_dir
    )

    train_labels = collect_labels(
        train_labels_dir
    )

    augmented_images = collect_images(
        augmented_images_dir
    )

    augmented_labels = collect_labels(
        augmented_labels_dir
    )

    val_images = collect_images(
        val_images_dir
    )

    test_images = collect_images(
        test_images_dir
    )

    train_stems = set(
        train_images
    )

    train_label_stems = set(
        train_labels
    )

    val_stems = set(
        val_images
    )

    test_stems = set(
        test_images
    )

    augmented_image_stems = set(
        augmented_images
    )

    augmented_label_stems = set(
        augmented_labels
    )

    missing_train_labels = sorted(
        train_stems
        - train_label_stems
    )

    labels_without_train_images = sorted(
        train_label_stems
        - train_stems
    )

    leakage_train_val = sorted(
        train_stems
        & val_stems
    )

    leakage_train_test = sorted(
        train_stems
        & test_stems
    )

    leakage_val_test = sorted(
        val_stems
        & test_stems
    )

    augmented_images_without_labels = sorted(
        augmented_image_stems
        - augmented_label_stems
    )

    augmented_labels_without_images = sorted(
        augmented_label_stems
        - augmented_image_stems
    )

    missing_augmented_sources = []

    augmented_box_increases = []

    duplicate_augmented_annotations = []

    augmented_class_mismatches = []

    source_box_total = 0
    augmented_box_total = 0

    common_augmented_stems = sorted(
        augmented_image_stems
        & augmented_label_stems
    )

    source_cache: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    total = len(
        common_augmented_stems
    )

    print(
        f"Checking {total} "
        f"augmented samples..."
    )

    for index, augmented_name in enumerate(
        common_augmented_stems,
        start=1,
    ):
        if (
            index == 1
            or index % 100 == 0
            or index == total
        ):
            print(
                f"  Processed "
                f"{index}/{total}"
            )

        try:
            original_stem = source_stem(
                augmented_name
            )

        except ValueError:
            missing_augmented_sources.append(
                augmented_name
            )
            continue

        if original_stem not in train_labels:
            missing_augmented_sources.append(
                augmented_name
            )
            continue

        if original_stem not in source_cache:
            (
                original_annotations,
                _,
            ) = read_label_file(
                train_labels[
                    original_stem
                ]
            )

            source_cache[
                original_stem
            ] = original_annotations

        original_annotations = (
            source_cache[
                original_stem
            ]
        )

        (
            augmented_annotations,
            augmented_raw_lines,
        ) = read_label_file(
            augmented_labels[
                augmented_name
            ]
        )

        source_box_count = len(
            original_annotations
        )

        augmented_box_count = len(
            augmented_annotations
        )

        source_box_total += (
            source_box_count
        )

        augmented_box_total += (
            augmented_box_count
        )

        if (
            augmented_box_count
            > source_box_count
        ):
            augmented_box_increases.append(
                {
                    "augmented_stem":
                        augmented_name,

                    "source_stem":
                        original_stem,

                    "source_boxes":
                        source_box_count,

                    "augmented_boxes":
                        augmented_box_count,
                }
            )

        source_classes = {
            annotation["class_id"]
            for annotation
            in original_annotations
        }

        augmented_classes = {
            annotation["class_id"]
            for annotation
            in augmented_annotations
        }

        if not (
            augmented_classes
            .issubset(
                source_classes
            )
        ):
            augmented_class_mismatches.append(
                augmented_name
            )

        duplicates = find_duplicate_lines(
            augmented_raw_lines
        )

        if duplicates:
            duplicate_augmented_annotations.append(
                {
                    "stem":
                        augmented_name,

                    "duplicates":
                        duplicates,
                }
            )

    passed = not any(
        [
            missing_train_labels,
            labels_without_train_images,
            leakage_train_val,
            leakage_train_test,
            leakage_val_test,
            augmented_images_without_labels,
            augmented_labels_without_images,
            missing_augmented_sources,
            augmented_box_increases,
            duplicate_augmented_annotations,
            augmented_class_mismatches,
        ]
    )

    return {
        "train_image_count":
            len(train_images),

        "train_label_count":
            len(train_labels),

        "augmented_image_count":
            len(augmented_images),

        "augmented_label_count":
            len(augmented_labels),

        "validation_image_count":
            len(val_images),

        "test_image_count":
            len(test_images),

        "source_box_total":
            source_box_total,

        "augmented_box_total":
            augmented_box_total,

        "missing_train_labels":
            missing_train_labels,

        "labels_without_train_images":
            labels_without_train_images,

        "leakage_train_val":
            leakage_train_val,

        "leakage_train_test":
            leakage_train_test,

        "leakage_val_test":
            leakage_val_test,

        "augmented_images_without_labels":
            augmented_images_without_labels,

        "augmented_labels_without_images":
            augmented_labels_without_images,

        "missing_augmented_sources":
            missing_augmented_sources,

        "augmented_box_increases":
            augmented_box_increases,

        "duplicate_augmented_annotations":
            duplicate_augmented_annotations,

        "augmented_class_mismatches":
            augmented_class_mismatches,

        "passed":
            passed,
    }


def save_preflight_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )