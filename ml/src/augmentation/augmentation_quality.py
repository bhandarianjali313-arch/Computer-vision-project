from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from ml.src.data.neu_inspector import IMAGE_EXTENSIONS
from ml.src.data.voc_to_yolo import CLASS_NAMES
from ml.src.data.yolo_visualizer import (
    draw_yolo_annotations,
    read_yolo_label,
)


AUGMENTED_STEM_PATTERN = re.compile(
    r"^(?P<source>.+)_aug(?P<number>\d+)$"
)


def source_stem_from_augmented_stem(
    augmented_stem: str,
) -> str:
    """
    Recover the original image stem.

    Example:
        crazing_12_aug01 -> crazing_12
    """

    match = AUGMENTED_STEM_PATTERN.match(
        augmented_stem
    )

    if match is None:
        raise ValueError(
            "Invalid augmented filename stem: "
            f"{augmented_stem}"
        )

    return match.group("source")


def collect_images(
    directory: Path,
) -> dict[str, Path]:
    """
    Collect supported images from a flat directory.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Image directory not found: {directory}"
        )

    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    }


def collect_labels(
    directory: Path,
) -> dict[str, Path]:
    """
    Collect YOLO label files from a flat directory.
    """

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


def validate_augmented_dataset(
    source_images_dir: Path,
    source_labels_dir: Path,
    augmented_images_dir: Path,
    augmented_labels_dir: Path,
) -> dict[str, Any]:
    """
    Validate the structural and annotation integrity
    of the generated augmentation dataset.
    """

    source_images = collect_images(
        source_images_dir
    )

    source_labels = collect_labels(
        source_labels_dir
    )

    augmented_images = collect_images(
        augmented_images_dir
    )

    augmented_labels = collect_labels(
        augmented_labels_dir
    )

    image_stems = set(
        augmented_images
    )

    label_stems = set(
        augmented_labels
    )

    images_without_labels = sorted(
        image_stems - label_stems
    )

    labels_without_images = sorted(
        label_stems - image_stems
    )

    common_stems = sorted(
        image_stems & label_stems
    )

    missing_source_images = []
    missing_source_labels = []
    unreadable_images = []
    dimension_mismatches = []
    empty_augmented_labels = []
    class_mismatches = []

    original_box_count = 0
    augmented_box_count = 0

    class_counts: Counter[str] = Counter()

    for augmented_stem in common_stems:
        try:
            source_stem = (
                source_stem_from_augmented_stem(
                    augmented_stem
                )
            )
        except ValueError:
            missing_source_images.append(
                augmented_stem
            )
            continue

        if source_stem not in source_images:
            missing_source_images.append(
                augmented_stem
            )
            continue

        if source_stem not in source_labels:
            missing_source_labels.append(
                augmented_stem
            )
            continue

        source_image = cv2.imread(
            str(source_images[source_stem])
        )

        augmented_image = cv2.imread(
            str(augmented_images[augmented_stem])
        )

        if source_image is None:
            unreadable_images.append(
                str(source_images[source_stem])
            )
            continue

        if augmented_image is None:
            unreadable_images.append(
                str(
                    augmented_images[
                        augmented_stem
                    ]
                )
            )
            continue

        if (
            source_image.shape[:2]
            != augmented_image.shape[:2]
        ):
            dimension_mismatches.append(
                augmented_stem
            )

        source_annotations = read_yolo_label(
            source_labels[source_stem]
        )

        augmented_annotations = (
            read_yolo_label(
                augmented_labels[
                    augmented_stem
                ]
            )
        )

        original_box_count += len(
            source_annotations
        )

        augmented_box_count += len(
            augmented_annotations
        )

        if not augmented_annotations:
            empty_augmented_labels.append(
                augmented_stem
            )
            continue

        source_classes = {
            annotation["class_id"]
            for annotation
            in source_annotations
        }

        augmented_classes = {
            annotation["class_id"]
            for annotation
            in augmented_annotations
        }

        # Augmentation can remove a partially visible
        # bounding box, but it must never invent
        # a new defect class.
        if not augmented_classes.issubset(
            source_classes
        ):
            class_mismatches.append(
                augmented_stem
            )

        for annotation in (
            augmented_annotations
        ):
            class_name = (
                annotation["class_name"]
            )

            class_counts[
                class_name
            ] += 1

    if original_box_count > 0:
        box_retention_rate = (
            augmented_box_count
            / original_box_count
        )
    else:
        box_retention_rate = 0.0

    passed = not any(
        [
            images_without_labels,
            labels_without_images,
            missing_source_images,
            missing_source_labels,
            unreadable_images,
            dimension_mismatches,
            empty_augmented_labels,
            class_mismatches,
        ]
    )

    return {
        "source_image_count":
            len(source_images),
        "source_label_count":
            len(source_labels),
        "augmented_image_count":
            len(augmented_images),
        "augmented_label_count":
            len(augmented_labels),
        "validated_pair_count":
            len(common_stems),
        "original_box_count":
            original_box_count,
        "augmented_box_count":
            augmented_box_count,
        "box_retention_rate":
            box_retention_rate,
        "class_counts": {
            class_name:
                class_counts[class_name]
            for class_name
            in CLASS_NAMES
        },
        "images_without_labels":
            images_without_labels,
        "labels_without_images":
            labels_without_images,
        "missing_source_images":
            missing_source_images,
        "missing_source_labels":
            missing_source_labels,
        "unreadable_images":
            unreadable_images,
        "dimension_mismatches":
            dimension_mismatches,
        "empty_augmented_labels":
            empty_augmented_labels,
        "class_mismatches":
            class_mismatches,
        "passed":
            passed,
    }


def add_title(
    image,
    title: str,
):
    """
    Add a title strip above an image.
    """

    result = cv2.copyMakeBorder(
        image,
        32,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    cv2.putText(
        result,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return result


def build_comparison_image(
    source_image_path: Path,
    source_label_path: Path,
    augmented_image_path: Path,
    augmented_label_path: Path,
):
    """
    Create an original-vs-augmented annotated preview.
    """

    source_image = cv2.imread(
        str(source_image_path)
    )

    augmented_image = cv2.imread(
        str(augmented_image_path)
    )

    if source_image is None:
        raise ValueError(
            f"Could not read {source_image_path}"
        )

    if augmented_image is None:
        raise ValueError(
            f"Could not read {augmented_image_path}"
        )

    source_annotations = read_yolo_label(
        source_label_path
    )

    augmented_annotations = read_yolo_label(
        augmented_label_path
    )

    source_drawn = draw_yolo_annotations(
        source_image,
        source_annotations,
    )

    augmented_drawn = draw_yolo_annotations(
        augmented_image,
        augmented_annotations,
    )

    if (
        source_drawn.shape[:2]
        != augmented_drawn.shape[:2]
    ):
        augmented_drawn = cv2.resize(
            augmented_drawn,
            (
                source_drawn.shape[1],
                source_drawn.shape[0],
            ),
        )

    source_drawn = add_title(
        source_drawn,
        "ORIGINAL",
    )

    augmented_drawn = add_title(
        augmented_drawn,
        "AUGMENTED",
    )

    return cv2.hconcat(
        [
            source_drawn,
            augmented_drawn,
        ]
    )


def generate_comparison_previews(
    source_images_dir: Path,
    source_labels_dir: Path,
    augmented_images_dir: Path,
    augmented_labels_dir: Path,
    output_dir: Path,
    sample_count: int = 18,
    seed: int = 42,
) -> list[str]:
    """
    Generate reproducible original-vs-augmented
    visual comparison images.
    """

    source_images = collect_images(
        source_images_dir
    )

    source_labels = collect_labels(
        source_labels_dir
    )

    augmented_images = collect_images(
        augmented_images_dir
    )

    augmented_labels = collect_labels(
        augmented_labels_dir
    )

    valid_stems = sorted(
        set(augmented_images)
        & set(augmented_labels)
    )

    if not valid_stems:
        raise ValueError(
            "No valid augmented image/label pairs found."
        )

    rng = random.Random(seed)

    sample_count = min(
        sample_count,
        len(valid_stems),
    )

    selected_stems = rng.sample(
        valid_stems,
        sample_count,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated = []

    for augmented_stem in selected_stems:
        source_stem = (
            source_stem_from_augmented_stem(
                augmented_stem
            )
        )

        if (
            source_stem not in source_images
            or source_stem not in source_labels
        ):
            continue

        comparison = build_comparison_image(
            source_image_path=(
                source_images[source_stem]
            ),
            source_label_path=(
                source_labels[source_stem]
            ),
            augmented_image_path=(
                augmented_images[
                    augmented_stem
                ]
            ),
            augmented_label_path=(
                augmented_labels[
                    augmented_stem
                ]
            ),
        )

        output_path = (
            output_dir
            / f"{augmented_stem}_comparison.jpg"
        )

        success = cv2.imwrite(
            str(output_path),
            comparison,
        )

        if not success:
            raise RuntimeError(
                f"Failed to save {output_path}"
            )

        generated.append(
            str(output_path)
        )

    return generated


def save_quality_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save validation results as JSON.
    """

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