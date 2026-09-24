from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import albumentations as A
import cv2

from ml.src.data.neu_inspector import IMAGE_EXTENSIONS


def build_train_augmentation() -> A.Compose:
    """
    Build the augmentation pipeline used only for training images.

    The transformations are intentionally moderate because industrial
    defects are often small and aggressive transformations could destroy
    their visual characteristics.
    """

    return A.Compose(
        [
            A.HorizontalFlip(
                p=0.5,
            ),

            A.VerticalFlip(
                p=0.2,
            ),

            A.Rotate(
              limit=10,
              border_mode=cv2.BORDER_CONSTANT,
              p=0.4,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.20,
                p=0.5,
            ),

            A.OneOf(
                [
                    A.GaussNoise(
                        p=1.0,
                    ),

                    A.GaussianBlur(
                        blur_limit=(3, 5),
                        p=1.0,
                    ),
                ],
                p=0.25,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.20,
        ),
    )


def load_yolo_annotations(
    label_path: Path,
) -> tuple[list[list[float]], list[int]]:
    """
    Read a YOLO label file.

    Each line:
        class_id x_center y_center width height
    """

    bboxes: list[list[float]] = []
    class_labels: list[int] = []

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO annotation in "
                f"{label_path}, line {line_number}"
            )

        try:
            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError as error:
            raise ValueError(
                f"Non-numeric YOLO annotation in "
                f"{label_path}, line {line_number}"
            ) from error

        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise ValueError(
                f"Invalid YOLO coordinates in "
                f"{label_path}, line {line_number}"
            )

        bboxes.append(
            [
                x_center,
                y_center,
                width,
                height,
            ]
        )

        class_labels.append(
            class_id
        )

    return bboxes, class_labels




def save_yolo_annotations(
    output_path: Path,
    bboxes: list,
    class_labels: list[int],
) -> None:
    """
    Save augmented annotations in YOLO format.

    Exact duplicate boxes are removed so that the
    same physical object is not written twice after
    augmentation.
    """

    if len(bboxes) != len(class_labels):
        raise ValueError(
            "Bounding-box count and class-label "
            "count do not match."
        )

    lines = []

    # Used to prevent exact duplicate YOLO annotations.
    seen_annotations = set()

    for class_id, bbox in zip(
        class_labels,
        bboxes,
    ):
        x_center = float(bbox[0])
        y_center = float(bbox[1])
        width = float(bbox[2])
        height = float(bbox[3])

        # Use the same precision that will be written
        # to the YOLO label file.
        annotation_key = (
            int(class_id),
            round(x_center, 6),
            round(y_center, 6),
            round(width, 6),
            round(height, 6),
        )

        if annotation_key in seen_annotations:
            continue

        seen_annotations.add(
            annotation_key
        )

        line = (
            f"{annotation_key[0]} "
            f"{annotation_key[1]:.6f} "
            f"{annotation_key[2]:.6f} "
            f"{annotation_key[3]:.6f} "
            f"{annotation_key[4]:.6f}"
        )

        lines.append(line)

    if not lines:
        raise ValueError(
            f"Augmentation produced no valid "
            f"annotations for {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def augment_sample(
    image,
    bboxes: list[list[float]],
    class_labels: list[int],
    transform: A.Compose,
) -> tuple[Any, list, list[int]]:
    """
    Apply augmentation while keeping YOLO boxes synchronized.
    """

    transformed = transform(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels,
    )

    augmented_image = transformed[
        "image"
    ]

    augmented_bboxes = list(
        transformed["bboxes"]
    )

    augmented_classes = [
        int(class_id)
        for class_id
        in transformed["class_labels"]
    ]

    return (
        augmented_image,
        augmented_bboxes,
        augmented_classes,
    )


def find_training_images(
    images_dir: Path,
) -> list[Path]:
    """
    Find all supported training images.
    """

    return sorted(
        path
        for path in images_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


def generate_augmented_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    augmentations_per_image: int = 1,
    limit: int | None = None,
    clean_output: bool = False,
    max_attempts: int = 5,
) -> dict[str, Any]:
    """
    Generate an offline augmented copy of the training split.

    Validation and test data must never be passed to this function.
    """

    if augmentations_per_image <= 0:
        raise ValueError(
            "augmentations_per_image must be greater than zero."
        )

    if clean_output:
        if output_images_dir.exists():
            shutil.rmtree(
                output_images_dir
            )

        if output_labels_dir.exists():
            shutil.rmtree(
                output_labels_dir
            )

    output_images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_labels_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = find_training_images(
        images_dir
    )

    if limit is not None:
        image_paths = image_paths[:limit]

    if not image_paths:
        raise FileNotFoundError(
            f"No training images found in {images_dir}"
        )

    transform = build_train_augmentation()

    generated_images = 0
    generated_boxes = 0
    skipped_samples = []
    original_boxes = 0

    for image_index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        label_path = (
            labels_dir
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Label not found for "
                f"{image_path.name}"
            )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise ValueError(
                f"Could not read image: "
                f"{image_path}"
            )

        bboxes, class_labels = (
            load_yolo_annotations(
                label_path
            )
        )

        original_boxes += len(
            bboxes
        )

        for augmentation_number in range(
            1,
            augmentations_per_image + 1,
        ):
            successful = False

            for _ in range(
                max_attempts
            ):
                (
                    augmented_image,
                    augmented_bboxes,
                    augmented_classes,
                ) = augment_sample(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels,
                    transform=transform,
                )

                # We avoid creating an augmented
                # training image with no defect boxes.
                if augmented_bboxes:
                    successful = True
                    break

            if not successful:
                skipped_samples.append(
                    image_path.name
                )
                continue

            output_stem = (
                f"{image_path.stem}"
                f"_aug{augmentation_number:02d}"
            )

            output_image_path = (
                output_images_dir
                / f"{output_stem}.jpg"
            )

            output_label_path = (
                output_labels_dir
                / f"{output_stem}.txt"
            )

            image_saved = cv2.imwrite(
                str(output_image_path),
                augmented_image,
            )

            if not image_saved:
                raise RuntimeError(
                    f"Failed to save "
                    f"{output_image_path}"
                )

            save_yolo_annotations(
                output_path=output_label_path,
                bboxes=augmented_bboxes,
                class_labels=augmented_classes,
            )

            generated_images += 1
            generated_boxes += len(
                augmented_bboxes
            )

        if (
            image_index % 100 == 0
            or image_index == len(image_paths)
        ):
            print(
                f"Processed "
                f"{image_index}/"
                f"{len(image_paths)} images"
            )

    return {
        "source_image_count": len(
            image_paths
        ),
        "original_box_count": original_boxes,
        "augmentations_per_image":
            augmentations_per_image,
        "generated_image_count":
            generated_images,
        "generated_box_count":
            generated_boxes,
        "skipped_count":
            len(skipped_samples),
        "skipped_samples":
            skipped_samples,
    }


def save_augmentation_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save an augmentation-generation summary.
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