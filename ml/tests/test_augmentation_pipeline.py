from pathlib import Path

import cv2
import numpy as np
import pytest

from ml.src.augmentation.augmentation_quality import (
    source_stem_from_augmented_stem,
    validate_augmented_dataset,
)


def write_label(
    path: Path,
    class_id: int = 0,
):
    path.write_text(
        (
            f"{class_id} "
            "0.500000 "
            "0.500000 "
            "0.400000 "
            "0.400000\n"
        ),
        encoding="utf-8",
    )


def write_image(
    path: Path,
):
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    cv2.imwrite(
        str(path),
        image,
    )


def create_directories(
    root: Path,
):
    source_images = (
        root / "source_images"
    )

    source_labels = (
        root / "source_labels"
    )

    augmented_images = (
        root / "augmented_images"
    )

    augmented_labels = (
        root / "augmented_labels"
    )

    for directory in (
        source_images,
        source_labels,
        augmented_images,
        augmented_labels,
    ):
        directory.mkdir()

    return (
        source_images,
        source_labels,
        augmented_images,
        augmented_labels,
    )


def test_source_stem_from_augmented_stem():
    assert (
        source_stem_from_augmented_stem(
            "crazing_25_aug01"
        )
        == "crazing_25"
    )

    assert (
        source_stem_from_augmented_stem(
            "pitted_surface_7_aug12"
        )
        == "pitted_surface_7"
    )


def test_invalid_augmented_stem():
    with pytest.raises(
        ValueError
    ):
        source_stem_from_augmented_stem(
            "crazing_25"
        )


def test_valid_augmented_dataset(
    tmp_path: Path,
):
    (
        source_images,
        source_labels,
        augmented_images,
        augmented_labels,
    ) = create_directories(
        tmp_path
    )

    write_image(
        source_images / "crazing_1.jpg"
    )

    write_label(
        source_labels / "crazing_1.txt",
        class_id=0,
    )

    write_image(
        augmented_images
        / "crazing_1_aug01.jpg"
    )

    write_label(
        augmented_labels
        / "crazing_1_aug01.txt",
        class_id=0,
    )

    report = validate_augmented_dataset(
        source_images_dir=source_images,
        source_labels_dir=source_labels,
        augmented_images_dir=(
            augmented_images
        ),
        augmented_labels_dir=(
            augmented_labels
        ),
    )

    assert report["passed"] is True

    assert (
        report["augmented_image_count"]
        == 1
    )

    assert (
        report["augmented_label_count"]
        == 1
    )

    assert (
        report["augmented_box_count"]
        == 1
    )


def test_detect_image_without_label(
    tmp_path: Path,
):
    (
        source_images,
        source_labels,
        augmented_images,
        augmented_labels,
    ) = create_directories(
        tmp_path
    )

    write_image(
        source_images / "crazing_1.jpg"
    )

    write_label(
        source_labels / "crazing_1.txt"
    )

    write_image(
        augmented_images
        / "crazing_1_aug01.jpg"
    )

    report = validate_augmented_dataset(
        source_images_dir=source_images,
        source_labels_dir=source_labels,
        augmented_images_dir=(
            augmented_images
        ),
        augmented_labels_dir=(
            augmented_labels
        ),
    )

    assert report["passed"] is False

    assert (
        "crazing_1_aug01"
        in report["images_without_labels"]
    )