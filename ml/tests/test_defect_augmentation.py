from pathlib import Path

import albumentations as A
import numpy as np
import pytest

from ml.src.augmentation.defect_augmentation import (
    augment_sample,
    load_yolo_annotations,
    save_yolo_annotations,
)


def test_load_yolo_annotations(
    tmp_path: Path,
):
    label_path = (
        tmp_path / "sample.txt"
    )

    label_path.write_text(
        "2 0.500000 0.400000 "
        "0.300000 0.200000\n",
        encoding="utf-8",
    )

    bboxes, classes = (
        load_yolo_annotations(
            label_path
        )
    )

    assert classes == [2]

    assert bboxes == [
        [
            0.5,
            0.4,
            0.3,
            0.2,
        ]
    ]


def test_save_and_reload_annotations(
    tmp_path: Path,
):
    output_path = (
        tmp_path / "labels.txt"
    )

    save_yolo_annotations(
        output_path=output_path,
        bboxes=[
            [
                0.5,
                0.5,
                0.25,
                0.30,
            ]
        ],
        class_labels=[4],
    )

    bboxes, classes = (
        load_yolo_annotations(
            output_path
        )
    )

    assert classes == [4]

    assert bboxes[0][0] == pytest.approx(
        0.5
    )

    assert bboxes[0][1] == pytest.approx(
        0.5
    )

    assert bboxes[0][2] == pytest.approx(
        0.25
    )

    assert bboxes[0][3] == pytest.approx(
        0.30
    )


def test_horizontal_flip_updates_bbox():
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    transform = A.Compose(
        [
            A.HorizontalFlip(
                p=1.0
            )
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=[
                "class_labels"
            ],
        ),
    )

    (
        _,
        transformed_boxes,
        transformed_classes,
    ) = augment_sample(
        image=image,
        bboxes=[
            [
                0.25,
                0.50,
                0.20,
                0.30,
            ]
        ],
        class_labels=[1],
        transform=transform,
    )

    assert len(
        transformed_boxes
    ) == 1

    assert (
        transformed_boxes[0][0]
        == pytest.approx(
            0.75,
            abs=1e-6,
        )
    )

    assert (
        transformed_boxes[0][1]
        == pytest.approx(
            0.50,
            abs=1e-6,
        )
    )

    assert transformed_classes == [1]


def test_augmentation_keeps_coordinates_valid():
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    transform = A.Compose(
        [
            A.HorizontalFlip(
                p=1.0
            )
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=[
                "class_labels"
            ],
        ),
    )

    (
        _,
        bboxes,
        _,
    ) = augment_sample(
        image=image,
        bboxes=[
            [
                0.4,
                0.4,
                0.25,
                0.25,
            ]
        ],
        class_labels=[3],
        transform=transform,
    )

    for bbox in bboxes:
        x_center = bbox[0]
        y_center = bbox[1]
        width = bbox[2]
        height = bbox[3]

        assert (
            0.0
            <= x_center
            <= 1.0
        )

        assert (
            0.0
            <= y_center
            <= 1.0
        )

        assert (
            0.0
            < width
            <= 1.0
        )

        assert (
            0.0
            < height
            <= 1.0
        )