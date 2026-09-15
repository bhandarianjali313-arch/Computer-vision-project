from pathlib import Path

import cv2
import numpy as np

from ml.src.data.yolo_visualizer import (
    find_image_by_stem,
    read_yolo_label,
    yolo_to_pixel_bbox,
)


def test_yolo_to_pixel_bbox():
    bbox = yolo_to_pixel_bbox(
        x_center=0.5,
        y_center=0.5,
        width=0.5,
        height=0.5,
        image_width=200,
        image_height=200,
    )

    assert bbox == (
        50,
        50,
        150,
        150,
    )


def test_read_yolo_label(
    tmp_path: Path,
):
    label_path = (
        tmp_path / "sample.txt"
    )

    label_path.write_text(
        "0 0.500000 0.500000 "
        "0.500000 0.500000\n",
        encoding="utf-8",
    )

    annotations = read_yolo_label(
        label_path
    )

    assert len(annotations) == 1

    annotation = annotations[0]

    assert annotation["class_id"] == 0
    assert (
        annotation["class_name"]
        == "crazing"
    )

    assert (
        annotation["x_center"]
        == 0.5
    )


def test_find_image_by_stem(
    tmp_path: Path,
):
    images_dir = (
        tmp_path / "IMAGES"
    )

    images_dir.mkdir()

    image_path = (
        images_dir / "sample.jpg"
    )

    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    cv2.imwrite(
        str(image_path),
        image,
    )

    result = find_image_by_stem(
        tmp_path,
        "sample",
    )

    assert result == image_path