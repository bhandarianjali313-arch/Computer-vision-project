from pathlib import Path

import cv2
import numpy as np

from ml.src.data.neu_inspector import (
    inspect_dataset,
    normalize_class_name,
)


def create_test_annotation(
    path: Path,
    filename: str,
) -> None:
    xml_content = f"""
<annotation>
    <filename>{filename}</filename>

    <size>
        <width>100</width>
        <height>100</height>
        <depth>3</depth>
    </size>

    <object>
        <name>crazing</name>

        <bndbox>
            <xmin>10</xmin>
            <ymin>20</ymin>
            <xmax>70</xmax>
            <ymax>80</ymax>
        </bndbox>
    </object>
</annotation>
"""

    path.write_text(
        xml_content.strip(),
        encoding="utf-8",
    )


def test_normalize_class_names():
    assert (
        normalize_class_name("Pitted Surface")
        == "pitted_surface"
    )

    assert (
        normalize_class_name("rolled-in scale")
        == "rolled-in_scale"
    )

    assert (
        normalize_class_name("Scratches")
        == "scratches"
    )


def test_dataset_inspection(tmp_path):
    images_dir = tmp_path / "IMAGES"
    annotations_dir = tmp_path / "ANNOTATIONS"

    images_dir.mkdir()
    annotations_dir.mkdir()

    image_path = images_dir / "crazing_1.jpg"

    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    cv2.imwrite(
        str(image_path),
        image,
    )

    annotation_path = (
        annotations_dir / "crazing_1.xml"
    )

    create_test_annotation(
        annotation_path,
        image_path.name,
    )

    report = inspect_dataset(tmp_path)

    assert report["image_count"] == 1
    assert report["annotation_count"] == 1
    assert report["total_objects"] == 1

    assert (
        report["class_object_counts"]["crazing"]
        == 1
    )

    assert report["invalid_bbox_count"] == 0
    assert report["invalid_xml_count"] == 0

    assert (
        report["images_without_annotations"]
        == []
    )

    assert (
        report["annotations_without_images"]
        == []
    )