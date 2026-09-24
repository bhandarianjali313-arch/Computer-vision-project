from pathlib import Path

import pytest

from ml.src.data.voc_to_yolo import (
    CLASS_TO_ID,
    convert_annotation,
    validate_yolo_bbox,
    voc_bbox_to_yolo,
)


def test_class_mapping():
    assert (
        CLASS_TO_ID["crazing"]
        == 0
    )

    assert (
        CLASS_TO_ID["inclusion"]
        == 1
    )

    assert (
        CLASS_TO_ID["patches"]
        == 2
    )

    assert (
        CLASS_TO_ID["pitted_surface"]
        == 3
    )

    assert (
        CLASS_TO_ID[
            "rolled-in_scale"
        ]
        == 4
    )

    assert (
        CLASS_TO_ID["scratches"]
        == 5
    )


def test_voc_bbox_to_yolo():
    result = voc_bbox_to_yolo(
        xmin=50,
        ymin=50,
        xmax=150,
        ymax=150,
        image_width=200,
        image_height=200,
    )

    (
        x_center,
        y_center,
        width,
        height,
    ) = result

    assert (
        x_center
        == pytest.approx(0.5)
    )

    assert (
        y_center
        == pytest.approx(0.5)
    )

    assert (
        width
        == pytest.approx(0.5)
    )

    assert (
        height
        == pytest.approx(0.5)
    )


def test_valid_yolo_bbox():
    assert validate_yolo_bbox(
        0.5,
        0.5,
        0.25,
        0.30,
    )


def test_invalid_yolo_bbox():
    assert not validate_yolo_bbox(
        1.2,
        0.5,
        0.25,
        0.30,
    )

    assert not validate_yolo_bbox(
        0.5,
        0.5,
        0.0,
        0.30,
    )


def test_convert_single_annotation(
    tmp_path: Path,
):
    xml_path = (
        tmp_path / "crazing_1.xml"
    )

    xml_path.write_text(
        """
<annotation>
    <filename>crazing_1.jpg</filename>

    <size>
        <width>200</width>
        <height>200</height>
        <depth>3</depth>
    </size>

    <object>
        <name>crazing</name>

        <bndbox>
            <xmin>50</xmin>
            <ymin>50</ymin>
            <xmax>150</xmax>
            <ymax>150</ymax>
        </bndbox>
    </object>
</annotation>
""".strip(),
        encoding="utf-8",
    )

    (
        lines,
        counts,
        duplicates,
    ) = convert_annotation(
        xml_path
    )

    assert len(lines) == 1

    assert (
        counts["crazing"]
        == 1
    )

    assert duplicates == 0

    values = lines[0].split()

    assert values[0] == "0"

    assert (
        float(values[1])
        == pytest.approx(0.5)
    )

    assert (
        float(values[2])
        == pytest.approx(0.5)
    )

    assert (
        float(values[3])
        == pytest.approx(0.5)
    )

    assert (
        float(values[4])
        == pytest.approx(0.5)
    )


def test_duplicate_voc_annotations_are_removed(
    tmp_path: Path,
):
    xml_path = (
        tmp_path / "duplicate.xml"
    )

    xml_path.write_text(
        """
<annotation>
    <filename>duplicate.jpg</filename>

    <size>
        <width>200</width>
        <height>200</height>
        <depth>3</depth>
    </size>

    <object>
        <name>inclusion</name>

        <bndbox>
            <xmin>50</xmin>
            <ymin>40</ymin>
            <xmax>100</xmax>
            <ymax>120</ymax>
        </bndbox>
    </object>

    <object>
        <name>inclusion</name>

        <bndbox>
            <xmin>50</xmin>
            <ymin>40</ymin>
            <xmax>100</xmax>
            <ymax>120</ymax>
        </bndbox>
    </object>
</annotation>
""".strip(),
        encoding="utf-8",
    )

    (
        lines,
        counts,
        duplicates,
    ) = convert_annotation(
        xml_path
    )

    assert len(lines) == 1

    assert (
        counts["inclusion"]
        == 1
    )

    assert duplicates == 1

    values = lines[0].split()

    assert values[0] == "1"


def test_same_bbox_different_classes_not_removed(
    tmp_path: Path,
):
    """
    Two boxes with identical coordinates but different
    classes must NOT be considered duplicates.
    """

    xml_path = (
        tmp_path
        / "different_classes.xml"
    )

    xml_path.write_text(
        """
<annotation>
    <filename>sample.jpg</filename>

    <size>
        <width>200</width>
        <height>200</height>
        <depth>3</depth>
    </size>

    <object>
        <name>inclusion</name>

        <bndbox>
            <xmin>50</xmin>
            <ymin>40</ymin>
            <xmax>100</xmax>
            <ymax>120</ymax>
        </bndbox>
    </object>

    <object>
        <name>patches</name>

        <bndbox>
            <xmin>50</xmin>
            <ymin>40</ymin>
            <xmax>100</xmax>
            <ymax>120</ymax>
        </bndbox>
    </object>
</annotation>
""".strip(),
        encoding="utf-8",
    )

    (
        lines,
        counts,
        duplicates,
    ) = convert_annotation(
        xml_path
    )

    assert len(lines) == 2

    assert (
        counts["inclusion"]
        == 1
    )

    assert (
        counts["patches"]
        == 1
    )

    assert duplicates == 0


def test_invalid_voc_bbox_raises_error(
    tmp_path: Path,
):
    xml_path = (
        tmp_path
        / "invalid_bbox.xml"
    )

    xml_path.write_text(
        """
<annotation>
    <filename>invalid.jpg</filename>

    <size>
        <width>200</width>
        <height>200</height>
        <depth>3</depth>
    </size>

    <object>
        <name>scratches</name>

        <bndbox>
            <xmin>150</xmin>
            <ymin>50</ymin>
            <xmax>100</xmax>
            <ymax>150</ymax>
        </bndbox>
    </object>
</annotation>
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError
    ):
        convert_annotation(
            xml_path
        )