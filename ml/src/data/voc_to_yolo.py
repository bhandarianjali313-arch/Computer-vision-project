from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ml.src.data.neu_inspector import (
    find_annotations,
    parse_annotation,
)


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_TO_ID = {
    class_name: class_id
    for class_id, class_name in enumerate(CLASS_NAMES)
}


def voc_bbox_to_yolo(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert a Pascal VOC bounding box to normalized YOLO format.

    Pascal VOC:
        xmin, ymin, xmax, ymax

    YOLO:
        x_center, y_center, width, height
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            "Image width and height must be greater than zero."
        )

    if not (
        0 <= xmin < xmax <= image_width
        and 0 <= ymin < ymax <= image_height
    ):
        raise ValueError(
            "Invalid bounding box: "
            f"({xmin}, {ymin}, {xmax}, {ymax}) "
            f"for image size "
            f"{image_width}x{image_height}"
        )

    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0

    x_center /= image_width
    y_center /= image_height

    normalized_width = bbox_width / image_width
    normalized_height = bbox_height / image_height

    return (
        x_center,
        y_center,
        normalized_width,
        normalized_height,
    )


def validate_yolo_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> bool:
    """
    Validate normalized YOLO bounding-box coordinates.
    """

    return (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    )


def convert_annotation(
    xml_path: Path,
) -> tuple[list[str], Counter[str]]:
    """
    Convert one Pascal VOC XML annotation to YOLO label lines.
    """

    annotation = parse_annotation(xml_path)

    image_width = annotation["width"]
    image_height = annotation["height"]

    yolo_lines: list[str] = []
    class_counts: Counter[str] = Counter()

    for object_data in annotation["objects"]:
        class_name = object_data["class_name"]

        if class_name not in CLASS_TO_ID:
            raise ValueError(
                f"Unknown class '{class_name}' "
                f"in {xml_path}"
            )

        class_id = CLASS_TO_ID[class_name]

        bbox = object_data["bbox"]

        yolo_bbox = voc_bbox_to_yolo(
            xmin=bbox["xmin"],
            ymin=bbox["ymin"],
            xmax=bbox["xmax"],
            ymax=bbox["ymax"],
            image_width=image_width,
            image_height=image_height,
        )

        if not validate_yolo_bbox(*yolo_bbox):
            raise ValueError(
                f"Invalid YOLO coordinates generated "
                f"from {xml_path}: {yolo_bbox}"
            )

        x_center, y_center, width, height = yolo_bbox

        line = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        yolo_lines.append(line)
        class_counts[class_name] += 1

    return yolo_lines, class_counts


def convert_dataset(
    dataset_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Convert all Pascal VOC XML annotations in a dataset
    to YOLO-format text files.
    """

    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_root}"
        )

    annotations = find_annotations(dataset_root)

    if not annotations:
        raise FileNotFoundError(
            f"No XML annotations found in: {dataset_root}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stems = [
        annotation.stem
        for annotation in annotations
    ]

    if len(stems) != len(set(stems)):
        raise ValueError(
            "Duplicate annotation filenames detected. "
            "YOLO labels require unique file stems."
        )

    total_boxes = 0
    class_counts: Counter[str] = Counter()

    for xml_path in annotations:
        yolo_lines, annotation_counts = (
            convert_annotation(xml_path)
        )

        output_path = (
            output_dir / f"{xml_path.stem}.txt"
        )

        output_path.write_text(
            "\n".join(yolo_lines) + "\n",
            encoding="utf-8",
        )

        total_boxes += len(yolo_lines)
        class_counts.update(annotation_counts)

    return {
        "annotation_count": len(annotations),
        "label_file_count": len(annotations),
        "total_boxes": total_boxes,
        "class_counts": {
            class_name: class_counts[class_name]
            for class_name in CLASS_NAMES
        },
        "output_dir": str(output_dir),
    }