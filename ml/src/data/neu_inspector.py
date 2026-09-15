from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

EXPECTED_CLASSES = {
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
}


def normalize_class_name(name: str) -> str:
    """
    Normalize common variations of NEU defect class names.
    """
    cleaned = name.strip().lower()

    aliases = {
        "crazing": "crazing",
        "inclusion": "inclusion",
        "patches": "patches",
        "pitted surface": "pitted_surface",
        "pitted_surface": "pitted_surface",
        "pitted-surface": "pitted_surface",
        "rolled-in scale": "rolled-in_scale",
        "rolled-in_scale": "rolled-in_scale",
        "rolled in scale": "rolled-in_scale",
        "rolled-in-scale": "rolled-in_scale",
        "scratches": "scratches",
        "scratch": "scratches",
    }

    return aliases.get(cleaned, cleaned)


def find_images(dataset_root: Path) -> list[Path]:
    """
    Recursively find all supported image files.
    """
    return sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_annotations(dataset_root: Path) -> list[Path]:
    """
    Recursively find all XML annotation files.
    """
    return sorted(dataset_root.rglob("*.xml"))


def parse_annotation(xml_path: Path) -> dict[str, Any]:
    """
    Parse a Pascal VOC XML annotation.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")

    if size_node is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    width_node = size_node.find("width")
    height_node = size_node.find("height")

    if width_node is None or height_node is None:
        raise ValueError(f"Missing image dimensions in {xml_path}")

    width = int(width_node.text)
    height = int(height_node.text)

    objects = []

    for object_node in root.findall("object"):
        name_node = object_node.find("name")
        bbox_node = object_node.find("bndbox")

        if name_node is None or bbox_node is None:
            continue

        class_name = normalize_class_name(name_node.text or "")

        xmin = int(float(bbox_node.findtext("xmin", "0")))
        ymin = int(float(bbox_node.findtext("ymin", "0")))
        xmax = int(float(bbox_node.findtext("xmax", "0")))
        ymax = int(float(bbox_node.findtext("ymax", "0")))

        objects.append(
            {
                "class_name": class_name,
                "bbox": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                },
            }
        )

    return {
        "width": width,
        "height": height,
        "objects": objects,
    }


def is_valid_bbox(
    bbox: dict[str, int],
    width: int,
    height: int,
) -> bool:
    """
    Validate bounding-box coordinates.
    """
    xmin = bbox["xmin"]
    ymin = bbox["ymin"]
    xmax = bbox["xmax"]
    ymax = bbox["ymax"]

    return (
        0 <= xmin < xmax <= width
        and 0 <= ymin < ymax <= height
    )


def inspect_dataset(dataset_root: Path) -> dict[str, Any]:
    """
    Perform a complete structural audit of the NEU dataset.
    """
    dataset_root = dataset_root.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_root}"
        )

    images = find_images(dataset_root)
    annotations = find_annotations(dataset_root)

    image_by_stem = {image.stem: image for image in images}
    annotation_by_stem = {
        annotation.stem: annotation
        for annotation in annotations
    }

    class_object_counts: Counter[str] = Counter()
    images_per_class: dict[str, set[str]] = defaultdict(set)
    dimension_counts: Counter[str] = Counter()

    invalid_xml_files: list[str] = []
    image_read_errors: list[str] = []
    dimension_mismatches: list[str] = []
    invalid_boxes: list[dict[str, Any]] = []

    total_objects = 0

    for annotation_path in annotations:
        try:
            annotation = parse_annotation(annotation_path)
        except (ET.ParseError, ValueError, TypeError) as error:
            invalid_xml_files.append(
                f"{annotation_path}: {error}"
            )
            continue

        width = annotation["width"]
        height = annotation["height"]

        stem = annotation_path.stem
        image_path = image_by_stem.get(stem)

        if image_path is not None:
            image = cv2.imread(str(image_path))

            if image is None:
                image_read_errors.append(str(image_path))
            else:
                actual_height, actual_width = image.shape[:2]

                dimension_counts[
                    f"{actual_width}x{actual_height}"
                ] += 1

                if (
                    actual_width != width
                    or actual_height != height
                ):
                    dimension_mismatches.append(stem)

        for object_data in annotation["objects"]:
            total_objects += 1

            class_name = object_data["class_name"]
            bbox = object_data["bbox"]

            class_object_counts[class_name] += 1
            images_per_class[class_name].add(stem)

            if not is_valid_bbox(bbox, width, height):
                invalid_boxes.append(
                    {
                        "annotation": str(annotation_path),
                        "class_name": class_name,
                        "bbox": bbox,
                        "image_size": [width, height],
                    }
                )

    image_stems = set(image_by_stem)
    annotation_stems = set(annotation_by_stem)

    images_without_annotations = sorted(
        image_stems - annotation_stems
    )

    annotations_without_images = sorted(
        annotation_stems - image_stems
    )

    detected_classes = set(class_object_counts)

    missing_expected_classes = sorted(
        EXPECTED_CLASSES - detected_classes
    )

    unexpected_classes = sorted(
        detected_classes - EXPECTED_CLASSES
    )

    return {
        "dataset_root": str(dataset_root),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "total_objects": total_objects,
        "detected_classes": sorted(detected_classes),
        "class_object_counts": dict(
            sorted(class_object_counts.items())
        ),
        "images_per_class": {
            class_name: len(image_names)
            for class_name, image_names
            in sorted(images_per_class.items())
        },
        "image_dimensions": dict(
            sorted(dimension_counts.items())
        ),
        "images_without_annotations":
            images_without_annotations,
        "annotations_without_images":
            annotations_without_images,
        "invalid_xml_count": len(invalid_xml_files),
        "invalid_xml_files": invalid_xml_files,
        "image_read_error_count": len(image_read_errors),
        "image_read_errors": image_read_errors,
        "dimension_mismatch_count":
            len(dimension_mismatches),
        "dimension_mismatches": dimension_mismatches,
        "invalid_bbox_count": len(invalid_boxes),
        "invalid_boxes": invalid_boxes,
        "missing_expected_classes":
            missing_expected_classes,
        "unexpected_classes":
            unexpected_classes,
    }