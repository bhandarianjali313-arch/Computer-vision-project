from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ml.src.data.neu_inspector import IMAGE_EXTENSIONS
from ml.src.data.voc_to_yolo import CLASS_NAMES


def yolo_to_pixel_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert normalized YOLO coordinates to pixel coordinates.

    YOLO format:
        x_center, y_center, width, height

    Returns:
        xmin, ymin, xmax, ymax
    """

    x_center_px = x_center * image_width
    y_center_px = y_center * image_height

    width_px = width * image_width
    height_px = height * image_height

    xmin = int(round(x_center_px - width_px / 2))
    ymin = int(round(y_center_px - height_px / 2))

    xmax = int(round(x_center_px + width_px / 2))
    ymax = int(round(y_center_px + height_px / 2))

    xmin = max(0, min(xmin, image_width - 1))
    ymin = max(0, min(ymin, image_height - 1))

    xmax = max(0, min(xmax, image_width))
    ymax = max(0, min(ymax, image_height))

    return xmin, ymin, xmax, ymax


def find_image_by_stem(
    dataset_root: Path,
    stem: str,
) -> Path | None:
    """
    Find an image recursively using its filename stem.
    """

    for path in dataset_root.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.stem == stem
        ):
            return path

    return None


def read_yolo_label(
    label_path: Path,
) -> list[dict]:
    """
    Read one YOLO annotation file.
    """

    annotations = []

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

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(
                f"Invalid class ID {class_id} "
                f"in {label_path}"
            )

        annotations.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            }
        )

    return annotations


def draw_yolo_annotations(
    image: np.ndarray,
    annotations: list[dict],
) -> np.ndarray:
    """
    Draw YOLO bounding boxes and class names
    on an OpenCV image.
    """

    output = image.copy()

    image_height, image_width = output.shape[:2]

    for annotation in annotations:
        xmin, ymin, xmax, ymax = (
            yolo_to_pixel_bbox(
                x_center=annotation["x_center"],
                y_center=annotation["y_center"],
                width=annotation["width"],
                height=annotation["height"],
                image_width=image_width,
                image_height=image_height,
            )
        )

        class_id = annotation["class_id"]
        class_name = annotation["class_name"]

        # Different visual color for each class.
        color = (
            int((37 * (class_id + 1)) % 255),
            int((97 * (class_id + 1)) % 255),
            int((157 * (class_id + 1)) % 255),
        )

        cv2.rectangle(
            output,
            (xmin, ymin),
            (xmax, ymax),
            color,
            2,
        )

        label = (
            f"{class_id}: {class_name}"
        )

        text_position_y = max(
            ymin - 6,
            12,
        )

        cv2.putText(
            output,
            label,
            (xmin, text_position_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )

    return output


def visualize_label_file(
    label_path: Path,
    dataset_root: Path,
    output_dir: Path,
) -> Path:
    """
    Generate a visualization for one YOLO label file.
    """

    image_path = find_image_by_stem(
        dataset_root,
        label_path.stem,
    )

    if image_path is None:
        raise FileNotFoundError(
            f"Image not found for label: "
            f"{label_path.name}"
        )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not read image: "
            f"{image_path}"
        )

    annotations = read_yolo_label(
        label_path
    )

    visualized_image = draw_yolo_annotations(
        image,
        annotations,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{label_path.stem}_visualized.jpg"
    )

    success = cv2.imwrite(
        str(output_path),
        visualized_image,
    )

    if not success:
        raise RuntimeError(
            f"Failed to save image: "
            f"{output_path}"
        )

    return output_path