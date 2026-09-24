from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import yaml

from ml.src.data.voc_to_yolo import CLASS_NAMES
from ml.src.data.yolo_visualizer import (
    read_yolo_label,
    yolo_to_pixel_bbox,
)


def load_error_analysis_config(
    config_path: Path,
) -> dict[str, Any]:
    """
    Load error-analysis configuration.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Error-analysis config not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Error-analysis config must be a mapping."
        )

    analysis = config.get("analysis")

    if not isinstance(analysis, dict):
        raise ValueError(
            "Configuration must contain an 'analysis' section."
        )

    required = {
        "confidence_threshold",
        "match_iou_threshold",
        "imgsz",
        "max_error_previews",
    }

    missing = required - set(analysis)

    if missing:
        raise ValueError(
            f"Missing error-analysis settings: {sorted(missing)}"
        )

    return analysis


def calculate_iou(
    box_a: list[float] | tuple[float, float, float, float],
    box_b: list[float] | tuple[float, float, float, float],
) -> float:
    """
    Calculate intersection-over-union for two XYXY boxes.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - intersection_area
    )

    if union <= 0:
        return 0.0

    return float(
        intersection_area / union
    )


def load_ground_truth(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """
    Load YOLO GT labels and convert boxes to pixel XYXY.
    """

    annotations = read_yolo_label(
        label_path
    )

    ground_truth = []

    for annotation in annotations:
        xyxy = yolo_to_pixel_bbox(
            x_center=annotation["x_center"],
            y_center=annotation["y_center"],
            width=annotation["width"],
            height=annotation["height"],
            image_width=image_width,
            image_height=image_height,
        )

        ground_truth.append(
            {
                "class_id":
                    annotation["class_id"],

                "class_name":
                    annotation["class_name"],

                "bbox":
                    [
                        float(value)
                        for value in xyxy
                    ],
            }
        )

    return ground_truth


def match_detections(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Greedily match predictions to ground-truth boxes by IoU.

    Matching is class-independent first. Once spatially matched:
      - same class -> true positive
      - different class -> class confusion

    Unmatched predictions -> false positives.
    Unmatched GT boxes -> false negatives.

    This is intended for diagnostic error analysis,
    not as a replacement for COCO/Ultralytics mAP evaluation.
    """

    candidates = []

    for gt_index, gt in enumerate(
        ground_truth
    ):
        for pred_index, prediction in enumerate(
            predictions
        ):
            iou = calculate_iou(
                gt["bbox"],
                prediction["bbox"],
            )

            if iou >= iou_threshold:
                candidates.append(
                    (
                        iou,
                        gt_index,
                        pred_index,
                    )
                )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    matched_gt = set()
    matched_predictions = set()

    true_positives = []
    class_confusions = []

    for (
        iou,
        gt_index,
        pred_index,
    ) in candidates:
        if gt_index in matched_gt:
            continue

        if pred_index in matched_predictions:
            continue

        matched_gt.add(
            gt_index
        )

        matched_predictions.add(
            pred_index
        )

        gt = ground_truth[
            gt_index
        ]

        prediction = predictions[
            pred_index
        ]

        record = {
            "iou":
                float(iou),

            "ground_truth_class_id":
                gt["class_id"],

            "ground_truth_class":
                gt["class_name"],

            "predicted_class_id":
                prediction["class_id"],

            "predicted_class":
                prediction["class_name"],

            "confidence":
                prediction["confidence"],

            "ground_truth_bbox":
                gt["bbox"],

            "predicted_bbox":
                prediction["bbox"],
        }

        if (
            gt["class_id"]
            == prediction["class_id"]
        ):
            true_positives.append(
                record
            )

        else:
            class_confusions.append(
                record
            )

    false_positives = []

    for pred_index, prediction in enumerate(
        predictions
    ):
        if pred_index not in matched_predictions:
            false_positives.append(
                prediction
            )

    false_negatives = []

    for gt_index, gt in enumerate(
        ground_truth
    ):
        if gt_index not in matched_gt:
            false_negatives.append(
                gt
            )

    return {
        "true_positives":
            true_positives,

        "class_confusions":
            class_confusions,

        "false_positives":
            false_positives,

        "false_negatives":
            false_negatives,
    }


def summarize_image_errors(
    image_name: str,
    matches: dict[str, Any],
) -> dict[str, Any]:
    """
    Create a compact per-image error summary.
    """

    tp = len(
        matches["true_positives"]
    )

    fp = len(
        matches["false_positives"]
    )

    fn = len(
        matches["false_negatives"]
    )

    confusion = len(
        matches["class_confusions"]
    )

    total_errors = (
        fp
        + fn
        + confusion
    )

    return {
        "image":
            image_name,

        "true_positives":
            tp,

        "false_positives":
            fp,

        "false_negatives":
            fn,

        "class_confusions":
            confusion,

        "total_errors":
            total_errors,
    }


def build_global_summary(
    image_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate error counts over the complete validation set.
    """

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_confusions = 0

    fp_classes: Counter[str] = Counter()
    fn_classes: Counter[str] = Counter()

    confusion_pairs: Counter[str] = Counter()

    for image_result in image_results:
        matches = image_result[
            "matches"
        ]

        total_tp += len(
            matches["true_positives"]
        )

        total_fp += len(
            matches["false_positives"]
        )

        total_fn += len(
            matches["false_negatives"]
        )

        total_confusions += len(
            matches["class_confusions"]
        )

        for prediction in matches[
            "false_positives"
        ]:
            fp_classes[
                prediction["class_name"]
            ] += 1

        for ground_truth in matches[
            "false_negatives"
        ]:
            fn_classes[
                ground_truth["class_name"]
            ] += 1

        for confusion in matches[
            "class_confusions"
        ]:
            key = (
                f"{confusion['ground_truth_class']}"
                " -> "
                f"{confusion['predicted_class']}"
            )

            confusion_pairs[
                key
            ] += 1

    difficult_images = sorted(
        (
            image_result[
                "summary"
            ]
            for image_result
            in image_results
        ),
        key=lambda row: (
            row["total_errors"],
            row["false_negatives"],
            row["false_positives"],
        ),
        reverse=True,
    )

    return {
        "images_analyzed":
            len(image_results),

        "true_positives":
            total_tp,

        "false_positives":
            total_fp,

        "false_negatives":
            total_fn,

        "class_confusions":
            total_confusions,

        "false_positives_by_class": {
            class_name:
                fp_classes[class_name]
            for class_name
            in CLASS_NAMES
        },

        "false_negatives_by_class": {
            class_name:
                fn_classes[class_name]
            for class_name
            in CLASS_NAMES
        },

        "class_confusion_pairs":
            dict(
                confusion_pairs.most_common()
            ),

        "difficult_images":
            difficult_images,
    }


def draw_error_preview(
    image,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
):
    """
    Draw ground truth and predictions on one diagnostic image.

    GT:
        GT:<class>

    Prediction:
        P:<class> <confidence>
    """

    output = image.copy()

    for gt in ground_truth:
        x1, y1, x2, y2 = [
            int(value)
            for value in gt["bbox"]
        ]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            f"GT:{gt['class_name']}",
            (
                x1,
                max(
                    y1 - 4,
                    12,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    for prediction in predictions:
        x1, y1, x2, y2 = [
            int(value)
            for value
            in prediction["bbox"]
        ]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            1,
        )

        cv2.putText(
            output,
            (
                f"P:{prediction['class_name']} "
                f"{prediction['confidence']:.2f}"
            ),
            (
                x1,
                min(
                    y2 + 12,
                    output.shape[0] - 2,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save JSON report.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )