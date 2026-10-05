from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_quality_policy(
    config_path: Path,
) -> dict[str, Any]:

    if not config_path.exists():
        raise FileNotFoundError(
            f"Quality policy not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Quality policy must be a YAML mapping."
        )

    required_sections = {
        "postprocessing",
        "triage",
        "decision",
    }

    missing = (
        required_sections
        - set(config)
    )

    if missing:
        raise ValueError(
            "Missing quality-policy sections: "
            f"{sorted(missing)}"
        )

    return config


def get_bbox(
    detection: dict[str, Any],
) -> dict[str, float]:

    bbox = detection.get(
        "bbox_xyxy"
    )

    if bbox is None:
        bbox = detection.get(
            "bbox"
        )

    if not isinstance(bbox, dict):
        raise ValueError(
            "Detection does not contain "
            "'bbox_xyxy' or 'bbox'."
        )

    required = {
        "x1",
        "y1",
        "x2",
        "y2",
    }

    missing = required - set(bbox)

    if missing:
        raise ValueError(
            "Bounding box is missing coordinates: "
            f"{sorted(missing)}"
        )

    return {
        key: float(bbox[key])
        for key in required
    }


def bbox_iou(
    first: dict[str, float],
    second: dict[str, float],
) -> float:

    x1 = max(
        first["x1"],
        second["x1"],
    )

    y1 = max(
        first["y1"],
        second["y1"],
    )

    x2 = min(
        first["x2"],
        second["x2"],
    )

    y2 = min(
        first["y2"],
        second["y2"],
    )

    intersection_width = max(
        0.0,
        x2 - x1,
    )

    intersection_height = max(
        0.0,
        y2 - y1,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    first_area = (
        max(
            0.0,
            first["x2"] - first["x1"],
        )
        *
        max(
            0.0,
            first["y2"] - first["y1"],
        )
    )

    second_area = (
        max(
            0.0,
            second["x2"] - second["x1"],
        )
        *
        max(
            0.0,
            second["y2"] - second["y1"],
        )
    )

    union = (
        first_area
        + second_area
        - intersection
    )

    if union <= 0:
        return 0.0

    return float(
        intersection / union
    )


def filter_confidence(
    detections: list[dict[str, Any]],
    minimum_confidence: float,
) -> list[dict[str, Any]]:

    return [
        detection
        for detection in detections
        if float(
            detection.get(
                "confidence",
                0.0,
            )
        )
        >= minimum_confidence
    ]


def deduplicate_detections(
    detections: list[dict[str, Any]],
    iou_threshold: float,
) -> list[dict[str, Any]]:

    ordered = sorted(
        detections,
        key=lambda item: float(
            item.get(
                "confidence",
                0.0,
            )
        ),
        reverse=True,
    )

    kept = []

    for candidate in ordered:

        candidate_class = (
            candidate.get(
                "class_name"
            )
        )

        candidate_bbox = get_bbox(
            candidate
        )

        duplicate = False

        for accepted in kept:

            if (
                accepted.get(
                    "class_name"
                )
                != candidate_class
            ):
                continue

            overlap = bbox_iou(
                candidate_bbox,
                get_bbox(
                    accepted
                ),
            )

            if overlap >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(
                candidate
            )

    return kept


def calculate_area_ratio(
    detection: dict[str, Any],
    image_width: int,
    image_height: int,
) -> float:

    if (
        image_width <= 0
        or image_height <= 0
    ):
        raise ValueError(
            "Image dimensions must be positive."
        )

    bbox = get_bbox(
        detection
    )

    width = max(
        0.0,
        bbox["x2"] - bbox["x1"],
    )

    height = max(
        0.0,
        bbox["y2"] - bbox["y1"],
    )

    image_area = float(
        image_width
        * image_height
    )

    return float(
        (width * height)
        / image_area
    )


def classify_triage_level(
    confidence: float,
    area_ratio: float,
    policy: dict[str, Any],
) -> str:

    triage = policy[
        "triage"
    ]

    high_confidence = float(
        triage[
            "high_confidence"
        ]
    )

    medium_confidence = float(
        triage[
            "medium_confidence"
        ]
    )

    high_area_ratio = float(
        triage[
            "high_area_ratio"
        ]
    )

    medium_area_ratio = float(
        triage[
            "medium_area_ratio"
        ]
    )

    if (
        confidence >= high_confidence
        and area_ratio >= high_area_ratio
    ):
        return "HIGH"

    if (
        confidence >= medium_confidence
        or area_ratio >= medium_area_ratio
    ):
        return "MEDIUM"

    return "LOW"


def enrich_detections(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    policy: dict[str, Any],
) -> list[dict[str, Any]]:

    enriched = []

    for detection in detections:

        area_ratio = calculate_area_ratio(
            detection=detection,
            image_width=image_width,
            image_height=image_height,
        )

        confidence = float(
            detection[
                "confidence"
            ]
        )

        triage_level = (
            classify_triage_level(
                confidence=confidence,
                area_ratio=area_ratio,
                policy=policy,
            )
        )

        item = dict(
            detection
        )

        item[
            "area_ratio"
        ] = float(
            area_ratio
        )

        item[
            "triage_level"
        ] = triage_level

        enriched.append(
            item
        )

    return enriched


def make_quality_decision(
    detections: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[str, list[str]]:

    decision_config = policy[
        "decision"
    ]

    pass_label = decision_config[
        "pass_label"
    ]

    review_label = decision_config[
        "review_label"
    ]

    reject_label = decision_config[
        "reject_label"
    ]

    if not detections:
        return (
            pass_label,
            [
                "No detections remained "
                "after post-processing."
            ],
        )

    reasons = []

    high_detections = [
        detection
        for detection in detections
        if detection.get(
            "triage_level"
        )
        == "HIGH"
    ]

    reject_count = int(
        policy[
            "triage"
        ][
            "reject_detection_count"
        ]
    )

    if high_detections:
        reasons.append(
            "At least one HIGH operational "
            "triage detection was present."
        )

    if len(detections) >= reject_count:
        reasons.append(
            "Detection count reached the "
            "configured reject threshold."
        )

    if reasons:
        return (
            reject_label,
            reasons,
        )

    return (
        review_label,
        [
            "Defects were detected, but "
            "configured reject conditions "
            "were not reached."
        ],
    )


def apply_quality_policy(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    policy: dict[str, Any],
) -> dict[str, Any]:

    minimum_confidence = float(
        policy[
            "postprocessing"
        ][
            "minimum_confidence"
        ]
    )

    duplicate_iou = float(
        policy[
            "postprocessing"
        ][
            "duplicate_iou_threshold"
        ]
    )

    original_count = len(
        detections
    )

    confidence_filtered = (
        filter_confidence(
            detections,
            minimum_confidence,
        )
    )

    deduplicated = (
        deduplicate_detections(
            confidence_filtered,
            duplicate_iou,
        )
    )

    enriched = (
        enrich_detections(
            detections=deduplicated,
            image_width=image_width,
            image_height=image_height,
            policy=policy,
        )
    )

    decision, reasons = (
        make_quality_decision(
            enriched,
            policy,
        )
    )

    triage_counts = {
        "LOW": 0,
        "MEDIUM": 0,
        "HIGH": 0,
    }

    for detection in enriched:

        level = detection[
            "triage_level"
        ]

        triage_counts[
            level
        ] += 1

    return {
        "decision":
            decision,

        "decision_reasons":
            reasons,

        "input_detection_count":
            original_count,

        "after_confidence_filter":
            len(
                confidence_filtered
            ),

        "final_detection_count":
            len(
                enriched
            ),

        "triage_counts":
            triage_counts,

        "detections":
            enriched,

        "policy_note": (
            "Operational triage is a "
            "configurable image-space heuristic "
            "and is not a calibrated physical "
            "defect-severity measurement."
        ),
    }


def save_quality_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:

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