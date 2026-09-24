from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_evaluation_config(
    config_path: Path,
) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Evaluation config not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Evaluation configuration must be a YAML mapping."
        )

    evaluation = config.get("evaluation")

    if not isinstance(evaluation, dict):
        raise ValueError(
            "Configuration must contain an 'evaluation' section."
        )

    return evaluation


def to_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_float_list(value: Any) -> list[float]:
    if value is None:
        return []

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()

    array = np.asarray(
        value,
        dtype=float,
    ).reshape(-1)

    return [float(item) for item in array]


def to_int_list(value: Any) -> list[int]:
    if value is None:
        return []

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()

    array = np.asarray(value).reshape(-1)

    return [int(item) for item in array]


def get_metric_value(
    values: list[float],
    metric_index: int | None,
) -> float | None:
    if metric_index is None:
        return None

    if not 0 <= metric_index < len(values):
        return None

    return float(values[metric_index])


def extract_detection_report(
    metrics: Any,
    class_names: list[str],
) -> dict[str, Any]:
    if not hasattr(metrics, "box"):
        raise ValueError(
            "Evaluation metrics do not contain bounding-box results."
        )

    box = metrics.box

    precision_values = to_float_list(
        getattr(box, "p", None)
    )

    recall_values = to_float_list(
        getattr(box, "r", None)
    )

    map50_values = to_float_list(
        getattr(box, "ap50", None)
    )

    map50_95_values = to_float_list(
        getattr(box, "ap", None)
    )

    class_indices = to_int_list(
        getattr(box, "ap_class_index", None)
    )

    if not class_indices:
        count = max(
            len(precision_values),
            len(recall_values),
            len(map50_values),
            len(map50_95_values),
        )

        class_indices = list(range(count))

    index_lookup = {
        class_id: metric_index
        for metric_index, class_id
        in enumerate(class_indices)
    }

    per_class = []

    for class_id, class_name in enumerate(class_names):
        metric_index = index_lookup.get(class_id)

        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": get_metric_value(
                    precision_values,
                    metric_index,
                ),
                "recall": get_metric_value(
                    recall_values,
                    metric_index,
                ),
                "map50": get_metric_value(
                    map50_values,
                    metric_index,
                ),
                "map50_95": get_metric_value(
                    map50_95_values,
                    metric_index,
                ),
            }
        )

    speed = getattr(metrics, "speed", {})

    normalized_speed = {}

    if isinstance(speed, dict):
        normalized_speed = {
            str(key): to_float(value)
            for key, value in speed.items()
        }

    return {
        "overall": {
            "precision": to_float(
                getattr(box, "mp", None)
            ),
            "recall": to_float(
                getattr(box, "mr", None)
            ),
            "map50": to_float(
                getattr(box, "map50", None)
            ),
            "map50_95": to_float(
                getattr(box, "map", None)
            ),
        },
        "per_class": per_class,
        "speed_ms_per_image": normalized_speed,
    }


def save_evaluation_json(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def save_per_class_csv(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        report["per_class"]
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )