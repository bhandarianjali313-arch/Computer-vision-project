from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_CANDIDATE_FIELDS = {
    "name",
    "model",
    "imgsz",
    "batch",
}


def load_tuning_config(
    config_path: Path,
) -> dict[str, Any]:
    """
    Load and validate the hyperparameter-screening configuration.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Tuning configuration not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Tuning configuration must be a YAML mapping."
        )

    screening = config.get("screening")
    candidates = config.get("candidates")
    augmentation = config.get(
        "augmentation",
        {},
    )

    if not isinstance(
        screening,
        dict,
    ):
        raise ValueError(
            "Configuration must contain a screening section."
        )

    if not isinstance(
        candidates,
        list,
    ) or not candidates:
        raise ValueError(
            "Configuration must contain at least one candidate."
        )

    candidate_names = []

    for candidate in candidates:
        if not isinstance(
            candidate,
            dict,
        ):
            raise ValueError(
                "Each tuning candidate must be a mapping."
            )

        missing = (
            REQUIRED_CANDIDATE_FIELDS
            - set(candidate)
        )

        if missing:
            raise ValueError(
                "Candidate is missing fields: "
                f"{sorted(missing)}"
            )

        candidate_names.append(
            candidate["name"]
        )

    if (
        len(candidate_names)
        != len(set(candidate_names))
    ):
        raise ValueError(
            "Tuning candidate names must be unique."
        )

    config["augmentation"] = augmentation

    return config


def build_screening_arguments(
    screening: dict[str, Any],
    augmentation: dict[str, Any],
    candidate: dict[str, Any],
    dataset_yaml: Path,
    device: str,
    project_dir: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    """
    Build Ultralytics training arguments for one candidate.
    """

    arguments = {
        **screening,
        **augmentation,

        "data": str(
            dataset_yaml.resolve()
        ),

        "imgsz": int(
            candidate["imgsz"]
        ),

        "batch": int(
            candidate["batch"]
        ),

        "device":
            device,

        "project": str(
            project_dir.resolve()
        ),

        "name":
            candidate["name"],

        "exist_ok":
            True,
    }

    if smoke:
        arguments.update(
            {
                "epochs": 1,
                "fraction": 0.05,
                "patience": 1,
                "plots": False,
            }
        )

    return arguments


def select_screening_candidate(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Select the candidate with highest validation mAP50-95.

    Recall is used only as a secondary tie-breaker.
    """

    valid_results = [
        result
        for result in results
        if result.get("map50_95") is not None
    ]

    if not valid_results:
        raise ValueError(
            "No candidate contains a valid mAP50-95 value."
        )

    return max(
        valid_results,
        key=lambda result: (
            float(
                result["map50_95"]
            ),
            float(
                result.get(
                    "recall",
                    0.0,
                )
                or 0.0
            ),
        ),
    )


def save_tuning_results(
    results: list[dict[str, Any]],
    selected: dict[str, Any],
    json_path: Path,
    csv_path: Path,
) -> None:
    """
    Save screening results in JSON and CSV formats.
    """

    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "selection_metric":
            "validation mAP50-95",

        "secondary_metric":
            "validation recall",

        "selected_candidate":
            selected,

        "candidates":
            results,
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not results:
        raise ValueError(
            "Cannot save empty tuning results."
        )

    fieldnames = [
        "name",
        "model",
        "imgsz",
        "batch",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "training_seconds",
        "best_weights",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    field:
                        result.get(field)
                    for field in fieldnames
                }
            )