from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SELECTED_FIELDS = {
    "name",
    "model",
    "imgsz",
    "batch",
}


def load_selected_candidate(
    screening_results_path: Path,
) -> dict[str, Any]:
    """
    Load the candidate selected during Day 12 screening.
    """

    if not screening_results_path.exists():
        raise FileNotFoundError(
            "Day 12 screening results not found: "
            f"{screening_results_path}"
        )

    with screening_results_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            "Screening results must contain a JSON object."
        )

    candidate = payload.get(
        "selected_candidate"
    )

    if not isinstance(candidate, dict):
        raise ValueError(
            "Screening results do not contain "
            "'selected_candidate'."
        )

    missing = (
        REQUIRED_SELECTED_FIELDS
        - set(candidate)
    )

    if missing:
        raise ValueError(
            "Selected candidate is missing fields: "
            f"{sorted(missing)}"
        )

    return candidate


def load_optimized_config(
    config_path: Path,
) -> dict[str, Any]:
    """
    Load full optimized-training settings.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Optimized training config not found: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Optimized training config "
            "must be a YAML mapping."
        )

    training = config.get(
        "training"
    )

    if not isinstance(
        training,
        dict,
    ):
        raise ValueError(
            "Configuration must contain "
            "a 'training' section."
        )

    augmentation = config.get(
        "augmentation",
        {},
    )

    if not isinstance(
        augmentation,
        dict,
    ):
        raise ValueError(
            "'augmentation' must be a mapping."
        )

    return config


def build_optimized_training_arguments(
    config: dict[str, Any],
    selected_candidate: dict[str, Any],
    dataset_yaml: Path,
    device: str,
    project_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    """
    Build the full Ultralytics training arguments.

    Architecture, image size and batch size come
    directly from Day 12 screening.
    """

    training = dict(
        config["training"]
    )

    augmentation = dict(
        config.get(
            "augmentation",
            {},
        )
    )

    return {
        **training,
        **augmentation,

        "data": str(
            dataset_yaml.resolve()
        ),

        "imgsz": int(
            selected_candidate[
                "imgsz"
            ]
        ),

        "batch": int(
            selected_candidate[
                "batch"
            ]
        ),

        "device":
            device,

        "project": str(
            project_dir.resolve()
        ),

        "name":
            run_name,

        "exist_ok":
            True,
    }


def save_training_manifest(
    selected_candidate: dict[str, Any],
    training_arguments: dict[str, Any],
    elapsed_seconds: float,
    best_weights: Path,
    last_weights: Path,
    output_path: Path,
) -> None:
    """
    Save reproducibility metadata for the Day 13 run.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "selected_from":
            "Day 12 validation screening",

        "selected_candidate":
            selected_candidate,

        "training": {
            "epochs":
                training_arguments.get(
                    "epochs"
                ),

            "imgsz":
                training_arguments.get(
                    "imgsz"
                ),

            "batch":
                training_arguments.get(
                    "batch"
                ),

            "seed":
                training_arguments.get(
                    "seed"
                ),

            "device":
                training_arguments.get(
                    "device"
                ),
        },

        "elapsed_seconds":
            float(elapsed_seconds),

        "best_weights":
            str(
                best_weights.resolve()
            ),

        "best_weights_exists":
            best_weights.exists(),

        "last_weights":
            str(
                last_weights.resolve()
            ),

        "last_weights_exists":
            last_weights.exists(),
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )