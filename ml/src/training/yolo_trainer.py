from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def load_training_config(
    config_path: Path,
) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Training config not found: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "Training configuration must be a mapping."
        )

    if "model" not in config:
        raise ValueError(
            "Training config is missing 'model'."
        )

    return config


def resolve_device(
    requested_device: str | None = None,
) -> str:
    """
    Resolve training device.

    requested_device examples:
        "cpu"
        "0"
    """

    if requested_device:
        return requested_device

    if torch.cuda.is_available():
        return "0"

    return "cpu"


def build_training_arguments(
    config: dict[str, Any],
    dataset_yaml: Path,
    device: str,
    smoke: bool = False,
) -> dict[str, Any]:
    training = dict(
        config.get(
            "training",
            {},
        )
    )

    augmentation = dict(
        config.get(
            "augmentation",
            {},
        )
    )

    arguments = {
        **training,
        **augmentation,
        "data": str(
            dataset_yaml.resolve()
        ),
        "device": device,
        "project": str(
            Path(
                "outputs/training"
            ).resolve()
        ),
        "name": "yolov8n_baseline",
        "exist_ok": True,
    }

    if smoke:
        arguments.update(
            {
                "epochs": 1,
                "batch": 4,
                "workers": 0,
                "fraction": 0.05,
                "patience": 1,
                "name": "yolov8n_smoke",
            }
        )

    return arguments