from pathlib import Path

import yaml

from ml.src.training.yolo_trainer import (
    build_training_arguments,
    load_training_config,
)


def test_load_training_config(
    tmp_path: Path,
):
    config_path = (
        tmp_path / "training.yaml"
    )

    config_path.write_text(
        yaml.safe_dump(
            {
                "model": "yolov8n.pt",
                "training": {
                    "epochs": 50,
                    "imgsz": 320,
                    "batch": 16,
                },
                "augmentation": {
                    "mosaic": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_training_config(
        config_path
    )

    assert (
        config["model"]
        == "yolov8n.pt"
    )

    assert (
        config["training"]["epochs"]
        == 50
    )


def test_build_regular_arguments(
    tmp_path: Path,
):
    dataset = (
        tmp_path / "dataset.yaml"
    )

    dataset.write_text(
        "names: {}\n",
        encoding="utf-8",
    )

    config = {
        "model": "yolov8n.pt",

        "training": {
            "epochs": 50,
            "imgsz": 320,
            "batch": 16,
        },

        "augmentation": {
            "mosaic": 0.0,
        },
    }

    arguments = build_training_arguments(
        config=config,
        dataset_yaml=dataset,
        device="cpu",
        smoke=False,
    )

    assert (
        arguments["epochs"]
        == 50
    )

    assert (
        arguments["imgsz"]
        == 320
    )

    assert (
        arguments["device"]
        == "cpu"
    )

    assert (
        arguments["mosaic"]
        == 0.0
    )


def test_smoke_training_overrides(
    tmp_path: Path,
):
    dataset = (
        tmp_path / "dataset.yaml"
    )

    dataset.write_text(
        "names: {}\n",
        encoding="utf-8",
    )

    config = {
        "model": "yolov8n.pt",

        "training": {
            "epochs": 50,
            "imgsz": 320,
            "batch": 16,
        },

        "augmentation": {},
    }

    arguments = build_training_arguments(
        config=config,
        dataset_yaml=dataset,
        device="cpu",
        smoke=True,
    )

    assert (
        arguments["epochs"]
        == 1
    )

    assert (
        arguments["batch"]
        == 4
    )

    assert (
        arguments["fraction"]
        == 0.05
    )

    assert (
        arguments["name"]
        == "yolov8n_smoke"
    )