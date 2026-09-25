import json
from pathlib import Path

import yaml

from ml.src.training.optimized_training import (
    build_optimized_training_arguments,
    load_optimized_config,
    load_selected_candidate,
)


def test_load_selected_candidate(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "screening.json"
    )

    path.write_text(
        json.dumps(
            {
                "selected_candidate": {
                    "name":
                        "nano_416_resolution",

                    "model":
                        "yolov8n.pt",

                    "imgsz":
                        416,

                    "batch":
                        8,

                    "map50_95":
                        0.50,
                }
            }
        ),
        encoding="utf-8",
    )

    candidate = (
        load_selected_candidate(
            path
        )
    )

    assert (
        candidate["name"]
        == "nano_416_resolution"
    )

    assert (
        candidate["imgsz"]
        == 416
    )

    assert (
        candidate["batch"]
        == 8
    )


def test_load_optimized_config(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "optimized.yaml"
    )

    path.write_text(
        yaml.safe_dump(
            {
                "training": {
                    "epochs":
                        50,

                    "seed":
                        42,
                },

                "augmentation": {
                    "mosaic":
                        0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    config = (
        load_optimized_config(
            path
        )
    )

    assert (
        config["training"]["epochs"]
        == 50
    )

    assert (
        config["training"]["seed"]
        == 42
    )


def test_build_optimized_arguments(
    tmp_path: Path,
):
    dataset = (
        tmp_path
        / "dataset.yaml"
    )

    dataset.write_text(
        "names: {}\n",
        encoding="utf-8",
    )

    config = {
        "training": {
            "epochs": 50,
            "patience": 15,
            "seed": 42,
        },

        "augmentation": {
            "mosaic": 0.0,
        },
    }

    candidate = {
        "name":
            "small_320_capacity",

        "model":
            "yolov8s.pt",

        "imgsz":
            320,

        "batch":
            8,
    }

    arguments = (
        build_optimized_training_arguments(
            config=config,
            selected_candidate=(
                candidate
            ),
            dataset_yaml=dataset,
            device="cpu",
            project_dir=(
                tmp_path
                / "runs"
            ),
            run_name=(
                "optimized"
            ),
        )
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
        arguments["batch"]
        == 8
    )

    assert (
        arguments["seed"]
        == 42
    )

    assert (
        arguments["device"]
        == "cpu"
    )

    assert (
        arguments["mosaic"]
        == 0.0
    )