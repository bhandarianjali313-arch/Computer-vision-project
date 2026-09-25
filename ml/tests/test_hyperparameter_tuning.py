from pathlib import Path

import yaml

from ml.src.training.hyperparameter_tuning import (
    build_screening_arguments,
    load_tuning_config,
    select_screening_candidate,
)


def test_load_tuning_config(
    tmp_path: Path,
):
    config_path = (
        tmp_path / "tuning.yaml"
    )

    config = {
        "screening": {
            "epochs": 10,
            "fraction": 0.25,
        },

        "augmentation": {
            "mosaic": 0.0,
        },

        "candidates": [
            {
                "name":
                    "nano_320",

                "model":
                    "yolov8n.pt",

                "imgsz":
                    320,

                "batch":
                    16,
            }
        ],
    }

    config_path.write_text(
        yaml.safe_dump(
            config
        ),
        encoding="utf-8",
    )

    loaded = load_tuning_config(
        config_path
    )

    assert (
        len(
            loaded[
                "candidates"
            ]
        )
        == 1
    )

    assert (
        loaded[
            "candidates"
        ][0]["imgsz"]
        == 320
    )


def test_build_screening_arguments(
    tmp_path: Path,
):
    dataset_yaml = (
        tmp_path / "dataset.yaml"
    )

    dataset_yaml.write_text(
        "names: {}\n",
        encoding="utf-8",
    )

    arguments = (
        build_screening_arguments(
            screening={
                "epochs": 10,
                "fraction": 0.25,
                "seed": 42,
            },
            augmentation={
                "mosaic": 0.0,
            },
            candidate={
                "name":
                    "nano_416",

                "model":
                    "yolov8n.pt",

                "imgsz":
                    416,

                "batch":
                    8,
            },
            dataset_yaml=(
                dataset_yaml
            ),
            device="cpu",
            project_dir=(
                tmp_path
                / "runs"
            ),
        )
    )

    assert (
        arguments["epochs"]
        == 10
    )

    assert (
        arguments["imgsz"]
        == 416
    )

    assert (
        arguments["batch"]
        == 8
    )

    assert (
        arguments["device"]
        == "cpu"
    )

    assert (
        arguments["mosaic"]
        == 0.0
    )


def test_smoke_overrides():
    arguments = (
        build_screening_arguments(
            screening={
                "epochs": 10,
                "fraction": 0.25,
            },
            augmentation={},
            candidate={
                "name":
                    "nano",

                "model":
                    "yolov8n.pt",

                "imgsz":
                    320,

                "batch":
                    16,
            },
            dataset_yaml=Path(
                "dataset.yaml"
            ),
            device="cpu",
            project_dir=Path(
                "runs"
            ),
            smoke=True,
        )
    )

    assert (
        arguments["epochs"]
        == 1
    )

    assert (
        arguments["fraction"]
        == 0.05
    )


def test_select_candidate_by_map():
    results = [
        {
            "name": "a",
            "map50_95": 0.41,
            "recall": 0.70,
        },
        {
            "name": "b",
            "map50_95": 0.48,
            "recall": 0.66,
        },
        {
            "name": "c",
            "map50_95": 0.44,
            "recall": 0.75,
        },
    ]

    selected = (
        select_screening_candidate(
            results
        )
    )

    assert (
        selected["name"]
        == "b"
    )


def test_recall_breaks_map_tie():
    results = [
        {
            "name": "a",
            "map50_95": 0.48,
            "recall": 0.65,
        },
        {
            "name": "b",
            "map50_95": 0.48,
            "recall": 0.72,
        },
    ]

    selected = (
        select_screening_candidate(
            results
        )
    )

    assert (
        selected["name"]
        == "b"
    )