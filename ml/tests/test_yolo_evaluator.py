from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from ml.src.evaluation.yolo_evaluator import (
    extract_detection_report,
    load_evaluation_config,
    save_per_class_csv,
)


def test_load_evaluation_config(
    tmp_path: Path,
):
    config_path = (
        tmp_path / "eval.yaml"
    )

    config_path.write_text(
        yaml.safe_dump(
            {
                "evaluation": {
                    "imgsz": 320,
                    "batch": 16,
                    "plots": True,
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_evaluation_config(
        config_path
    )

    assert config["imgsz"] == 320
    assert config["batch"] == 16
    assert config["plots"] is True


def test_extract_detection_report():
    box = SimpleNamespace(
        mp=0.80,
        mr=0.70,
        map50=0.85,
        map=0.55,

        p=np.array(
            [
                0.80,
                0.81,
            ]
        ),

        r=np.array(
            [
                0.70,
                0.71,
            ]
        ),

        ap50=np.array(
            [
                0.85,
                0.86,
            ]
        ),

        ap=np.array(
            [
                0.55,
                0.56,
            ]
        ),

        ap_class_index=np.array(
            [
                0,
                1,
            ]
        ),
    )

    metrics = SimpleNamespace(
        box=box,

        speed={
            "preprocess": 0.5,
            "inference": 2.0,
            "postprocess": 0.7,
        },
    )

    report = extract_detection_report(
        metrics,
        [
            "crazing",
            "inclusion",
        ],
    )

    assert (
        report["overall"]["precision"]
        == 0.80
    )

    assert (
        report["overall"]["map50"]
        == 0.85
    )

    assert (
        report["per_class"][0][
            "class_name"
        ]
        == "crazing"
    )

    assert (
        report["per_class"][1][
            "map50_95"
        ]
        == 0.56
    )


def test_save_per_class_csv(
    tmp_path: Path,
):
    output_path = (
        tmp_path / "per_class.csv"
    )

    report = {
        "per_class": [
            {
                "class_id": 0,
                "class_name": "crazing",
                "precision": 0.8,
                "recall": 0.7,
                "map50": 0.85,
                "map50_95": 0.55,
            }
        ]
    }

    save_per_class_csv(
        report,
        output_path,
    )

    dataframe = pd.read_csv(
        output_path
    )

    assert len(dataframe) == 1

    assert (
        dataframe.iloc[0][
            "class_name"
        ]
        == "crazing"
    )

    assert (
        dataframe.iloc[0][
            "map50"
        ]
        == 0.85
    )