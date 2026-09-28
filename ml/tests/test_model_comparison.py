from pathlib import Path

import pytest
import yaml

from ml.src.evaluation.model_comparison import (
    calculate_deltas,
    get_model_size_mb,
    load_comparison_config,
    summarize_latency,
)


def test_load_comparison_config(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "comparison.yaml"
    )

    path.write_text(
        yaml.safe_dump(
            {
                "comparison": {
                    "baseline_imgsz": 320,

                    "validation": {
                        "batch": 8,
                    },

                    "benchmark": {
                        "confidence": 0.25,
                        "sample_count": 30,
                        "warmup_count": 3,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    config = (
        load_comparison_config(
            path
        )
    )

    assert (
        config["baseline_imgsz"]
        == 320
    )

    assert (
        config["benchmark"][
            "sample_count"
        ]
        == 30
    )


def test_model_size(
    tmp_path: Path,
):
    path = (
        tmp_path / "model.pt"
    )

    path.write_bytes(
        b"x" * 1024 * 1024
    )

    size = get_model_size_mb(
        path
    )

    assert size == pytest.approx(
        1.0,
        abs=0.01,
    )


def test_latency_summary():
    result = summarize_latency(
        [
            10.0,
            20.0,
            30.0,
        ]
    )

    assert (
        result["mean_latency_ms"]
        == pytest.approx(20.0)
    )

    assert (
        result["median_latency_ms"]
        == pytest.approx(20.0)
    )

    assert (
        result["approx_fps"]
        == pytest.approx(50.0)
    )


def test_calculate_deltas():
    baseline = {
        "precision": 0.60,
        "recall": 0.65,
        "map50": 0.70,
        "map50_95": 0.40,
        "model_size_mb": 6.0,
        "mean_latency_ms": 20.0,
        "approx_fps": 50.0,
    }

    optimized = {
        "precision": 0.70,
        "recall": 0.72,
        "map50": 0.78,
        "map50_95": 0.48,
        "model_size_mb": 6.0,
        "mean_latency_ms": 30.0,
        "approx_fps": 33.0,
    }

    result = calculate_deltas(
        baseline,
        optimized,
    )

    assert (
        result["precision"]
        == pytest.approx(0.10)
    )

    assert (
        result["map50_95"]
        == pytest.approx(0.08)
    )

    assert (
        result["mean_latency_ms"]
        == pytest.approx(10.0)
    )