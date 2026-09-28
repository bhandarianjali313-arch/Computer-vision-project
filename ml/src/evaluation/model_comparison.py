from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def load_comparison_config(
    config_path: Path,
) -> dict[str, Any]:
    """
    Load Day 14 model-comparison configuration.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Comparison config not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Comparison config must be a YAML mapping."
        )

    comparison = config.get(
        "comparison"
    )

    if not isinstance(
        comparison,
        dict,
    ):
        raise ValueError(
            "Configuration must contain "
            "a 'comparison' section."
        )

    return comparison


def get_model_size_mb(
    weights_path: Path,
) -> float:
    """
    Return model-file size in megabytes.
    """

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}"
        )

    return (
        weights_path.stat().st_size
        / (1024 * 1024)
    )


def collect_benchmark_images(
    images_dir: Path,
    sample_count: int,
) -> list:
    """
    Load benchmark images into memory so disk I/O is
    excluded from the timed model.predict calls.
    """

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Benchmark image directory not found: "
            f"{images_dir}"
        )

    image_paths = sorted(
        [
            path
            for path in images_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                }
            )
        ]
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No benchmark images found in "
            f"{images_dir}"
        )

    image_paths = image_paths[
        :min(
            sample_count,
            len(image_paths),
        )
    ]

    images = []

    for image_path in image_paths:
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise ValueError(
                f"Could not read image: "
                f"{image_path}"
            )

        images.append(image)

    return images


def summarize_latency(
    latency_ms: list[float],
) -> dict[str, float]:
    """
    Summarize per-image end-to-end inference latency.
    """

    if not latency_ms:
        raise ValueError(
            "Latency list cannot be empty."
        )

    mean_ms = statistics.mean(
        latency_ms
    )

    median_ms = statistics.median(
        latency_ms
    )

    p95_ms = float(
        np.percentile(
            latency_ms,
            95,
        )
    )

    fps = (
        1000.0 / mean_ms
        if mean_ms > 0
        else 0.0
    )

    return {
        "mean_latency_ms":
            float(mean_ms),

        "median_latency_ms":
            float(median_ms),

        "p95_latency_ms":
            float(p95_ms),

        "approx_fps":
            float(fps),
    }


def benchmark_model(
    model,
    images: list,
    imgsz: int,
    device: str,
    confidence: float,
    warmup_count: int = 3,
) -> dict[str, float]:
    """
    Benchmark one model.

    Images are already in memory, so reported latency
    excludes disk loading but includes the model.predict
    preprocessing, inference and postprocessing path.
    """

    if not images:
        raise ValueError(
            "At least one benchmark image is required."
        )

    warmup_count = min(
        warmup_count,
        len(images),
    )

    for index in range(
        warmup_count
    ):
        model.predict(
            source=images[index],
            imgsz=imgsz,
            conf=confidence,
            device=device,
            verbose=False,
        )

    latency_ms = []

    for image in images:
        start = time.perf_counter()

        model.predict(
            source=image,
            imgsz=imgsz,
            conf=confidence,
            device=device,
            verbose=False,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        latency_ms.append(
            elapsed * 1000.0
        )

    summary = summarize_latency(
        latency_ms
    )

    summary[
        "benchmark_image_count"
    ] = len(images)

    return summary


def calculate_deltas(
    baseline: dict[str, Any],
    optimized: dict[str, Any],
) -> dict[str, float | None]:
    """
    Compute optimized - baseline differences.
    """

    fields = [
        "precision",
        "recall",
        "map50",
        "map50_95",
        "model_size_mb",
        "mean_latency_ms",
        "approx_fps",
    ]

    deltas = {}

    for field in fields:
        baseline_value = baseline.get(
            field
        )

        optimized_value = optimized.get(
            field
        )

        if (
            baseline_value is None
            or optimized_value is None
        ):
            deltas[field] = None
            continue

        deltas[field] = (
            float(optimized_value)
            - float(baseline_value)
        )

    return deltas


def save_comparison(
    baseline: dict[str, Any],
    optimized: dict[str, Any],
    deltas: dict[str, Any],
    metadata: dict[str, Any],
    json_path: Path,
    csv_path: Path,
) -> None:
    """
    Save Day 14 comparison as JSON and CSV.
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
        "metadata":
            metadata,

        "baseline":
            baseline,

        "optimized":
            optimized,

        "optimized_minus_baseline":
            deltas,
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    fieldnames = [
        "configuration",
        "model",
        "imgsz",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "model_size_mb",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "approx_fps",
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

        for row in (
            baseline,
            optimized,
        ):
            writer.writerow(
                {
                    field:
                        row.get(field)
                    for field in fieldnames
                }
            )