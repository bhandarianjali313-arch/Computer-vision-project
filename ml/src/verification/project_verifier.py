from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

from app.config import (
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    IOU_THRESHOLD,
    MODEL_PATH,
    QUALITY_POLICY_PATH,
)

from ml.src.data.voc_to_yolo import (
    CLASS_NAMES,
)


EXPECTED_CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


REQUIRED_SOURCE_PATHS = [
    "app/config.py",
    "app/detector.py",
    "app/main.py",

    "ml/configs/quality_policy.yaml",
    "ml/configs/onnx_export.yaml",
    "ml/configs/onnx_runtime.yaml",
    "ml/configs/tensorrt.yaml",

    "ml/scripts/predict_image.py",
    "ml/scripts/run_realtime_inference.py",
    "ml/scripts/export_onnx.py",
    "ml/scripts/benchmark_onnx_runtime.py",
    "ml/scripts/prepare_tensorrt.py",
    "ml/scripts/evaluate_quality_image.py",

    "requirements.txt",
    "requirements-ml.txt",
]


ARTIFACT_PATHS = {
    "raw_dataset":
        "data/raw/NEU-DET",

    "processed_validation_images":
        (
            "data/processed/"
            "neu_yolo/images/val"
        ),

    "optimized_pytorch_model":
        (
            "models/optimized/"
            "yolo_neu_optimized_best.pt"
        ),

    "onnx_model":
        (
            "models/onnx/"
            "yolo_neu_optimized.onnx"
        ),
}


def verify_required_sources(
    project_root: Path,
) -> dict[str, Any]:
    """
    Verify that the important tracked source
    files required by the final pipeline exist.
    """

    checks = {}

    for relative_path in (
        REQUIRED_SOURCE_PATHS
    ):
        full_path = (
            project_root
            / relative_path
        )

        checks[
            relative_path
        ] = full_path.exists()

    return {
        "passed":
            all(
                checks.values()
            ),

        "checks":
            checks,
    }


def verify_class_mapping(
) -> dict[str, Any]:
    """
    Verify that the project still uses the
    fixed six-class NEU-DET mapping.
    """

    actual = list(
        CLASS_NAMES
    )

    return {
        "passed":
            actual
            == EXPECTED_CLASS_NAMES,

        "expected":
            EXPECTED_CLASS_NAMES,

        "actual":
            actual,
    }


def verify_quality_policy(
) -> dict[str, Any]:
    """
    Validate the Day 20 quality-policy file.
    """

    if not (
        QUALITY_POLICY_PATH
        .exists()
    ):
        return {
            "passed":
                False,

            "path":
                str(
                    QUALITY_POLICY_PATH
                ),

            "error":
                "Quality policy is missing.",
        }

    with (
        QUALITY_POLICY_PATH
        .open(
            "r",
            encoding="utf-8",
        )
    ) as file:
        policy = yaml.safe_load(
            file
        )

    required_sections = {
        "postprocessing",
        "triage",
        "decision",
    }

    if not isinstance(
        policy,
        dict,
    ):
        return {
            "passed":
                False,

            "path":
                str(
                    QUALITY_POLICY_PATH
                ),

            "error":
                (
                    "Quality policy must "
                    "be a YAML mapping."
                ),
        }

    missing = (
        required_sections
        - set(
            policy
        )
    )

    return {
        "passed":
            not missing,

        "path":
            str(
                QUALITY_POLICY_PATH
            ),

        "missing_sections":
            sorted(
                missing
            ),
    }


def verify_runtime_configuration(
) -> dict[str, Any]:
    """
    Verify that backend inference still matches
    the optimized ML configuration.
    """

    normalized_model_path = (
        str(
            MODEL_PATH
        )
        .replace(
            "\\",
            "/",
        )
    )

    checks = {
        "image_size_is_416":
            IMAGE_SIZE
            == 416,

        "confidence_is_0_25":
            abs(
                CONFIDENCE_THRESHOLD
                - 0.25
            )
            < 1e-9,

        "iou_is_0_45":
            abs(
                IOU_THRESHOLD
                - 0.45
            )
            < 1e-9,

        "optimized_model_path":
            normalized_model_path.endswith(
                "models/optimized/"
                "yolo_neu_optimized_best.pt"
            ),
    }

    return {
        "passed":
            all(
                checks.values()
            ),

        "checks":
            checks,

        "values": {
            "image_size":
                IMAGE_SIZE,

            "confidence_threshold":
                CONFIDENCE_THRESHOLD,

            "iou_threshold":
                IOU_THRESHOLD,

            "model_path":
                str(
                    MODEL_PATH
                ),
        },
    }


def verify_artifacts(
    project_root: Path,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Check local data/model artifacts.

    These files are intentionally ignored by Git,
    so they are optional for source-code CI.

    In strict mode every artifact must exist.
    """

    checks = {}

    for (
        name,
        relative_path,
    ) in ARTIFACT_PATHS.items():

        full_path = (
            project_root
            / relative_path
        )

        checks[
            name
        ] = {
            "path":
                str(
                    full_path
                ),

            "exists":
                full_path.exists(),
        }

    if strict:
        passed = all(
            item[
                "exists"
            ]
            for item
            in checks.values()
        )

    else:
        passed = True

    return {
        "passed":
            passed,

        "strict":
            strict,

        "checks":
            checks,
    }


def build_verification_report(
    project_root: Path,
    strict_artifacts: bool = False,
) -> dict[str, Any]:
    """
    Build the complete final verification report.
    """

    project_root = (
        project_root
        .resolve()
    )

    source_result = (
        verify_required_sources(
            project_root
        )
    )

    class_result = (
        verify_class_mapping()
    )

    runtime_result = (
        verify_runtime_configuration()
    )

    policy_result = (
        verify_quality_policy()
    )

    artifact_result = (
        verify_artifacts(
            project_root,
            strict=(
                strict_artifacts
            ),
        )
    )

    sections = {
        "required_sources":
            source_result,

        "class_mapping":
            class_result,

        "runtime_configuration":
            runtime_result,

        "quality_policy":
            policy_result,

        "artifacts":
            artifact_result,
    }

    overall_passed = all(
        section[
            "passed"
        ]
        for section
        in sections.values()
    )

    return {
        "passed":
            overall_passed,

        "project_root":
            str(
                project_root
            ),

        "environment": {
            "python_version":
                sys.version.split()[0],

            "platform":
                platform.platform(),

            "python_executable":
                sys.executable,
        },

        "sections":
            sections,

        "notes": [
            (
                "Dataset, trained-model and "
                "generated-output directories "
                "are intentionally ignored "
                "by Git."
            ),
            (
                "TensorRT engine generation "
                "requires compatible NVIDIA "
                "hardware and was not executed "
                "on the CPU-only development "
                "machine."
            ),
            (
                "PASS/REVIEW/REJECT is an "
                "operational image-space "
                "heuristic rather than a "
                "calibrated physical severity "
                "measurement."
            ),
        ],
    }


def save_verification_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save final verification report to JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )