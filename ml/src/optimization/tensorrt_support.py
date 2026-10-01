from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import onnxruntime as ort
import yaml

from ml.src.optimization.onnx_export import (
    calculate_sha256,
)


def load_tensorrt_config(
    config_path: Path,
) -> dict[str, Any]:
    """
    Load TensorRT deployment configuration.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"TensorRT config not found: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "TensorRT configuration must "
            "be a YAML mapping."
        )

    tensorrt = config.get(
        "tensorrt"
    )

    fallback = config.get(
        "fallback"
    )

    if not isinstance(
        tensorrt,
        dict,
    ):
        raise ValueError(
            "Configuration must contain "
            "a 'tensorrt' section."
        )

    if not isinstance(
        fallback,
        dict,
    ):
        raise ValueError(
            "Configuration must contain "
            "a 'fallback' section."
        )

    required_tensorrt = {
        "input_name",
        "batch_size",
        "channels",
        "height",
        "width",
        "precision",
        "onnx_model",
        "engine_path",
        "metadata_path",
    }

    required_fallback = {
        "backend",
        "provider",
    }

    missing_tensorrt = (
        required_tensorrt
        - set(tensorrt)
    )

    missing_fallback = (
        required_fallback
        - set(fallback)
    )

    if missing_tensorrt:
        raise ValueError(
            "Missing TensorRT settings: "
            f"{sorted(missing_tensorrt)}"
        )

    if missing_fallback:
        raise ValueError(
            "Missing fallback settings: "
            f"{sorted(missing_fallback)}"
        )

    precision = str(
        tensorrt["precision"]
    ).lower()

    if precision not in {
        "fp32",
        "fp16",
    }:
        raise ValueError(
            "TensorRT precision must be "
            "'fp32' or 'fp16'."
        )

    return config


def command_exists(
    command: str,
) -> bool:
    """
    Check whether an executable is available
    on the current PATH.
    """

    return (
        shutil.which(
            command
        )
        is not None
    )


def python_module_exists(
    module_name: str,
) -> bool:
    """
    Check whether a Python module is installed.
    """

    return (
        importlib.util.find_spec(
            module_name
        )
        is not None
    )


def query_nvidia_gpu() -> dict[str, Any]:
    """
    Query NVIDIA GPU information.

    Returns a safe result when nvidia-smi is
    unavailable.
    """

    if not command_exists(
        "nvidia-smi"
    ):
        return {
            "available":
                False,

            "gpus":
                [],
        }

    command = [
        "nvidia-smi",
        "--query-gpu="
        "name,driver_version,memory.total",
        "--format=csv,noheader",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )

    except (
        OSError,
        subprocess.TimeoutExpired,
    ):
        return {
            "available":
                False,

            "gpus":
                [],
        }

    if result.returncode != 0:
        return {
            "available":
                False,

            "gpus":
                [],
        }

    lines = [
        line.strip()
        for line
        in result.stdout.splitlines()
        if line.strip()
    ]

    return {
        "available":
            bool(lines),

        "gpus":
            lines,
    }


def detect_deployment_environment(
) -> dict[str, Any]:
    """
    Inspect GPU, TensorRT, trtexec and
    ONNX Runtime capabilities.
    """

    gpu = query_nvidia_gpu()

    trtexec_path = shutil.which(
        "trtexec"
    )

    tensorrt_python = (
        python_module_exists(
            "tensorrt"
        )
    )

    providers = (
        ort.get_available_providers()
    )

    trt_ort_provider = (
        "TensorrtExecutionProvider"
        in providers
    )

    cuda_ort_provider = (
        "CUDAExecutionProvider"
        in providers
    )

    build_possible = bool(
        gpu["available"]
        and trtexec_path
    )

    return {
        "nvidia_gpu_available":
            gpu["available"],

        "gpus":
            gpu["gpus"],

        "trtexec_available":
            trtexec_path
            is not None,

        "trtexec_path":
            trtexec_path,

        "tensorrt_python_available":
            tensorrt_python,

        "onnxruntime_providers":
            providers,

        "onnxruntime_tensorrt_provider":
            trt_ort_provider,

        "onnxruntime_cuda_provider":
            cuda_ort_provider,

        "tensorrt_engine_build_possible":
            build_possible,
    }


def recommend_backend(
    environment: dict[str, Any],
) -> dict[str, str]:
    """
    Choose the safest available deployment backend.
    """

    if environment.get(
        "onnxruntime_tensorrt_provider"
    ):
        return {
            "backend":
                "onnxruntime-tensorrt",

            "reason":
                (
                    "ONNX Runtime TensorRT "
                    "Execution Provider is available."
                ),
        }

    if environment.get(
        "tensorrt_engine_build_possible"
    ):
        return {
            "backend":
                "tensorrt-engine",

            "reason":
                (
                    "NVIDIA GPU and trtexec "
                    "are available."
                ),
        }

    if environment.get(
        "onnxruntime_cuda_provider"
    ):
        return {
            "backend":
                "onnxruntime-cuda",

            "reason":
                (
                    "CUDA Execution Provider "
                    "is available."
                ),
        }

    return {
        "backend":
            "onnxruntime-cpu",

        "reason":
            (
                "TensorRT/CUDA deployment "
                "is unavailable. Using ONNX "
                "Runtime CPU fallback."
            ),
    }


def build_trtexec_command(
    trtexec_path: str,
    onnx_path: Path,
    engine_path: Path,
    precision: str,
) -> list[str]:
    """
    Build a minimal, portable trtexec command.

    The ONNX model already has a fixed
    1x3x416x416 input shape.
    """

    command = [
        trtexec_path,
        f"--onnx={onnx_path.resolve()}",
        f"--saveEngine={engine_path.resolve()}",
    ]

    precision = precision.lower()

    if precision == "fp16":
        command.append(
            "--fp16"
        )

    elif precision != "fp32":
        raise ValueError(
            f"Unsupported precision: "
            f"{precision}"
        )

    return command


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    environment: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a TensorRT engine using trtexec.

    This function must only be used when the
    environment has NVIDIA/TensorRT support.
    """

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: "
            f"{onnx_path}"
        )

    if not environment.get(
        "tensorrt_engine_build_possible"
    ):
        raise RuntimeError(
            "TensorRT engine build is not "
            "available in this environment."
        )

    trtexec_path = environment.get(
        "trtexec_path"
    )

    if not trtexec_path:
        raise RuntimeError(
            "trtexec path could not be resolved."
        )

    engine_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = build_trtexec_command(
        trtexec_path=trtexec_path,
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision=precision,
    )

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    success = (
        result.returncode == 0
        and engine_path.exists()
    )

    return {
        "success":
            success,

        "return_code":
            int(
                result.returncode
            ),

        "command":
            command,

        "stdout":
            result.stdout,

        "stderr":
            result.stderr,

        "engine_exists":
            engine_path.exists(),
    }


def create_deployment_metadata(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    environment: dict[str, Any],
    backend: dict[str, str],
    build_result: (
        dict[str, Any]
        | None
    ) = None,
) -> dict[str, Any]:
    """
    Create reproducibility metadata for
    TensorRT/deployment preparation.
    """

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: "
            f"{onnx_path}"
        )

    metadata = {
        "source_onnx":
            str(
                onnx_path.resolve()
            ),

        "source_onnx_sha256":
            calculate_sha256(
                onnx_path
            ),

        "requested_precision":
            precision,

        "environment":
            environment,

        "recommended_backend":
            backend,

        "engine": {
            "path":
                str(
                    engine_path.resolve()
                ),

            "exists":
                engine_path.exists(),

            "sha256":
                (
                    calculate_sha256(
                        engine_path
                    )
                    if engine_path.exists()
                    else None
                ),
        },

        "build_result":
            build_result,
    }

    return metadata


def save_metadata(
    metadata: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save deployment preparation metadata.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )