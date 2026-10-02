from pathlib import Path

import pytest
import yaml

from ml.src.optimization.tensorrt_support import (
    build_trtexec_command,
    load_tensorrt_config,
    recommend_backend,
)


def test_load_tensorrt_config(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "tensorrt.yaml"
    )

    path.write_text(
        yaml.safe_dump(
            {
                "tensorrt": {
                    "input_name":
                        "images",

                    "batch_size":
                        1,

                    "channels":
                        3,

                    "height":
                        416,

                    "width":
                        416,

                    "precision":
                        "fp16",

                    "onnx_model":
                        "model.onnx",

                    "engine_path":
                        "model.engine",

                    "metadata_path":
                        "metadata.json",
                },

                "fallback": {
                    "backend":
                        "onnxruntime",

                    "provider":
                        "CPUExecutionProvider",
                },
            }
        ),
        encoding="utf-8",
    )

    config = (
        load_tensorrt_config(
            path
        )
    )

    assert (
        config["tensorrt"][
            "height"
        ]
        == 416
    )

    assert (
        config["tensorrt"][
            "precision"
        ]
        == "fp16"
    )


def test_cpu_backend_fallback():
    environment = {
        "onnxruntime_tensorrt_provider":
            False,

        "tensorrt_engine_build_possible":
            False,

        "onnxruntime_cuda_provider":
            False,
    }

    backend = (
        recommend_backend(
            environment
        )
    )

    assert (
        backend["backend"]
        == "onnxruntime-cpu"
    )


def test_cuda_fallback():
    environment = {
        "onnxruntime_tensorrt_provider":
            False,

        "tensorrt_engine_build_possible":
            False,

        "onnxruntime_cuda_provider":
            True,
    }

    backend = (
        recommend_backend(
            environment
        )
    )

    assert (
        backend["backend"]
        == "onnxruntime-cuda"
    )


def test_tensorrt_engine_backend():
    environment = {
        "onnxruntime_tensorrt_provider":
            False,

        "tensorrt_engine_build_possible":
            True,

        "onnxruntime_cuda_provider":
            True,
    }

    backend = (
        recommend_backend(
            environment
        )
    )

    assert (
        backend["backend"]
        == "tensorrt-engine"
    )


def test_ort_tensorrt_preferred():
    environment = {
        "onnxruntime_tensorrt_provider":
            True,

        "tensorrt_engine_build_possible":
            True,

        "onnxruntime_cuda_provider":
            True,
    }

    backend = (
        recommend_backend(
            environment
        )
    )

    assert (
        backend["backend"]
        == "onnxruntime-tensorrt"
    )


def test_build_fp16_trtexec_command(
    tmp_path: Path,
):
    onnx_path = (
        tmp_path
        / "model.onnx"
    )

    engine_path = (
        tmp_path
        / "model.engine"
    )

    command = (
        build_trtexec_command(
            trtexec_path=(
                "trtexec"
            ),
            onnx_path=(
                onnx_path
            ),
            engine_path=(
                engine_path
            ),
            precision="fp16",
        )
    )

    assert (
        "--fp16"
        in command
    )

    assert any(
        item.startswith(
            "--onnx="
        )
        for item in command
    )

    assert any(
        item.startswith(
            "--saveEngine="
        )
        for item in command
    )


def test_build_fp32_command_has_no_fp16(
    tmp_path: Path,
):
    command = (
        build_trtexec_command(
            trtexec_path=(
                "trtexec"
            ),
            onnx_path=(
                tmp_path
                / "model.onnx"
            ),
            engine_path=(
                tmp_path
                / "model.engine"
            ),
            precision="fp32",
        )
    )

    assert (
        "--fp16"
        not in command
    )


def test_invalid_precision():
    with pytest.raises(
        ValueError
    ):
        build_trtexec_command(
            trtexec_path=(
                "trtexec"
            ),
            onnx_path=Path(
                "model.onnx"
            ),
            engine_path=Path(
                "model.engine"
            ),
            precision="int4",
        )