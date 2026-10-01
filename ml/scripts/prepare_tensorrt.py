import argparse
from pathlib import Path

from ml.src.optimization.tensorrt_support import (
    build_tensorrt_engine,
    create_deployment_metadata,
    detect_deployment_environment,
    load_tensorrt_config,
    recommend_backend,
    save_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect TensorRT deployment "
            "capabilities and optionally build "
            "a TensorRT engine."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "ml/configs/tensorrt.yaml"
        ),
    )

    parser.add_argument(
        "--build",
        action="store_true",
        help=(
            "Attempt TensorRT engine generation "
            "using trtexec."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_tensorrt_config(
        args.config
    )

    trt = config[
        "tensorrt"
    ]

    onnx_path = Path(
        trt["onnx_model"]
    )

    engine_path = Path(
        trt["engine_path"]
    )

    metadata_path = Path(
        trt["metadata_path"]
    )

    if not onnx_path.exists():
        raise FileNotFoundError(
            "The Day 17 ONNX model is missing:\n"
            f"{onnx_path}"
        )

    environment = (
        detect_deployment_environment()
    )

    backend = recommend_backend(
        environment
    )

    print("=" * 80)
    print("DAY 19 - TENSORRT DEPLOYMENT PREPARATION")
    print("=" * 80)

    print(
        f"ONNX model       : "
        f"{onnx_path}"
    )

    print(
        f"Requested engine : "
        f"{engine_path}"
    )

    print(
        f"Precision        : "
        f"{trt['precision']}"
    )

    print(
        f"Input shape      : "
        f"{trt['batch_size']} x "
        f"{trt['channels']} x "
        f"{trt['height']} x "
        f"{trt['width']}"
    )

    print("\nEnvironment:")

    print(
        f"  NVIDIA GPU        : "
        f"{environment['nvidia_gpu_available']}"
    )

    if environment[
        "gpus"
    ]:
        for gpu in environment[
            "gpus"
        ]:
            print(
                f"    {gpu}"
            )

    print(
        f"  trtexec           : "
        f"{environment['trtexec_available']}"
    )

    print(
        f"  TensorRT Python   : "
        f"{environment['tensorrt_python_available']}"
    )

    print(
        f"  ORT TensorRT EP   : "
        f"{environment['onnxruntime_tensorrt_provider']}"
    )

    print(
        f"  ORT CUDA EP       : "
        f"{environment['onnxruntime_cuda_provider']}"
    )

    print(
        f"  Engine build      : "
        f"{environment['tensorrt_engine_build_possible']}"
    )

    print("\nRecommended backend:")

    print(
        f"  {backend['backend']}"
    )

    print(
        f"  Reason: "
        f"{backend['reason']}"
    )

    build_result = None

    if args.build:
        print("\nTensorRT build requested.")

        if not environment[
            "tensorrt_engine_build_possible"
        ]:
            print(
                "Build skipped: NVIDIA GPU "
                "and/or trtexec is unavailable."
            )

        else:
            build_result = (
                build_tensorrt_engine(
                    onnx_path=onnx_path,
                    engine_path=engine_path,
                    precision=str(
                        trt["precision"]
                    ),
                    environment=environment,
                )
            )

            print(
                f"Build success: "
                f"{build_result['success']}"
            )

            print(
                f"Engine exists: "
                f"{build_result['engine_exists']}"
            )

            if not build_result[
                "success"
            ]:
                print(
                    "\ntrtexec stderr:"
                )

                print(
                    build_result[
                        "stderr"
                    ]
                )

    metadata = (
        create_deployment_metadata(
            onnx_path=onnx_path,
            engine_path=engine_path,
            precision=str(
                trt["precision"]
            ),
            environment=environment,
            backend=backend,
            build_result=build_result,
        )
    )

    save_metadata(
        metadata,
        metadata_path,
    )

    print(
        f"\nMetadata report : "
        f"{metadata_path}"
    )

    print("=" * 80)

    if not environment[
        "tensorrt_engine_build_possible"
    ]:
        print(
            "\nTensorRT was not executed on "
            "this machine."
        )

        print(
            "Current deployment path:"
        )

        print(
            "ONNX -> ONNX Runtime CPU"
        )

        print(
            "\nThe repository is now prepared "
            "for TensorRT engine generation on "
            "an NVIDIA deployment machine."
        )


if __name__ == "__main__":
    main()