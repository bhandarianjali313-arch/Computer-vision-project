import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from ml.src.optimization.onnx_export import (
    inspect_onnx_model,
    load_onnx_export_config,
    save_json,
)
from ml.src.training.yolo_trainer import (
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export the optimized NEU-DET "
            "YOLO model to ONNX."
        )
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(
            "models/optimized/"
            "yolo_neu_optimized_best.pt"
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "ml/configs/"
            "onnx_export.yaml"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "models/onnx/"
            "yolo_neu_optimized.onnx"
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "outputs/optimization/"
            "day17_onnx_export.json"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(
            f"PyTorch weights not found: "
            f"{args.weights}"
        )

    config = (
        load_onnx_export_config(
            args.config
        )
    )

    export_config = (
        config["export"]
    )

    device = resolve_device(
        args.device
    )

    print("=" * 76)
    print("DAY 17 - YOLO ONNX EXPORT")
    print("=" * 76)

    print(
        f"Source model : "
        f"{args.weights}"
    )

    print(
        f"Image size   : "
        f"{export_config['imgsz']}"
    )

    print(
        f"Batch        : "
        f"{export_config['batch']}"
    )

    print(
        f"Opset        : "
        f"{export_config['opset']}"
    )

    print(
        f"Dynamic      : "
        f"{export_config['dynamic']}"
    )

    print(
        f"Simplify     : "
        f"{export_config['simplify']}"
    )

    print(
        f"Device       : "
        f"{device}"
    )

    print("=" * 76)

    model = YOLO(
        str(args.weights)
    )

    exported = model.export(
        format="onnx",
        imgsz=int(
            export_config["imgsz"]
        ),
        batch=int(
            export_config["batch"]
        ),
        opset=int(
            export_config["opset"]
        ),
        dynamic=bool(
            export_config["dynamic"]
        ),
        simplify=bool(
            export_config["simplify"]
        ),
        half=bool(
            export_config["half"]
        ),
        device=device,
    )

    exported_path = Path(
        str(exported)
    )

    if not exported_path.exists():
        raise FileNotFoundError(
            "Ultralytics reported export "
            "completion but the ONNX model "
            f"was not found: {exported_path}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        exported_path.resolve()
        != args.output.resolve()
    ):
        shutil.copy2(
            exported_path,
            args.output,
        )

    inspection = inspect_onnx_model(
        args.output
    )

    manifest = {
        "source_model":
            str(
                args.weights.resolve()
            ),

        "onnx_model":
            str(
                args.output.resolve()
            ),

        "export_configuration":
            export_config,

        "onnx":
            inspection,
    }

    save_json(
        manifest,
        args.manifest,
    )

    print("\n" + "=" * 76)
    print("ONNX EXPORT COMPLETED")
    print("=" * 76)

    print(
        f"ONNX model : "
        f"{args.output}"
    )

    print(
        f"Valid      : "
        f"{inspection['valid']}"
    )

    print(
        f"Size       : "
        f"{inspection['file_size_mb']:.2f} MB"
    )

    print(
        f"Graph nodes: "
        f"{inspection['node_count']}"
    )

    print(
        "\nInputs:"
    )

    for item in inspection[
        "inputs"
    ]:
        print(
            f"  {item['name']}: "
            f"{item['shape']}"
        )

    print(
        "\nOutputs:"
    )

    for item in inspection[
        "outputs"
    ]:
        print(
            f"  {item['name']}: "
            f"{item['shape']}"
        )

    print(
        f"\nSHA-256   : "
        f"{inspection['sha256']}"
    )

    print(
        f"Manifest  : "
        f"{args.manifest}"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()