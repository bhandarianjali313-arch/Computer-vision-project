import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from ml.src.inference.image_inference import (
    extract_detections,
)
from ml.src.optimization.onnx_export import (
    compare_detections,
    load_onnx_export_config,
    save_json,
)
from ml.src.training.yolo_trainer import (
    resolve_device,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare PyTorch and ONNX "
            "YOLO predictions."
        )
    )

    parser.add_argument(
        "--pytorch",
        type=Path,
        default=Path(
            "models/optimized/"
            "yolo_neu_optimized_best.pt"
        ),
    )

    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path(
            "models/onnx/"
            "yolo_neu_optimized.onnx"
        ),
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_yolo/images/val"
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
        "--report",
        type=Path,
        default=Path(
            "outputs/optimization/"
            "day17_onnx_parity.json"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    for required in (
        args.pytorch,
        args.onnx,
        args.images,
    ):
        if not required.exists():
            raise FileNotFoundError(
                f"Required path not found: "
                f"{required}"
            )

    config = (
        load_onnx_export_config(
            args.config
        )
    )

    export_config = config[
        "export"
    ]

    parity_config = config[
        "parity"
    ]

    device = resolve_device(
        args.device
    )

    image_paths = sorted(
        [
            path
            for path
            in args.images.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    image_paths = image_paths[
        :int(
            parity_config[
                "sample_count"
            ]
        )
    ]

    if not image_paths:
        raise FileNotFoundError(
            "No parity-check images found."
        )

    pytorch_model = YOLO(
        str(args.pytorch)
    )

    onnx_model = YOLO(
        str(args.onnx)
    )

    print("=" * 78)
    print("PYTORCH VS ONNX PREDICTION PARITY")
    print("=" * 78)

    print(
        f"Images     : "
        f"{len(image_paths)}"
    )

    print(
        f"Image size : "
        f"{export_config['imgsz']}"
    )

    print(
        f"Confidence : "
        f"{parity_config['confidence_threshold']}"
    )

    print(
        f"Device     : "
        f"{device}"
    )

    print("=" * 78)

    image_reports = []

    total_pytorch = 0
    total_onnx = 0
    total_matches = 0

    for image_path in image_paths:
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise ValueError(
                f"Could not read: "
                f"{image_path}"
            )

        pytorch_result = (
            pytorch_model.predict(
                source=image,
                imgsz=int(
                    export_config[
                        "imgsz"
                    ]
                ),
                conf=float(
                    parity_config[
                        "confidence_threshold"
                    ]
                ),
                iou=float(
                    parity_config[
                        "iou_threshold"
                    ]
                ),
                device=device,
                verbose=False,
            )[0]
        )

        onnx_result = (
            onnx_model.predict(
                source=image,
                imgsz=int(
                    export_config[
                        "imgsz"
                    ]
                ),
                conf=float(
                    parity_config[
                        "confidence_threshold"
                    ]
                ),
                iou=float(
                    parity_config[
                        "iou_threshold"
                    ]
                ),
                device="cpu",
                verbose=False,
            )[0]
        )

        pytorch_detections = (
            extract_detections(
                pytorch_result
            )
        )

        onnx_detections = (
            extract_detections(
                onnx_result
            )
        )

        comparison = compare_detections(
            pytorch_detections=(
                pytorch_detections
            ),
            onnx_detections=(
                onnx_detections
            ),
            match_iou_threshold=float(
                parity_config[
                    "match_iou_threshold"
                ]
            ),
            confidence_tolerance=float(
                parity_config[
                    "confidence_tolerance"
                ]
            ),
        )

        total_pytorch += (
            comparison[
                "pytorch_count"
            ]
        )

        total_onnx += (
            comparison[
                "onnx_count"
            ]
        )

        total_matches += (
            comparison[
                "matched_count"
            ]
        )

        image_reports.append(
            {
                "image":
                    image_path.name,

                **comparison,
            }
        )

        print(
            f"{image_path.name:<28} "
            f"PT={comparison['pytorch_count']:<3} "
            f"ONNX={comparison['onnx_count']:<3} "
            f"matched="
            f"{comparison['matched_count']:<3} "
            f"rate="
            f"{comparison['match_rate']:.2%}"
        )

    denominator = max(
        total_pytorch,
        total_onnx,
    )

    if denominator == 0:
        overall_match_rate = 1.0
    else:
        overall_match_rate = (
            total_matches
            / denominator
        )

    minimum_match_rate = float(
        parity_config[
            "minimum_match_rate"
        ]
    )

    passed = (
        overall_match_rate
        >= minimum_match_rate
    )

    report = {
        "pytorch_model":
            str(
                args.pytorch.resolve()
            ),

        "onnx_model":
            str(
                args.onnx.resolve()
            ),

        "images_checked":
            len(image_paths),

        "total_pytorch_detections":
            total_pytorch,

        "total_onnx_detections":
            total_onnx,

        "total_matched_detections":
            total_matches,

        "overall_match_rate":
            float(
                overall_match_rate
            ),

        "minimum_required_match_rate":
            minimum_match_rate,

        "passed":
            passed,

        "images":
            image_reports,
    }

    save_json(
        report,
        args.report,
    )

    print("\n" + "=" * 78)
    print("PARITY SUMMARY")
    print("=" * 78)

    print(
        f"PyTorch detections : "
        f"{total_pytorch}"
    )

    print(
        f"ONNX detections    : "
        f"{total_onnx}"
    )

    print(
        f"Matched detections : "
        f"{total_matches}"
    )

    print(
        f"Overall match rate : "
        f"{overall_match_rate:.2%}"
    )

    print(
        f"Required match rate: "
        f"{minimum_match_rate:.2%}"
    )

    print(
        f"Status             : "
        f"{'PASSED' if passed else 'FAILED'}"
    )

    print(
        f"Report             : "
        f"{args.report}"
    )

    print("=" * 78)

    if not passed:
        raise RuntimeError(
            "PyTorch/ONNX prediction "
            "parity check failed."
        )


if __name__ == "__main__":
    main()