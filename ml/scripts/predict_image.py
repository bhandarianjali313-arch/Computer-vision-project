import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from ml.src.inference.image_inference import (
    annotate_image,
    build_prediction_report,
    extract_detections,
    load_inference_config,
    save_prediction_json,
)
from ml.src.training.yolo_trainer import (
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run industrial defect detection "
            "on a single image."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Input image.",
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
            "inference.yaml"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/inference/day15"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.source.exists():
        raise FileNotFoundError(
            f"Input image not found: "
            f"{args.source}"
        )

    if not args.weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: "
            f"{args.weights}"
        )

    config = load_inference_config(
        args.config
    )

    device = resolve_device(
        args.device
    )

    image = cv2.imread(
        str(args.source)
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: "
            f"{args.source}"
        )

    print("=" * 74)
    print("INDUSTRIAL DEFECT IMAGE INFERENCE")
    print("=" * 74)

    print(
        f"Image      : "
        f"{args.source}"
    )

    print(
        f"Model      : "
        f"{args.weights}"
    )

    print(
        f"Device     : "
        f"{device}"
    )

    print(
        f"Image size : "
        f"{config['imgsz']}"
    )

    print(
        f"Confidence : "
        f"{config['confidence_threshold']}"
    )

    print("=" * 74)

    model = YOLO(
        str(args.weights)
    )

    results = model.predict(
        source=image,
        imgsz=int(
            config["imgsz"]
        ),
        conf=float(
            config[
                "confidence_threshold"
            ]
        ),
        iou=float(
            config[
                "iou_threshold"
            ]
        ),
        max_det=int(
            config[
                "max_detections"
            ]
        ),
        device=device,
        verbose=False,
    )

    if len(results) != 1:
        raise RuntimeError(
            "Expected exactly one "
            "prediction result."
        )

    detections = (
        extract_detections(
            results[0]
        )
    )

    annotated = (
        annotate_image(
            image,
            detections,
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotated_path = (
        args.output_dir
        / (
            f"{args.source.stem}"
            "_prediction.jpg"
        )
    )

    json_path = (
        args.output_dir
        / (
            f"{args.source.stem}"
            "_prediction.json"
        )
    )

    success = cv2.imwrite(
        str(annotated_path),
        annotated,
    )

    if not success:
        raise RuntimeError(
            "Unable to save annotated "
            f"image: {annotated_path}"
        )

    report = (
        build_prediction_report(
            source_path=args.source,
            weights_path=args.weights,
            image_shape=image.shape,
            detections=detections,
            inference_config=config,
        )
    )

    save_prediction_json(
        report,
        json_path,
    )

    print(
        f"\nDetections : "
        f"{len(detections)}"
    )

    if detections:
        print("\nDetected defects:")

        for index, detection in enumerate(
            detections,
            start=1,
        ):
            print(
                f"  {index}. "
                f"{detection['class_name']:<20} "
                f"confidence="
                f"{detection['confidence']:.4f}"
            )

    else:
        print(
            "\nNo defects detected above "
            "the configured threshold."
        )

    print(
        f"\nAnnotated image : "
        f"{annotated_path}"
    )

    print(
        f"JSON output     : "
        f"{json_path}"
    )

    print("=" * 74)


if __name__ == "__main__":
    main()