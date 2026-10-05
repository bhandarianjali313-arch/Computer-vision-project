import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from ml.src.inference.image_inference import (
    extract_detections,
)
from ml.src.inference.quality_decision import (
    apply_quality_policy,
    load_quality_policy,
    save_quality_report,
)
from ml.src.training.yolo_trainer import (
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run defect detection and "
            "manufacturing quality triage."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
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
        "--policy",
        type=Path,
        default=Path(
            "ml/configs/"
            "quality_policy.yaml"
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
            "outputs/inference/day20/"
            "quality_decision.json"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.source.exists():
        raise FileNotFoundError(
            f"Image not found: "
            f"{args.source}"
        )

    if not args.weights.exists():
        raise FileNotFoundError(
            f"Model not found: "
            f"{args.weights}"
        )

    if not args.policy.exists():
        raise FileNotFoundError(
            f"Quality policy not found: "
            f"{args.policy}"
        )

    image = cv2.imread(
        str(args.source)
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: "
            f"{args.source}"
        )

    policy = load_quality_policy(
        args.policy
    )

    device = resolve_device(
        args.device
    )

    model = YOLO(
        str(args.weights)
    )

    confidence = float(
        policy[
            "postprocessing"
        ][
            "minimum_confidence"
        ]
    )

    results = model.predict(
        source=image,
        imgsz=416,
        conf=confidence,
        iou=0.45,
        device=device,
        verbose=False,
    )

    if len(results) != 1:
        raise RuntimeError(
            "Expected exactly one prediction result."
        )

    detections = extract_detections(
        results[0]
    )

    height, width = image.shape[:2]

    quality_report = apply_quality_policy(
        detections=detections,
        image_width=width,
        image_height=height,
        policy=policy,
    )

    report = {
        "source_image": str(
            args.source.resolve()
        ),

        "model": str(
            args.weights.resolve()
        ),

        "image": {
            "width": width,
            "height": height,
        },

        **quality_report,
    }

    save_quality_report(
        report,
        args.output,
    )

    print("=" * 76)
    print(
        "DAY 20 - MANUFACTURING "
        "QUALITY DECISION"
    )
    print("=" * 76)

    print(
        f"Source image      : "
        f"{args.source}"
    )

    print(
        f"Device            : "
        f"{device}"
    )

    print(
        f"Raw detections    : "
        f"{report['input_detection_count']}"
    )

    print(
        f"After confidence  : "
        f"{report['after_confidence_filter']}"
    )

    print(
        f"Final detections  : "
        f"{report['final_detection_count']}"
    )

    print(
        f"LOW               : "
        f"{report['triage_counts']['LOW']}"
    )

    print(
        f"MEDIUM            : "
        f"{report['triage_counts']['MEDIUM']}"
    )

    print(
        f"HIGH              : "
        f"{report['triage_counts']['HIGH']}"
    )

    print(
        f"\nQuality decision  : "
        f"{report['decision']}"
    )

    print("\nReasons:")

    for reason in report[
        "decision_reasons"
    ]:
        print(
            f"  - {reason}"
        )

    print(
        f"\nReport            : "
        f"{args.output}"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()