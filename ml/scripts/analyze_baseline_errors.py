import argparse
import csv
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

from ml.src.data.voc_to_yolo import (
    CLASS_NAMES,
)
from ml.src.evaluation.error_analysis import (
    build_global_summary,
    draw_error_preview,
    load_error_analysis_config,
    load_ground_truth,
    match_detections,
    save_json,
    summarize_image_errors,
)
from ml.src.training.yolo_trainer import (
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze false positives, false negatives "
            "and class confusions for the YOLOv8 baseline."
        )
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(
            "models/baseline/"
            "yolov8n_neu_baseline_best.pt"
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
        "--labels",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_yolo/labels/val"
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "ml/configs/error_analysis.yaml"
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
            "outputs/error_analysis/"
            "baseline_validation"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(
            f"Weights not found: "
            f"{args.weights}"
        )

    if not args.images.exists():
        raise FileNotFoundError(
            f"Validation images not found: "
            f"{args.images}"
        )

    if not args.labels.exists():
        raise FileNotFoundError(
            f"Validation labels not found: "
            f"{args.labels}"
        )

    config = load_error_analysis_config(
        args.config
    )

    confidence_threshold = float(
        config["confidence_threshold"]
    )

    match_iou_threshold = float(
        config["match_iou_threshold"]
    )

    imgsz = int(
        config["imgsz"]
    )

    max_previews = int(
        config["max_error_previews"]
    )

    device = resolve_device(
        args.device
    )

    print("=" * 78)
    print("YOLOV8 BASELINE ERROR ANALYSIS")
    print("=" * 78)

    print(
        f"Weights              : "
        f"{args.weights}"
    )

    print(
        f"Validation images    : "
        f"{args.images}"
    )

    print(
        f"Confidence threshold : "
        f"{confidence_threshold}"
    )

    print(
        f"Match IoU threshold  : "
        f"{match_iou_threshold}"
    )

    print(
        f"Image size           : "
        f"{imgsz}"
    )

    print(
        f"Device               : "
        f"{device}"
    )

    print("=" * 78)

    output_dir = (
        args.output_dir
    )

    preview_dir = (
        output_dir
        / "difficult_samples"
    )

    if output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = YOLO(
        str(args.weights)
    )

    predictions_stream = model.predict(
        source=str(
            args.images.resolve()
        ),
        imgsz=imgsz,
        conf=confidence_threshold,
        device=device,
        stream=True,
        verbose=False,
    )

    image_results = []

    processed = 0

    for result in predictions_stream:
        image_path = Path(
            result.path
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise ValueError(
                f"Could not read "
                f"{image_path}"
            )

        height, width = (
            image.shape[:2]
        )

        label_path = (
            args.labels
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"GT label not found: "
                f"{label_path}"
            )

        ground_truth = (
            load_ground_truth(
                label_path=label_path,
                image_width=width,
                image_height=height,
            )
        )

        predictions = []

        boxes = result.boxes

        if boxes is not None:
            xyxy_values = (
                boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            confidence_values = (
                boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            class_values = (
                boxes.cls
                .detach()
                .cpu()
                .numpy()
            )

            for (
                xyxy,
                confidence,
                class_id,
            ) in zip(
                xyxy_values,
                confidence_values,
                class_values,
            ):
                class_id = int(
                    class_id
                )

                predictions.append(
                    {
                        "class_id":
                            class_id,

                        "class_name":
                            CLASS_NAMES[
                                class_id
                            ],

                        "confidence":
                            float(
                                confidence
                            ),

                        "bbox":
                            [
                                float(value)
                                for value
                                in xyxy.tolist()
                            ],
                    }
                )

        matches = match_detections(
            ground_truth=ground_truth,
            predictions=predictions,
            iou_threshold=(
                match_iou_threshold
            ),
        )

        summary = (
            summarize_image_errors(
                image_name=image_path.name,
                matches=matches,
            )
        )

        image_results.append(
            {
                "image":
                    image_path.name,

                "image_path":
                    str(image_path),

                "ground_truth_count":
                    len(ground_truth),

                "prediction_count":
                    len(predictions),

                "ground_truth":
                    ground_truth,

                "predictions":
                    predictions,

                "matches":
                    matches,

                "summary":
                    summary,
            }
        )

        processed += 1

        if (
            processed % 50 == 0
        ):
            print(
                f"Processed "
                f"{processed} images"
            )

    global_summary = (
        build_global_summary(
            image_results
        )
    )

    difficult_images = (
        global_summary[
            "difficult_images"
        ]
    )

    image_result_lookup = {
        item["image"]: item
        for item in image_results
    }

    preview_count = 0

    for difficult in difficult_images:
        if (
            difficult["total_errors"]
            <= 0
        ):
            continue

        if (
            preview_count
            >= max_previews
        ):
            break

        item = image_result_lookup[
            difficult["image"]
        ]

        image = cv2.imread(
            item["image_path"]
        )

        preview = draw_error_preview(
            image=image,
            ground_truth=(
                item["ground_truth"]
            ),
            predictions=(
                item["predictions"]
            ),
        )

        output_path = (
            preview_dir
            / difficult["image"]
        )

        success = cv2.imwrite(
            str(output_path),
            preview,
        )

        if not success:
            raise RuntimeError(
                f"Failed to write "
                f"{output_path}"
            )

        preview_count += 1

    report = {
        "experiment": {
            "model":
                "YOLOv8n baseline",

            "split":
                "validation",

            "confidence_threshold":
                confidence_threshold,

            "match_iou_threshold":
                match_iou_threshold,

            "image_size":
                imgsz,

            "device":
                device,
        },

        "summary":
            global_summary,

        "images":
            image_results,
    }

    json_path = (
        output_dir
        / "error_analysis.json"
    )

    save_json(
        report,
        json_path,
    )

    csv_path = (
        output_dir
        / "difficult_images.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "image",
            "true_positives",
            "false_positives",
            "false_negatives",
            "class_confusions",
            "total_errors",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            difficult_images
        )

    print("\n" + "=" * 78)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 78)

    print(
        f"Images analyzed     : "
        f"{global_summary['images_analyzed']}"
    )

    print(
        f"True positives      : "
        f"{global_summary['true_positives']}"
    )

    print(
        f"False positives     : "
        f"{global_summary['false_positives']}"
    )

    print(
        f"False negatives     : "
        f"{global_summary['false_negatives']}"
    )

    print(
        f"Class confusions    : "
        f"{global_summary['class_confusions']}"
    )

    print(
        "\nFalse positives by class:"
    )

    for (
        class_name,
        count,
    ) in global_summary[
        "false_positives_by_class"
    ].items():
        print(
            f"  {class_name:<20} "
            f"{count}"
        )

    print(
        "\nFalse negatives by class:"
    )

    for (
        class_name,
        count,
    ) in global_summary[
        "false_negatives_by_class"
    ].items():
        print(
            f"  {class_name:<20} "
            f"{count}"
        )

    print(
        "\nClass confusion pairs:"
    )

    confusion_pairs = (
        global_summary[
            "class_confusion_pairs"
        ]
    )

    if confusion_pairs:
        for (
            pair,
            count,
        ) in confusion_pairs.items():
            print(
                f"  {pair:<40} "
                f"{count}"
            )

    else:
        print(
            "  None detected"
        )

    print(
        f"\nDifficult previews : "
        f"{preview_count}"
    )

    print(
        f"JSON report        : "
        f"{json_path}"
    )

    print(
        f"CSV report         : "
        f"{csv_path}"
    )

    print(
        f"Preview directory  : "
        f"{preview_dir}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()