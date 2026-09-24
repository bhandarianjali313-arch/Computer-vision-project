import argparse
from pathlib import Path

from ml.src.augmentation.defect_augmentation import (
    augment_sample,
    validate_yolo_box,
)
from ml.src.augmentation.sample_loader import (
    load_sample,
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
            "Smoke-test augmentation using "
            "real NEU training samples."
        )
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(
            "data/processed/neu_yolo/images/train"
        ),
    )

    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path(
            "data/processed/neu_yolo/labels/train"
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=50,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(
            f"Training image directory not found: "
            f"{args.images_dir}"
        )

    if not args.labels_dir.exists():
        raise FileNotFoundError(
            f"Training label directory not found: "
            f"{args.labels_dir}"
        )

    image_paths = sorted(
        path
        for path in args.images_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No training images found in "
            f"{args.images_dir}"
        )

    selected_images = image_paths[
        : args.samples
    ]

    processed = 0
    original_boxes = 0
    augmented_boxes = 0

    print("=" * 72)
    print(
        "NEU-DET AUGMENTATION PIPELINE SMOKE TEST"
    )
    print("=" * 72)

    for image_path in selected_images:
        label_path = (
            args.labels_dir
            / f"{image_path.stem}.txt"
        )

        image, bboxes, class_labels = (
            load_sample(
                image_path=image_path,
                label_path=label_path,
            )
        )

        original_boxes += len(
            bboxes
        )

        result = augment_sample(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels,
        )

        for bbox in result["bboxes"]:
            if not validate_yolo_box(
                bbox
            ):
                raise RuntimeError(
                    f"Invalid augmented box "
                    f"for {image_path.name}"
                )

        processed += 1

        augmented_boxes += len(
            result["bboxes"]
        )

    print(
        f"Samples processed        : {processed}"
    )

    print(
        f"Original boxes           : {original_boxes}"
    )

    print(
        f"Boxes after augmentation : {augmented_boxes}"
    )

    print(
        "Invalid augmented boxes : 0"
    )

    print(
        "\nAugmentation status      : PASSED"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()