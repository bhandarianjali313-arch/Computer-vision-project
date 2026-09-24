import argparse
from pathlib import Path

from ml.src.training.dataset_config import (
    write_training_dataset_yaml,
)
from ml.src.training.training_preflight import (
    save_preflight_report,
    validate_training_data,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the NEU training dataset "
            "and prepare YOLO training configuration."
        )
    )

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(
            "data/processed"
        ),
    )

    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_training_dataset.yaml"
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "outputs/"
            "day08_training_preflight.json"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    processed = args.processed_root

    print("=" * 74)
    print("YOLO TRAINING DATA PREFLIGHT")
    print("=" * 74)

    report = validate_training_data(
        train_images_dir=(
            processed
            / "neu_yolo/images/train"
        ),
        train_labels_dir=(
            processed
            / "neu_yolo/labels/train"
        ),
        augmented_images_dir=(
            processed
            / "neu_yolo_augmented/images/train"
        ),
        augmented_labels_dir=(
            processed
            / "neu_yolo_augmented/labels/train"
        ),
        val_images_dir=(
            processed
            / "neu_yolo/images/val"
        ),
        test_images_dir=(
            processed
            / "neu_yolo/images/test"
        ),
    )

    print(
        f"Original train images : "
        f"{report['train_image_count']}"
    )

    print(
        f"Original train labels : "
        f"{report['train_label_count']}"
    )

    print(
        f"Augmented images      : "
        f"{report['augmented_image_count']}"
    )

    print(
        f"Augmented labels      : "
        f"{report['augmented_label_count']}"
    )

    print(
        f"Validation images     : "
        f"{report['validation_image_count']}"
    )

    print(
        f"Test images           : "
        f"{report['test_image_count']}"
    )

    print(
        f"\nSource boxes checked  : "
        f"{report['source_box_total']}"
    )

    print(
        f"Augmented boxes       : "
        f"{report['augmented_box_total']}"
    )

    checks = {
        "Missing train labels":
            report["missing_train_labels"],

        "Labels without images":
            report["labels_without_train_images"],

        "Train/val leakage":
            report["leakage_train_val"],

        "Train/test leakage":
            report["leakage_train_test"],

        "Val/test leakage":
            report["leakage_val_test"],

        "Aug images without labels":
            report[
                "augmented_images_without_labels"
            ],

        "Aug labels without images":
            report[
                "augmented_labels_without_images"
            ],

        "Missing augmentation source":
            report[
                "missing_augmented_sources"
            ],

        "Augmented box increases":
            report[
                "augmented_box_increases"
            ],

        "Duplicate augmented boxes":
            report[
                "duplicate_augmented_annotations"
            ],

        "Augmented class mismatch":
            report[
                "augmented_class_mismatches"
            ],
    }

    print("\nStrict checks:")

    for name, problems in checks.items():
        print(
            f"  {name:<30}: "
            f"{len(problems)}"
        )

    save_preflight_report(
        report,
        args.report,
    )

    if not report["passed"]:
        print("\n" + "=" * 74)

        print(
            "TRAINING PREFLIGHT : FAILED"
        )

        print(
            f"Detailed report   : "
            f"{args.report}"
        )

        if report[
            "augmented_box_increases"
        ]:
            print(
                "\nFirst box-count increases:"
            )

            for issue in report[
                "augmented_box_increases"
            ][:10]:
                print(
                    "  "
                    f"{issue['augmented_stem']}: "
                    f"{issue['source_boxes']} "
                    f"-> "
                    f"{issue['augmented_boxes']}"
                )

        raise RuntimeError(
            "Training blocked because strict "
            "preflight validation failed."
        )

    yaml_path = (
        write_training_dataset_yaml(
            processed_root=processed,
            output_path=args.output_yaml,
        )
    )

    print("\n" + "=" * 74)

    print(
        "TRAINING PREFLIGHT : PASSED"
    )

    print(
        f"Training dataset  : "
        f"{yaml_path}"
    )

    print(
        f"Preflight report  : "
        f"{args.report}"
    )

    print("=" * 74)


if __name__ == "__main__":
    main()