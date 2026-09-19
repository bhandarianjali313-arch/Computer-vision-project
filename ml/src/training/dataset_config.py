from __future__ import annotations

from pathlib import Path

import yaml

from ml.src.data.voc_to_yolo import CLASS_NAMES


def write_training_dataset_yaml(
    processed_root: Path,
    output_path: Path,
) -> Path:
    """
    Build a YOLO dataset configuration using both
    original and offline-augmented training images.

    Validation and test remain untouched.
    """

    processed_root = processed_root.resolve()

    config = {
        "path": processed_root.as_posix(),

        "train": [
            "neu_yolo/images/train",
            "neu_yolo_augmented/images/train",
        ],

        "val": "neu_yolo/images/val",

        "test": "neu_yolo/images/test",

        "names": {
            class_id: class_name
            for class_id, class_name
            in enumerate(CLASS_NAMES)
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )

    return output_path.resolve()