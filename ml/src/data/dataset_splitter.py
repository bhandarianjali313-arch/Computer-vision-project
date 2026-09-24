from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from ml.src.data.neu_inspector import find_images
from ml.src.data.voc_to_yolo import CLASS_NAMES


def infer_primary_class(stem: str) -> str:
    """
    Infer the original NEU defect category from the image filename.

    Examples:
        crazing_1 -> crazing
        pitted_surface_10 -> pitted_surface
        rolled-in_scale_7 -> rolled-in_scale
    """

    stem_lower = stem.lower()

    # Longest names first to avoid accidental partial matches.
    for class_name in sorted(
        CLASS_NAMES,
        key=len,
        reverse=True,
    ):
        if stem_lower.startswith(class_name.lower()):
            return class_name

    raise ValueError(
        f"Could not infer primary class from filename: {stem}"
    )


def validate_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            "Train, validation and test ratios must sum to 1.0. "
            f"Received total={total}"
        )

    if any(
        ratio <= 0
        for ratio in (
            train_ratio,
            val_ratio,
            test_ratio,
        )
    ):
        raise ValueError(
            "All split ratios must be greater than zero."
        )


def split_class_samples(
    samples: list[Path],
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[Path]]:
    """
    Shuffle and split samples belonging to one primary class.
    """

    samples = samples.copy()
    rng.shuffle(samples)

    total = len(samples)

    train_count = round(
        total * train_ratio
    )

    val_count = round(
        total * val_ratio
    )

    # Remaining samples automatically become test samples.
    train_samples = samples[:train_count]

    val_samples = samples[
        train_count:
        train_count + val_count
    ]

    test_samples = samples[
        train_count + val_count:
    ]

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }


def create_split_assignments(
    image_paths: list[Path],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
) -> dict[str, list[Path]]:
    """
    Create deterministic class-stratified dataset assignments.
    """

    validate_ratios(
        train_ratio,
        val_ratio,
        test_ratio,
    )

    grouped_images: dict[
        str,
        list[Path],
    ] = defaultdict(list)

    for image_path in image_paths:
        primary_class = infer_primary_class(
            image_path.stem
        )

        grouped_images[
            primary_class
        ].append(image_path)

    missing_classes = (
        set(CLASS_NAMES)
        - set(grouped_images)
    )

    if missing_classes:
        raise ValueError(
            "Dataset is missing primary classes: "
            f"{sorted(missing_classes)}"
        )

    assignments = {
        "train": [],
        "val": [],
        "test": [],
    }

    rng = random.Random(seed)

    for class_name in CLASS_NAMES:
        class_split = split_class_samples(
            samples=grouped_images[class_name],
            rng=rng,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        for split_name in assignments:
            assignments[split_name].extend(
                class_split[split_name]
            )

    # Sort after splitting so generated dataset
    # structure is stable and easy to inspect.
    for split_name in assignments:
        assignments[split_name] = sorted(
            assignments[split_name],
            key=lambda path: path.name,
        )

    return assignments


def reset_directory(
    directory: Path,
) -> None:
    """
    Delete a generated directory if it exists and recreate it.
    """

    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def count_objects(
    label_paths: list[Path],
) -> dict[str, int]:
    """
    Count YOLO bounding boxes per class.
    """

    counts: Counter[str] = Counter()

    for label_path in label_paths:
        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:
            if not line.strip():
                continue

            class_id = int(
                line.split()[0]
            )

            if not 0 <= class_id < len(
                CLASS_NAMES
            ):
                raise ValueError(
                    f"Invalid class ID {class_id} "
                    f"in {label_path}"
                )

            class_name = CLASS_NAMES[class_id]

            counts[class_name] += 1

    return {
        class_name: counts[class_name]
        for class_name in CLASS_NAMES
    }


def verify_no_leakage(
    assignments: dict[str, list[Path]],
) -> None:
    """
    Ensure no physical image appears in multiple splits.
    """

    train = {
        path.stem
        for path in assignments["train"]
    }

    val = {
        path.stem
        for path in assignments["val"]
    }

    test = {
        path.stem
        for path in assignments["test"]
    }

    if train & val:
        raise RuntimeError(
            "Data leakage detected between train and val."
        )

    if train & test:
        raise RuntimeError(
            "Data leakage detected between train and test."
        )

    if val & test:
        raise RuntimeError(
            "Data leakage detected between val and test."
        )


def write_dataset_yaml(
    output_root: Path,
) -> Path:
    """
    Generate the Ultralytics YOLO dataset configuration.
    """

    yaml_path = (
        output_root / "dataset.yaml"
    )

    config = {
        "path": output_root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            class_id: class_name
            for class_id, class_name
            in enumerate(CLASS_NAMES)
        },
    }

    with yaml_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )

    return yaml_path


def build_yolo_dataset(
    raw_dataset_root: Path,
    all_labels_dir: Path,
    output_root: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
) -> dict[str, Any]:
    """
    Create the complete train/val/test YOLO dataset.
    """

    raw_dataset_root = (
        raw_dataset_root.resolve()
    )

    all_labels_dir = (
        all_labels_dir.resolve()
    )

    output_root = (
        output_root.resolve()
    )

    if not raw_dataset_root.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: "
            f"{raw_dataset_root}"
        )

    if not all_labels_dir.exists():
        raise FileNotFoundError(
            f"YOLO labels not found: "
            f"{all_labels_dir}"
        )

    image_paths = find_images(
        raw_dataset_root
    )

    if not image_paths:
        raise FileNotFoundError(
            "No images found in raw dataset."
        )

    image_by_stem = {
        image_path.stem: image_path
        for image_path in image_paths
    }

    label_paths = sorted(
        all_labels_dir.glob("*.txt")
    )

    label_by_stem = {
        label_path.stem: label_path
        for label_path in label_paths
    }

    image_stems = set(
        image_by_stem
    )

    label_stems = set(
        label_by_stem
    )

    if image_stems != label_stems:
        missing_labels = sorted(
            image_stems - label_stems
        )

        missing_images = sorted(
            label_stems - image_stems
        )

        raise ValueError(
            "Image/label mismatch detected.\n"
            f"Missing labels: {missing_labels[:10]}\n"
            f"Missing images: {missing_images[:10]}"
        )

    assignments = create_split_assignments(
        image_paths=image_paths,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    verify_no_leakage(
        assignments
    )

    for split_name in (
        "train",
        "val",
        "test",
    ):
        image_output_dir = (
            output_root
            / "images"
            / split_name
        )

        label_output_dir = (
            output_root
            / "labels"
            / split_name
        )

        reset_directory(
            image_output_dir
        )

        reset_directory(
            label_output_dir
        )

        for image_path in assignments[
            split_name
        ]:
            label_path = label_by_stem[
                image_path.stem
            ]

            shutil.copy2(
                image_path,
                image_output_dir
                / image_path.name,
            )

            shutil.copy2(
                label_path,
                label_output_dir
                / label_path.name,
            )

    split_summary = {}

    for split_name in (
        "train",
        "val",
        "test",
    ):
        split_images = assignments[
            split_name
        ]

        split_labels = [
            label_by_stem[
                image_path.stem
            ]
            for image_path in split_images
        ]

        primary_class_counts = Counter(
            infer_primary_class(
                image_path.stem
            )
            for image_path in split_images
        )

        split_summary[
            split_name
        ] = {
            "image_count": len(
                split_images
            ),
            "label_count": len(
                split_labels
            ),
            "primary_class_counts": {
                class_name:
                    primary_class_counts[
                        class_name
                    ]
                for class_name
                in CLASS_NAMES
            },
            "object_counts": count_objects(
                split_labels
            ),
            "stems": [
                image_path.stem
                for image_path in split_images
            ],
        }

    manifest = {
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "total_images": len(
            image_paths
        ),
        "splits": split_summary,
    }

    manifest_path = (
        output_root
        / "split_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    yaml_path = write_dataset_yaml(
        output_root
    )

    return {
        "total_images": len(
            image_paths
        ),
        "splits": split_summary,
        "yaml_path": str(
            yaml_path
        ),
        "manifest_path": str(
            manifest_path
        ),
    }