from pathlib import Path

from ml.src.data.dataset_splitter import (
    create_split_assignments,
    infer_primary_class,
    verify_no_leakage,
)


def test_infer_primary_class():
    assert (
        infer_primary_class(
            "crazing_1"
        )
        == "crazing"
    )

    assert (
        infer_primary_class(
            "pitted_surface_25"
        )
        == "pitted_surface"
    )

    assert (
        infer_primary_class(
            "rolled-in_scale_100"
        )
        == "rolled-in_scale"
    )


def test_reproducible_split():
    class_names = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]

    images = []

    for class_name in class_names:
        for index in range(10):
            images.append(
                Path(
                    f"{class_name}_{index}.jpg"
                )
            )

    first = create_split_assignments(
        images,
        seed=42,
    )

    second = create_split_assignments(
        images,
        seed=42,
    )

    assert first == second


def test_split_has_no_leakage():
    class_names = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]

    images = []

    for class_name in class_names:
        for index in range(10):
            images.append(
                Path(
                    f"{class_name}_{index}.jpg"
                )
            )

    assignments = (
        create_split_assignments(
            images,
            seed=42,
        )
    )

    verify_no_leakage(
        assignments
    )

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

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)

    assert (
        len(train)
        + len(val)
        + len(test)
        == 60
    )