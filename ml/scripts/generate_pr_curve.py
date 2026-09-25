from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO


WEIGHTS = Path(
    "models/baseline/yolov8n_neu_baseline_best.pt"
)

DATASET = Path(
    "data/processed/neu_training_dataset.yaml"
)

OUTPUT_DIR = Path(
    "outputs/evaluation/baseline_validation"
)

OUTPUT_FILE = OUTPUT_DIR / "BoxPR_curve.png"


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Model not found: {WEIGHTS}"
        )

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATASET}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("GENERATING PRECISION-RECALL CURVE")
    print("=" * 70)

    model = YOLO(
        str(WEIGHTS)
    )

    metrics = model.val(
        data=str(DATASET.resolve()),
        split="val",
        imgsz=320,
        batch=16,
        workers=0,
        device="cpu",
        plots=False,
        verbose=False,
    )

    box = metrics.box

    px = getattr(
        box,
        "px",
        None,
    )

    precision_values = getattr(
        box,
        "prec_values",
        None,
    )

    if (
        px is None
        or precision_values is None
    ):
        raise RuntimeError(
            "This Ultralytics version does not expose "
            "PR-curve data through px/prec_values."
        )

    import numpy as np

    recall = np.asarray(
        px,
        dtype=float,
    )

    precision = np.asarray(
        precision_values,
        dtype=float,
    )

    plt.figure(
        figsize=(8, 6)
    )

    if precision.ndim == 1:
        plt.plot(
            recall,
            precision,
            label="all classes",
        )

    else:
        names = model.names

        for class_id in range(
            min(
                precision.shape[0],
                len(names),
            )
        ):
            plt.plot(
                recall,
                precision[class_id],
                label=names[class_id],
            )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "YOLOv8 Baseline Precision-Recall Curve"
    )

    plt.xlim(
        0.0,
        1.0,
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        loc="lower left",
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=200,
    )

    plt.close()

    print(
        f"PR curve saved to:\n{OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()