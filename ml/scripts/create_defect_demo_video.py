from pathlib import Path

import cv2


INPUT_DIR = Path(
    "data/processed/neu_yolo/images/val"
)

OUTPUT_PATH = Path(
    "demo/defect_video.mp4"
)

FPS = 5.0

FRAME_SIZE = (
    416,
    416,
)


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}"
        )

    image_paths = sorted(
        [
            path
            for path in INPUT_DIR.iterdir()
            if path.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
            }
        ]
    )

    if not image_paths:
        raise RuntimeError(
            "No images found in validation directory."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        FPS,
        FRAME_SIZE,
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create output video."
        )

    frames_written = 0

    try:
        for image_path in image_paths:
            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                print(
                    f"Skipping unreadable image: "
                    f"{image_path}"
                )
                continue

            frame = cv2.resize(
                image,
                FRAME_SIZE,
                interpolation=cv2.INTER_LINEAR,
            )

            # Keep every image visible for
            # approximately half a second.
            repeat_count = max(
                1,
                int(FPS * 0.5),
            )

            for _ in range(
                repeat_count
            ):
                writer.write(
                    frame
                )

                frames_written += 1

    finally:
        writer.release()

    print("=" * 70)
    print("DEFECT DEMO VIDEO CREATED")
    print("=" * 70)

    print(
        f"Source images : "
        f"{len(image_paths)}"
    )

    print(
        f"Frames written: "
        f"{frames_written}"
    )

    print(
        f"FPS           : "
        f"{FPS}"
    )

    print(
        f"Output        : "
        f"{OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()