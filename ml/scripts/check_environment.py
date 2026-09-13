import platform
import sys

import cv2
import numpy as np
import torch
import ultralytics


def check_pytorch() -> bool:
    """Run a small tensor operation to verify PyTorch."""
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = tensor * 2

    return result.tolist() == [2.0, 4.0, 6.0]


def main() -> None:
    print("=" * 65)
    print("REAL-TIME INDUSTRIAL DEFECT DETECTION")
    print("ML ENVIRONMENT CHECK")
    print("=" * 65)

    print(f"Operating System     : {platform.system()} {platform.release()}")
    print(f"Python version       : {sys.version.split()[0]}")
    print(f"PyTorch version      : {torch.__version__}")
    print(f"OpenCV version       : {cv2.__version__}")
    print(f"NumPy version        : {np.__version__}")
    print(f"Ultralytics version  : {ultralytics.__version__}")

    print("-" * 65)

    cuda_available = torch.cuda.is_available()

    print(f"CUDA available       : {cuda_available}")

    if cuda_available:
        print(f"GPU                   : {torch.cuda.get_device_name(0)}")
        print(f"PyTorch CUDA version  : {torch.version.cuda}")
        device = "cuda"
    else:
        print("GPU                   : CUDA GPU not detected")
        device = "cpu"

    print(f"Selected device       : {device}")

    print("-" * 65)

    pytorch_ok = check_pytorch()

    if pytorch_ok:
        print("PyTorch tensor test   : PASSED")
    else:
        print("PyTorch tensor test   : FAILED")
        raise RuntimeError("PyTorch environment verification failed.")

    print("-" * 65)
    print("Environment status    : READY")
    print("=" * 65)


if __name__ == "__main__":
    main()