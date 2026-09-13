import cv2
import numpy as np
import torch


def test_pytorch_tensor_operation():
    tensor = torch.tensor([1, 2, 3])
    result = tensor * 2

    assert result.tolist() == [2, 4, 6]


def test_numpy_array_creation():
    array = np.array([10, 20, 30])

    assert array.shape == (3,)
    assert array.sum() == 60


def test_opencv_import():
    assert cv2.__version__ is not None


def test_pytorch_device_available():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tensor = torch.tensor([1.0, 2.0]).to(device)

    assert tensor.device.type in {"cpu", "cuda"}