"""Tests for src.utils.device_detect.

The detector reports the available compute devices for transcription.
CPU is always present; CUDA devices are enumerated via torch when
available. Tests inject a fake `torch` module via `sys.modules` so they
run identically whether torch is installed or not.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def no_cuda_torch(mocker):
    """Inject a fake torch module that reports no CUDA available."""
    fake = MagicMock(name="torch")
    fake.cuda.is_available.return_value = False
    mocker.patch.dict(sys.modules, {"torch": fake})
    return fake


@pytest.fixture
def torch_unavailable(mocker):
    """Make `import torch` raise ImportError."""
    mocker.patch.dict(sys.modules, {"torch": None})


@pytest.fixture
def cuda_torch(mocker):
    """Inject a fake torch with one RTX 3080-like CUDA device."""
    fake = MagicMock(name="torch")
    fake.cuda.is_available.return_value = True
    fake.cuda.device_count.return_value = 1

    props = MagicMock()
    props.name = "NVIDIA GeForce RTX 3080"
    props.total_memory = 10 * 1024**3  # 10 GiB
    fake.cuda.get_device_properties.return_value = props

    mocker.patch.dict(sys.modules, {"torch": fake})
    return fake


def test_cpu_is_always_present(no_cuda_torch) -> None:
    from src.utils.device_detect import detect_devices

    devices = detect_devices()
    assert any(d.kind == "cpu" for d in devices)


def test_no_cuda_returns_cpu_only(no_cuda_torch) -> None:
    from src.utils.device_detect import detect_devices

    devices = detect_devices()
    assert [d.kind for d in devices] == ["cpu"]


def test_torch_missing_returns_cpu_only(torch_unavailable) -> None:
    from src.utils.device_detect import detect_devices

    devices = detect_devices()
    assert [d.kind for d in devices] == ["cpu"]


def test_cuda_present_lists_gpu(cuda_torch) -> None:
    from src.utils.device_detect import detect_devices

    devices = detect_devices()
    cuda_devices = [d for d in devices if d.kind == "cuda"]
    assert len(cuda_devices) == 1
    gpu = cuda_devices[0]
    assert gpu.name == "NVIDIA GeForce RTX 3080"
    assert gpu.index == 0
    assert gpu.vram_gb == pytest.approx(10.0, rel=0.01)


def test_cuda_multi_gpu_enumerates_all(mocker) -> None:
    from src.utils.device_detect import detect_devices

    fake = MagicMock(name="torch")
    fake.cuda.is_available.return_value = True
    fake.cuda.device_count.return_value = 2

    def props_for(i):
        p = MagicMock()
        p.name = f"GPU {i}"
        p.total_memory = (8 + i) * 1024**3
        return p

    fake.cuda.get_device_properties.side_effect = props_for
    mocker.patch.dict(sys.modules, {"torch": fake})

    devices = detect_devices()
    cuda = [d for d in devices if d.kind == "cuda"]
    assert [d.index for d in cuda] == [0, 1]
    assert [d.name for d in cuda] == ["GPU 0", "GPU 1"]


def test_cpu_appears_first(cuda_torch) -> None:
    from src.utils.device_detect import detect_devices

    devices = detect_devices()
    assert devices[0].kind == "cpu"
