"""Compute device detection (SPEC §3.5, §5.2).

Reports CPU plus any CUDA devices visible to torch. CUDA absence is
expected (no GPU, no driver, no torch install) — the function never
raises in those cases; it just returns CPU only.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Device:
    kind: str  # "cpu" or "cuda"
    name: str
    index: int | None = None
    vram_gb: float | None = None

    @property
    def label(self) -> str:
        if self.kind == "cpu":
            return "CPU"
        vram = f" ({self.vram_gb:.1f} GB)" if self.vram_gb is not None else ""
        return f"CUDA — {self.name}{vram}"


def detect_devices() -> list[Device]:
    devices: list[Device] = [Device(kind="cpu", name="CPU")]
    devices.extend(_detect_cuda_devices())
    return devices


def _detect_cuda_devices() -> list[Device]:
    try:
        import torch
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []

    out: list[Device] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        out.append(Device(kind="cuda", name=props.name, index=i, vram_gb=vram_gb))
    return out
