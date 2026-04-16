"""Shared pytest fixtures."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tiny_wav(tmp_path: Path) -> Path:
    """Write a 1-second 440 Hz mono sine wave at 16 kHz, return the path."""
    sample_rate = 16_000
    duration_s = 1.0
    freq_hz = 440.0

    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    samples = (0.5 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    pcm16 = (samples * 32767).astype(np.int16)

    path = tmp_path / "tiny.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return path
