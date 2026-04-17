"""Smoke tests for WaveformWorker (SPEC §3.2, §5.3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.audio_processor import Waveform
from src.workers.waveform_worker import WaveformWorker


def test_waveform_worker_emits_samples_ready(qtbot, tiny_wav: Path) -> None:
    worker = WaveformWorker(tiny_wav, target_points=256)
    with qtbot.waitSignal(worker.samples_ready, timeout=5_000) as blocker:
        worker.start()
    payload = blocker.args[0]
    assert isinstance(payload, Waveform)
    assert payload.amplitudes.shape == (256,)
    assert payload.times.shape == (256,)
    worker.wait(5_000)


def test_waveform_worker_emits_failed_for_missing_file(qtbot, tmp_path: Path) -> None:
    worker = WaveformWorker(tmp_path / "does_not_exist.wav", target_points=100)
    with qtbot.waitSignal(worker.failed, timeout=5_000) as blocker:
        worker.start()
    assert "FileNotFoundError" in blocker.args[0]
    worker.wait(5_000)
