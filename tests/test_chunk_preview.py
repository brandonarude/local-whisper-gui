"""Smoke tests for chunk-boundary preview (SPEC §3.3)."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from src.utils.device_detect import Device


@pytest.fixture
def long_wav_with_silences(tmp_path: Path) -> Path:
    """22-second mono wav with two clear silence gaps (see test_audio_processor)."""
    sample_rate = 16_000
    freq_hz = 440.0

    def tone(duration_s: float) -> np.ndarray:
        n = int(sample_rate * duration_s)
        t = np.linspace(0.0, duration_s, n, endpoint=False)
        return (0.5 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)

    def silence(duration_s: float) -> np.ndarray:
        return np.zeros(int(sample_rate * duration_s), dtype=np.float32)

    segments = [
        tone(6.0), silence(1.0),
        tone(6.0), silence(2.0),
        tone(6.0), silence(1.0),
    ]
    pcm16 = (np.concatenate(segments) * 32767).astype(np.int16)

    path = tmp_path / "long.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return path


def test_chunk_preview_worker_emits_boundaries(qtbot, long_wav_with_silences: Path) -> None:
    from src.workers.chunk_preview_worker import ChunkPreviewWorker

    worker = ChunkPreviewWorker(
        long_wav_with_silences,
        min_silence_ms=500,
        silence_thresh_db=-40.0,
        min_chunk_s=1.0,
        max_chunk_s=30.0,
        overlap_s=1.0,
    )
    with qtbot.waitSignal(worker.boundaries_ready, timeout=10_000) as blocker:
        worker.start()
    boundaries = blocker.args[0]
    assert isinstance(boundaries, list)
    # Three chunks → two boundaries between them.
    assert len(boundaries) == 2
    assert boundaries == sorted(boundaries)
    worker.wait(5_000)


def test_chunk_preview_worker_failure_signal(qtbot, tmp_path: Path) -> None:
    from src.workers.chunk_preview_worker import ChunkPreviewWorker

    worker = ChunkPreviewWorker(
        tmp_path / "missing.wav",
        min_silence_ms=500,
        silence_thresh_db=-40.0,
        min_chunk_s=1.0,
        max_chunk_s=30.0,
        overlap_s=1.0,
    )
    with qtbot.waitSignal(worker.failed, timeout=5_000) as blocker:
        worker.start()
    assert "FileNotFoundError" in blocker.args[0]
    worker.wait(5_000)


def test_main_window_renders_chunk_boundaries_on_load(
    qtbot, long_wav_with_silences: Path
) -> None:
    from src.ui.main_window import MainWindow

    w = MainWindow(devices=[Device(kind="cpu", name="CPU")])
    qtbot.addWidget(w)
    # Short chunks so the tiny test file produces multiple chunks.
    w._settings_panel._min_chunk_spin.setValue(0.05)  # min 3s
    w._settings_panel._max_chunk_spin.setValue(10.0)
    w._settings_panel._min_silence_spin.setValue(500)

    w._file_header.load_path(long_wav_with_silences)
    assert w._waveform_worker is not None
    w._waveform_worker.wait(5_000)

    assert w._chunk_preview_worker is not None
    w._chunk_preview_worker.wait(10_000)
    # Process the queued boundaries_ready signal into the waveform widget.
    qtbot.waitUntil(lambda: len(w._waveform._boundary_lines) >= 1, timeout=5_000)


def test_main_window_clears_boundaries_when_chunking_disabled(
    qtbot, long_wav_with_silences: Path
) -> None:
    from src.ui.main_window import MainWindow

    w = MainWindow(devices=[Device(kind="cpu", name="CPU")])
    qtbot.addWidget(w)
    w._file_header.load_path(long_wav_with_silences)
    w._waveform_worker.wait(5_000)
    if w._chunk_preview_worker is not None:
        w._chunk_preview_worker.wait(10_000)

    w._settings_panel.set_chunking_enabled(False)
    # Disabling should clear lines synchronously.
    assert w._waveform._boundary_lines == []
