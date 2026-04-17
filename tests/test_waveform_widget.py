"""Smoke tests for WaveformWidget (SPEC §3.2)."""
from __future__ import annotations

import numpy as np

from src.core.audio_processor import Waveform


def test_waveform_widget_instantiates(qtbot) -> None:
    from src.ui.waveform_widget import WaveformWidget

    w = WaveformWidget()
    qtbot.addWidget(w)
    assert w is not None


def test_waveform_widget_accepts_samples_and_boundaries(qtbot) -> None:
    from src.ui.waveform_widget import WaveformWidget

    w = WaveformWidget()
    qtbot.addWidget(w)

    times = np.linspace(0.0, 10.0, 500)
    amps = np.sin(2 * np.pi * 1.0 * times).astype(np.float32)
    w.set_samples(Waveform(times=times, amplitudes=amps))
    w.set_chunk_boundaries([2.5, 5.0, 7.5])
    # Replacing boundaries shouldn't leak items.
    w.set_chunk_boundaries([1.0, 9.0])
    w.clear()


def test_waveform_widget_set_samples_replaces_existing_curve(qtbot) -> None:
    from src.ui.waveform_widget import WaveformWidget

    w = WaveformWidget()
    qtbot.addWidget(w)

    t1 = np.linspace(0.0, 5.0, 200)
    a1 = np.zeros_like(t1, dtype=np.float32)
    w.set_samples(Waveform(times=t1, amplitudes=a1))

    t2 = np.linspace(0.0, 8.0, 400)
    a2 = np.ones_like(t2, dtype=np.float32) * 0.5
    # Second call must not raise nor leak curves.
    w.set_samples(Waveform(times=t2, amplitudes=a2))
