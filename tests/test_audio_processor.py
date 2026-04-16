"""Tests for src.core.audio_processor.

Probe reads basic metadata (duration, sample rate, channels, size) from any
ffmpeg-decodable audio file. Backend is `pydub.utils.mediainfo`, but tests
only care about the public shape: a probe result with those four fields,
and clear errors for missing or non-audio files.

Waveform extraction downsamples the audio to a fixed number of peak-amplitude
points suitable for rendering in pyqtgraph. Output is mono, float, in [-1, 1],
paired with a monotonic time axis in seconds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_probe_duration(tiny_wav: Path) -> None:
    from src.core.audio_processor import probe

    info = probe(tiny_wav)
    assert info.duration_s == pytest.approx(1.0, abs=0.05)


def test_probe_sample_rate(tiny_wav: Path) -> None:
    from src.core.audio_processor import probe

    info = probe(tiny_wav)
    assert info.sample_rate == 16_000


def test_probe_channels(tiny_wav: Path) -> None:
    from src.core.audio_processor import probe

    info = probe(tiny_wav)
    assert info.channels == 1


def test_probe_size_matches_filesystem(tiny_wav: Path) -> None:
    from src.core.audio_processor import probe

    info = probe(tiny_wav)
    assert info.size_bytes == tiny_wav.stat().st_size


def test_probe_accepts_str_path(tiny_wav: Path) -> None:
    from src.core.audio_processor import probe

    info = probe(str(tiny_wav))
    assert info.sample_rate == 16_000
    assert info.channels == 1


def test_probe_missing_file_raises(tmp_path: Path) -> None:
    from src.core.audio_processor import probe

    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "does_not_exist.wav")


def test_probe_non_audio_file_raises(tmp_path: Path) -> None:
    from src.core.audio_processor import probe

    bogus = tmp_path / "not_audio.txt"
    bogus.write_text("this is plain text, not an audio stream")
    with pytest.raises(ValueError):
        probe(bogus)


# --- waveform extraction ---------------------------------------------------


def test_waveform_returns_exact_target_points(tiny_wav: Path) -> None:
    from src.core.audio_processor import generate_waveform_samples

    wf = generate_waveform_samples(tiny_wav, target_points=500)
    assert wf.amplitudes.shape == (500,)
    assert wf.times.shape == (500,)


@pytest.mark.parametrize("n", [100, 500, 2000])
def test_waveform_honours_varying_target_points(tiny_wav: Path, n: int) -> None:
    from src.core.audio_processor import generate_waveform_samples

    wf = generate_waveform_samples(tiny_wav, target_points=n)
    assert wf.amplitudes.shape == (n,)
    assert wf.times.shape == (n,)


def test_waveform_amplitudes_are_mono_float_in_unit_range(tiny_wav: Path) -> None:
    from src.core.audio_processor import generate_waveform_samples

    wf = generate_waveform_samples(tiny_wav, target_points=500)
    assert wf.amplitudes.ndim == 1
    assert np.issubdtype(wf.amplitudes.dtype, np.floating)
    assert wf.amplitudes.min() >= -1.0
    assert wf.amplitudes.max() <= 1.0


def test_waveform_captures_expected_peak_amplitude(tiny_wav: Path) -> None:
    # tiny_wav is 0.5 * sin(2π·440·t) quantised to int16; peak buckets should
    # recover a magnitude close to 0.5 after normalisation.
    from src.core.audio_processor import generate_waveform_samples

    wf = generate_waveform_samples(tiny_wav, target_points=200)
    assert float(np.abs(wf.amplitudes).max()) == pytest.approx(0.5, abs=0.05)


def test_waveform_time_axis_is_strictly_monotonic(tiny_wav: Path) -> None:
    from src.core.audio_processor import generate_waveform_samples

    wf = generate_waveform_samples(tiny_wav, target_points=500)
    assert np.all(np.diff(wf.times) > 0)


def test_waveform_time_axis_spans_audio_duration(tiny_wav: Path) -> None:
    from src.core.audio_processor import generate_waveform_samples

    wf = generate_waveform_samples(tiny_wav, target_points=500)
    assert wf.times[0] >= 0.0
    assert wf.times[-1] == pytest.approx(1.0, abs=0.05)


def test_waveform_missing_file_raises(tmp_path: Path) -> None:
    from src.core.audio_processor import generate_waveform_samples

    with pytest.raises(FileNotFoundError):
        generate_waveform_samples(tmp_path / "nope.wav", target_points=100)
