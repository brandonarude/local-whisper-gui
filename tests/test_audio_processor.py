"""Tests for src.core.audio_processor.

Probe reads basic metadata (duration, sample rate, channels, size) from any
ffmpeg-decodable audio file. Backend is `pydub.utils.mediainfo`, but tests
only care about the public shape: a probe result with those four fields,
and clear errors for missing or non-audio files.

Waveform extraction downsamples the audio to a fixed number of peak-amplitude
points suitable for rendering in pyqtgraph. Output is mono, float, in [-1, 1],
paired with a monotonic time axis in seconds.

Chunking splits long audio on silence (pydub.silence), enforces min/max chunk
durations, and adds a small overlap at each boundary so the transcription
stitcher can align chunk outputs (SPEC §3.3, §6).
"""
from __future__ import annotations

import wave
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


# --- chunking --------------------------------------------------------------


@pytest.fixture
def long_wav_with_silences(tmp_path: Path) -> Path:
    """22-second mono wav: three 6s tones separated by silence gaps of 1s and 2s.

    Layout (seconds): [0..6] tone, [6..7] silence, [7..13] tone,
                      [13..15] silence, [15..21] tone, [21..22] silence.
    """
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


def test_chunk_count_matches_silence_gaps(long_wav_with_silences: Path) -> None:
    from src.core.audio_processor import chunk_audio

    chunks = chunk_audio(
        long_wav_with_silences,
        min_silence_ms=500,
        silence_thresh_db=-40.0,
        min_chunk_s=1.0,
        max_chunk_s=30.0,
        overlap_s=1.0,
    )
    # Three tones separated by two silences of length > min_silence_ms.
    assert len(chunks) == 3


def test_chunks_respect_min_chunk_duration(long_wav_with_silences: Path) -> None:
    from src.core.audio_processor import chunk_audio

    chunks = chunk_audio(
        long_wav_with_silences,
        min_silence_ms=500,
        silence_thresh_db=-40.0,
        min_chunk_s=2.0,
        max_chunk_s=30.0,
        overlap_s=0.5,
    )
    for c in chunks:
        assert (c.end_s - c.start_s) >= 2.0 - 1e-3


def test_chunks_respect_max_chunk_duration(long_wav_with_silences: Path) -> None:
    from src.core.audio_processor import chunk_audio

    # Force further splitting: each silence-split chunk is ~6s, max is 4s.
    chunks = chunk_audio(
        long_wav_with_silences,
        min_silence_ms=500,
        silence_thresh_db=-40.0,
        min_chunk_s=0.5,
        max_chunk_s=4.0,
        overlap_s=0.5,
    )
    for c in chunks:
        assert (c.end_s - c.start_s) <= 4.0 + 0.2


def test_consecutive_chunks_overlap_by_requested_amount(long_wav_with_silences: Path) -> None:
    from src.core.audio_processor import chunk_audio

    overlap_s = 1.0
    chunks = chunk_audio(
        long_wav_with_silences,
        min_silence_ms=500,
        silence_thresh_db=-40.0,
        min_chunk_s=1.0,
        max_chunk_s=30.0,
        overlap_s=overlap_s,
    )
    assert len(chunks) >= 2
    for prev, curr in zip(chunks, chunks[1:]):
        assert prev.end_s - curr.start_s == pytest.approx(overlap_s, abs=0.1)


def test_chunks_cover_audio_content(long_wav_with_silences: Path) -> None:
    from src.core.audio_processor import chunk_audio

    overlap_s = 1.0
    chunks = chunk_audio(
        long_wav_with_silences,
        min_silence_ms=500,
        silence_thresh_db=-40.0,
        min_chunk_s=1.0,
        max_chunk_s=30.0,
        overlap_s=overlap_s,
    )
    # Sum of non-overlap durations ≈ content duration (18s of tone + some
    # silence padding left by pydub's keep_silence trimming). Loose bounds
    # to tolerate padding without accepting a completely wrong result.
    total_non_overlap = sum(c.end_s - c.start_s for c in chunks) - overlap_s * (len(chunks) - 1)
    assert 16.0 < total_non_overlap < 22.0


def test_chunks_have_monotonic_start_times(long_wav_with_silences: Path) -> None:
    from src.core.audio_processor import chunk_audio

    chunks = chunk_audio(
        long_wav_with_silences,
        min_silence_ms=500,
        silence_thresh_db=-40.0,
        min_chunk_s=1.0,
        max_chunk_s=30.0,
        overlap_s=1.0,
    )
    starts = [c.start_s for c in chunks]
    assert starts == sorted(starts)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_audio_missing_file_raises(tmp_path: Path) -> None:
    from src.core.audio_processor import chunk_audio

    with pytest.raises(FileNotFoundError):
        chunk_audio(tmp_path / "nope.wav")
