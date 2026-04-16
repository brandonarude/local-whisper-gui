"""Tests for src.core.audio_processor.

Probe reads basic metadata (duration, sample rate, channels, size) from any
ffmpeg-decodable audio file. Backend is `pydub.utils.mediainfo`, but tests
only care about the public shape: a probe result with those four fields,
and clear errors for missing or non-audio files.
"""
from __future__ import annotations

from pathlib import Path

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
