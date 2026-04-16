"""Audio file probing, waveform extraction, and (later) chunking helpers.

Probing uses `pydub.utils.mediainfo`, which shells out to ffprobe. For missing
files we raise `FileNotFoundError`; for files ffprobe cannot recognise as
audio we raise `ValueError`.

Waveform extraction loads the file via pydub, downmixes to mono, and
peak-downsamples the PCM samples into a fixed number of points suitable for
rendering in pyqtgraph.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from pydub.utils import mediainfo


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    duration_s: float
    sample_rate: int
    channels: int
    size_bytes: int


def probe(path: str | Path) -> AudioInfo:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    info = mediainfo(str(p))
    if not info or info.get("codec_type") != "audio":
        raise ValueError(f"Not a recognised audio file: {p}")

    try:
        duration_s = float(info["duration"])
        sample_rate = int(info["sample_rate"])
        channels = int(info["channels"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Incomplete audio metadata for {p}: {e}") from e

    return AudioInfo(
        path=p,
        duration_s=duration_s,
        sample_rate=sample_rate,
        channels=channels,
        size_bytes=p.stat().st_size,
    )


@dataclass(frozen=True)
class Waveform:
    times: np.ndarray
    amplitudes: np.ndarray


def generate_waveform_samples(
    path: str | Path, target_points: int = 2000
) -> Waveform:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    audio = AudioSegment.from_file(str(p))
    if audio.channels > 1:
        audio = audio.set_channels(1)

    samples = np.asarray(audio.get_array_of_samples(), dtype=np.float64)
    max_amp = float(1 << (8 * audio.sample_width - 1))
    samples /= max_amp

    amplitudes = np.zeros(target_points, dtype=np.float32)
    for i, chunk in enumerate(np.array_split(samples, target_points)):
        if chunk.size:
            amplitudes[i] = chunk[np.argmax(np.abs(chunk))]

    duration_s = len(samples) / float(audio.frame_rate)
    bucket = duration_s / target_points
    times = np.linspace(0.0, duration_s, target_points, endpoint=False) + bucket / 2
    return Waveform(times=times, amplitudes=amplitudes)
