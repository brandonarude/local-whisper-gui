"""Audio file probing and (later) chunking / waveform helpers.

Probing uses `pydub.utils.mediainfo`, which shells out to ffprobe. For missing
files we raise `FileNotFoundError`; for files ffprobe cannot recognise as
audio we raise `ValueError`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
