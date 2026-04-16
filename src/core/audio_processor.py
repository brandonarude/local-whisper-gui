"""Audio file probing, waveform extraction, and silence-based chunking.

Probing uses `pydub.utils.mediainfo`, which shells out to ffprobe. For missing
files we raise `FileNotFoundError`; for files ffprobe cannot recognise as
audio we raise `ValueError`.

Waveform extraction loads the file via pydub, downmixes to mono, and
peak-downsamples the PCM samples into a fixed number of points suitable for
rendering in pyqtgraph.

Chunking splits on silence via `pydub.silence.detect_nonsilent`, then enforces
a max chunk duration (splitting any natural chunk that exceeds it) and
prepends ~overlap_s of preceding audio to each chunk after the first. The
overlap inflates chunks that cross a silence gap — max-enforcement accounts
for that inflation so the final chunk duration stays within `max_chunk_s`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.utils import mediainfo

from src.utils.constants import (
    CHUNK_OVERLAP_SECONDS,
    DEFAULT_MAX_CHUNK_MINUTES,
    DEFAULT_MIN_CHUNK_MINUTES,
    DEFAULT_MIN_SILENCE_MS,
    DEFAULT_SILENCE_THRESHOLD_DBFS,
)


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


@dataclass(frozen=True)
class Chunk:
    index: int
    start_s: float
    end_s: float
    audio: AudioSegment


def _split_natural_range(
    start_ms: int, end_ms: int, first_allow_ms: int, rest_allow_ms: int
) -> list[tuple[int, int]]:
    """Greedy split into pieces: first ≤ first_allow_ms, rest ≤ rest_allow_ms."""
    length = end_ms - start_ms
    if length <= first_allow_ms:
        return [(start_ms, end_ms)]

    first_allow_ms = max(first_allow_ms, 1)
    rest_allow_ms = max(rest_allow_ms, 1)

    pieces: list[tuple[int, int]] = []
    cut = start_ms + first_allow_ms
    pieces.append((start_ms, cut))
    while end_ms - cut > rest_allow_ms:
        nxt = cut + rest_allow_ms
        pieces.append((cut, nxt))
        cut = nxt
    if cut < end_ms:
        pieces.append((cut, end_ms))
    return pieces


def chunk_audio(
    path: str | Path,
    *,
    min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
    silence_thresh_db: float = float(DEFAULT_SILENCE_THRESHOLD_DBFS),
    min_chunk_s: float = DEFAULT_MIN_CHUNK_MINUTES * 60.0,
    max_chunk_s: float = DEFAULT_MAX_CHUNK_MINUTES * 60.0,
    overlap_s: float = CHUNK_OVERLAP_SECONDS,
) -> list[Chunk]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    audio = AudioSegment.from_file(str(p))
    total_ms = len(audio)

    ranges_ms = detect_nonsilent(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db,
    )
    if not ranges_ms:
        ranges_ms = [[0, total_ms]]

    max_ms = int(max_chunk_s * 1000)
    overlap_ms = int(overlap_s * 1000)
    rest_allow_ms = max(max_ms - overlap_ms, 1)

    pieces: list[tuple[int, int]] = []
    prev_natural_end: int | None = None
    for nat_start, nat_end in ranges_ms:
        if prev_natural_end is None:
            first_allow_ms = max_ms
        else:
            silence_gap_ms = max(0, nat_start - prev_natural_end)
            first_allow_ms = max(max_ms - silence_gap_ms - overlap_ms, 1)
        pieces.extend(
            _split_natural_range(nat_start, nat_end, first_allow_ms, rest_allow_ms)
        )
        prev_natural_end = nat_end

    chunks: list[Chunk] = []
    prev_end_ms = 0
    for i, (piece_start_ms, piece_end_ms) in enumerate(pieces):
        if i == 0:
            actual_start_ms = piece_start_ms
        else:
            actual_start_ms = max(0, prev_end_ms - overlap_ms)
        actual_end_ms = piece_end_ms

        segment = audio[actual_start_ms:actual_end_ms]
        chunks.append(
            Chunk(
                index=i,
                start_s=actual_start_ms / 1000.0,
                end_s=actual_end_ms / 1000.0,
                audio=segment,
            )
        )
        prev_end_ms = actual_end_ms

    return chunks
