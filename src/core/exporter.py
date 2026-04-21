"""Exporters for transcription results in the four formats SPEC §3.7 lists.

The exporter consumes a list of :class:`Segment` dataclasses (optionally
carrying per-word timings) and writes them to disk. The dataclasses are
defined here rather than in ``transcriber.py`` because the exporter is the
first consumer in the build order (PLAN commit 17 lands before the
transcriber); downstream modules (stitcher, transcriber) import them from
here.

Time formatting helpers:

- :func:`_fmt_hms_ms` renders ``HH:MM:SS<sep>mmm`` — ``,`` for SRT, ``.``
  for VTT.
- :func:`_fmt_ms` renders ``M:SS`` for the optional timestamp prefix in the
  plain-text writer.

Plain-text timestamp cadence (issue #5): when timestamps are enabled,
``write_txt`` emits one ``[M:SS]``-prefixed line per ``timestamp_cadence_s``
window rather than one per faster-whisper segment. Word-level timings are
used to split inside a segment when available, so a single huge segment
still gets regular timestamps instead of a single marker at 0:00.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    words: tuple[Word, ...] | None = None


def _fmt_hms_ms(seconds: float, ms_sep: str) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    s = (total_ms // 1000) % 60
    m = (total_ms // 60_000) % 60
    h = total_ms // 3_600_000
    return f"{h:02d}:{m:02d}:{s:02d}{ms_sep}{ms:03d}"


def _fmt_ms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


DEFAULT_TIMESTAMP_CADENCE_S: float = 30.0


def write_txt(
    segments: Sequence[Segment],
    path: str | Path,
    *,
    include_timestamps: bool,
    timestamp_cadence_s: float = DEFAULT_TIMESTAMP_CADENCE_S,
) -> None:
    if not include_timestamps:
        lines = [seg.text.strip() for seg in segments]
        body = "\n".join(lines)
        if body:
            body += "\n"
        Path(path).write_text(body, encoding="utf-8")
        return

    if timestamp_cadence_s <= 0:
        raise ValueError("timestamp_cadence_s must be > 0")

    lines: list[str] = []
    block_start: float | None = None
    block_parts: list[str] = []

    def flush() -> None:
        if block_start is None or not block_parts:
            return
        text = " ".join(p for p in block_parts if p).strip()
        lines.append(f"[{_fmt_ms(block_start)}] {text}")

    for start, text in _timestamp_atoms(segments):
        if not text:
            continue
        if block_start is None:
            block_start = start
            block_parts = [text]
            continue
        if start - block_start >= timestamp_cadence_s:
            flush()
            block_start = start
            block_parts = [text]
        else:
            block_parts.append(text)
    flush()

    body = "\n".join(lines)
    if body:
        body += "\n"
    Path(path).write_text(body, encoding="utf-8")


def _timestamp_atoms(
    segments: Sequence[Segment],
) -> Iterator[tuple[float, str]]:
    """Yield ``(start_s, text)`` atoms for cadenced txt output.

    Word-level timings are preferred so a single large segment can be
    split mid-way on a word boundary; segments without words fall back to
    one atom per segment (the cadence can only snap at segment boundaries
    in that case).
    """
    for seg in segments:
        if seg.words:
            for w in seg.words:
                yield (float(w.start), w.text.strip())
        else:
            yield (float(seg.start), seg.text.strip())


def cadence_exceeds_duration(
    duration_s: float, cadence_s: float
) -> bool:
    """True when ``cadence_s`` is longer than the audio ``duration_s``.

    The UI keys its "cadence longer than audio" warning off this predicate.
    A non-positive duration is treated as exceeded so a caller that doesn't
    yet know the duration still surfaces the warning.
    """
    if duration_s <= 0:
        return True
    return float(cadence_s) > float(duration_s)


def write_srt(segments: Sequence[Segment], path: str | Path) -> None:
    blocks: list[str] = []
    for i, seg in enumerate(segments, start=1):
        cue = (
            f"{_fmt_hms_ms(seg.start, ',')} --> {_fmt_hms_ms(seg.end, ',')}"
        )
        blocks.append(f"{i}\n{cue}\n{seg.text.strip()}")
    body = "\n\n".join(blocks)
    if body:
        body += "\n"
    Path(path).write_text(body, encoding="utf-8")


def write_vtt(segments: Sequence[Segment], path: str | Path) -> None:
    parts: list[str] = ["WEBVTT", ""]
    for seg in segments:
        cue = (
            f"{_fmt_hms_ms(seg.start, '.')} --> {_fmt_hms_ms(seg.end, '.')}"
        )
        parts.append(f"{cue}\n{seg.text.strip()}")
    body = "\n\n".join(parts).rstrip() + "\n"
    Path(path).write_text(body, encoding="utf-8")


def write_json(segments: Sequence[Segment], path: str | Path) -> None:
    out: list[dict] = []
    for seg in segments:
        entry: dict = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        }
        if seg.words:
            entry["words"] = [
                {
                    "start": w.start,
                    "end": w.end,
                    "text": w.text,
                    "probability": w.probability,
                }
                for w in seg.words
            ]
        out.append(entry)
    payload = {"segments": out}
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
