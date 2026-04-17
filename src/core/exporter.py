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
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


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


def write_txt(
    segments: Sequence[Segment],
    path: str | Path,
    *,
    include_timestamps: bool,
) -> None:
    lines: list[str] = []
    for seg in segments:
        text = seg.text.strip()
        if include_timestamps:
            lines.append(f"[{_fmt_ms(seg.start)}] {text}")
        else:
            lines.append(text)
    body = "\n".join(lines)
    if body:
        body += "\n"
    Path(path).write_text(body, encoding="utf-8")


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
