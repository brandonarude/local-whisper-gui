"""Output-size estimation and free-space probe (SPEC §7, disk-full case).

The goal is a fast pre-check: catch the "export won't fit on this USB
stick" case before we start writing, without trying to model the
exporter's exact output byte count. Estimates are intentionally
conservative — over-counting a little is fine; under-counting is not,
because under-counting would let a borderline export slip past the
guard and fail mid-write with a corrupt partial file.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from src.core.exporter import Segment


# Rough bytes-per-segment per format. Derived from eyeballing the exporter
# output on realistic transcripts; the JSON number includes a baseline for
# the envelope (braces + metadata) plus per-segment overhead on top of the
# per-word contribution handled separately below.
_FORMAT_BYTES_PER_SEGMENT: dict[str, int] = {
    "txt": 120,
    "srt": 200,
    "vtt": 200,
    "json": 300,
}

# JSON is the only format that serialises per-word data; count those
# separately so a word-heavy segment inflates the estimate as it should.
_JSON_BYTES_PER_WORD = 80


def estimate_export_size(
    segments: Iterable[Segment], formats: Iterable[str]
) -> int:
    seg_list = list(segments)
    n = len(seg_list)
    word_count = sum(len(s.words) if s.words else 0 for s in seg_list)

    total = 0
    for fmt in formats:
        per_seg = _FORMAT_BYTES_PER_SEGMENT.get(fmt)
        if per_seg is None:
            continue
        total += per_seg * n
        if fmt == "json":
            total += _JSON_BYTES_PER_WORD * word_count
    return total


def has_sufficient_free_space(path: str | Path, needed_bytes: int) -> bool:
    target = Path(path)
    probe = target
    while not probe.exists():
        parent = probe.parent
        if parent == probe:  # reached filesystem root
            break
        probe = parent

    try:
        usage = shutil.disk_usage(str(probe))
    except OSError:
        # Rather than fail-closed on a flaky probe, let the actual write
        # surface any real error — the OS message will be clearer than a
        # pre-check best-guess.
        return True
    return usage.free >= int(needed_bytes)
