"""Chunk stitching — the accuracy-critical path (SPEC §3.7, §6).

Each transcribed chunk has timestamps in its own *local* frame and overlaps
the previous chunk by ``overlap_s`` seconds of audio. Both chunks transcribe
that overlap region; left untouched, the merged output would either drop
words (if we just dropped the overlap from one side) or duplicate them (if
we concatenated naively).

:func:`stitch` walks the chunk list left-to-right, maintaining a running
``kept`` word sequence in *absolute* time. For each new chunk it:

1. Identifies the trailing kept words that fall inside the overlap window
   (``tail``) and the leading words from the new chunk that fall in the
   same window (``head``).
2. Tokenises both sides (lowercased, trailing-punctuation stripped) and
   uses :class:`difflib.SequenceMatcher` to locate the longest matching
   block.
3. Splices on the match: keeps ``tail`` through the end of the match, then
   takes the new chunk's words from the end of the match onward.
4. Logs a warning on the ``src.core.stitcher`` logger when the match
   covers a small fraction of the shorter side — the boundary may be
   inexact and the user (via the UI log panel, SPEC §3.8) should know.
5. Falls back to a time midpoint splice when no tokens match at all,
   so a complete disagreement still produces a well-ordered output.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from src.core.exporter import Segment, Word

_logger = logging.getLogger(__name__)

_DEFAULT_LOW_CONFIDENCE = 0.5
_PUNCT = ".,;:!?\"'()[]{}"
_EPS = 1e-6


@dataclass(frozen=True)
class ChunkResult:
    start_s: float
    duration_s: float
    segments: list[Segment]


def _normalize_token(text: str) -> str:
    return text.lower().strip().strip(_PUNCT)


def _shift_words(words: Sequence[Word], offset: float) -> list[Word]:
    return [
        Word(
            start=w.start + offset,
            end=w.end + offset,
            text=w.text,
            probability=w.probability,
        )
        for w in words
    ]


def _chunk_words_absolute(chunk: ChunkResult) -> list[Word]:
    out: list[Word] = []
    for seg in chunk.segments:
        if seg.words:
            out.extend(_shift_words(seg.words, chunk.start_s))
    return out


def stitch(
    chunks: Sequence[ChunkResult],
    overlap_s: float,
    *,
    low_confidence_ratio: float = _DEFAULT_LOW_CONFIDENCE,
) -> list[Segment]:
    if not chunks:
        return []

    kept: list[Word] = _chunk_words_absolute(chunks[0])

    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        curr = chunks[i]
        prev_end_abs = prev.start_s + prev.duration_s
        actual_overlap = max(0.0, min(overlap_s, prev_end_abs - curr.start_s))
        overlap_end_abs = curr.start_s + actual_overlap

        curr_words = _chunk_words_absolute(curr)

        # Tail: contiguous suffix of kept whose start lies inside the overlap.
        tail_start_idx = len(kept)
        for j, w in enumerate(kept):
            if w.start >= curr.start_s - _EPS:
                tail_start_idx = j
                break
        tail = kept[tail_start_idx:]

        # Head: leading prefix of curr_words inside the overlap window.
        head_count = 0
        for w in curr_words:
            if w.start < overlap_end_abs - _EPS:
                head_count += 1
            else:
                break
        head = curr_words[:head_count]
        rest = curr_words[head_count:]

        tail_tokens = [_normalize_token(w.text) for w in tail]
        head_tokens = [_normalize_token(w.text) for w in head]

        sm = SequenceMatcher(None, tail_tokens, head_tokens, autojunk=False)
        match = sm.find_longest_match(0, len(tail_tokens), 0, len(head_tokens))

        denom = min(len(tail_tokens), len(head_tokens))
        confidence = match.size / denom if denom else 1.0

        if denom > 0 and confidence < low_confidence_ratio:
            _logger.warning(
                "Low-confidence stitch at chunk %d boundary "
                "(confidence=%.2f, tail=%r, head=%r); overlap may be inexact.",
                i,
                confidence,
                tail_tokens,
                head_tokens,
            )

        if match.size > 0:
            new_tail = tail[: match.a + match.size]
            head_remainder = head[match.b + match.size :]
            kept = kept[:tail_start_idx] + new_tail + head_remainder + rest
        else:
            # No matching tokens at all: split at the overlap midpoint so the
            # merged output is at least temporally well-ordered.
            midpoint = (curr.start_s + overlap_end_abs) / 2.0
            kept = [w for w in kept if w.start < midpoint - _EPS]
            kept.extend(w for w in curr_words if w.start >= midpoint - _EPS)

    if not kept:
        return []

    text = " ".join(w.text for w in kept)
    return [
        Segment(
            start=kept[0].start,
            end=kept[-1].end,
            text=text,
            words=tuple(kept),
        )
    ]
