"""Tests for src.core.stitcher — the accuracy-critical path (SPEC §3.7, §6).

The stitcher merges per-chunk transcription output into a single list of
segments with timestamps in the *original* audio's frame. For any two
adjacent chunks, the ~2s audio overlap is transcribed twice (once at the
tail of chunk N, once at the head of chunk N+1). The stitcher uses
sequence alignment (``difflib.SequenceMatcher``) on the tokenised overlap
text to locate a splice point and then concatenates:

- chunk N's words up to the splice point, plus
- chunk N+1's words *from* the splice point onward.

Acceptance bar (PLAN commit 20):

- No words dropped at boundaries (ground truth is preserved in full).
- No words duplicated at boundaries (each overlap word appears exactly
  once in the final output).
- Timestamps are offset by cumulative chunk ``start_s`` so the merged
  segments are in the original audio's frame.
- Adversarial overlap: when the overlap region contains a repeated
  phrase (e.g. "and then ... and then"), the stitcher still picks a
  splice that preserves the ground truth without duplicating or
  dropping words.
- Low-confidence splices (overlap texts disagree significantly) emit a
  warning on the ``src.core.stitcher`` logger so users are aware the
  boundary may be imperfect.
"""
from __future__ import annotations

import logging
from typing import Sequence

import pytest


# --- test helpers ----------------------------------------------------------


def _seg(word_tuples: Sequence[tuple[str, float, float]]):
    """One Segment whose text is the joined word list; word-level data attached.

    Each tuple is ``(text, start, end)`` in the chunk's *local* frame.
    """
    from src.core.exporter import Segment, Word

    words = tuple(
        Word(start=s, end=e, text=t, probability=0.95) for t, s, e in word_tuples
    )
    return Segment(
        start=word_tuples[0][1],
        end=word_tuples[-1][2],
        text=" ".join(t for t, _, _ in word_tuples),
        words=words,
    )


def _flatten_words(segments):
    out = []
    for s in segments:
        if s.words:
            out.extend(s.words)
    return out


def _chunk(start_s: float, duration_s: float, word_tuples):
    from src.core.stitcher import ChunkResult

    return ChunkResult(
        start_s=start_s,
        duration_s=duration_s,
        segments=[_seg(word_tuples)],
    )


# --- degenerate inputs -----------------------------------------------------


def test_empty_chunk_list_returns_empty() -> None:
    from src.core.stitcher import stitch

    assert stitch([], overlap_s=2.0) == []


def test_single_chunk_passes_through_with_absolute_timestamps() -> None:
    from src.core.stitcher import stitch

    chunk = _chunk(
        start_s=10.0,
        duration_s=5.0,
        word_tuples=[("alpha", 0.0, 0.5), ("beta", 1.0, 1.5)],
    )
    out = stitch([chunk], overlap_s=2.0)
    words = _flatten_words(out)
    assert [w.text for w in words] == ["alpha", "beta"]
    # chunk.start_s = 10.0 → word starts shifted by 10.
    assert [w.start for w in words] == pytest.approx([10.0, 11.0])
    assert [w.end for w in words] == pytest.approx([10.5, 11.5])


# --- timestamp offset ------------------------------------------------------


def test_non_overlapping_chunks_concatenate_with_absolute_timestamps() -> None:
    from src.core.stitcher import stitch

    chunks = [
        _chunk(0.0, 5.0, [("a", 0.0, 0.5), ("b", 1.0, 1.5)]),
        _chunk(10.0, 5.0, [("c", 0.0, 0.5), ("d", 1.0, 1.5)]),
    ]
    out = stitch(chunks, overlap_s=0.0)
    words = _flatten_words(out)
    assert [w.text for w in words] == ["a", "b", "c", "d"]
    assert [w.start for w in words] == pytest.approx([0.0, 1.0, 10.0, 11.0])


# --- exact-overlap happy path ---------------------------------------------


def test_exact_overlap_preserves_ground_truth_words() -> None:
    """Ground truth: 'the quick brown fox jumps over the lazy dog'.

    Chunk 1 covers words at absolute 0..5s; chunk 2 covers 3..8s (2s
    overlap). Both chunks transcribe the overlap identically — the
    stitcher must produce the full sequence with no words lost or
    repeated at the boundary.
    """
    from src.core.stitcher import stitch

    chunks = [
        _chunk(
            0.0, 5.0,
            [
                ("the", 0.0, 0.4),
                ("quick", 1.0, 1.4),
                ("brown", 2.0, 2.4),
                ("fox", 3.0, 3.4),
                ("jumps", 4.0, 4.4),
            ],
        ),
        _chunk(
            3.0, 5.0,
            [
                ("fox", 0.0, 0.4),   # overlap: abs 3.0
                ("jumps", 1.0, 1.4), # overlap: abs 4.0
                ("over", 2.0, 2.4),
                ("the", 3.0, 3.4),
                ("lazy", 4.0, 4.4),
            ],
        ),
    ]
    out = stitch(chunks, overlap_s=2.0)
    words = _flatten_words(out)
    assert [w.text for w in words] == [
        "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy",
    ]


def test_exact_overlap_timestamps_are_monotonic_and_offset() -> None:
    from src.core.stitcher import stitch

    chunks = [
        _chunk(
            0.0, 5.0,
            [("the", 0.0, 0.4), ("fox", 3.0, 3.4), ("jumps", 4.0, 4.4)],
        ),
        _chunk(
            3.0, 5.0,
            [
                ("fox", 0.0, 0.4),   # abs 3.0
                ("jumps", 1.0, 1.4), # abs 4.0
                ("over", 2.0, 2.4),  # abs 5.0
                ("lazy", 4.0, 4.4),  # abs 7.0
            ],
        ),
    ]
    out = stitch(chunks, overlap_s=2.0)
    words = _flatten_words(out)
    starts = [w.start for w in words]
    assert starts == sorted(starts), "word starts must be monotonic"
    # Words from chunk 2 that survive the splice must have +3.0 offset.
    texts_to_starts = {w.text: w.start for w in words}
    assert texts_to_starts["over"] == pytest.approx(5.0)
    assert texts_to_starts["lazy"] == pytest.approx(7.0)


# --- adversarial cases ----------------------------------------------------


def test_adversarial_repeated_phrase_in_overlap() -> None:
    """Overlap text 'and then' appears twice in the ground truth.

    Ground truth: 'we ran and then we jumped and then we fell'.
    Chunk 1 covers 'we ran and then we' (abs 0..5). Chunk 2 covers
    'then we jumped and then we fell' (abs 3..10, 2s overlap).

    Overlap region (abs 3..5): 'and then we' from chunk 1 (tail) vs
    'then we' + next word from chunk 2 (head). A naive concatenation
    would either duplicate 'and then we' or drop it; the stitcher must
    preserve the ground truth exactly.
    """
    from src.core.stitcher import stitch

    chunks = [
        _chunk(
            0.0, 5.0,
            [
                ("we", 0.0, 0.4),
                ("ran", 1.0, 1.4),
                ("and", 2.0, 2.4),
                ("then", 3.0, 3.4),  # overlap starts
                ("we", 4.0, 4.4),    # overlap continues
            ],
        ),
        _chunk(
            3.0, 7.0,
            [
                ("then", 0.0, 0.4),   # abs 3.0 — overlap
                ("we", 1.0, 1.4),     # abs 4.0 — overlap
                ("jumped", 2.0, 2.4), # abs 5.0
                ("and", 3.0, 3.4),    # abs 6.0
                ("then", 4.0, 4.4),   # abs 7.0
                ("we", 5.0, 5.4),     # abs 8.0
                ("fell", 6.0, 6.4),   # abs 9.0
            ],
        ),
    ]
    out = stitch(chunks, overlap_s=2.0)
    words = _flatten_words(out)
    assert [w.text for w in words] == [
        "we", "ran", "and", "then", "we", "jumped", "and", "then", "we", "fell",
    ]


def test_adversarial_mismatch_with_small_overlap_match() -> None:
    """Chunk 2's head only partially matches chunk 1's tail.

    Chunk 1 tail: 'brown fox'. Chunk 2 head: 'fox jumps'. They share the
    single word 'fox' — SequenceMatcher should splice after 'fox' without
    dropping or duplicating it.
    """
    from src.core.stitcher import stitch

    chunks = [
        _chunk(
            0.0, 5.0,
            [
                ("the", 0.0, 0.4),
                ("quick", 1.0, 1.4),
                ("brown", 3.0, 3.4),  # overlap region: abs 3.0-5.0
                ("fox", 4.0, 4.4),
            ],
        ),
        _chunk(
            3.0, 5.0,
            [
                ("fox", 1.0, 1.4),    # abs 4.0 — matches chunk1 tail
                ("jumps", 2.0, 2.4),  # abs 5.0
                ("over", 3.0, 3.4),   # abs 6.0
            ],
        ),
    ]
    out = stitch(chunks, overlap_s=2.0)
    words = _flatten_words(out)
    assert [w.text for w in words] == [
        "the", "quick", "brown", "fox", "jumps", "over",
    ]


# --- low-confidence logging -----------------------------------------------


def test_low_confidence_splice_emits_warning(caplog) -> None:
    """When overlap texts don't match well, the stitcher must log a warning.

    The caller (UI log panel, SPEC §3.8) surfaces these so users know
    stitching may have guessed at the boundary.
    """
    from src.core.stitcher import stitch

    caplog.set_level(logging.WARNING, logger="src.core.stitcher")

    chunks = [
        _chunk(
            0.0, 5.0,
            [
                ("the", 0.0, 0.4),
                ("quick", 1.0, 1.4),
                ("brown", 2.0, 2.4),
                ("fox", 3.0, 3.4),   # overlap tail
                ("jumps", 4.0, 4.4), # overlap tail
            ],
        ),
        _chunk(
            3.0, 5.0,
            # Overlap head is completely different words — alignment will
            # be poor.
            [
                ("elephant", 0.0, 0.4), # abs 3.0
                ("walks", 1.0, 1.4),    # abs 4.0
                ("over", 2.0, 2.4),     # abs 5.0
                ("the", 3.0, 3.4),      # abs 6.0
                ("lazy", 4.0, 4.4),     # abs 7.0
            ],
        ),
    ]
    stitch(chunks, overlap_s=2.0)

    warnings = [
        r for r in caplog.records
        if r.name == "src.core.stitcher" and r.levelno >= logging.WARNING
    ]
    assert warnings, "expected at least one low-confidence warning"


def test_high_confidence_splice_does_not_warn(caplog) -> None:
    from src.core.stitcher import stitch

    caplog.set_level(logging.WARNING, logger="src.core.stitcher")

    chunks = [
        _chunk(
            0.0, 5.0,
            [
                ("the", 0.0, 0.4),
                ("quick", 1.0, 1.4),
                ("brown", 2.0, 2.4),
                ("fox", 3.0, 3.4),
                ("jumps", 4.0, 4.4),
            ],
        ),
        _chunk(
            3.0, 5.0,
            [
                ("fox", 0.0, 0.4),
                ("jumps", 1.0, 1.4),
                ("over", 2.0, 2.4),
                ("the", 3.0, 3.4),
                ("lazy", 4.0, 4.4),
            ],
        ),
    ]
    stitch(chunks, overlap_s=2.0)

    warnings = [
        r for r in caplog.records
        if r.name == "src.core.stitcher" and r.levelno >= logging.WARNING
    ]
    assert not warnings, f"unexpected warning(s): {[r.message for r in warnings]}"


# --- no dropped / duplicated words on happy path --------------------------


def test_no_word_appears_at_overlap_timestamp_twice() -> None:
    """Positional check: every unique (text, start) pair must be unique."""
    from src.core.stitcher import stitch

    chunks = [
        _chunk(
            0.0, 5.0,
            [
                ("the", 0.0, 0.4),
                ("quick", 1.0, 1.4),
                ("brown", 2.0, 2.4),
                ("fox", 3.0, 3.4),
                ("jumps", 4.0, 4.4),
            ],
        ),
        _chunk(
            3.0, 5.0,
            [
                ("fox", 0.0, 0.4),
                ("jumps", 1.0, 1.4),
                ("over", 2.0, 2.4),
                ("the", 3.0, 3.4),
                ("lazy", 4.0, 4.4),
            ],
        ),
    ]
    out = stitch(chunks, overlap_s=2.0)
    words = _flatten_words(out)
    positions = [(w.text, round(w.start, 3)) for w in words]
    assert len(positions) == len(set(positions)), (
        f"duplicate (text, start) positions found: {positions}"
    )


def test_three_chunk_stitch_preserves_ground_truth() -> None:
    """Three chunks with overlaps — cumulative splicing must not drift."""
    from src.core.stitcher import stitch

    chunks = [
        _chunk(
            0.0, 5.0,
            [
                ("one", 0.0, 0.4),
                ("two", 1.0, 1.4),
                ("three", 2.0, 2.4),
                ("four", 3.0, 3.4),
                ("five", 4.0, 4.4),
            ],
        ),
        _chunk(
            3.0, 5.0,
            [
                ("four", 0.0, 0.4),   # overlap: abs 3.0
                ("five", 1.0, 1.4),   # overlap: abs 4.0
                ("six", 2.0, 2.4),
                ("seven", 3.0, 3.4),
                ("eight", 4.0, 4.4),
            ],
        ),
        _chunk(
            6.0, 5.0,
            [
                ("seven", 0.0, 0.4),  # overlap: abs 6.0
                ("eight", 1.0, 1.4),  # overlap: abs 7.0
                ("nine", 2.0, 2.4),
                ("ten", 3.0, 3.4),
            ],
        ),
    ]
    out = stitch(chunks, overlap_s=2.0)
    words = _flatten_words(out)
    assert [w.text for w in words] == [
        "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten",
    ]
