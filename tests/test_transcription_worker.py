"""Smoke tests for TranscriptionWorker (SPEC §3.8, §5.4).

The worker runs on a QThread and talks to a real
:class:`~src.core.transcriber.Transcriber`, but we inject a fake whose
``transcribe`` returns pre-canned :class:`~src.core.exporter.Segment`
lists. That keeps faster-whisper out of the test entirely and lets us
assert the signal protocol (``chunk_started → segment → chunk_completed
→ completed``) directly.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from src.core.exporter import Segment, Word
from src.workers.transcription_worker import ChunkParams, TranscriptionWorker


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def long_wav_with_silences(tmp_path: Path) -> Path:
    """22-second mono wav with clear silence gaps (see test_chunk_preview)."""
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


class FakeTranscriber:
    """Returns a fixed segment list every call; records each invocation."""

    def __init__(self, segments_per_call):
        self._batches = list(segments_per_call)
        self.calls: list[dict] = []

    def transcribe(
        self,
        audio_path,
        *,
        language=None,
        word_timestamps=True,
        should_cancel=None,
    ):
        self.calls.append(
            {
                "path": Path(audio_path),
                "language": language,
                "word_timestamps": word_timestamps,
                "should_cancel": should_cancel,
            }
        )
        if self._batches:
            return self._batches.pop(0)
        return []


def _seg(start: float, end: float, text: str) -> Segment:
    return Segment(
        start=start,
        end=end,
        text=text,
        words=(Word(start=start, end=end, text=text, probability=0.99),),
    )


def _default_chunk_params() -> ChunkParams:
    return ChunkParams(
        min_silence_ms=500,
        silence_thresh_db=-40.0,
        min_chunk_s=1.0,
        max_chunk_s=30.0,
        overlap_s=1.0,
    )


# --- whole-file path (no chunking) ----------------------------------------


def test_whole_file_signal_order_and_completion(qtbot, tiny_wav: Path) -> None:
    transcriber = FakeTranscriber(
        [[_seg(0.0, 0.5, "hello"), _seg(0.5, 1.0, "world")]]
    )
    worker = TranscriptionWorker(
        audio_path=tiny_wav,
        total_duration_s=1.0,
        transcriber=transcriber,
        chunking_enabled=False,
        chunk_params=_default_chunk_params(),
        language="en",
    )

    received: list[tuple[str, object]] = []
    worker.chunk_started.connect(lambda i, n: received.append(("chunk_started", (i, n))))
    worker.segment.connect(lambda i, s: received.append(("segment", (i, s.text))))
    worker.chunk_completed.connect(lambda i: received.append(("chunk_completed", i)))
    worker.completed.connect(lambda r: received.append(("completed", len(r))))

    with qtbot.waitSignal(worker.completed, timeout=10_000):
        worker.start()
    worker.wait(5_000)

    names = [name for name, _ in received]
    assert names == [
        "chunk_started",
        "segment",
        "segment",
        "chunk_completed",
        "completed",
    ]
    # Transcriber was called with the whole file path and the language.
    assert len(transcriber.calls) == 1
    assert transcriber.calls[0]["path"] == tiny_wav
    assert transcriber.calls[0]["language"] == "en"


# --- chunked path ---------------------------------------------------------


def test_chunked_run_emits_per_chunk_signals(
    qtbot, long_wav_with_silences: Path
) -> None:
    transcriber = FakeTranscriber(
        [
            [_seg(0.0, 1.0, "a")],
            [_seg(0.0, 1.0, "b")],
            [_seg(0.0, 1.0, "c")],
        ]
    )
    worker = TranscriptionWorker(
        audio_path=long_wav_with_silences,
        total_duration_s=22.0,
        transcriber=transcriber,
        chunking_enabled=True,
        chunk_params=ChunkParams(
            min_silence_ms=500,
            silence_thresh_db=-40.0,
            min_chunk_s=1.0,
            max_chunk_s=10.0,
            overlap_s=1.0,
        ),
        language="en",
    )

    starts: list[tuple[int, int]] = []
    completes: list[int] = []
    worker.chunk_started.connect(lambda i, n: starts.append((i, n)))
    worker.chunk_completed.connect(lambda i: completes.append(i))

    with qtbot.waitSignal(worker.completed, timeout=20_000) as blocker:
        worker.start()
    worker.wait(5_000)

    results = blocker.args[0]
    assert len(results) == 3
    assert [i for i, _ in starts] == [0, 1, 2]
    # All three totals match the chunk count.
    assert {n for _, n in starts} == {3}
    assert completes == [0, 1, 2]
    # Each chunk got its own transcribe call with a distinct temp wav path.
    assert len(transcriber.calls) == 3
    paths = [c["path"] for c in transcriber.calls]
    assert len({p for p in paths}) == 3


# --- cancellation ---------------------------------------------------------


def test_cancel_between_chunks_preserves_completed_results(
    qtbot, long_wav_with_silences: Path
) -> None:
    """Interruption requested after chunk 0 completes → results has chunk 0,
    worker emits `cancelled` and does not emit `completed`."""
    transcriber = FakeTranscriber(
        [
            [_seg(0.0, 1.0, "a")],
            [_seg(0.0, 1.0, "b")],
            [_seg(0.0, 1.0, "c")],
        ]
    )
    worker = TranscriptionWorker(
        audio_path=long_wav_with_silences,
        total_duration_s=22.0,
        transcriber=transcriber,
        chunking_enabled=True,
        chunk_params=ChunkParams(
            min_silence_ms=500,
            silence_thresh_db=-40.0,
            min_chunk_s=1.0,
            max_chunk_s=10.0,
            overlap_s=1.0,
        ),
        language="en",
    )

    # Request interruption as soon as the first chunk finishes.
    worker.chunk_completed.connect(lambda _i: worker.requestInterruption())

    completed_hits: list[object] = []
    worker.completed.connect(lambda r: completed_hits.append(r))

    with qtbot.waitSignal(worker.cancelled, timeout=20_000) as blocker:
        worker.start()
    worker.wait(5_000)

    partial = blocker.args[0]
    assert len(partial) == 1
    assert partial[0].segments[0].text == "a"
    # `completed` must not fire when we cancelled.
    assert completed_hits == []
