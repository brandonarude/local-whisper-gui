"""Transcription orchestration on a background QThread (SPEC §3.8, §5.4).

The worker coordinates chunking + per-chunk faster-whisper transcription
without blocking the GUI. Chunks are exported to temp WAV files (the
:class:`~src.core.transcriber.Transcriber` takes a path, not an
``AudioSegment``) inside a :class:`tempfile.TemporaryDirectory` scoped to
the run, so the filesystem is clean whether we complete, fail, or cancel.

Signal order for a successful run:

    chunk_started → segment* → chunk_completed  (repeat per chunk)
    → completed → QThread.finished

On cancellation the current in-flight chunk is dropped from ``results``
(the user asked for *completed* chunks only — partial transcripts inside
an interrupted chunk are not "completed"); ``cancelled`` is emitted with
the list of fully-completed :class:`~src.core.stitcher.ChunkResult`
instances so the caller can offer partial export (SPEC §5.4).

When chunking is disabled the whole file is treated as a single chunk,
keeping the signal protocol uniform for downstream consumers.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.audio_processor import chunk_audio
from src.core.stitcher import ChunkResult
from src.core.transcriber import Transcriber


@dataclass(frozen=True)
class ChunkParams:
    min_silence_ms: int
    silence_thresh_db: float
    min_chunk_s: float
    max_chunk_s: float
    overlap_s: float


class TranscriptionWorker(QThread):
    chunk_started = pyqtSignal(int, int)          # chunk_index, total_chunks
    segment = pyqtSignal(int, object)             # chunk_index, Segment
    chunk_completed = pyqtSignal(int)             # chunk_index
    progress = pyqtSignal(int, float, float)      # percent, audio_done_s, wall_elapsed_s
    log = pyqtSignal(str)
    completed = pyqtSignal(list)                  # list[ChunkResult]
    cancelled = pyqtSignal(list)                  # list[ChunkResult] (partial)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        audio_path: str | Path,
        total_duration_s: float,
        transcriber: Transcriber,
        chunking_enabled: bool,
        chunk_params: ChunkParams,
        language: str,
        word_timestamps: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._audio_path = Path(audio_path)
        self._total_duration_s = max(float(total_duration_s), 0.0)
        self._transcriber = transcriber
        self._chunking_enabled = chunking_enabled
        self._chunk_params = chunk_params
        self._language = language
        self._word_timestamps = word_timestamps

    # --- cancellation ---------------------------------------------------

    def _should_cancel(self) -> bool:
        return self.isInterruptionRequested()

    # --- main loop ------------------------------------------------------

    def run(self) -> None:
        start_wall = time.monotonic()
        results: list[ChunkResult] = []
        try:
            plan = self._plan_chunks()
            if self._should_cancel():
                self.cancelled.emit(results)
                return

            total_chunks = len(plan)
            audio_done = 0.0

            with tempfile.TemporaryDirectory(prefix="lwg_chunks_") as td:
                td_path = Path(td)
                for idx, (start_s, duration_s, exporter) in enumerate(plan):
                    if self._should_cancel():
                        self.cancelled.emit(results)
                        return

                    self.chunk_started.emit(idx, total_chunks)
                    self.log.emit(f"Chunk {idx + 1}/{total_chunks} started")

                    chunk_path = exporter(td_path, idx)

                    segs = self._transcriber.transcribe(
                        chunk_path,
                        language=self._language,
                        word_timestamps=self._word_timestamps,
                        should_cancel=self._should_cancel,
                    )
                    for s in segs:
                        self.segment.emit(idx, s)

                    if self._should_cancel():
                        # Drop partial chunk: user asked for completed chunks only.
                        self.cancelled.emit(results)
                        return

                    results.append(
                        ChunkResult(
                            start_s=start_s,
                            duration_s=duration_s,
                            segments=list(segs),
                        )
                    )

                    audio_done += duration_s
                    elapsed = time.monotonic() - start_wall
                    pct = (
                        int(min(100, round(100 * audio_done / self._total_duration_s)))
                        if self._total_duration_s > 0
                        else 100
                    )
                    self.progress.emit(pct, audio_done, elapsed)
                    self.chunk_completed.emit(idx)
                    self.log.emit(f"Chunk {idx + 1}/{total_chunks} completed")

            self.completed.emit(results)
        except Exception as exc:  # pragma: no cover - surfaced via `failed`
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    # --- planning -------------------------------------------------------

    def _plan_chunks(self):
        """Return a list of ``(start_s, duration_s, exporter)`` triples.

        ``exporter(tmp_dir, idx)`` writes the chunk audio to a wav file in
        ``tmp_dir`` and returns its path. Deferring the export lets us skip
        it altogether for a cancellation between chunks.
        """
        if self._chunking_enabled:
            self.log.emit("Splitting audio into chunks...")
            chunks = chunk_audio(
                self._audio_path,
                min_silence_ms=self._chunk_params.min_silence_ms,
                silence_thresh_db=self._chunk_params.silence_thresh_db,
                min_chunk_s=self._chunk_params.min_chunk_s,
                max_chunk_s=self._chunk_params.max_chunk_s,
                overlap_s=self._chunk_params.overlap_s,
            )

            def make_exporter(chunk):
                def export(td: Path, idx: int) -> Path:
                    out = td / f"chunk_{idx:03d}.wav"
                    chunk.audio.export(str(out), format="wav")
                    return out

                return export

            return [
                (c.start_s, c.end_s - c.start_s, make_exporter(c)) for c in chunks
            ]

        # Whole-file path — skip pydub entirely and hand the original file
        # straight to the transcriber.
        def export_whole(_td: Path, _idx: int) -> Path:
            return self._audio_path

        return [(0.0, self._total_duration_s, export_whole)]
