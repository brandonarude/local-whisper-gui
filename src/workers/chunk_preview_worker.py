"""Background chunk-boundary preview (SPEC §3.3).

Running the silence-based chunker on an hour-long wav takes real time
(pydub has to decode the whole file), so it has to live off the GUI
thread. This worker emits only the *boundary times* (start of each
chunk after the first) rather than the full :class:`~src.core.audio_processor.Chunk`
list — the waveform widget draws lines, not audio, and shipping
:class:`pydub.AudioSegment` across thread boundaries is expensive and
pointless.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.audio_processor import chunk_audio


class ChunkPreviewWorker(QThread):
    boundaries_ready = pyqtSignal(list)  # list[float] in seconds
    failed = pyqtSignal(str)

    def __init__(
        self,
        path: str | Path,
        *,
        min_silence_ms: int,
        silence_thresh_db: float,
        min_chunk_s: float,
        max_chunk_s: float,
        overlap_s: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._path = Path(path)
        self._min_silence_ms = min_silence_ms
        self._silence_thresh_db = silence_thresh_db
        self._min_chunk_s = min_chunk_s
        self._max_chunk_s = max_chunk_s
        self._overlap_s = overlap_s

    def run(self) -> None:
        try:
            chunks = chunk_audio(
                self._path,
                min_silence_ms=self._min_silence_ms,
                silence_thresh_db=self._silence_thresh_db,
                min_chunk_s=self._min_chunk_s,
                max_chunk_s=self._max_chunk_s,
                overlap_s=self._overlap_s,
            )
        except Exception as exc:  # pragma: no cover - surfaced via `failed`
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return

        # Boundaries = starts of chunks after the first. Sorted and
        # deduplicated for defensiveness — upstream already gives us
        # monotonic starts.
        boundaries = sorted({c.start_s for c in chunks[1:]})
        self.boundaries_ready.emit(list(boundaries))
