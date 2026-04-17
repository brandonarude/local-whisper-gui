"""Background worker that triggers a faster-whisper model download (SPEC §8.3).

Instantiating :class:`faster_whisper.WhisperModel` is what actually fetches
the model from HuggingFace. There is no public progress hook, so the worker
just runs the instantiation on a QThread, emits ``completed`` when it
returns, and ``failed`` with a classified error kind on exception. The
calling dialog shows an indeterminate progress bar while the worker runs.
"""
from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from src.utils.errors import classify


class ModelDownloadWorker(QThread):
    completed = pyqtSignal()
    failed = pyqtSignal(str, str)  # message, ErrorKind.value

    def __init__(
        self,
        *,
        model: str,
        device: str = "cpu",
        compute_type: str = "int8",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._device = device
        self._compute_type = compute_type

    def run(self) -> None:  # pragma: no cover - network heavy
        try:
            from faster_whisper import WhisperModel

            WhisperModel(
                self._model,
                device=self._device,
                compute_type=self._compute_type,
            )
            self.completed.emit()
        except Exception as exc:
            kind = classify(exc)
            self.failed.emit(f"{type(exc).__name__}: {exc}", kind.value)
