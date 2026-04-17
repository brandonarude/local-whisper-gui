"""Background waveform generation (SPEC §3.2, §5.3).

Waveform peak extraction decodes the full audio file, which is cheap for a
tiny wav but can take seconds for hour-long inputs. Running it on the GUI
thread would block the event loop — this worker moves it onto a QThread so
the rest of the UI (metadata labels, settings, menu) stays responsive while
the render catches up.

Emits :attr:`samples_ready` with the :class:`~src.core.audio_processor.Waveform`
on success, or :attr:`failed` with the exception message on error. Consumers
connect to both and treat them as terminal.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.audio_processor import Waveform, generate_waveform_samples


class WaveformWorker(QThread):
    samples_ready = pyqtSignal(object)  # Waveform
    failed = pyqtSignal(str)

    def __init__(
        self,
        path: str | Path,
        target_points: int = 2000,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._path = Path(path)
        self._target_points = target_points

    def run(self) -> None:
        try:
            wf: Waveform = generate_waveform_samples(
                self._path, target_points=self._target_points
            )
        except Exception as exc:  # pragma: no cover - exercised via failed signal
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.samples_ready.emit(wf)
