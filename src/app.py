"""QApplication + top-level window setup.

The main window so far holds the file-header row (Load button + metadata
labels) stacked above the waveform widget. Loading a file probes via
pydub, displays the metadata, and spawns a `WaveformWorker` to render the
waveform without blocking the UI. Later commits in `PLAN.md` plug in the
settings panel (27), progress panel (28), and the full main-window layout
(29).
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from src.core.audio_processor import AudioInfo
from src.ui.file_header import FileHeader
from src.ui.waveform_widget import WaveformWidget
from src.workers.waveform_worker import WaveformWorker

APP_NAME = "Local Whisper GUI"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(960, 720)

        central = QWidget(self)
        layout = QVBoxLayout(central)

        self._file_header = FileHeader(self)
        self._file_header.setObjectName("fileHeader")
        layout.addWidget(self._file_header)

        self._waveform = WaveformWidget(self)
        self._waveform.setObjectName("waveformWidget")
        layout.addWidget(self._waveform, stretch=1)

        self.setCentralWidget(central)

        self._waveform_worker: WaveformWorker | None = None

        self._file_header.file_loaded.connect(self._on_file_loaded)
        self._file_header.load_failed.connect(self._on_load_failed)

    def _on_file_loaded(self, info: AudioInfo) -> None:
        self._waveform.clear()
        self._start_waveform_worker(info)

    def _on_load_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Could not load file", message)

    def _start_waveform_worker(self, info: AudioInfo) -> None:
        # Replace any previous worker cleanly; .quit()+.wait() keeps the
        # thread from being torn down mid-run if the user loads a second
        # file before the first finishes.
        if self._waveform_worker is not None and self._waveform_worker.isRunning():
            self._waveform_worker.requestInterruption()
            self._waveform_worker.quit()
            self._waveform_worker.wait(2_000)

        worker = WaveformWorker(info.path, parent=self)
        worker.samples_ready.connect(self._waveform.set_samples)
        worker.failed.connect(self._on_load_failed)
        worker.finished.connect(worker.deleteLater)
        self._waveform_worker = worker
        worker.start()


def create_application(argv: list[str]) -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    return app
