"""Main window composition (SPEC §4, §9).

Stacks the audio header, waveform, settings panel, and progress panel
into the canonical top-to-bottom layout from the spec. A status bar
reports the current device and model (updated whenever the settings
panel fires ``values_changed``). Transcription wiring lands in commit 34;
until then the Start/Cancel signals are routed through the progress
panel but perform no work.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt

from src.core.audio_processor import AudioInfo
from src.ui.file_header import FileHeader
from src.ui.progress_panel import ProgressPanel
from src.ui.settings_panel import SettingsPanel
from src.ui.waveform_widget import WaveformWidget
from src.utils import constants as C
from src.utils.device_detect import Device, detect_devices
from src.workers.waveform_worker import WaveformWorker

APP_NAME = "Local Whisper GUI"


class MainWindow(QMainWindow):
    def __init__(self, devices: list[Device] | None = None) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1100, 820)

        self._devices = devices if devices is not None else detect_devices()

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._file_header = FileHeader(self)
        self._file_header.setObjectName("fileHeader")
        layout.addWidget(self._file_header)

        self._waveform = WaveformWidget(self)
        self._waveform.setObjectName("waveformWidget")
        self._waveform.setMinimumHeight(120)
        layout.addWidget(self._waveform)

        # Settings above progress, splittable so long logs can grow.
        splitter = QSplitter(Qt.Orientation.Vertical, self)
        self._settings_panel = SettingsPanel(devices=self._devices, parent=self)
        self._settings_panel.setObjectName("settingsPanel")
        splitter.addWidget(self._settings_panel)

        self._progress_panel = ProgressPanel(self)
        self._progress_panel.setObjectName("progressPanel")
        splitter.addWidget(self._progress_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)

        self.setCentralWidget(central)

        self._status = self.statusBar()
        self._refresh_status_bar()

        self._waveform_worker: WaveformWorker | None = None

        self._file_header.file_loaded.connect(self._on_file_loaded)
        self._file_header.load_failed.connect(self._on_load_failed)
        self._settings_panel.values_changed.connect(self._refresh_status_bar)

    # --- file loading ---------------------------------------------------

    def _on_file_loaded(self, info: AudioInfo) -> None:
        self._waveform.clear()
        self._progress_panel.set_file_loaded(True)
        self._progress_panel.reset_progress()
        self._progress_panel.append_log(f"Loaded {info.path.name} ({info.duration_s:.1f}s)")
        self._start_waveform_worker(info)

    def _on_load_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Could not load file", message)
        self._progress_panel.append_log(f"Error: {message}")

    def _start_waveform_worker(self, info: AudioInfo) -> None:
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

    # --- status bar -----------------------------------------------------

    def _device_label_for(self, device_key: str) -> str:
        for d in self._devices:
            key = "cpu" if d.kind == "cpu" else f"cuda:{d.index or 0}"
            if key == device_key:
                return d.label
        return device_key

    def _refresh_status_bar(self) -> None:
        v = self._settings_panel.values()
        model_label = next(
            (m for m in C.MODELS if m.name == v.model), None
        )
        model_text = (
            f"{v.model} (~{model_label.vram_hint_gb:.1f} GB)"
            if model_label is not None
            else v.model
        )
        self._status.showMessage(
            f"Device: {self._device_label_for(v.device_key)}  |  Model: {model_text}"
        )
