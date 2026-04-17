"""Audio file header: Load button + metadata readout (SPEC §3.1, §4).

Owns only the header row UI (button + labels) and exposes
:attr:`file_loaded` as a `pyqtSignal[AudioInfo]` when the user picks a file
that successfully probes. The main window wires that into the waveform
worker and the rest of the pipeline; keeping the probe here means an
unreadable file surfaces an error before any downstream widgets see it.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
)

from src.core.audio_processor import AudioInfo, probe


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1 << 30:
        return f"{size_bytes / (1 << 30):.2f} GB"
    if size_bytes >= 1 << 20:
        return f"{size_bytes / (1 << 20):.1f} MB"
    if size_bytes >= 1 << 10:
        return f"{size_bytes / (1 << 10):.1f} KB"
    return f"{size_bytes} B"


class FileHeader(QWidget):
    file_loaded = pyqtSignal(object)  # AudioInfo
    load_failed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._button = QPushButton("Load Audio File")
        self._button.setObjectName("loadAudioButton")
        self._button.clicked.connect(self._on_load_clicked)
        layout.addWidget(self._button)

        self._filename_label = QLabel("No file loaded")
        self._filename_label.setObjectName("filenameLabel")
        layout.addWidget(self._filename_label)

        layout.addStretch(1)

        self._metadata_label = QLabel("")
        self._metadata_label.setObjectName("metadataLabel")
        layout.addWidget(self._metadata_label)

        self._info: AudioInfo | None = None

    @property
    def current_info(self) -> AudioInfo | None:
        return self._info

    def load_path(self, path: str | Path) -> AudioInfo | None:
        try:
            info = probe(path)
        except (FileNotFoundError, ValueError) as exc:
            message = f"Could not load {Path(path).name}: {exc}"
            self.load_failed.emit(message)
            return None

        self._info = info
        self._filename_label.setText(info.path.name)
        self._metadata_label.setText(
            " | ".join(
                [
                    _format_duration(info.duration_s),
                    f"{info.sample_rate / 1000:.1f} kHz",
                    f"{info.channels} ch",
                    _format_size(info.size_bytes),
                ]
            )
        )
        self.file_loaded.emit(info)
        return info

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            "",
            "Audio files (*.mp3 *.wav *.flac *.m4a *.ogg *.opus *.wma *.aac *.webm);;All files (*)",
        )
        if not path:
            return
        info = self.load_path(path)
        if info is None:
            # load_failed has already fired for any listeners; also surface
            # the error directly for users with no main-window wiring yet.
            QMessageBox.warning(
                self,
                "Could not load file",
                f"{Path(path).name} is not a recognised audio file.",
            )
