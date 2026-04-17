"""Transcription progress panel (SPEC §3.8, §4).

Widget skeleton — Start/Cancel buttons, an overall progress bar, an ETA
label, and a read-only status log. The panel stays dumb: it exposes
:attr:`start_clicked` and :attr:`cancel_clicked` signals plus setter
methods, and lets the transcription worker (commit 34) drive them. The
Start button is disabled until a file is loaded and until not already
transcribing; Cancel mirrors that (enabled only while running).
"""
from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class TranscriptionState(str, Enum):
    IDLE_NO_FILE = "idle_no_file"
    READY = "ready"
    RUNNING = "running"


class ProgressPanel(QWidget):
    start_clicked = pyqtSignal()
    cancel_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        button_row = QHBoxLayout()
        self._start_button = QPushButton("▶ Start Transcription")
        self._start_button.setObjectName("startButton")
        self._start_button.clicked.connect(self.start_clicked)
        button_row.addWidget(self._start_button)

        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setObjectName("cancelButton")
        self._cancel_button.clicked.connect(self.cancel_clicked)
        button_row.addWidget(self._cancel_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("overallProgressBar")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p%")
        layout.addWidget(self._progress_bar)

        self._eta_label = QLabel("ETA: —")
        self._eta_label.setObjectName("etaLabel")
        layout.addWidget(self._eta_label)

        self._log = QTextEdit()
        self._log.setObjectName("transcriptionLog")
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Transcription status messages appear here.")
        layout.addWidget(self._log, stretch=1)

        self._state = TranscriptionState.IDLE_NO_FILE
        self._apply_state()

    # --- state ----------------------------------------------------------

    def set_file_loaded(self, loaded: bool) -> None:
        if self._state is TranscriptionState.RUNNING:
            return
        self._state = (
            TranscriptionState.READY if loaded else TranscriptionState.IDLE_NO_FILE
        )
        self._apply_state()

    def set_running(self, running: bool) -> None:
        if running:
            self._state = TranscriptionState.RUNNING
        else:
            # Coming out of RUNNING: buttons wait for the main window to
            # call set_file_loaded again (the file is still there, but the
            # loaded-gate is the caller's job).
            self._state = TranscriptionState.READY
        self._apply_state()

    def state(self) -> TranscriptionState:
        return self._state

    # --- progress -------------------------------------------------------

    def set_progress_percent(self, pct: int) -> None:
        pct = max(0, min(100, int(pct)))
        self._progress_bar.setValue(pct)

    def set_progress_text(self, text: str) -> None:
        self._progress_bar.setFormat(text)

    def set_eta(self, eta_text: str) -> None:
        self._eta_label.setText(f"ETA: {eta_text}")

    def reset_progress(self) -> None:
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p%")
        self._eta_label.setText("ETA: —")

    # --- log ------------------------------------------------------------

    def append_log(self, message: str) -> None:
        self._log.append(message)
        # Keep newest line visible when the widget has focus or is hidden.
        self._log.moveCursor(QTextCursor.MoveOperation.End)

    def clear_log(self) -> None:
        self._log.clear()

    # --- internals ------------------------------------------------------

    def _apply_state(self) -> None:
        if self._state is TranscriptionState.IDLE_NO_FILE:
            self._start_button.setEnabled(False)
            self._cancel_button.setEnabled(False)
        elif self._state is TranscriptionState.READY:
            self._start_button.setEnabled(True)
            self._cancel_button.setEnabled(False)
        else:  # RUNNING
            self._start_button.setEnabled(False)
            self._cancel_button.setEnabled(True)
