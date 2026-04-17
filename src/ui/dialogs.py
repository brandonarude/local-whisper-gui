"""Shared modal dialogs (SPEC §3.3, §5.4, §7).

These are thin, testable wrappers over :class:`QMessageBox` rather than
ad-hoc call sites scattered across the UI — consolidating them here means
the wording, button layout, and return-type contract stay consistent and
can be swapped (e.g., to a custom widget) in one place later.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QCheckBox, QMessageBox, QWidget


def prompt_chunk_long_file(
    parent: QWidget | None, duration_hours: float
) -> bool:
    """Suggest chunking for files longer than ~1h (SPEC §3.3).

    Returns True if the user wants to enable chunking, False to keep the
    file as-is.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Long audio file")
    box.setText(
        f"This file is approximately {duration_hours:.1f} hours long. "
        "Splitting it into chunks is recommended for reliability and accuracy."
    )
    box.setInformativeText("Enable chunking for this transcription?")
    yes = box.addButton("Enable chunking", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Keep as one file", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(yes)
    box.exec()
    return box.clickedButton() is yes


def show_error(
    parent: QWidget | None, title: str, message: str, details: str | None = None
) -> None:
    """Error dialog with an optional details-on-demand expander."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(title)
    box.setText(message)
    if details:
        box.setDetailedText(details)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()


def prompt_partial_output_on_cancel(
    parent: QWidget | None, completed_chunks: int, total_chunks: int
) -> bool:
    """After a cancel, ask whether to export the partial result (SPEC §5.4)."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Transcription cancelled")
    box.setText(
        f"Transcription was cancelled after {completed_chunks} of {total_chunks} "
        "chunk(s) completed."
    )
    box.setInformativeText("Export the partial transcript?")
    yes = box.addButton("Export partial", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Discard", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(yes)
    box.exec()
    return box.clickedButton() is yes


def show_missing_ffmpeg(parent: QWidget | None) -> None:
    """Startup warning when ffmpeg isn't on PATH (SPEC §7)."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("ffmpeg not found")
    box.setText(
        "ffmpeg is required to decode audio files but was not found on PATH."
    )
    box.setInformativeText(
        "Install ffmpeg (e.g., `sudo apt install ffmpeg` on Debian/Ubuntu, "
        "`brew install ffmpeg` on macOS, or download from "
        "https://ffmpeg.org/download.html on Windows) and relaunch the app."
    )
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()


def show_missing_faster_whisper(parent: QWidget | None) -> None:
    """Fatal: faster-whisper not importable (SPEC §5.2)."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle("faster-whisper not installed")
    box.setText(
        "faster-whisper is required for transcription but is not installed."
    )
    box.setInformativeText(
        "Install it into the app's Python environment with "
        "`pip install faster-whisper`, then relaunch."
    )
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()


def show_startup_warning(
    parent: QWidget | None,
    *,
    title: str,
    message: str,
    detail: str = "",
    allow_suppress: bool = True,
) -> bool:
    """Non-fatal startup warning with an optional "Don't show again" checkbox.

    Returns True if the user asked to suppress this warning in future
    launches, False otherwise. When ``allow_suppress`` is False the
    checkbox is omitted and the return value is always False.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(message)
    if detail:
        box.setInformativeText(detail)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)

    suppress_cb: QCheckBox | None = None
    if allow_suppress:
        suppress_cb = QCheckBox("Don't show this again")
        box.setCheckBox(suppress_cb)

    box.exec()
    return bool(suppress_cb is not None and suppress_cb.isChecked())


def describe_load_error(path: str | Path, exc: BaseException) -> str:
    """Format a consistent 'failed to load X' message used by the main window."""
    return f"Could not load {Path(path).name}: {type(exc).__name__}: {exc}"


def show_export_complete(
    parent: QWidget | None,
    output_dir: str | Path,
    files: list[str | Path],
) -> None:
    """Success notification with an "Open folder" button (SPEC §5.4)."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Information)
    box.setWindowTitle("Transcription complete")
    file_list = "\n".join(f"• {Path(f).name}" for f in files)
    box.setText(f"Wrote {len(files)} file(s) to:\n{output_dir}")
    if file_list:
        box.setInformativeText(file_list)
    open_btn = box.addButton("Open folder", QMessageBox.ButtonRole.ActionRole)
    box.addButton(QMessageBox.StandardButton.Ok)
    box.exec()
    if box.clickedButton() is open_btn:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir)))
