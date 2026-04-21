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


def prompt_cadence_exceeds_duration(
    parent: QWidget | None, duration_s: float, cadence_s: int
) -> bool:
    """Warn that the .txt timestamp cadence is longer than the loaded audio.

    Returns True to proceed (the .txt will contain only a single ``[0:00]``
    marker); False to cancel so the user can shorten the cadence first.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Timestamp cadence longer than audio")
    box.setText(
        f"The timestamp cadence ({cadence_s}s) is longer than the loaded "
        f"audio ({duration_s:.1f}s)."
    )
    box.setInformativeText(
        "The .txt transcript will contain only a single timestamp at the "
        "start. Continue anyway?"
    )
    yes = box.addButton("Continue", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(yes)
    box.exec()
    return box.clickedButton() is yes


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


def prompt_oom_retry(parent: QWidget | None, detail: str = "") -> bool:
    """GPU ran out of memory (SPEC §7). Returns True if the user wants to
    retry with current settings; False if they'll adjust first."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle("Out of memory")
    box.setText(
        "Transcription ran out of GPU memory. Try a smaller model, "
        "switch to CPU, or close other GPU-using apps."
    )
    if detail:
        box.setDetailedText(detail)
    retry = box.addButton("Retry", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(retry)
    box.exec()
    return box.clickedButton() is retry


def prompt_download_retry(parent: QWidget | None, detail: str = "") -> bool:
    """Model download failed (SPEC §7). Returns True to retry."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Model download failed")
    box.setText(
        "The faster-whisper model could not be downloaded. Check your "
        "network connection and try again."
    )
    if detail:
        box.setDetailedText(detail)
    retry = box.addButton("Retry", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(retry)
    box.exec()
    return box.clickedButton() is retry


def _format_bytes(n: int) -> str:
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def prompt_low_disk_space(
    parent: QWidget | None,
    *,
    needed_bytes: int,
    free_bytes: int,
    output_dir: str | Path,
) -> bool:
    """Pre-export disk-space warning (SPEC §7). Returns True to proceed
    anyway, False to cancel the export."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Low disk space")
    box.setText(
        f"Output directory {Path(output_dir)} has {_format_bytes(free_bytes)} free, "
        f"but the export is estimated at {_format_bytes(needed_bytes)}."
    )
    box.setInformativeText("Proceed with the export anyway?")
    proceed = box.addButton("Export anyway", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(proceed)
    box.exec()
    return box.clickedButton() is proceed


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


def prompt_select_model_to_download(
    parent: QWidget | None, models: list[str], default: str | None = None
) -> str | None:
    """Let the user pick a model from the Whisper list to pre-download.
    Returns the selected name, or None if cancelled."""
    from PyQt6.QtWidgets import QInputDialog

    idx = 0
    if default is not None and default in models:
        idx = models.index(default)
    name, ok = QInputDialog.getItem(
        parent,
        "Pre-download Model",
        "Select a Whisper model to download now:",
        models,
        idx,
        editable=False,
    )
    return name if ok and name else None


def confirm_clear_model_cache(
    parent: QWidget | None, *, model_names: list[str], total_bytes: int
) -> bool:
    """Confirm deleting cached Whisper models. Returns True to proceed."""
    if not model_names:
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Clear Cached Models")
        box.setText("No cached Whisper models found.")
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()
        return False

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Clear Cached Models")
    box.setText(
        f"Delete {len(model_names)} cached Whisper model(s) "
        f"({_format_bytes(total_bytes)})?"
    )
    box.setInformativeText("\n".join(f"• {n}" for n in model_names))
    yes = box.addButton("Delete", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(yes)
    box.exec()
    return box.clickedButton() is yes


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
