"""QApplication factory and startup orchestration.

Main-window composition lives in :mod:`src.ui.main_window`; this module
handles the steps that need to run before the window is shown — in
particular the §5.2 startup checks (ffmpeg, faster-whisper, CUDA).

Failing checks surface as dialogs and are logged, but never block the
main window from opening. The transcription flow (SPEC §5.4) is the
point where a missing backend actually stops work, and the user is
better served by seeing the app state than by a cold exit.
"""
from __future__ import annotations

from typing import Callable, Iterable

from PyQt6.QtWidgets import QApplication, QWidget

from src.ui import dialogs
from src.ui.main_window import APP_NAME, MainWindow
from src.utils import startup_checks
from src.utils.config import Config
from src.utils.startup_checks import CheckResult

__all__ = [
    "APP_NAME",
    "MainWindow",
    "create_application",
    "handle_startup_checks",
]

# Non-fatal probes whose warning is suppressible via a "Don't show again"
# checkbox. Fatal backend failures (ffmpeg, faster-whisper) are always
# re-shown on launch — the app can't transcribe without them, so a sticky
# reminder is the point.
_SUPPRESSIBLE_WARNINGS = frozenset({"cuda"})


def create_application(argv: list[str]) -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    return app


def handle_startup_checks(
    parent: QWidget | None,
    *,
    config: Config | None = None,
    results: Iterable[CheckResult] | None = None,
    check_runner: Callable[[], list[CheckResult]] = startup_checks.run_startup_checks,
) -> list[CheckResult]:
    """Run the startup probes and present any failures/warnings.

    Fatal failures (ffmpeg missing, faster-whisper missing) show a modal
    dialog every launch. Non-fatal warnings (CUDA absent) show a dialog
    with a "Don't show again" checkbox that persists through ``config``.

    The window is opened regardless; returned results let callers (or
    the main window) adapt their UI (e.g., disable Start when a fatal
    backend is missing).
    """
    cfg = config if config is not None else Config()
    result_list = list(results) if results is not None else check_runner()

    for result in result_list:
        if result.ok:
            if result.severity == "warning":
                _show_non_fatal(parent, cfg, result)
            continue
        _show_fatal(parent, result)

    return result_list


def _show_fatal(parent: QWidget | None, result: CheckResult) -> None:
    if result.name == "ffmpeg":
        dialogs.show_missing_ffmpeg(parent)
    elif result.name == "faster_whisper":
        dialogs.show_missing_faster_whisper(parent)
    else:
        dialogs.show_error(
            parent,
            title=f"Startup check failed: {result.name}",
            message=result.message,
            details=result.detail or None,
        )


def _show_non_fatal(parent: QWidget | None, cfg: Config, result: CheckResult) -> None:
    suppressible = result.name in _SUPPRESSIBLE_WARNINGS
    if suppressible and cfg.is_warning_suppressed(result.name):
        return
    title = _warning_title_for(result.name)
    suppress = dialogs.show_startup_warning(
        parent,
        title=title,
        message=result.message,
        detail=result.detail,
        allow_suppress=suppressible,
    )
    if suppress and suppressible:
        cfg.set_warning_suppressed(result.name, True)


def _warning_title_for(name: str) -> str:
    return {
        "cuda": "CUDA not available",
    }.get(name, f"Startup warning: {name}")
