"""QApplication factory; main-window composition lives in :mod:`src.ui.main_window`."""
from __future__ import annotations

from PyQt6.QtWidgets import QApplication

from src.ui.main_window import APP_NAME, MainWindow

__all__ = ["APP_NAME", "MainWindow", "create_application"]


def create_application(argv: list[str]) -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    return app
