"""QApplication + top-level window setup.

The window is intentionally empty at this stage; later commits in `PLAN.md`
will assemble the real layout (audio/waveform/settings/progress panels) in
`src/ui/main_window.py` and route it through here.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QApplication, QMainWindow

APP_NAME = "Local Whisper GUI"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(960, 720)


def create_application(argv: list[str]) -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    return app
