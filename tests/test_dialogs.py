"""Smoke tests for src.ui.dialogs (SPEC §3.3, §5.4, §7)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PyQt6.QtWidgets import QMessageBox


def test_prompt_chunk_long_file_accept(qtbot) -> None:
    from src.ui import dialogs

    def fake_exec(self):
        return 0

    def fake_clicked(self):
        return next(
            b for b in self.buttons()
            if self.buttonRole(b) == QMessageBox.ButtonRole.AcceptRole
        )

    with patch.object(QMessageBox, "exec", fake_exec), patch.object(
        QMessageBox, "clickedButton", fake_clicked
    ):
        assert dialogs.prompt_chunk_long_file(None, 1.5) is True


def test_prompt_chunk_long_file_reject(qtbot) -> None:
    from src.ui import dialogs

    def fake_exec(self):
        return 0

    def fake_clicked(self):
        return next(
            b for b in self.buttons()
            if self.buttonRole(b) == QMessageBox.ButtonRole.RejectRole
        )

    with patch.object(QMessageBox, "exec", fake_exec), patch.object(
        QMessageBox, "clickedButton", fake_clicked
    ):
        assert dialogs.prompt_chunk_long_file(None, 2.0) is False


def test_prompt_partial_output_accept(qtbot) -> None:
    from src.ui import dialogs

    def fake_exec(self):
        return 0

    def fake_clicked(self):
        return next(
            b for b in self.buttons()
            if self.buttonRole(b) == QMessageBox.ButtonRole.AcceptRole
        )

    with patch.object(QMessageBox, "exec", fake_exec), patch.object(
        QMessageBox, "clickedButton", fake_clicked
    ):
        assert dialogs.prompt_partial_output_on_cancel(None, 3, 5) is True


def test_prompt_partial_output_reject(qtbot) -> None:
    from src.ui import dialogs

    def fake_exec(self):
        return 0

    def fake_clicked(self):
        return next(
            b for b in self.buttons()
            if self.buttonRole(b) == QMessageBox.ButtonRole.RejectRole
        )

    with patch.object(QMessageBox, "exec", fake_exec), patch.object(
        QMessageBox, "clickedButton", fake_clicked
    ):
        assert dialogs.prompt_partial_output_on_cancel(None, 0, 5) is False


def test_show_error_does_not_raise(qtbot) -> None:
    from src.ui import dialogs

    with patch.object(QMessageBox, "exec", return_value=0):
        dialogs.show_error(None, "Something broke", "details here", details="traceback…")


def test_show_missing_ffmpeg_does_not_raise(qtbot) -> None:
    from src.ui import dialogs

    with patch.object(QMessageBox, "exec", return_value=0):
        dialogs.show_missing_ffmpeg(None)


def test_describe_load_error_formats_message(tmp_path: Path) -> None:
    from src.ui.dialogs import describe_load_error

    msg = describe_load_error(tmp_path / "song.mp3", ValueError("bad header"))
    assert "song.mp3" in msg
    assert "ValueError" in msg
    assert "bad header" in msg
