"""Smoke tests for the Settings menu model cache actions (SPEC §5, §8.3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.device_detect import Device


def _devices() -> list[Device]:
    return [Device(kind="cpu", name="CPU")]


def test_clear_cache_cancel_does_not_delete(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    (tmp_path / "models--Systran--faster-whisper-small").mkdir()
    (tmp_path / "models--Systran--faster-whisper-small" / "f").write_bytes(b"x" * 100)

    monkeypatch.setattr(mw_mod.model_cache, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        mw_mod.dialogs, "confirm_clear_model_cache",
        lambda parent, *, model_names, total_bytes: False,
    )

    w = MainWindow(devices=_devices())
    qtbot.addWidget(w)
    w._on_clear_model_cache()

    assert (tmp_path / "models--Systran--faster-whisper-small").exists()


def test_clear_cache_confirm_deletes_whisper_dirs(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PyQt6.QtWidgets import QMessageBox

    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    whisper = tmp_path / "models--Systran--faster-whisper-small"
    whisper.mkdir()
    (whisper / "f").write_bytes(b"x" * 100)
    other = tmp_path / "models--openai--clip-vit-base"
    other.mkdir()
    (other / "f").write_bytes(b"y" * 100)

    monkeypatch.setattr(mw_mod.model_cache, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        mw_mod.dialogs, "confirm_clear_model_cache",
        lambda parent, *, model_names, total_bytes: True,
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: None)

    w = MainWindow(devices=_devices())
    qtbot.addWidget(w)
    w._on_clear_model_cache()

    assert not whisper.exists()
    assert other.exists()


def test_predownload_cancel_does_nothing(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    monkeypatch.setattr(
        mw_mod.dialogs, "prompt_select_model_to_download",
        lambda parent, models, default=None: None,
    )

    w = MainWindow(devices=_devices())
    qtbot.addWidget(w)
    w._on_predownload_model()  # must not raise

    assert getattr(w, "_model_download_worker", None) is None


def test_predownload_starts_worker_when_model_selected(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selecting a model constructs a ModelDownloadWorker and starts it."""
    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    started: list = []

    class FakeWorker:
        def __init__(self, *, model, parent=None, **_kw):  # noqa: ARG002
            started.append(model)
            self.completed = _FakeSignal()
            self.failed = _FakeSignal()
            self.finished = _FakeSignal()

        def start(self):
            self.did_start = True

        def requestInterruption(self):  # noqa: N802
            self.interrupted = True

        def deleteLater(self):  # noqa: N802
            pass

    monkeypatch.setattr(mw_mod, "ModelDownloadWorker", FakeWorker)
    monkeypatch.setattr(
        mw_mod.dialogs, "prompt_select_model_to_download",
        lambda parent, models, default=None: "tiny",
    )

    w = MainWindow(devices=_devices())
    qtbot.addWidget(w)
    w._on_predownload_model()

    assert started == ["tiny"]


class _FakeSignal:
    def connect(self, _cb):  # noqa: D401
        pass
