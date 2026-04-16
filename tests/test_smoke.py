"""Smoke tests: the package imports and a QApplication can be created."""
from __future__ import annotations

import wave
from pathlib import Path


def test_src_package_imports() -> None:
    import src  # noqa: F401


def test_qapplication_can_be_created(qapp) -> None:
    from PyQt6.QtWidgets import QApplication

    assert QApplication.instance() is qapp
    assert isinstance(qapp, QApplication)


def test_tiny_wav_fixture(tiny_wav: Path) -> None:
    assert tiny_wav.exists()
    with wave.open(str(tiny_wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16_000
        assert wf.getnframes() == 16_000
