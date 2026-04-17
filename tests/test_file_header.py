"""Smoke tests for the file header row (SPEC §3.1)."""
from __future__ import annotations

from pathlib import Path


def test_file_header_instantiates(qtbot) -> None:
    from src.ui.file_header import FileHeader

    h = FileHeader()
    qtbot.addWidget(h)
    assert h.current_info is None


def test_load_path_populates_metadata_and_emits(qtbot, tiny_wav: Path) -> None:
    from src.ui.file_header import FileHeader

    h = FileHeader()
    qtbot.addWidget(h)
    with qtbot.waitSignal(h.file_loaded, timeout=5_000) as blocker:
        info = h.load_path(tiny_wav)
    assert info is not None
    assert blocker.args[0] is info
    assert h.current_info is info
    metadata = h._metadata_label.text()
    assert "16.0 kHz" in metadata
    assert "1 ch" in metadata


def test_load_path_missing_file_emits_failure(qtbot, tmp_path: Path) -> None:
    from src.ui.file_header import FileHeader

    h = FileHeader()
    qtbot.addWidget(h)
    with qtbot.waitSignal(h.load_failed, timeout=5_000) as blocker:
        result = h.load_path(tmp_path / "missing.wav")
    assert result is None
    assert "missing.wav" in blocker.args[0]


def test_main_window_wires_header_to_waveform(qtbot, tiny_wav: Path) -> None:
    from src.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    window._file_header.load_path(tiny_wav)
    worker = window._waveform_worker
    assert worker is not None
    with qtbot.waitSignal(worker.samples_ready, timeout=5_000):
        pass
    worker.wait(5_000)
    assert window._file_header.current_info is not None
