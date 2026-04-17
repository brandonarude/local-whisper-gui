"""Wiring tests for classified transcription failures and the pre-export
disk-space check (SPEC §7)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.exporter import Segment, Word
from src.utils.device_detect import Device
from src.utils.errors import ErrorKind


def _devices() -> list[Device]:
    return [Device(kind="cpu", name="CPU")]


def _factory(transcriber):
    def make(*, model, device, compute_type):  # noqa: ARG001
        return transcriber
    return make


def _seg(text: str) -> Segment:
    return Segment(
        start=0.0, end=1.0, text=text,
        words=(Word(start=0.0, end=1.0, text=text, probability=0.9),),
    )


def test_oom_dispatches_to_prompt_oom_retry(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    """A CUDA OOM failure should trigger the OOM retry dialog, not a generic error."""
    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    oom_calls: list = []
    generic_calls: list = []
    monkeypatch.setattr(
        mw_mod.dialogs, "prompt_oom_retry",
        lambda parent, detail="": oom_calls.append(detail) or False,
    )
    monkeypatch.setattr(
        mw_mod.dialogs, "show_error",
        lambda *a, **kw: generic_calls.append((a, kw)),
    )

    w = MainWindow(devices=_devices(), transcriber_factory=_factory(None))
    qtbot.addWidget(w)

    w._on_transcription_failed("OutOfMemoryError: cuda oom", ErrorKind.CUDA_OOM.value)

    assert len(oom_calls) == 1
    assert generic_calls == []


def test_oom_retry_restarts_transcription(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    monkeypatch.setattr(
        mw_mod.dialogs, "prompt_oom_retry",
        lambda parent, detail="": True,
    )

    w = MainWindow(devices=_devices(), transcriber_factory=_factory(None))
    qtbot.addWidget(w)

    restart_calls: list = []
    monkeypatch.setattr(
        w, "_on_start_transcription",
        lambda: restart_calls.append(True),
    )

    w._on_transcription_failed("OOM", ErrorKind.CUDA_OOM.value)

    assert restart_calls == [True]


def test_model_download_dispatches_to_download_retry(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    download_calls: list = []
    monkeypatch.setattr(
        mw_mod.dialogs, "prompt_download_retry",
        lambda parent, detail="": download_calls.append(detail) or False,
    )

    w = MainWindow(devices=_devices(), transcriber_factory=_factory(None))
    qtbot.addWidget(w)

    w._on_transcription_failed(
        "HTTPError: connection reset", ErrorKind.MODEL_DOWNLOAD.value
    )
    assert len(download_calls) == 1


def test_audio_decode_shows_generic_error(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    error_calls: list = []
    monkeypatch.setattr(
        mw_mod.dialogs, "show_error",
        lambda parent, title, msg, details=None: error_calls.append((title, msg)),
    )

    w = MainWindow(devices=_devices(), transcriber_factory=_factory(None))
    qtbot.addWidget(w)

    w._on_transcription_failed(
        "CouldntDecodeError: bad header", ErrorKind.AUDIO_DECODE.value
    )
    assert len(error_calls) == 1
    assert "decode" in error_calls[0][0].lower()


def test_unknown_failure_falls_through_to_show_error(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    error_calls: list = []
    monkeypatch.setattr(
        mw_mod.dialogs, "show_error",
        lambda parent, title, msg, details=None: error_calls.append(title),
    )

    w = MainWindow(devices=_devices(), transcriber_factory=_factory(None))
    qtbot.addWidget(w)

    w._on_transcription_failed("boom", ErrorKind.UNKNOWN.value)
    assert len(error_calls) == 1


def test_export_skipped_when_disk_space_declined(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the disk-space prompt returns False, no files should be written."""
    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    monkeypatch.setattr(
        mw_mod.disk_space, "has_sufficient_free_space",
        lambda path, needed: False,
    )
    monkeypatch.setattr(
        mw_mod.dialogs, "prompt_low_disk_space",
        lambda parent, *, needed_bytes, free_bytes, output_dir: False,
    )
    export_calls: list = []
    monkeypatch.setattr(
        mw_mod.dialogs, "show_export_complete",
        lambda *a, **kw: export_calls.append(a),
    )

    w = MainWindow(devices=_devices(), transcriber_factory=_factory(None))
    qtbot.addWidget(w)

    out_dir = tmp_path / "outs"
    w._settings_panel.set_output_formats(["txt"])
    w._settings_panel.set_output_dir(str(out_dir))

    # Seed a loaded file so _export_results has info.path
    from src.core.audio_processor import AudioInfo

    w._file_header._info = AudioInfo(
        path=tmp_path / "fake.wav",
        duration_s=1.0,
        sample_rate=16_000,
        channels=1,
        size_bytes=1024,
    )

    from src.core.stitcher import ChunkResult

    results = [ChunkResult(start_s=0.0, duration_s=1.0, segments=[_seg("hello")])]
    w._export_results(results, partial=False)

    assert export_calls == []
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_export_proceeds_when_disk_space_confirmed(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    monkeypatch.setattr(
        mw_mod.disk_space, "has_sufficient_free_space",
        lambda path, needed: False,
    )
    proceed_calls: list = []
    monkeypatch.setattr(
        mw_mod.dialogs, "prompt_low_disk_space",
        lambda parent, *, needed_bytes, free_bytes, output_dir: proceed_calls.append(True) or True,
    )
    monkeypatch.setattr(mw_mod.dialogs, "show_export_complete", lambda *a, **kw: None)

    w = MainWindow(devices=_devices(), transcriber_factory=_factory(None))
    qtbot.addWidget(w)

    out_dir = tmp_path / "outs"
    w._settings_panel.set_output_formats(["txt"])
    w._settings_panel.set_output_dir(str(out_dir))

    from src.core.audio_processor import AudioInfo

    w._file_header._info = AudioInfo(
        path=tmp_path / "fake.wav",
        duration_s=1.0,
        sample_rate=16_000,
        channels=1,
        size_bytes=1024,
    )

    from src.core.stitcher import ChunkResult

    results = [ChunkResult(start_s=0.0, duration_s=1.0, segments=[_seg("hello")])]
    w._export_results(results, partial=False)

    assert proceed_calls == [True]
    assert (out_dir / "fake.txt").exists()
