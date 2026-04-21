"""Smoke tests for post-transcription export (SPEC §5.4)."""
from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np
import pytest

from src.core.exporter import Segment, Word
from src.utils.device_detect import Device


def _devices() -> list[Device]:
    return [Device(kind="cpu", name="CPU")]


def _seg(start: float, end: float, text: str) -> Segment:
    return Segment(
        start=start, end=end, text=text,
        words=(Word(start=start, end=end, text=text, probability=0.9),),
    )


class FakeTranscriber:
    def __init__(self, segments_per_call, *, block_event: threading.Event | None = None):
        self._batches = list(segments_per_call)
        self._block = block_event

    def transcribe(self, audio_path, *, language=None, word_timestamps=True, should_cancel=None):
        if self._block is not None:
            self._block.wait(timeout=5.0)
            self._block.clear()
        if self._batches:
            return self._batches.pop(0)
        return []


def _factory(t):
    def make(*, model, device, compute_type):  # noqa: ARG001
        return t
    return make


@pytest.fixture
def long_wav(tmp_path: Path) -> Path:
    sr = 16_000

    def tone(s):
        t = np.linspace(0.0, s, int(sr * s), endpoint=False)
        return (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    def silence(s):
        return np.zeros(int(sr * s), dtype=np.float32)

    pcm = (
        np.concatenate(
            [tone(6.0), silence(1.0), tone(6.0), silence(2.0), tone(6.0), silence(1.0)]
        )
        * 32767
    ).astype(np.int16)
    path = tmp_path / "long.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def test_successful_run_writes_selected_formats(
    qtbot, tiny_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ui import dialogs as dlg_mod
    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    export_calls: list = []
    monkeypatch.setattr(
        mw_mod.dialogs,
        "show_export_complete",
        lambda parent, out_dir, files: export_calls.append((out_dir, list(files))),
    )
    monkeypatch.setattr(
        mw_mod.dialogs,
        "prompt_cadence_exceeds_duration",
        lambda *a, **kw: True,
    )

    out_dir = tmp_path / "outs"
    fake = FakeTranscriber([[_seg(0.0, 0.5, "hello world")]])
    w = MainWindow(devices=_devices(), transcriber_factory=_factory(fake))
    qtbot.addWidget(w)
    w._settings_panel.set_chunking_enabled(False)
    w._settings_panel.set_output_formats(["txt", "srt", "vtt", "json"])
    w._settings_panel.set_output_dir(str(out_dir))

    w._file_header.load_path(tiny_wav)
    w._waveform_worker.wait(5_000)

    w._progress_panel._start_button.click()
    qtbot.waitUntil(lambda: w._transcription_worker is None, timeout=10_000)

    stem = tiny_wav.stem
    assert (out_dir / f"{stem}.txt").exists()
    assert (out_dir / f"{stem}.srt").exists()
    assert (out_dir / f"{stem}.vtt").exists()
    assert (out_dir / f"{stem}.json").exists()

    # SRT should parse loosely (starts with "1\n<time range>").
    srt_text = (out_dir / f"{stem}.srt").read_text()
    assert srt_text.startswith("1\n")
    assert "-->" in srt_text

    # VTT must begin with the WEBVTT magic.
    assert (out_dir / f"{stem}.vtt").read_text().startswith("WEBVTT")

    assert len(export_calls) == 1
    assert export_calls[0][0] == out_dir
    assert {Path(p).suffix for p in export_calls[0][1]} == {".txt", ".srt", ".vtt", ".json"}


def test_cancel_declined_skips_export(
    qtbot, long_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    monkeypatch.setattr(
        mw_mod.dialogs, "prompt_partial_output_on_cancel",
        lambda parent, completed, total: False,
    )
    export_complete_calls: list = []
    monkeypatch.setattr(
        mw_mod.dialogs, "show_export_complete",
        lambda *a, **kw: export_complete_calls.append(a),
    )
    monkeypatch.setattr(
        mw_mod.dialogs, "prompt_cadence_exceeds_duration",
        lambda *a, **kw: True,
    )

    out_dir = tmp_path / "partials"
    gate = threading.Event()
    fake = FakeTranscriber(
        [[_seg(0.0, 1.0, "a")], [_seg(0.0, 1.0, "b")], [_seg(0.0, 1.0, "c")]],
        block_event=gate,
    )

    w = MainWindow(devices=_devices(), transcriber_factory=_factory(fake))
    qtbot.addWidget(w)
    w._settings_panel._min_silence_spin.setValue(500)
    w._settings_panel._min_chunk_spin.setValue(0.05)
    w._settings_panel._max_chunk_spin.setValue(10.0)
    w._settings_panel.set_output_formats(["txt"])
    w._settings_panel.set_output_dir(str(out_dir))
    w._file_header.load_path(long_wav)
    w._waveform_worker.wait(5_000)
    if w._chunk_preview_worker is not None:
        w._chunk_preview_worker.wait(10_000)

    w._progress_panel._start_button.click()
    worker = w._transcription_worker
    assert worker is not None
    seen: list[int] = []
    worker.chunk_completed.connect(seen.append)

    gate.set()
    qtbot.waitUntil(lambda: 0 in seen, timeout=10_000)

    w._progress_panel._cancel_button.click()
    gate.set()
    qtbot.waitUntil(lambda: w._transcription_worker is None, timeout=10_000)

    assert export_complete_calls == []
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_cancel_accepted_writes_partial_files(
    qtbot, long_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    monkeypatch.setattr(
        mw_mod.dialogs, "prompt_partial_output_on_cancel",
        lambda parent, completed, total: True,
    )
    monkeypatch.setattr(mw_mod.dialogs, "show_export_complete", lambda *a, **kw: None)
    monkeypatch.setattr(
        mw_mod.dialogs, "prompt_cadence_exceeds_duration",
        lambda *a, **kw: True,
    )

    out_dir = tmp_path / "partials"
    gate = threading.Event()
    fake = FakeTranscriber(
        [[_seg(0.0, 1.0, "first")], [_seg(0.0, 1.0, "second")], [_seg(0.0, 1.0, "third")]],
        block_event=gate,
    )

    w = MainWindow(devices=_devices(), transcriber_factory=_factory(fake))
    qtbot.addWidget(w)
    w._settings_panel._min_silence_spin.setValue(500)
    w._settings_panel._min_chunk_spin.setValue(0.05)
    w._settings_panel._max_chunk_spin.setValue(10.0)
    w._settings_panel.set_output_formats(["txt"])
    w._settings_panel.set_output_dir(str(out_dir))
    w._file_header.load_path(long_wav)
    w._waveform_worker.wait(5_000)
    if w._chunk_preview_worker is not None:
        w._chunk_preview_worker.wait(10_000)

    w._progress_panel._start_button.click()
    worker = w._transcription_worker
    assert worker is not None
    seen: list[int] = []
    worker.chunk_completed.connect(seen.append)

    gate.set()
    qtbot.waitUntil(lambda: 0 in seen, timeout=10_000)

    w._progress_panel._cancel_button.click()
    gate.set()
    qtbot.waitUntil(lambda: w._transcription_worker is None, timeout=10_000)

    partial_txt = out_dir / f"{long_wav.stem}.partial.txt"
    assert partial_txt.exists()
    assert "first" in partial_txt.read_text()
