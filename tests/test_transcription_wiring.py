"""Smoke tests for the Start/Cancel wiring in MainWindow (SPEC §3.8, §5.4).

The real :class:`~src.core.transcriber.Transcriber` would try to import
``faster_whisper``; tests inject a fake via the ``transcriber_factory``
hook on :class:`MainWindow`.
"""
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
    """Fake transcriber returning pre-canned segments; optionally blocks
    each ``transcribe`` call on a ``threading.Event`` so tests can drive
    the worker one chunk at a time."""

    def __init__(
        self,
        segments_per_call=None,
        *,
        block_event: threading.Event | None = None,
    ):
        self._batches = list(segments_per_call or [])
        self.kwargs: dict | None = None
        self._block_event = block_event

    def transcribe(
        self,
        audio_path,
        *,
        language=None,
        word_timestamps=True,
        should_cancel=None,
    ):
        self.kwargs = {"language": language, "word_timestamps": word_timestamps}
        if self._block_event is not None:
            self._block_event.wait(timeout=5.0)
            self._block_event.clear()
        if self._batches:
            return self._batches.pop(0)
        return []


def _factory(transcriber: FakeTranscriber):
    def make(*, model, device, compute_type):  # noqa: ARG001
        return transcriber
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


def test_start_runs_transcription_and_populates_results(
    qtbot, tiny_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    # Export dialogs are modal QMessageBoxes — stub them out so the
    # completed-path doesn't block the test.
    monkeypatch.setattr(mw_mod.dialogs, "show_export_complete", lambda *a, **kw: None)

    fake = FakeTranscriber([[_seg(0.0, 0.5, "hello"), _seg(0.5, 1.0, "world")]])
    w = MainWindow(devices=_devices(), transcriber_factory=_factory(fake))
    qtbot.addWidget(w)

    # Disable chunking so the whole-file path runs (tiny_wav is 1s).
    w._settings_panel.set_chunking_enabled(False)
    w._file_header.load_path(tiny_wav)
    w._waveform_worker.wait(5_000)

    w._progress_panel._start_button.click()
    qtbot.waitUntil(lambda: w._transcription_worker is None, timeout=10_000)

    assert len(w._transcription_results) == 1
    assert w._transcription_results[0].segments[0].text == "hello"
    assert w._progress_panel._progress_bar.value() == 100
    assert fake.kwargs is not None
    assert fake.kwargs["language"] == "en"
    # Buttons reset to ready state after completion.
    assert w._progress_panel._start_button.isEnabled()
    assert not w._progress_panel._cancel_button.isEnabled()


def test_cancel_preserves_completed_chunks(
    qtbot, long_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ui import main_window as mw_mod
    from src.ui.main_window import MainWindow

    # Decline partial export so no follow-up dialog blocks the test.
    monkeypatch.setattr(
        mw_mod.dialogs, "prompt_partial_output_on_cancel",
        lambda parent, completed, total: False,
    )

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
    w._file_header.load_path(long_wav)
    w._waveform_worker.wait(5_000)
    if w._chunk_preview_worker is not None:
        w._chunk_preview_worker.wait(10_000)

    w._progress_panel._start_button.click()
    worker = w._transcription_worker
    assert worker is not None

    seen_chunks: list[int] = []
    worker.chunk_completed.connect(seen_chunks.append)

    # Release chunk 0 — it completes fully.
    gate.set()
    qtbot.waitUntil(lambda: 0 in seen_chunks, timeout=10_000)

    # Chunk 1 is now blocked inside transcribe(). Request cancellation, then
    # release the gate so transcribe() returns and the worker observes the
    # interrupt after the transcribe call.
    w._progress_panel._cancel_button.click()
    gate.set()

    qtbot.waitUntil(lambda: w._transcription_worker is None, timeout=10_000)

    assert len(w._transcription_results) == 1
    assert w._transcription_results[0].segments[0].text == "a"
    # Buttons reset to ready state after cancel.
    assert w._progress_panel._start_button.isEnabled()
    assert not w._progress_panel._cancel_button.isEnabled()
