"""Tests for src.core.transcriber.

The transcriber wraps ``faster_whisper.WhisperModel``. These tests stub the
whole ``faster_whisper`` module in :data:`sys.modules` so nothing has to be
installed in CI and the real heavyweight import never runs. The wrapper
must:

- Not load the model at construction time (lazy-load so the UI can show
  the download-progress dialog on first use; SPEC §8.3).
- Forward ``(model, device, compute_type)`` into ``WhisperModel(...)``.
- Load the model exactly once per wrapper, regardless of how many times
  ``transcribe`` is called.
- Forward ``language`` and ``word_timestamps`` into
  ``WhisperModel.transcribe`` and treat our sentinel auto-detect code as
  ``language=None``.
- Normalise faster-whisper's ``Segment`` / ``Word`` shapes into our own
  :class:`~src.core.exporter.Segment` / :class:`~src.core.exporter.Word`
  dataclasses. Notable shape differences: faster-whisper uses ``word.word``
  for the text; its text fields carry a leading space (Whisper tokenisation
  artifact) that the wrapper should strip.
- Cooperate with a caller-supplied ``should_cancel`` callable: check it
  between segments and stop iterating as soon as it returns true (SPEC
  §3.8, §5.4). Completed segments are returned — not lost.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# --- stubbing helpers ------------------------------------------------------


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    """Install a fake ``faster_whisper`` module; return the WhisperModel mock.

    Downstream tests set the mock's ``.return_value`` (the "model instance")
    and configure ``model_instance.transcribe.return_value`` to drive what
    segments the wrapper sees.
    """
    module = types.ModuleType("faster_whisper")
    ctor = MagicMock(name="WhisperModel")
    module.WhisperModel = ctor
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return ctor


def _fw_word(start: float, end: float, word_text: str, probability: float):
    """faster-whisper Word shape: ``.word`` carries text, often with a leading space."""
    return SimpleNamespace(
        start=start, end=end, word=word_text, probability=probability
    )


def _fw_segment(
    start: float, end: float, text: str, words=None
) -> SimpleNamespace:
    """faster-whisper Segment shape."""
    return SimpleNamespace(start=start, end=end, text=text, words=words)


def _configure_transcribe(ctor: MagicMock, fw_segments):
    """Wire ``ctor()`` → instance with ``.transcribe`` returning (iter, info)."""
    model_instance = MagicMock(name="WhisperModelInstance")
    model_instance.transcribe.return_value = (
        iter(fw_segments),
        SimpleNamespace(language="en", duration=0.0),
    )
    ctor.return_value = model_instance
    return model_instance


# --- lazy construction ----------------------------------------------------


def test_construction_does_not_load_model(fake_faster_whisper) -> None:
    from src.core.transcriber import Transcriber

    Transcriber(model="tiny", device="cpu", compute_type="int8")
    assert not fake_faster_whisper.called


def test_first_transcribe_loads_model_with_correct_args(
    tmp_path: Path, fake_faster_whisper
) -> None:
    from src.core.transcriber import Transcriber

    _configure_transcribe(fake_faster_whisper, [])

    t = Transcriber(model="small", device="cuda", compute_type="float16")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    t.transcribe(audio)

    assert fake_faster_whisper.call_count == 1
    args, kwargs = fake_faster_whisper.call_args
    # model name may be positional or keyword — accept either.
    assert (args and args[0] == "small") or kwargs.get("model_size_or_path") == "small"
    assert kwargs.get("device") == "cuda"
    assert kwargs.get("compute_type") == "float16"


def test_model_loaded_once_across_multiple_transcribes(
    tmp_path: Path, fake_faster_whisper
) -> None:
    from src.core.transcriber import Transcriber

    model_instance = _configure_transcribe(fake_faster_whisper, [])

    t = Transcriber(model="tiny", device="cpu")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    t.transcribe(audio)
    # Subsequent calls need a fresh iterator on the mock.
    model_instance.transcribe.return_value = (
        iter([]), SimpleNamespace(language="en", duration=0.0)
    )
    t.transcribe(audio)

    assert fake_faster_whisper.call_count == 1


# --- argument forwarding to WhisperModel.transcribe ----------------------


def test_transcribe_forwards_language_and_word_timestamps(
    tmp_path: Path, fake_faster_whisper
) -> None:
    from src.core.transcriber import Transcriber

    model_instance = _configure_transcribe(fake_faster_whisper, [])

    t = Transcriber(model="tiny", device="cpu")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    t.transcribe(audio, language="fr", word_timestamps=True)

    _, kwargs = model_instance.transcribe.call_args
    assert kwargs.get("language") == "fr"
    assert kwargs.get("word_timestamps") is True


def test_transcribe_maps_auto_language_to_none(
    tmp_path: Path, fake_faster_whisper
) -> None:
    """Our sentinel ``AUTO_DETECT_LANGUAGE`` must become ``language=None`` for
    faster-whisper (which does language detection when language is None)."""
    from src.core.transcriber import Transcriber
    from src.utils.constants import AUTO_DETECT_LANGUAGE

    model_instance = _configure_transcribe(fake_faster_whisper, [])

    t = Transcriber(model="tiny", device="cpu")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    t.transcribe(audio, language=AUTO_DETECT_LANGUAGE)

    _, kwargs = model_instance.transcribe.call_args
    assert kwargs.get("language") is None


# --- segment normalisation -----------------------------------------------


def test_transcribe_normalizes_segments_into_our_dataclass(
    tmp_path: Path, fake_faster_whisper
) -> None:
    from src.core.exporter import Segment, Word
    from src.core.transcriber import Transcriber

    fw_segments = [
        _fw_segment(
            0.0, 1.2, " Hello world.",
            words=[
                _fw_word(0.0, 0.5, " Hello", 0.95),
                _fw_word(0.6, 1.2, " world.", 0.90),
            ],
        ),
        _fw_segment(1.5, 2.5, " This is a test.", words=None),
    ]
    _configure_transcribe(fake_faster_whisper, fw_segments)

    t = Transcriber(model="tiny", device="cpu")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    segments = list(t.transcribe(audio, word_timestamps=True))

    assert len(segments) == 2
    assert all(isinstance(s, Segment) for s in segments)
    # Leading-space artefact from Whisper tokenisation is stripped.
    assert segments[0].text == "Hello world."
    assert segments[0].start == pytest.approx(0.0)
    assert segments[0].end == pytest.approx(1.2)

    assert segments[0].words is not None
    assert all(isinstance(w, Word) for w in segments[0].words)
    assert [w.text for w in segments[0].words] == ["Hello", "world."]
    assert segments[0].words[0].probability == pytest.approx(0.95)
    assert segments[0].words[0].start == pytest.approx(0.0)
    assert segments[0].words[0].end == pytest.approx(0.5)

    # Segment without words round-trips as words=None (or an empty tuple).
    assert not segments[1].words


# --- cancellation cooperation --------------------------------------------


def test_should_cancel_stops_iteration_and_preserves_completed_segments(
    tmp_path: Path, fake_faster_whisper
) -> None:
    from src.core.transcriber import Transcriber

    fw_segments = [
        _fw_segment(0.0, 1.0, " one"),
        _fw_segment(1.0, 2.0, " two"),
        _fw_segment(2.0, 3.0, " three"),
        _fw_segment(3.0, 4.0, " four"),
    ]
    _configure_transcribe(fake_faster_whisper, fw_segments)

    t = Transcriber(model="tiny", device="cpu")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    # Cancel after the second segment has been yielded.
    calls = {"n": 0}

    def should_cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    segments = list(t.transcribe(audio, should_cancel=should_cancel))

    # The wrapper must preserve already-completed segments when cancelled.
    assert [s.text for s in segments] == ["one", "two"]


def test_should_cancel_none_means_never_cancel(
    tmp_path: Path, fake_faster_whisper
) -> None:
    from src.core.transcriber import Transcriber

    fw_segments = [
        _fw_segment(0.0, 1.0, " one"),
        _fw_segment(1.0, 2.0, " two"),
    ]
    _configure_transcribe(fake_faster_whisper, fw_segments)

    t = Transcriber(model="tiny", device="cpu")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    segments = list(t.transcribe(audio, should_cancel=None))
    assert [s.text for s in segments] == ["one", "two"]
