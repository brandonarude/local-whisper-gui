"""faster-whisper wrapper with lazy loading and cooperative cancellation.

The real :mod:`faster_whisper` package pulls in torch / ctranslate2 and can
trigger a multi-GB model download on first use. Importing it at module
import time would block the app launch for seconds (or minutes, on a cold
cache) and prevent the test suite from running without the dependency
installed. The import therefore happens inside :meth:`Transcriber._ensure_loaded`
on the first ``.transcribe()`` call, giving the UI a chance to surface a
download-progress dialog (SPEC §8.3) and letting tests stub the module via
``sys.modules``.

Our :class:`~src.core.exporter.Segment` / :class:`~src.core.exporter.Word`
dataclasses are the canonical internal shape; faster-whisper's objects
(``SimpleNamespace``-like with ``.word`` for text and a leading-space
tokenisation artefact) are normalised into them here so the rest of the
pipeline (stitcher, exporter) doesn't know about the upstream differences.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.core.exporter import Segment, Word
from src.utils.constants import AUTO_DETECT_LANGUAGE


class Transcriber:
    def __init__(
        self,
        model: str,
        device: str,
        compute_type: str = "auto",
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
        )

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        word_timestamps: bool = True,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[Segment]:
        self._ensure_loaded()

        fw_language = None if language == AUTO_DETECT_LANGUAGE else language

        segments_iter, _info = self._model.transcribe(
            str(audio_path),
            language=fw_language,
            word_timestamps=word_timestamps,
        )

        results: list[Segment] = []
        for fw_seg in segments_iter:
            if should_cancel is not None and should_cancel():
                break
            results.append(_normalize_segment(fw_seg))
        return results


def _normalize_segment(fw_seg) -> Segment:
    raw_words = getattr(fw_seg, "words", None)
    words: tuple[Word, ...] | None = None
    if raw_words:
        words = tuple(_normalize_word(w) for w in raw_words)
    return Segment(
        start=float(fw_seg.start),
        end=float(fw_seg.end),
        text=str(fw_seg.text).strip(),
        words=words,
    )


def _normalize_word(fw_word) -> Word:
    probability = getattr(fw_word, "probability", None)
    return Word(
        start=float(fw_word.start),
        end=float(fw_word.end),
        text=str(fw_word.word).strip(),
        probability=(float(probability) if probability is not None else None),
    )
