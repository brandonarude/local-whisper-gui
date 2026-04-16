"""Sanity checks on the constants module."""
from __future__ import annotations

from src.utils import constants as C


def test_models_non_empty_and_have_required_fields() -> None:
    assert len(C.MODELS) > 0
    names = [m.name for m in C.MODELS]
    assert "large-v3" in names
    for m in C.MODELS:
        assert m.name
        assert m.vram_hint_gb > 0
        assert m.description


def test_default_model_is_in_catalogue() -> None:
    assert C.DEFAULT_MODEL in {m.name for m in C.MODELS}


def test_languages_contain_english_and_auto_detect() -> None:
    assert "en" in C.LANGUAGES
    assert C.LANGUAGES["en"] == "English"
    assert C.AUTO_DETECT_LANGUAGE in C.LANGUAGES


def test_default_language_is_english() -> None:
    assert C.DEFAULT_LANGUAGE == "en"


def test_chunking_defaults_in_range() -> None:
    assert C.DEFAULT_MIN_SILENCE_MS > 0
    assert C.DEFAULT_SILENCE_THRESHOLD_DBFS < 0
    assert 0 < C.DEFAULT_MIN_CHUNK_MINUTES < C.DEFAULT_MAX_CHUNK_MINUTES
    assert C.CHUNK_OVERLAP_SECONDS > 0
    assert C.LONG_FILE_PROMPT_THRESHOLD_SECONDS == 3600


def test_output_formats_cover_spec() -> None:
    assert set(C.OUTPUT_FORMATS) == {"txt", "srt", "vtt", "json"}
    assert set(C.DEFAULT_OUTPUT_FORMATS).issubset(set(C.OUTPUT_FORMATS))
