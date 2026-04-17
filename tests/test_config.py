"""Tests for src.utils.config (settings persistence, SPEC §5.5).

A `temp_settings` fixture hands the Config wrapper an isolated QSettings
backed by an INI file in `tmp_path`, so the dev's real config is never
touched.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import QByteArray, QSettings


@pytest.fixture
def temp_settings(tmp_path: Path, qapp) -> QSettings:
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def cfg(temp_settings):
    from src.utils.config import Config

    return Config(settings=temp_settings)


# --- Round-trip ---------------------------------------------------------

def test_round_trip_theme(cfg) -> None:
    cfg.set_theme("dark")
    assert cfg.theme() == "dark"
    cfg.set_theme("light")
    assert cfg.theme() == "light"


def test_round_trip_model(cfg) -> None:
    cfg.set_model("small")
    assert cfg.model() == "small"


def test_round_trip_device(cfg) -> None:
    cfg.set_device("cuda:0")
    assert cfg.device() == "cuda:0"


def test_round_trip_language(cfg) -> None:
    cfg.set_language("fr")
    assert cfg.language() == "fr"


def test_round_trip_output_formats(cfg) -> None:
    cfg.set_output_formats(["txt", "srt", "json"])
    assert cfg.output_formats() == ["txt", "srt", "json"]


def test_round_trip_output_formats_single(cfg) -> None:
    # A single-element list must not be flattened to a bare string by QSettings.
    cfg.set_output_formats(["vtt"])
    assert cfg.output_formats() == ["vtt"]


def test_round_trip_include_timestamps(cfg) -> None:
    cfg.set_include_timestamps(False)
    assert cfg.include_timestamps() is False
    cfg.set_include_timestamps(True)
    assert cfg.include_timestamps() is True


def test_round_trip_output_dir(cfg, tmp_path) -> None:
    cfg.set_output_dir(str(tmp_path))
    assert cfg.output_dir() == str(tmp_path)


def test_round_trip_chunking_params(cfg) -> None:
    cfg.set_min_silence_ms(900)
    cfg.set_silence_threshold_dbfs(-35)
    cfg.set_min_chunk_minutes(7)
    cfg.set_max_chunk_minutes(45)
    cfg.set_chunking_enabled(False)
    assert cfg.min_silence_ms() == 900
    assert cfg.silence_threshold_dbfs() == -35
    assert cfg.min_chunk_minutes() == 7
    assert cfg.max_chunk_minutes() == 45
    assert cfg.chunking_enabled() is False


def test_round_trip_geometry(cfg) -> None:
    payload = QByteArray(b"some-geometry-bytes")
    cfg.set_window_geometry(payload)
    assert cfg.window_geometry() == payload


def test_round_trip_window_state(cfg) -> None:
    payload = QByteArray(b"some-state-bytes")
    cfg.set_window_state(payload)
    assert cfg.window_state() == payload


# --- Defaults on unset --------------------------------------------------

def test_default_theme_is_system(cfg) -> None:
    assert cfg.theme() == "system"


def test_default_model_matches_constants(cfg) -> None:
    from src.utils.constants import DEFAULT_MODEL

    assert cfg.model() == DEFAULT_MODEL


def test_default_device_is_cpu(cfg) -> None:
    assert cfg.device() == "cpu"


def test_default_language_matches_constants(cfg) -> None:
    from src.utils.constants import DEFAULT_LANGUAGE

    assert cfg.language() == DEFAULT_LANGUAGE


def test_default_output_formats_matches_constants(cfg) -> None:
    from src.utils.constants import DEFAULT_OUTPUT_FORMATS

    assert cfg.output_formats() == list(DEFAULT_OUTPUT_FORMATS)


def test_default_include_timestamps_is_true(cfg) -> None:
    assert cfg.include_timestamps() is True


def test_default_chunking_params_match_constants(cfg) -> None:
    from src.utils import constants as C

    assert cfg.min_silence_ms() == C.DEFAULT_MIN_SILENCE_MS
    assert cfg.silence_threshold_dbfs() == C.DEFAULT_SILENCE_THRESHOLD_DBFS
    assert cfg.min_chunk_minutes() == C.DEFAULT_MIN_CHUNK_MINUTES
    assert cfg.max_chunk_minutes() == C.DEFAULT_MAX_CHUNK_MINUTES


def test_default_chunking_enabled_is_true(cfg) -> None:
    assert cfg.chunking_enabled() is True


def test_default_window_geometry_is_none(cfg) -> None:
    assert cfg.window_geometry() is None


def test_default_window_state_is_none(cfg) -> None:
    assert cfg.window_state() is None


def test_default_output_dir_is_none(cfg) -> None:
    assert cfg.output_dir() is None


# --- Persistence across instances --------------------------------------

def test_values_persist_across_config_instances(temp_settings) -> None:
    from src.utils.config import Config

    Config(settings=temp_settings).set_model("medium")
    assert Config(settings=temp_settings).model() == "medium"


# --- Suppressed startup warnings ---------------------------------------

def test_default_warning_is_not_suppressed(cfg) -> None:
    assert cfg.is_warning_suppressed("cuda") is False


def test_suppressed_warning_round_trip(cfg) -> None:
    cfg.set_warning_suppressed("cuda", True)
    assert cfg.is_warning_suppressed("cuda") is True
    cfg.set_warning_suppressed("cuda", False)
    assert cfg.is_warning_suppressed("cuda") is False


def test_suppressed_warnings_are_per_key(cfg) -> None:
    cfg.set_warning_suppressed("cuda", True)
    assert cfg.is_warning_suppressed("cuda") is True
    assert cfg.is_warning_suppressed("ffmpeg") is False
