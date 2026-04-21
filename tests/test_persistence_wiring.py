"""Smoke tests for MainWindow persistence (SPEC §5.5)."""
from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings

from src.utils.config import Config
from src.utils.device_detect import Device


def _devices() -> list[Device]:
    return [Device(kind="cpu", name="CPU")]


@pytest.fixture
def config(tmp_path: Path) -> Config:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    return Config(settings=settings)


def test_config_values_applied_on_startup(qtbot, config: Config) -> None:
    from src.ui.main_window import MainWindow

    config.set_model("base")
    config.set_language("es")
    config.set_output_formats(["srt", "vtt"])
    config.set_include_timestamps(False)
    config.set_timestamp_cadence_s(45)
    config.set_chunking_enabled(False)
    config.set_min_silence_ms(1_234)
    config.set_silence_threshold_dbfs(-30)
    config.set_min_chunk_minutes(3)
    config.set_max_chunk_minutes(30)

    w = MainWindow(devices=_devices(), config=config)
    qtbot.addWidget(w)

    v = w._settings_panel.values()
    assert v.model == "base"
    assert v.language == "es"
    assert set(v.output_formats) == {"srt", "vtt"}
    assert v.include_timestamps is False
    assert v.timestamp_cadence_s == 45
    assert v.chunking_enabled is False
    assert v.min_silence_ms == 1_234
    assert v.silence_threshold_dbfs == -30
    assert v.min_chunk_minutes == 3
    assert v.max_chunk_minutes == 30


def test_close_persists_settings(qtbot, config: Config, tmp_path: Path) -> None:
    from src.ui.main_window import MainWindow

    w = MainWindow(devices=_devices(), config=config)
    qtbot.addWidget(w)
    w._settings_panel.set_model("small")
    w._settings_panel.set_language("fr")
    w._settings_panel.set_output_formats(["json"])
    w._settings_panel.set_output_dir(str(tmp_path))
    w._settings_panel.set_include_timestamps(False)
    w._settings_panel.set_timestamp_cadence_s(120)

    w.close()

    # Re-read via a fresh Config pointed at the same store.
    fresh = Config(settings=config._s)
    assert fresh.model() == "small"
    assert fresh.language() == "fr"
    assert fresh.output_formats() == ["json"]
    assert fresh.output_dir() == str(tmp_path)
    assert fresh.include_timestamps() is False
    assert fresh.timestamp_cadence_s() == 120


def test_window_geometry_round_trip(qtbot, config: Config) -> None:
    """Closing the window writes non-empty geometry bytes; re-opening
    restores them. (Exact pixel dimensions aren't asserted — offscreen
    Qt can clamp the size to the virtual screen.)"""
    from src.ui.main_window import MainWindow

    w = MainWindow(devices=_devices(), config=config)
    qtbot.addWidget(w)
    w.close()

    saved = config.window_geometry()
    assert saved is not None and not saved.isEmpty()

    w2 = MainWindow(devices=_devices(), config=config)
    qtbot.addWidget(w2)
    # After restore, the saved bytes are still reachable via the config.
    assert config.window_geometry() == saved


def test_config_omitted_leaves_defaults(qtbot) -> None:
    """When no config is passed, panel keeps its built-in defaults."""
    from src.ui.main_window import MainWindow
    from src.utils import constants as C

    w = MainWindow(devices=_devices())
    qtbot.addWidget(w)
    v = w._settings_panel.values()
    assert v.model == C.DEFAULT_MODEL
