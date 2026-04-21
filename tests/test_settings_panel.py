"""Smoke tests for SettingsPanel (SPEC §3.4–§3.7)."""
from __future__ import annotations

from src.utils import constants as C
from src.utils.device_detect import Device


def _fake_devices() -> list[Device]:
    return [
        Device(kind="cpu", name="CPU"),
        Device(kind="cuda", name="RTX 3080", index=0, vram_gb=8.0),
    ]


def test_settings_panel_instantiates_with_defaults(qtbot) -> None:
    from src.ui.settings_panel import SettingsPanel

    p = SettingsPanel(devices=_fake_devices())
    qtbot.addWidget(p)
    values = p.values()
    assert values.model == C.DEFAULT_MODEL
    assert values.device_key == "cpu"
    assert values.language == C.DEFAULT_LANGUAGE
    assert set(values.output_formats) == set(C.DEFAULT_OUTPUT_FORMATS)
    assert values.include_timestamps is True
    assert values.timestamp_cadence_s == C.DEFAULT_TIMESTAMP_CADENCE_S
    assert values.chunking_enabled is True
    assert values.min_silence_ms == C.DEFAULT_MIN_SILENCE_MS


def test_settings_panel_timestamp_cadence_round_trip(qtbot) -> None:
    from src.ui.settings_panel import SettingsPanel

    p = SettingsPanel(devices=_fake_devices())
    qtbot.addWidget(p)
    p.set_timestamp_cadence_s(75)
    assert p.values().timestamp_cadence_s == 75


def test_settings_panel_cadence_follows_timestamps_toggle(qtbot) -> None:
    from src.ui.settings_panel import SettingsPanel

    p = SettingsPanel(devices=_fake_devices())
    qtbot.addWidget(p)
    p.set_include_timestamps(False)
    assert p._cadence_spin.isEnabled() is False
    p.set_include_timestamps(True)
    assert p._cadence_spin.isEnabled() is True


def test_settings_panel_setters_round_trip(qtbot) -> None:
    from src.ui.settings_panel import SettingsPanel

    p = SettingsPanel(devices=_fake_devices())
    qtbot.addWidget(p)
    p.set_model("small")
    p.set_device("cuda:0")
    p.set_language("es")
    p.set_output_formats(["vtt", "json"])
    p.set_include_timestamps(False)
    p.set_output_dir("/tmp/out")
    p.set_chunking_enabled(False)

    v = p.values()
    assert v.model == "small"
    assert v.device_key == "cuda:0"
    assert v.language == "es"
    assert set(v.output_formats) == {"vtt", "json"}
    assert v.include_timestamps is False
    assert v.output_dir == "/tmp/out"
    assert v.chunking_enabled is False


def test_settings_panel_emits_values_changed(qtbot) -> None:
    from src.ui.settings_panel import SettingsPanel

    p = SettingsPanel(devices=_fake_devices())
    qtbot.addWidget(p)
    with qtbot.waitSignal(p.values_changed, timeout=1_000):
        p.set_model("base")


def test_settings_panel_cpu_only_builds_without_cuda(qtbot) -> None:
    from src.ui.settings_panel import SettingsPanel

    p = SettingsPanel(devices=[Device(kind="cpu", name="CPU")])
    qtbot.addWidget(p)
    assert p.values().device_key == "cpu"
    # Device help tooltip mentions CUDA absence.
    assert "No CUDA" in p._device_help.toolTip()
