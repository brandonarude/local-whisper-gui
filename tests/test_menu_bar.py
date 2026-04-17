"""Smoke tests for the main-window menu bar and theme toggle (SPEC §4, §5.1)."""
from __future__ import annotations

from src.utils.device_detect import Device


def _devices() -> list[Device]:
    return [Device(kind="cpu", name="CPU")]


def _menu_labels(window) -> list[str]:
    return [a.text() for a in window.menuBar().actions()]


def test_top_level_menus_present(qtbot) -> None:
    from src.ui.main_window import MainWindow

    w = MainWindow(devices=_devices())
    qtbot.addWidget(w)
    labels = _menu_labels(w)
    assert any("File" in s for s in labels)
    assert any("Settings" in s for s in labels)
    assert any("Help" in s for s in labels)


def test_theme_actions_exist_and_are_exclusive(qtbot) -> None:
    from src.ui.main_window import MainWindow

    w = MainWindow(devices=_devices())
    qtbot.addWidget(w)
    assert set(w._theme_actions.keys()) == {"light", "dark", "system"}
    checked = [t for t, a in w._theme_actions.items() if a.isChecked()]
    assert len(checked) == 1


def test_set_theme_updates_checked_action_and_state(qtbot, qapp) -> None:
    from src.ui.main_window import MainWindow

    w = MainWindow(devices=_devices())
    qtbot.addWidget(w)
    w.set_theme("dark")
    assert w.current_theme() == "dark"
    assert w._theme_actions["dark"].isChecked()
    assert not w._theme_actions["light"].isChecked()
    w.set_theme("light")
    assert w.current_theme() == "light"
    assert w._theme_actions["light"].isChecked()


def test_help_tooltips_on_model_and_device(qtbot) -> None:
    from src.ui.main_window import MainWindow

    w = MainWindow(devices=_devices())
    qtbot.addWidget(w)
    assert "Whisper model" in w._settings_panel._model_help.toolTip()
    assert w._settings_panel._device_help.toolTip() != ""
