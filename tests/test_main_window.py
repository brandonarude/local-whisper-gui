"""Smoke tests for the assembled main window (SPEC §4, §9)."""
from __future__ import annotations

from src.utils.device_detect import Device


def _fake_devices() -> list[Device]:
    return [
        Device(kind="cpu", name="CPU"),
        Device(kind="cuda", name="RTX 4090", index=0, vram_gb=24.0),
    ]


def test_main_window_exposes_all_panels(qtbot) -> None:
    from src.ui.main_window import MainWindow

    w = MainWindow(devices=_fake_devices())
    qtbot.addWidget(w)
    assert w._file_header is not None
    assert w._waveform is not None
    assert w._settings_panel is not None
    assert w._progress_panel is not None


def test_status_bar_reflects_settings(qtbot) -> None:
    from src.ui.main_window import MainWindow

    w = MainWindow(devices=_fake_devices())
    qtbot.addWidget(w)
    w._settings_panel.set_model("small")
    w._settings_panel.set_device("cuda:0")
    text = w._status.currentMessage()
    assert "small" in text
    assert "RTX 4090" in text


def test_file_load_enables_progress_panel_start(qtbot, tiny_wav) -> None:
    from src.ui.main_window import MainWindow

    w = MainWindow(devices=_fake_devices())
    qtbot.addWidget(w)
    assert not w._progress_panel._start_button.isEnabled()
    w._file_header.load_path(tiny_wav)
    assert w._progress_panel._start_button.isEnabled()
    # Waveform worker should have been kicked off.
    assert w._waveform_worker is not None
    w._waveform_worker.wait(5_000)


def test_app_module_reexports_main_window(qtbot) -> None:
    from src import app as app_mod
    from src.ui.main_window import MainWindow as MW

    assert app_mod.MainWindow is MW
    assert app_mod.APP_NAME == "Local Whisper GUI"
