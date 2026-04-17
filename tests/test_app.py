"""Smoke tests for the app entry point."""
from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings


def test_main_window_instantiates(qapp) -> None:
    from src.app import APP_NAME, MainWindow

    window = MainWindow()
    assert window.windowTitle() == APP_NAME


def test_create_application_returns_singleton(qapp) -> None:
    from PyQt6.QtWidgets import QApplication

    from src.app import create_application

    app = create_application([])
    assert app is QApplication.instance()


# --- startup checks -----------------------------------------------------

@pytest.fixture
def temp_config(tmp_path: Path, qapp):
    from src.utils.config import Config

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    return Config(settings=settings)


def _result(name: str, *, ok: bool, severity: str) -> object:
    from src.utils.startup_checks import CheckResult

    return CheckResult(name=name, ok=ok, severity=severity, message=f"{name} msg", detail="")


def test_handle_startup_checks_all_ok_shows_no_dialogs(qapp, temp_config, mocker) -> None:
    from src.app import handle_startup_checks

    results = [
        _result("ffmpeg", ok=True, severity="info"),
        _result("faster_whisper", ok=True, severity="info"),
        _result("cuda", ok=True, severity="info"),
    ]
    warn_spy = mocker.patch("src.app.dialogs.show_startup_warning")
    ffmpeg_spy = mocker.patch("src.app.dialogs.show_missing_ffmpeg")
    fw_spy = mocker.patch("src.app.dialogs.show_missing_faster_whisper")

    returned = handle_startup_checks(None, config=temp_config, results=results)
    assert returned == results
    warn_spy.assert_not_called()
    ffmpeg_spy.assert_not_called()
    fw_spy.assert_not_called()


def test_handle_startup_checks_cuda_warning_shown_with_suppress(
    qapp, temp_config, mocker
) -> None:
    from src.app import handle_startup_checks

    results = [_result("cuda", ok=True, severity="warning")]
    spy = mocker.patch("src.app.dialogs.show_startup_warning", return_value=False)

    handle_startup_checks(None, config=temp_config, results=results)
    spy.assert_called_once()
    # The CUDA warning is suppressible — caller should pass allow_suppress=True.
    _, kwargs = spy.call_args
    assert kwargs.get("allow_suppress") is True
    assert temp_config.is_warning_suppressed("cuda") is False


def test_handle_startup_checks_cuda_warning_suppression_persists(
    qapp, temp_config, mocker
) -> None:
    from src.app import handle_startup_checks

    results = [_result("cuda", ok=True, severity="warning")]
    mocker.patch("src.app.dialogs.show_startup_warning", return_value=True)

    handle_startup_checks(None, config=temp_config, results=results)
    assert temp_config.is_warning_suppressed("cuda") is True

    # Second call should not re-show the dialog.
    spy = mocker.patch("src.app.dialogs.show_startup_warning")
    handle_startup_checks(None, config=temp_config, results=results)
    spy.assert_not_called()


def test_handle_startup_checks_ffmpeg_missing_shows_fatal_dialog(
    qapp, temp_config, mocker
) -> None:
    from src.app import handle_startup_checks

    results = [_result("ffmpeg", ok=False, severity="error")]
    spy = mocker.patch("src.app.dialogs.show_missing_ffmpeg")

    handle_startup_checks(None, config=temp_config, results=results)
    spy.assert_called_once()


def test_handle_startup_checks_faster_whisper_missing_shows_fatal_dialog(
    qapp, temp_config, mocker
) -> None:
    from src.app import handle_startup_checks

    results = [_result("faster_whisper", ok=False, severity="error")]
    spy = mocker.patch("src.app.dialogs.show_missing_faster_whisper")

    handle_startup_checks(None, config=temp_config, results=results)
    spy.assert_called_once()


def test_handle_startup_checks_fatal_warnings_not_suppressible(
    qapp, temp_config, mocker
) -> None:
    """Even after marking the key as suppressed, fatal backend failures
    must re-show on every launch — the app literally can't work without
    them, so the reminder is the point."""
    from src.app import handle_startup_checks

    temp_config.set_warning_suppressed("ffmpeg", True)
    results = [_result("ffmpeg", ok=False, severity="error")]
    spy = mocker.patch("src.app.dialogs.show_missing_ffmpeg")

    handle_startup_checks(None, config=temp_config, results=results)
    spy.assert_called_once()


def test_handle_startup_checks_runs_probes_when_results_omitted(
    qapp, temp_config, mocker
) -> None:
    from src.app import handle_startup_checks

    mocker.patch("src.app.dialogs.show_startup_warning", return_value=False)
    runner = mocker.Mock(
        return_value=[_result("cuda", ok=True, severity="warning")]
    )

    handle_startup_checks(None, config=temp_config, check_runner=runner)
    runner.assert_called_once()
