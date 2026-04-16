"""Smoke tests for the app entry point."""
from __future__ import annotations


def test_main_window_instantiates(qapp) -> None:
    from src.app import APP_NAME, MainWindow

    window = MainWindow()
    assert window.windowTitle() == APP_NAME


def test_create_application_returns_singleton(qapp) -> None:
    from PyQt6.QtWidgets import QApplication

    from src.app import create_application

    app = create_application([])
    assert app is QApplication.instance()
