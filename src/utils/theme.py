"""Light/dark/system theme application (SPEC §5.1).

Excluded from the strict TDD pairing — this is mostly QPalette boilerplate
plumbed into the QApplication. A smoke test asserts each mode applies
without raising; the actual look is best validated visually.
"""
from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


class Theme(str, Enum):
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"


VALID_THEMES = {Theme.LIGHT.value, Theme.DARK.value, Theme.SYSTEM.value}


def apply_theme(app: QApplication, theme: str) -> str:
    """Apply `theme` to `app`. Returns the resolved theme actually applied
    ("light" or "dark") — i.e., "system" gets resolved to whichever the OS
    is currently set to."""
    if theme not in VALID_THEMES:
        raise ValueError(f"Unknown theme {theme!r}; expected one of {VALID_THEMES}")

    resolved = _resolve(app, theme)
    if resolved == Theme.DARK.value:
        app.setPalette(_dark_palette())
    else:
        app.setPalette(_light_palette())
    return resolved


def detect_system_theme(app: QApplication) -> str:
    """Return "light" or "dark" based on the OS color scheme, defaulting
    to "light" when the platform doesn't report one."""
    return _resolve(app, Theme.SYSTEM.value)


def _resolve(app: QApplication, theme: str) -> str:
    if theme != Theme.SYSTEM.value:
        return theme
    hints = app.styleHints()
    # Qt 6.5+ exposes colorScheme(); fall back to "light" otherwise.
    scheme = getattr(hints, "colorScheme", lambda: None)()
    if scheme is not None and scheme == Qt.ColorScheme.Dark:
        return Theme.DARK.value
    return Theme.LIGHT.value


def _light_palette() -> QPalette:
    # Plain default palette gives the platform's native light look.
    return QPalette()


def _dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    p.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    p.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    p.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    p.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    p.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    p.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    p.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    p.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    p.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    p.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Text,
        QColor(127, 127, 127),
    )
    p.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor(127, 127, 127),
    )
    return p
