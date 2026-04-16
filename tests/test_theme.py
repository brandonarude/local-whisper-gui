"""Smoke tests for the theme manager."""
from __future__ import annotations

import pytest

from src.utils.theme import VALID_THEMES, apply_theme, detect_system_theme


@pytest.mark.parametrize("theme", sorted(VALID_THEMES))
def test_apply_theme_runs(qapp, theme: str) -> None:
    resolved = apply_theme(qapp, theme)
    assert resolved in {"light", "dark"}


def test_apply_theme_light_resolves_to_light(qapp) -> None:
    assert apply_theme(qapp, "light") == "light"


def test_apply_theme_dark_resolves_to_dark(qapp) -> None:
    assert apply_theme(qapp, "dark") == "dark"


def test_apply_theme_unknown_raises(qapp) -> None:
    with pytest.raises(ValueError):
        apply_theme(qapp, "rainbow")


def test_detect_system_theme_returns_light_or_dark(qapp) -> None:
    assert detect_system_theme(qapp) in {"light", "dark"}
