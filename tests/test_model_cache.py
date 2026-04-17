"""Tests for src.utils.model_cache (SPEC §8.3, §5 Settings menu)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_cache_dir_honours_hf_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.utils import model_cache

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    cache = model_cache.cache_dir()
    assert cache == tmp_path / "hub"


def test_cache_dir_honours_explicit_cache_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.utils import model_cache

    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "custom_hub"))

    assert model_cache.cache_dir() == tmp_path / "custom_hub"


def test_cache_dir_defaults_under_user_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.utils import model_cache

    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert model_cache.cache_dir() == tmp_path / ".cache" / "huggingface" / "hub"


def test_total_cache_size_sums_files(tmp_path: Path) -> None:
    from src.utils import model_cache

    (tmp_path / "a").write_bytes(b"x" * 100)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b").write_bytes(b"y" * 250)
    assert model_cache.total_size_bytes(tmp_path) == 350


def test_total_cache_size_missing_dir_is_zero(tmp_path: Path) -> None:
    from src.utils import model_cache

    assert model_cache.total_size_bytes(tmp_path / "nope") == 0


def test_list_whisper_models_filters_to_whisper_dirs(tmp_path: Path) -> None:
    from src.utils import model_cache

    # HF cache layout: models--<org>--<name>/
    (tmp_path / "models--Systran--faster-whisper-small").mkdir()
    (tmp_path / "models--guillaumekln--faster-whisper-tiny").mkdir()
    (tmp_path / "models--openai--clip-vit-base").mkdir()
    (tmp_path / "not-a-model").mkdir()

    names = set(model_cache.list_whisper_models(tmp_path))
    assert "Systran/faster-whisper-small" in names
    assert "guillaumekln/faster-whisper-tiny" in names
    assert "openai/clip-vit-base" not in names


def test_clear_cache_removes_whisper_entries_only(tmp_path: Path) -> None:
    from src.utils import model_cache

    whisper = tmp_path / "models--Systran--faster-whisper-small"
    whisper.mkdir()
    (whisper / "file").write_bytes(b"x" * 100)

    other = tmp_path / "models--openai--clip-vit-base"
    other.mkdir()
    (other / "file").write_bytes(b"y" * 100)

    freed = model_cache.clear_whisper_cache(tmp_path)
    assert freed >= 100
    assert not whisper.exists()
    assert other.exists()


def test_clear_cache_missing_dir_is_zero(tmp_path: Path) -> None:
    from src.utils import model_cache

    assert model_cache.clear_whisper_cache(tmp_path / "nope") == 0


def test_format_size_readable() -> None:
    from src.utils import model_cache

    assert model_cache.format_size(0) == "0 B"
    assert model_cache.format_size(999) == "999 B"
    assert model_cache.format_size(1500) == "1.5 KB"
    assert model_cache.format_size(2 * (1 << 20)) == "2.0 MB"
    assert model_cache.format_size(3 * (1 << 30)) == "3.0 GB"
