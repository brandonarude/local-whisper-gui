"""HuggingFace model cache inspection/clearing (SPEC §5, Settings menu).

faster-whisper piggy-backs on ``huggingface_hub`` for model downloads,
which writes to ``$HF_HOME/hub`` (or ``$HUGGINGFACE_HUB_CACHE`` if set,
or ``~/.cache/huggingface/hub`` as the default). Model directories
follow the convention ``models--<org>--<name>``; we filter to entries
whose name contains ``whisper`` so "Clear cached models" doesn't touch
unrelated HF downloads the user may have.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable


def cache_dir() -> Path:
    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path(os.path.expanduser("~/.cache/huggingface/hub"))


def total_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                total += fp.stat().st_size
            except OSError:
                continue
    return total


def _is_whisper_model_dir(entry: Path) -> bool:
    name = entry.name
    if not name.startswith("models--"):
        return False
    return "whisper" in name.lower()


def _decode_model_name(entry_name: str) -> str:
    # "models--Systran--faster-whisper-small" -> "Systran/faster-whisper-small"
    return entry_name[len("models--"):].replace("--", "/")


def list_whisper_models(path: str | Path) -> list[str]:
    root = Path(path)
    if not root.exists():
        return []
    names = []
    for entry in root.iterdir():
        if entry.is_dir() and _is_whisper_model_dir(entry):
            names.append(_decode_model_name(entry.name))
    names.sort()
    return names


def whisper_model_dirs(path: str | Path) -> Iterable[Path]:
    root = Path(path)
    if not root.exists():
        return []
    return [e for e in root.iterdir() if e.is_dir() and _is_whisper_model_dir(e)]


def clear_whisper_cache(path: str | Path) -> int:
    """Remove whisper model directories under ``path``. Returns the total
    bytes freed (best-effort; silently skips files that can't be stat'd)."""
    freed = 0
    for entry in whisper_model_dirs(path):
        freed += total_size_bytes(entry)
        shutil.rmtree(entry, ignore_errors=True)
    return freed


def format_size(n: int) -> str:
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"
