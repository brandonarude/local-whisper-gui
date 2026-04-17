"""Startup environment probes (SPEC §5.2, §7).

Each probe returns a :class:`CheckResult` so the UI renders consistent
messaging. Fatal failures (ffmpeg missing, faster-whisper missing) block
transcription but should still allow the main window to open so the user
sees the guidance; the caller decides what to do with them via
:func:`has_fatal`.

CUDA absence is intentionally not fatal — the app must run on CPU-only
machines, and §7 mandates that the GPU dropdown entry is just grayed out
rather than erroring.
"""
from __future__ import annotations

import importlib
import importlib.util
import shutil
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    severity: str  # "info" | "warning" | "error"
    message: str
    detail: str = ""


def check_ffmpeg() -> CheckResult:
    path = shutil.which("ffmpeg")
    if path:
        return CheckResult(
            name="ffmpeg",
            ok=True,
            severity="info",
            message="ffmpeg is available.",
            detail=path,
        )
    return CheckResult(
        name="ffmpeg",
        ok=False,
        severity="error",
        message="ffmpeg was not found on PATH.",
        detail=(
            "Install ffmpeg via your package manager "
            "(`sudo apt install ffmpeg`, `brew install ffmpeg`) or from "
            "https://ffmpeg.org/download.html on Windows."
        ),
    )


def check_faster_whisper() -> CheckResult:
    try:
        spec = importlib.util.find_spec("faster_whisper")
    except (ValueError, ModuleNotFoundError, ImportError) as exc:
        return CheckResult(
            name="faster_whisper",
            ok=False,
            severity="error",
            message="faster-whisper could not be loaded.",
            detail=f"{type(exc).__name__}: {exc}",
        )
    if spec is None:
        return CheckResult(
            name="faster_whisper",
            ok=False,
            severity="error",
            message="faster-whisper is not installed.",
            detail="Install it with `pip install faster-whisper`.",
        )
    return CheckResult(
        name="faster_whisper",
        ok=True,
        severity="info",
        message="faster-whisper is available.",
    )


def check_cuda() -> CheckResult:
    # Absence of torch / CUDA is expected on CPU-only machines; report it
    # as a non-fatal warning so the UI can explain the grayed-out GPU row.
    torch = sys.modules.get("torch")
    if torch is None:
        try:
            import torch  # type: ignore
        except ImportError:
            return CheckResult(
                name="cuda",
                ok=True,
                severity="warning",
                message="CUDA is not available — transcription will run on CPU.",
                detail="torch is not installed.",
            )

    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover — defensive
        return CheckResult(
            name="cuda",
            ok=True,
            severity="warning",
            message="CUDA is not available — transcription will run on CPU.",
            detail=f"{type(exc).__name__}: {exc}",
        )

    if not available:
        return CheckResult(
            name="cuda",
            ok=True,
            severity="warning",
            message="CUDA is not available — transcription will run on CPU.",
            detail="torch.cuda.is_available() returned False.",
        )

    count = 0
    try:
        count = int(torch.cuda.device_count())
    except Exception:  # pragma: no cover
        pass
    return CheckResult(
        name="cuda",
        ok=True,
        severity="info",
        message=f"CUDA is available ({count} device(s)).",
    )


def run_startup_checks() -> list[CheckResult]:
    return [check_ffmpeg(), check_faster_whisper(), check_cuda()]


def has_fatal(results: list[CheckResult]) -> bool:
    return any((not r.ok) and r.severity == "error" for r in results)
