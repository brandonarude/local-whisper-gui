"""Tests for src.utils.startup_checks (SPEC §5.2, §7).

The checks are pure probes over the host environment: ffmpeg on PATH,
faster-whisper importable, CUDA visible to torch. Each returns a
``CheckResult`` so the UI can render consistent messaging and decide
which failures are fatal vs. non-fatal.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock


# --- ffmpeg --------------------------------------------------------------

def test_check_ffmpeg_present(mocker) -> None:
    from src.utils import startup_checks

    mocker.patch.object(startup_checks.shutil, "which", return_value="/usr/bin/ffmpeg")

    result = startup_checks.check_ffmpeg()
    assert result.ok is True
    assert result.name == "ffmpeg"
    assert result.severity == "info"
    assert "/usr/bin/ffmpeg" in result.detail


def test_check_ffmpeg_missing(mocker) -> None:
    from src.utils import startup_checks

    mocker.patch.object(startup_checks.shutil, "which", return_value=None)

    result = startup_checks.check_ffmpeg()
    assert result.ok is False
    assert result.name == "ffmpeg"
    assert result.severity == "error"
    assert "ffmpeg" in result.message.lower()


# --- faster-whisper ------------------------------------------------------

def test_check_faster_whisper_present(mocker) -> None:
    from src.utils import startup_checks

    fake_spec = MagicMock(name="fw_spec")
    mocker.patch.object(
        startup_checks.importlib.util, "find_spec", return_value=fake_spec
    )

    result = startup_checks.check_faster_whisper()
    assert result.ok is True
    assert result.name == "faster_whisper"
    assert result.severity == "info"


def test_check_faster_whisper_missing(mocker) -> None:
    from src.utils import startup_checks

    mocker.patch.object(
        startup_checks.importlib.util, "find_spec", return_value=None
    )

    result = startup_checks.check_faster_whisper()
    assert result.ok is False
    assert result.name == "faster_whisper"
    assert result.severity == "error"
    assert "faster-whisper" in result.message.lower() or "faster_whisper" in result.message.lower()


def test_check_faster_whisper_find_spec_raises(mocker) -> None:
    """A corrupted install can make find_spec raise; we must not crash."""
    from src.utils import startup_checks

    mocker.patch.object(
        startup_checks.importlib.util,
        "find_spec",
        side_effect=ValueError("no parent package"),
    )

    result = startup_checks.check_faster_whisper()
    assert result.ok is False
    assert result.severity == "error"


# --- CUDA ---------------------------------------------------------------

def test_check_cuda_available(mocker) -> None:
    from src.utils import startup_checks

    fake = MagicMock(name="torch")
    fake.cuda.is_available.return_value = True
    fake.cuda.device_count.return_value = 1
    mocker.patch.dict(sys.modules, {"torch": fake})

    result = startup_checks.check_cuda()
    assert result.ok is True
    assert result.name == "cuda"
    assert result.severity == "info"


def test_check_cuda_unavailable_is_not_fatal(mocker) -> None:
    """CUDA absence is expected, not an error — only flag it as a warning
    so the UI can explain the grayed-out GPU option."""
    from src.utils import startup_checks

    fake = MagicMock(name="torch")
    fake.cuda.is_available.return_value = False
    mocker.patch.dict(sys.modules, {"torch": fake})

    result = startup_checks.check_cuda()
    assert result.ok is True  # not fatal
    assert result.name == "cuda"
    assert result.severity == "warning"
    assert "cpu" in result.message.lower()


def test_check_cuda_torch_missing(mocker) -> None:
    from src.utils import startup_checks

    mocker.patch.dict(sys.modules, {"torch": None})

    result = startup_checks.check_cuda()
    assert result.ok is True  # not fatal
    assert result.severity == "warning"


# --- aggregate ----------------------------------------------------------

def test_run_startup_checks_reports_each_probe(mocker) -> None:
    from src.utils import startup_checks

    mocker.patch.object(startup_checks.shutil, "which", return_value="/usr/bin/ffmpeg")
    mocker.patch.object(
        startup_checks.importlib.util, "find_spec", return_value=MagicMock()
    )
    fake = MagicMock(name="torch")
    fake.cuda.is_available.return_value = True
    fake.cuda.device_count.return_value = 1
    mocker.patch.dict(sys.modules, {"torch": fake})

    results = startup_checks.run_startup_checks()
    names = [r.name for r in results]
    assert names == ["ffmpeg", "faster_whisper", "cuda"]
    assert all(r.ok for r in results)


def test_has_fatal_true_when_any_fatal(mocker) -> None:
    from src.utils import startup_checks

    mocker.patch.object(startup_checks.shutil, "which", return_value=None)
    mocker.patch.object(
        startup_checks.importlib.util, "find_spec", return_value=MagicMock()
    )
    fake = MagicMock(name="torch")
    fake.cuda.is_available.return_value = True
    fake.cuda.device_count.return_value = 1
    mocker.patch.dict(sys.modules, {"torch": fake})

    results = startup_checks.run_startup_checks()
    assert startup_checks.has_fatal(results) is True


def test_has_fatal_false_when_all_ok(mocker) -> None:
    from src.utils import startup_checks

    mocker.patch.object(startup_checks.shutil, "which", return_value="/usr/bin/ffmpeg")
    mocker.patch.object(
        startup_checks.importlib.util, "find_spec", return_value=MagicMock()
    )
    fake = MagicMock(name="torch")
    fake.cuda.is_available.return_value = False
    mocker.patch.dict(sys.modules, {"torch": fake})

    results = startup_checks.run_startup_checks()
    assert startup_checks.has_fatal(results) is False
