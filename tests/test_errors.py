"""Tests for src.utils.errors (SPEC §7).

Exception-classification helpers keep UI code ("was this OOM? a network
blip? disk full?") out of the widgets and concentrate the matching
rules in one place. Tests exercise the match boundaries so a typo in a
regex or class-name check won't go unnoticed.
"""
from __future__ import annotations

import errno

import pytest


# --- CUDA OOM -----------------------------------------------------------

def test_is_cuda_oom_by_class_name() -> None:
    from src.utils.errors import is_cuda_oom

    class OutOfMemoryError(RuntimeError):
        pass

    assert is_cuda_oom(OutOfMemoryError("CUDA out of memory")) is True


def test_is_cuda_oom_by_message() -> None:
    from src.utils.errors import is_cuda_oom

    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate ...")) is True
    assert is_cuda_oom(RuntimeError("cublas error: out of memory")) is True


def test_is_cuda_oom_negative() -> None:
    from src.utils.errors import is_cuda_oom

    assert is_cuda_oom(RuntimeError("something else")) is False
    assert is_cuda_oom(ValueError("out of range")) is False


# --- model download -----------------------------------------------------

def test_is_model_download_error_by_class() -> None:
    from src.utils.errors import is_model_download_error

    class HTTPError(Exception):
        pass

    class ConnectionError(Exception):
        pass

    class ReadTimeout(Exception):
        pass

    assert is_model_download_error(HTTPError("404")) is True
    assert is_model_download_error(ConnectionError("refused")) is True
    assert is_model_download_error(ReadTimeout("timed out")) is True


def test_is_model_download_error_by_oserror_message() -> None:
    from src.utils.errors import is_model_download_error

    assert is_model_download_error(OSError("Connection reset by peer")) is True
    assert is_model_download_error(OSError("Temporary failure in name resolution")) is True


def test_is_model_download_error_negative() -> None:
    from src.utils.errors import is_model_download_error

    assert is_model_download_error(ValueError("bad shape")) is False
    assert is_model_download_error(RuntimeError("CUDA out of memory")) is False


# --- disk full ----------------------------------------------------------

def test_is_disk_full_by_errno() -> None:
    from src.utils.errors import is_disk_full

    exc = OSError(errno.ENOSPC, "No space left on device")
    assert is_disk_full(exc) is True


def test_is_disk_full_by_message_fallback() -> None:
    """Some libraries surface disk-full as a raw OSError without errno."""
    from src.utils.errors import is_disk_full

    assert is_disk_full(OSError("No space left on device")) is True


def test_is_disk_full_negative() -> None:
    from src.utils.errors import is_disk_full

    assert is_disk_full(OSError(errno.EACCES, "permission denied")) is False
    assert is_disk_full(ValueError("nope")) is False


# --- audio decode -------------------------------------------------------

def test_is_audio_decode_error_by_class_name() -> None:
    from src.utils.errors import is_audio_decode_error

    class CouldntDecodeError(Exception):
        pass

    assert is_audio_decode_error(CouldntDecodeError("ffmpeg exited 1")) is True


def test_is_audio_decode_error_negative() -> None:
    from src.utils.errors import is_audio_decode_error

    assert is_audio_decode_error(ValueError("not audio")) is False


# --- classify ----------------------------------------------------------

def test_classify_prefers_most_specific(tmp_path) -> None:
    from src.utils.errors import ErrorKind, classify

    assert classify(RuntimeError("CUDA out of memory")) is ErrorKind.CUDA_OOM

    class HTTPError(Exception):
        pass

    assert classify(HTTPError("503")) is ErrorKind.MODEL_DOWNLOAD
    assert (
        classify(OSError(errno.ENOSPC, "No space left on device"))
        is ErrorKind.DISK_FULL
    )
    assert classify(ValueError("mystery")) is ErrorKind.UNKNOWN


@pytest.mark.parametrize(
    "kind",
    [
        "CUDA_OOM",
        "MODEL_DOWNLOAD",
        "AUDIO_DECODE",
        "DISK_FULL",
        "UNKNOWN",
    ],
)
def test_error_kind_enum_has_expected_members(kind: str) -> None:
    from src.utils.errors import ErrorKind

    assert hasattr(ErrorKind, kind)
