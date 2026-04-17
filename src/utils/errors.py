"""Exception classification (SPEC §7).

Predicates operate on class names and message substrings rather than
``isinstance`` against the real upstream types: ``torch.cuda.OutOfMemoryError``,
``huggingface_hub`` HTTP errors, and pydub's ``CouldntDecodeError`` aren't
guaranteed to be importable at app launch, so matching by name keeps the
classification robust to those dependencies being absent, mocked in tests,
or subclassed downstream.
"""
from __future__ import annotations

import errno
from enum import Enum


class ErrorKind(Enum):
    CUDA_OOM = "cuda_oom"
    MODEL_DOWNLOAD = "model_download"
    AUDIO_DECODE = "audio_decode"
    DISK_FULL = "disk_full"
    UNKNOWN = "unknown"


_OOM_MESSAGE_MARKERS = ("out of memory", "cuda oom")


def is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"OutOfMemoryError", "CUDAOutOfMemoryError"}:
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _OOM_MESSAGE_MARKERS)


_DOWNLOAD_CLASS_MARKERS = (
    "HTTPError",
    "ConnectionError",
    "ReadTimeout",
    "Timeout",
    "SSLError",
    "RepositoryNotFoundError",
    "RevisionNotFoundError",
)

_DOWNLOAD_OSERROR_MESSAGE_MARKERS = (
    "connection reset",
    "connection refused",
    "name resolution",
    "network is unreachable",
    "temporary failure",
)


def is_model_download_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if any(marker in name for marker in _DOWNLOAD_CLASS_MARKERS):
        return True
    if isinstance(exc, OSError):
        msg = str(exc).lower()
        return any(m in msg for m in _DOWNLOAD_OSERROR_MESSAGE_MARKERS)
    return False


def is_disk_full(exc: BaseException) -> bool:
    if isinstance(exc, OSError):
        if getattr(exc, "errno", None) == errno.ENOSPC:
            return True
        msg = str(exc).lower()
        return "no space left" in msg
    return False


def is_audio_decode_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    return "CouldntDecode" in name or "DecodeError" in name


def classify(exc: BaseException) -> ErrorKind:
    # Order matters: disk-full must be checked before the generic download
    # fall-through (both are OSError subtypes in practice).
    if is_cuda_oom(exc):
        return ErrorKind.CUDA_OOM
    if is_disk_full(exc):
        return ErrorKind.DISK_FULL
    if is_model_download_error(exc):
        return ErrorKind.MODEL_DOWNLOAD
    if is_audio_decode_error(exc):
        return ErrorKind.AUDIO_DECODE
    return ErrorKind.UNKNOWN
