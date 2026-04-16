"""Persistent settings (SPEC §5.5).

Thin typed wrapper over QSettings so the rest of the codebase doesn't
have to care about QSettings' string-y default behaviour or its
single-item-list flattening quirk.
"""
from __future__ import annotations

from PyQt6.QtCore import QByteArray, QSettings

from src.utils import constants as C

ORG = "LocalWhisperGUI"
APP = "LocalWhisperGUI"


class Config:
    def __init__(self, settings: QSettings | None = None) -> None:
        self._s = settings if settings is not None else QSettings(ORG, APP)

    # --- theme ----------------------------------------------------------
    def theme(self) -> str:
        return self._s.value("theme", "system", type=str)

    def set_theme(self, value: str) -> None:
        self._s.setValue("theme", value)

    # --- model / device / language --------------------------------------
    def model(self) -> str:
        return self._s.value("model", C.DEFAULT_MODEL, type=str)

    def set_model(self, value: str) -> None:
        self._s.setValue("model", value)

    def device(self) -> str:
        return self._s.value("device", "cpu", type=str)

    def set_device(self, value: str) -> None:
        self._s.setValue("device", value)

    def language(self) -> str:
        return self._s.value("language", C.DEFAULT_LANGUAGE, type=str)

    def set_language(self, value: str) -> None:
        self._s.setValue("language", value)

    # --- output ---------------------------------------------------------
    def output_formats(self) -> list[str]:
        # QSettings flattens a single-element list to a bare string when
        # round-tripping through INI; normalize back to list[str] here.
        v = self._s.value("output_formats", list(C.DEFAULT_OUTPUT_FORMATS))
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v]

    def set_output_formats(self, value: list[str]) -> None:
        self._s.setValue("output_formats", list(value))

    def include_timestamps(self) -> bool:
        return self._s.value("include_timestamps", True, type=bool)

    def set_include_timestamps(self, value: bool) -> None:
        self._s.setValue("include_timestamps", value)

    def output_dir(self) -> str | None:
        v = self._s.value("output_dir", None)
        return str(v) if v else None

    def set_output_dir(self, value: str) -> None:
        self._s.setValue("output_dir", value)

    # --- chunking -------------------------------------------------------
    def min_silence_ms(self) -> int:
        return self._s.value(
            "chunking/min_silence_ms", C.DEFAULT_MIN_SILENCE_MS, type=int
        )

    def set_min_silence_ms(self, value: int) -> None:
        self._s.setValue("chunking/min_silence_ms", int(value))

    def silence_threshold_dbfs(self) -> int:
        return self._s.value(
            "chunking/silence_threshold_dbfs",
            C.DEFAULT_SILENCE_THRESHOLD_DBFS,
            type=int,
        )

    def set_silence_threshold_dbfs(self, value: int) -> None:
        self._s.setValue("chunking/silence_threshold_dbfs", int(value))

    def min_chunk_minutes(self) -> int:
        return self._s.value(
            "chunking/min_chunk_minutes", C.DEFAULT_MIN_CHUNK_MINUTES, type=int
        )

    def set_min_chunk_minutes(self, value: int) -> None:
        self._s.setValue("chunking/min_chunk_minutes", int(value))

    def max_chunk_minutes(self) -> int:
        return self._s.value(
            "chunking/max_chunk_minutes", C.DEFAULT_MAX_CHUNK_MINUTES, type=int
        )

    def set_max_chunk_minutes(self, value: int) -> None:
        self._s.setValue("chunking/max_chunk_minutes", int(value))

    def chunking_enabled(self) -> bool:
        return self._s.value("chunking/enabled", True, type=bool)

    def set_chunking_enabled(self, value: bool) -> None:
        self._s.setValue("chunking/enabled", value)

    # --- window geometry -----------------------------------------------
    def window_geometry(self) -> QByteArray | None:
        return self._read_bytes("window/geometry")

    def set_window_geometry(self, value: QByteArray) -> None:
        self._s.setValue("window/geometry", value)

    def window_state(self) -> QByteArray | None:
        return self._read_bytes("window/state")

    def set_window_state(self, value: QByteArray) -> None:
        self._s.setValue("window/state", value)

    def _read_bytes(self, key: str) -> QByteArray | None:
        v = self._s.value(key, None)
        if v is None:
            return None
        if isinstance(v, QByteArray):
            return None if v.isEmpty() else v
        # INI may surface bytes/str; coerce.
        if isinstance(v, (bytes, bytearray)):
            return QByteArray(bytes(v)) if v else None
        if isinstance(v, str):
            return QByteArray(v.encode()) if v else None
        return v
