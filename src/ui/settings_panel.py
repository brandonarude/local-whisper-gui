"""Settings panel: model / device / language / output / chunking (SPEC §3.4–§3.7, §4).

The panel is a pure view — it renders controls populated from
:mod:`src.utils.constants` and :mod:`src.utils.device_detect`, surfaces
the current values through read/write helpers, and emits
:attr:`values_changed` whenever the user touches a control. Persistence
is deliberately out of scope (commit 38 wires QSettings load/save); the
main window does that after instantiating the panel.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.utils import constants as C
from src.utils.device_detect import Device, detect_devices


@dataclass(frozen=True)
class SettingsValues:
    model: str
    device_key: str  # "cpu" or "cuda:{index}"
    language: str  # ISO code or C.AUTO_DETECT_LANGUAGE
    output_formats: tuple[str, ...]
    include_timestamps: bool
    timestamp_cadence_s: int
    output_dir: str | None
    chunking_enabled: bool
    min_silence_ms: int
    silence_threshold_dbfs: int
    min_chunk_minutes: int
    max_chunk_minutes: int


def _device_key(device: Device) -> str:
    if device.kind == "cpu":
        return "cpu"
    return f"cuda:{device.index or 0}"


class SettingsPanel(QWidget):
    values_changed = pyqtSignal()

    def __init__(
        self,
        devices: list[Device] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._devices = devices if devices is not None else detect_devices()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        core_form = QFormLayout()
        core_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self._model_combo = QComboBox()
        for m in C.MODELS:
            self._model_combo.addItem(
                f"{m.name}  (~{m.vram_hint_gb:.1f} GB — {m.description})", m.name
            )
        self._set_combo_value(self._model_combo, C.DEFAULT_MODEL)
        self._model_help = QLabel("(?)")
        self._model_help.setToolTip(
            "Whisper model. Larger models are more accurate but slower and need "
            "more VRAM. See dropdown for approximate requirements."
        )
        core_form.addRow("Model:", self._row(self._model_combo, self._model_help))

        self._device_combo = QComboBox()
        for d in self._devices:
            self._device_combo.addItem(d.label, _device_key(d))
        if not any(d.kind == "cuda" for d in self._devices):
            tip = "No CUDA-capable GPU detected — CPU only."
        else:
            tip = "CUDA (GPU) is much faster but requires an NVIDIA GPU."
        self._device_help = QLabel("(?)")
        self._device_help.setToolTip(tip)
        core_form.addRow("Device:", self._row(self._device_combo, self._device_help))

        self._language_combo = QComboBox()
        self._language_combo.setEditable(True)
        for code, name in C.LANGUAGES.items():
            self._language_combo.addItem(name, code)
        self._language_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        completer = QCompleter([name for name in C.LANGUAGES.values()], self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._language_combo.setCompleter(completer)
        self._set_combo_value(self._language_combo, C.DEFAULT_LANGUAGE)
        core_form.addRow("Language:", self._language_combo)

        outer.addLayout(core_form)

        # Output group ----------------------------------------------------
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)

        formats_row = QHBoxLayout()
        self._format_checks: dict[str, QCheckBox] = {}
        for fmt in C.OUTPUT_FORMATS:
            cb = QCheckBox(f".{fmt}")
            cb.setChecked(fmt in C.DEFAULT_OUTPUT_FORMATS)
            cb.stateChanged.connect(self._emit_changed)
            self._format_checks[fmt] = cb
            formats_row.addWidget(cb)
        formats_row.addStretch(1)
        output_layout.addLayout(formats_row)

        self._timestamps_check = QCheckBox("Include timestamps in .txt")
        self._timestamps_check.setChecked(True)
        self._timestamps_check.stateChanged.connect(self._emit_changed)
        output_layout.addWidget(self._timestamps_check)

        cadence_row = QHBoxLayout()
        cadence_row.setContentsMargins(24, 0, 0, 0)
        self._cadence_label = QLabel("Timestamp every:")
        self._cadence_spin = QSpinBox()
        self._cadence_spin.setRange(
            C.MIN_TIMESTAMP_CADENCE_S, C.MAX_TIMESTAMP_CADENCE_S
        )
        self._cadence_spin.setSingleStep(5)
        self._cadence_spin.setSuffix(" s")
        self._cadence_spin.setValue(C.DEFAULT_TIMESTAMP_CADENCE_S)
        self._cadence_spin.setToolTip(
            "Insert a [M:SS] marker every N seconds of audio in the .txt "
            "transcript. If set longer than the loaded file, only a single "
            "marker appears and a warning is shown at transcription start."
        )
        self._cadence_spin.valueChanged.connect(self._emit_changed)
        self._timestamps_check.toggled.connect(self._sync_cadence_enabled)
        cadence_row.addWidget(self._cadence_label)
        cadence_row.addWidget(self._cadence_spin)
        cadence_row.addStretch(1)
        output_layout.addLayout(cadence_row)
        self._sync_cadence_enabled(self._timestamps_check.isChecked())

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Output dir:"))
        self._output_dir_edit = QLineEdit()
        self._output_dir_edit.setPlaceholderText(
            "Same directory as input file"
        )
        self._output_dir_edit.editingFinished.connect(self._emit_changed)
        dir_row.addWidget(self._output_dir_edit, stretch=1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._on_browse_output_dir)
        dir_row.addWidget(browse)
        output_layout.addLayout(dir_row)

        outer.addWidget(output_group)

        # Chunking group --------------------------------------------------
        self._chunking_group = QGroupBox("Chunking (recommended for >1h files)")
        self._chunking_group.setCheckable(True)
        self._chunking_group.setChecked(True)
        self._chunking_group.toggled.connect(self._emit_changed)

        chunking_form = QFormLayout(self._chunking_group)

        self._min_silence_spin = QSpinBox()
        self._min_silence_spin.setRange(100, 10_000)
        self._min_silence_spin.setSingleStep(100)
        self._min_silence_spin.setSuffix(" ms")
        self._min_silence_spin.setValue(C.DEFAULT_MIN_SILENCE_MS)
        self._min_silence_spin.valueChanged.connect(self._emit_changed)
        chunking_form.addRow("Min silence:", self._min_silence_spin)

        self._silence_thresh_spin = QSpinBox()
        self._silence_thresh_spin.setRange(-80, -10)
        self._silence_thresh_spin.setSuffix(" dBFS")
        self._silence_thresh_spin.setValue(C.DEFAULT_SILENCE_THRESHOLD_DBFS)
        self._silence_thresh_spin.valueChanged.connect(self._emit_changed)
        chunking_form.addRow("Silence threshold:", self._silence_thresh_spin)

        self._min_chunk_spin = QDoubleSpinBox()
        self._min_chunk_spin.setRange(0.5, 60.0)
        self._min_chunk_spin.setSingleStep(0.5)
        self._min_chunk_spin.setDecimals(1)
        self._min_chunk_spin.setSuffix(" min")
        self._min_chunk_spin.setValue(float(C.DEFAULT_MIN_CHUNK_MINUTES))
        self._min_chunk_spin.valueChanged.connect(self._emit_changed)
        chunking_form.addRow("Min chunk:", self._min_chunk_spin)

        self._max_chunk_spin = QDoubleSpinBox()
        self._max_chunk_spin.setRange(1.0, 120.0)
        self._max_chunk_spin.setSingleStep(1.0)
        self._max_chunk_spin.setDecimals(1)
        self._max_chunk_spin.setSuffix(" min")
        self._max_chunk_spin.setValue(float(C.DEFAULT_MAX_CHUNK_MINUTES))
        self._max_chunk_spin.valueChanged.connect(self._emit_changed)
        chunking_form.addRow("Max chunk:", self._max_chunk_spin)

        outer.addWidget(self._chunking_group)
        outer.addStretch(1)

        self._model_combo.currentIndexChanged.connect(self._emit_changed)
        self._device_combo.currentIndexChanged.connect(self._emit_changed)
        self._language_combo.currentIndexChanged.connect(self._emit_changed)

    # --- public API -----------------------------------------------------

    def values(self) -> SettingsValues:
        formats = tuple(
            fmt for fmt, cb in self._format_checks.items() if cb.isChecked()
        )
        dir_text = self._output_dir_edit.text().strip()
        return SettingsValues(
            model=self._model_combo.currentData() or C.DEFAULT_MODEL,
            device_key=self._device_combo.currentData() or "cpu",
            language=self._language_combo.currentData() or C.DEFAULT_LANGUAGE,
            output_formats=formats,
            include_timestamps=self._timestamps_check.isChecked(),
            timestamp_cadence_s=int(self._cadence_spin.value()),
            output_dir=dir_text if dir_text else None,
            chunking_enabled=self._chunking_group.isChecked(),
            min_silence_ms=self._min_silence_spin.value(),
            silence_threshold_dbfs=self._silence_thresh_spin.value(),
            min_chunk_minutes=int(round(self._min_chunk_spin.value())),
            max_chunk_minutes=int(round(self._max_chunk_spin.value())),
        )

    def set_model(self, name: str) -> None:
        self._set_combo_value(self._model_combo, name)

    def set_device(self, key: str) -> None:
        self._set_combo_value(self._device_combo, key)

    def set_language(self, code: str) -> None:
        self._set_combo_value(self._language_combo, code)

    def set_output_formats(self, formats: list[str] | tuple[str, ...]) -> None:
        wanted = set(formats)
        for fmt, cb in self._format_checks.items():
            cb.setChecked(fmt in wanted)

    def set_include_timestamps(self, enabled: bool) -> None:
        self._timestamps_check.setChecked(enabled)

    def set_timestamp_cadence_s(self, seconds: int) -> None:
        self._cadence_spin.setValue(int(seconds))

    def set_output_dir(self, path: str | None) -> None:
        self._output_dir_edit.setText(path or "")

    def set_chunking_enabled(self, enabled: bool) -> None:
        self._chunking_group.setChecked(enabled)

    def set_min_silence_ms(self, ms: int) -> None:
        self._min_silence_spin.setValue(int(ms))

    def set_silence_threshold_dbfs(self, dbfs: int) -> None:
        self._silence_thresh_spin.setValue(int(dbfs))

    def set_min_chunk_minutes(self, minutes: float) -> None:
        self._min_chunk_spin.setValue(float(minutes))

    def set_max_chunk_minutes(self, minutes: float) -> None:
        self._max_chunk_spin.setValue(float(minutes))

    # --- helpers --------------------------------------------------------

    def _set_combo_value(self, combo: QComboBox, data_value: str) -> None:
        idx = combo.findData(data_value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _emit_changed(self, *_args) -> None:
        self.values_changed.emit()

    def _sync_cadence_enabled(self, enabled: bool) -> None:
        self._cadence_label.setEnabled(bool(enabled))
        self._cadence_spin.setEnabled(bool(enabled))

    def _row(self, *widgets: QWidget) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        for w in widgets:
            h.addWidget(w)
        h.addStretch(1)
        return row

    def _on_browse_output_dir(self) -> None:
        start = self._output_dir_edit.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Select output directory", start)
        if chosen:
            self._output_dir_edit.setText(chosen)
            self._emit_changed()
