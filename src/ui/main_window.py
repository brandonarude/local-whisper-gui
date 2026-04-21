"""Main window composition (SPEC §4, §9).

Stacks the audio header, waveform, settings panel, and progress panel
into the canonical top-to-bottom layout from the spec. A status bar
reports the current device and model (updated whenever the settings
panel fires ``values_changed``). Transcription wiring lands in commit 34;
until then the Start/Cancel signals are routed through the progress
panel but perform no work.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QActionGroup
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.core import exporter
from src.core.audio_processor import AudioInfo
from src.core.estimator import Estimator
from src.core.stitcher import ChunkResult, stitch
from src.core.transcriber import Transcriber
from src.ui import dialogs
from src.ui.file_header import FileHeader
from src.ui.progress_panel import ProgressPanel
from src.ui.settings_panel import SettingsPanel
from src.ui.waveform_widget import WaveformWidget
from src.utils import constants as C
from src.utils import disk_space, model_cache
from src.utils.config import Config
from src.utils.device_detect import Device, detect_devices
from src.utils.errors import ErrorKind
from src.utils.theme import Theme, apply_theme
from src.workers.chunk_preview_worker import ChunkPreviewWorker
from src.workers.model_download_worker import ModelDownloadWorker
from src.workers.transcription_worker import ChunkParams, TranscriptionWorker
from src.workers.waveform_worker import WaveformWorker

APP_NAME = "Local Whisper GUI"


class MainWindow(QMainWindow):
    def __init__(
        self,
        devices: list[Device] | None = None,
        *,
        transcriber_factory=None,
        config: Config | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1100, 820)

        self._devices = devices if devices is not None else detect_devices()
        self._transcriber_factory = transcriber_factory or _default_transcriber_factory
        self._config = config

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._file_header = FileHeader(self)
        self._file_header.setObjectName("fileHeader")
        layout.addWidget(self._file_header)

        self._waveform = WaveformWidget(self)
        self._waveform.setObjectName("waveformWidget")
        self._waveform.setMinimumHeight(120)
        layout.addWidget(self._waveform)

        # Settings above progress, splittable so long logs can grow.
        splitter = QSplitter(Qt.Orientation.Vertical, self)
        self._settings_panel = SettingsPanel(devices=self._devices, parent=self)
        self._settings_panel.setObjectName("settingsPanel")
        splitter.addWidget(self._settings_panel)

        self._progress_panel = ProgressPanel(self)
        self._progress_panel.setObjectName("progressPanel")
        splitter.addWidget(self._progress_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)

        self.setCentralWidget(central)

        self._status = self.statusBar()
        self._refresh_status_bar()

        self._waveform_worker: WaveformWorker | None = None
        self._chunk_preview_worker: ChunkPreviewWorker | None = None
        self._transcription_worker: TranscriptionWorker | None = None
        self._estimator: Estimator | None = None
        self._transcription_results: list[ChunkResult] = []
        self._current_theme: str = Theme.SYSTEM.value

        self._build_menu_bar()

        self._file_header.file_loaded.connect(self._on_file_loaded)
        self._file_header.load_failed.connect(self._on_load_failed)
        self._settings_panel.values_changed.connect(self._refresh_status_bar)
        self._settings_panel.values_changed.connect(self._refresh_chunk_preview)
        self._progress_panel.start_clicked.connect(self._on_start_transcription)
        self._progress_panel.cancel_clicked.connect(self._on_cancel_transcription)

        if self._config is not None:
            self._apply_config(self._config)

    # --- persistence ----------------------------------------------------

    def _apply_config(self, config: Config) -> None:
        geom = config.window_geometry()
        if geom is not None:
            self.restoreGeometry(geom)
        state = config.window_state()
        if state is not None:
            self.restoreState(state)

        self._settings_panel.set_model(config.model())
        self._settings_panel.set_device(config.device())
        self._settings_panel.set_language(config.language())
        self._settings_panel.set_output_formats(config.output_formats())
        self._settings_panel.set_include_timestamps(config.include_timestamps())
        self._settings_panel.set_timestamp_cadence_s(config.timestamp_cadence_s())
        out_dir = config.output_dir()
        if out_dir:
            self._settings_panel.set_output_dir(out_dir)
        self._settings_panel.set_chunking_enabled(config.chunking_enabled())
        self._settings_panel.set_min_silence_ms(config.min_silence_ms())
        self._settings_panel.set_silence_threshold_dbfs(config.silence_threshold_dbfs())
        self._settings_panel.set_min_chunk_minutes(config.min_chunk_minutes())
        self._settings_panel.set_max_chunk_minutes(config.max_chunk_minutes())

        self.set_theme(config.theme())
        self._refresh_status_bar()

    def _save_config(self, config: Config) -> None:
        v = self._settings_panel.values()
        config.set_model(v.model)
        config.set_device(v.device_key)
        config.set_language(v.language)
        config.set_output_formats(list(v.output_formats))
        config.set_include_timestamps(v.include_timestamps)
        config.set_timestamp_cadence_s(v.timestamp_cadence_s)
        if v.output_dir:
            config.set_output_dir(v.output_dir)
        config.set_chunking_enabled(v.chunking_enabled)
        config.set_min_silence_ms(v.min_silence_ms)
        config.set_silence_threshold_dbfs(v.silence_threshold_dbfs)
        config.set_min_chunk_minutes(v.min_chunk_minutes)
        config.set_max_chunk_minutes(v.max_chunk_minutes)
        config.set_theme(self._current_theme)
        config.set_window_geometry(self.saveGeometry())
        config.set_window_state(self.saveState())

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._config is not None:
            self._save_config(self._config)
        super().closeEvent(event)

    # --- file loading ---------------------------------------------------

    def _on_file_loaded(self, info: AudioInfo) -> None:
        self._waveform.clear()
        self._progress_panel.set_file_loaded(True)
        self._progress_panel.reset_progress()
        self._progress_panel.append_log(f"Loaded {info.path.name} ({info.duration_s:.1f}s)")
        self._start_waveform_worker(info)
        self._refresh_chunk_preview()

    def _on_load_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Could not load file", message)
        self._progress_panel.append_log(f"Error: {message}")

    def _start_waveform_worker(self, info: AudioInfo) -> None:
        if self._waveform_worker is not None and self._waveform_worker.isRunning():
            self._waveform_worker.requestInterruption()
            self._waveform_worker.quit()
            self._waveform_worker.wait(2_000)

        worker = WaveformWorker(info.path, parent=self)
        worker.samples_ready.connect(self._waveform.set_samples)
        worker.failed.connect(self._on_load_failed)
        worker.finished.connect(worker.deleteLater)
        self._waveform_worker = worker
        worker.start()

    # --- chunk preview --------------------------------------------------

    def _refresh_chunk_preview(self) -> None:
        info = self._file_header.current_info
        if info is None:
            self._waveform.set_chunk_boundaries([])
            return

        values = self._settings_panel.values()
        if not values.chunking_enabled:
            self._cancel_chunk_preview_worker()
            self._waveform.set_chunk_boundaries([])
            return

        self._cancel_chunk_preview_worker()
        worker = ChunkPreviewWorker(
            info.path,
            min_silence_ms=values.min_silence_ms,
            silence_thresh_db=float(values.silence_threshold_dbfs),
            min_chunk_s=values.min_chunk_minutes * 60.0,
            max_chunk_s=values.max_chunk_minutes * 60.0,
            overlap_s=C.CHUNK_OVERLAP_SECONDS,
            parent=self,
        )
        worker.boundaries_ready.connect(self._on_chunk_boundaries_ready)
        worker.failed.connect(self._on_chunk_preview_failed)
        worker.finished.connect(worker.deleteLater)
        self._chunk_preview_worker = worker
        worker.start()

    def _cancel_chunk_preview_worker(self) -> None:
        w = self._chunk_preview_worker
        if w is not None and w.isRunning():
            w.requestInterruption()
            w.quit()
            w.wait(2_000)
        self._chunk_preview_worker = None

    def _on_chunk_boundaries_ready(self, boundaries: list) -> None:
        self._waveform.set_chunk_boundaries([float(b) for b in boundaries])

    def _on_chunk_preview_failed(self, message: str) -> None:
        self._progress_panel.append_log(f"Chunk preview failed: {message}")

    # --- status bar -----------------------------------------------------

    def _device_label_for(self, device_key: str) -> str:
        for d in self._devices:
            key = "cpu" if d.kind == "cpu" else f"cuda:{d.index or 0}"
            if key == device_key:
                return d.label
        return device_key

    # --- menu bar -------------------------------------------------------

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        load_action = file_menu.addAction("&Load Audio File…")
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._file_header._on_load_clicked)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("&Quit")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        settings_menu = mb.addMenu("&Settings")
        theme_menu = settings_menu.addMenu("&Theme")
        self._theme_actions: dict[str, "object"] = {}
        group = QActionGroup(self)
        group.setExclusive(True)
        for theme in (Theme.LIGHT, Theme.DARK, Theme.SYSTEM):
            act = theme_menu.addAction(theme.value.capitalize())
            act.setCheckable(True)
            act.setActionGroup(group)
            act.setChecked(theme.value == self._current_theme)
            act.triggered.connect(lambda _checked, t=theme.value: self.set_theme(t))
            self._theme_actions[theme.value] = act

        settings_menu.addSeparator()
        predownload_action = settings_menu.addAction("Pre-download Model…")
        predownload_action.triggered.connect(self._on_predownload_model)
        clear_cache_action = settings_menu.addAction("Clear Cached Models…")
        clear_cache_action.triggered.connect(self._on_clear_model_cache)

        help_menu = mb.addMenu("&Help")
        about_action = help_menu.addAction("&About")
        about_action.triggered.connect(self._on_about)

    def set_theme(self, theme: str) -> None:
        app = QApplication.instance()
        if app is None:
            return
        apply_theme(app, theme)
        self._current_theme = theme
        act = self._theme_actions.get(theme)
        if act is not None:
            act.setChecked(True)

    def current_theme(self) -> str:
        return self._current_theme

    # --- menu action stubs ----------------------------------------------

    def _on_predownload_model(self) -> None:
        model_names = [m.name for m in C.MODELS]
        current = self._settings_panel.values().model
        chosen = dialogs.prompt_select_model_to_download(
            self, model_names, default=current
        )
        if not chosen:
            return

        progress = QProgressDialog(
            f"Downloading {chosen}…", "Cancel", 0, 0, self
        )
        progress.setWindowTitle("Pre-download Model")
        progress.setAutoClose(False)
        progress.setMinimumDuration(0)

        worker = ModelDownloadWorker(model=chosen, parent=self)

        def _on_done():
            progress.close()
            QMessageBox.information(
                self, "Pre-download Model", f"Model '{chosen}' is ready."
            )

        def _on_failed(message: str, kind: str):
            progress.close()
            if kind == ErrorKind.MODEL_DOWNLOAD.value:
                if dialogs.prompt_download_retry(self, detail=message):
                    self._on_predownload_model()
            else:
                dialogs.show_error(self, "Download failed", message)

        worker.completed.connect(_on_done)
        worker.failed.connect(_on_failed)
        worker.finished.connect(worker.deleteLater)
        progress.canceled.connect(worker.requestInterruption)
        worker.start()
        self._model_download_worker = worker  # keep reference alive

    def _on_clear_model_cache(self) -> None:
        cache = model_cache.cache_dir()
        names = model_cache.list_whisper_models(cache)
        total = sum(
            model_cache.total_size_bytes(d)
            for d in model_cache.whisper_model_dirs(cache)
        )
        if not dialogs.confirm_clear_model_cache(
            self, model_names=names, total_bytes=total
        ):
            return
        freed = model_cache.clear_whisper_cache(cache)
        QMessageBox.information(
            self,
            "Clear Cached Models",
            f"Freed {model_cache.format_size(freed)}.",
        )

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            (
                f"{APP_NAME}\n\n"
                "Local transcription powered by faster-whisper.\n"
                "See SPEC.md for details."
            ),
        )

    def _refresh_status_bar(self) -> None:
        v = self._settings_panel.values()
        model_label = next(
            (m for m in C.MODELS if m.name == v.model), None
        )
        model_text = (
            f"{v.model} (~{model_label.vram_hint_gb:.1f} GB)"
            if model_label is not None
            else v.model
        )
        self._status.showMessage(
            f"Device: {self._device_label_for(v.device_key)}  |  Model: {model_text}"
        )

    # --- transcription control -----------------------------------------

    def _on_start_transcription(self) -> None:
        info = self._file_header.current_info
        if info is None:
            return
        if self._transcription_worker is not None and self._transcription_worker.isRunning():
            return

        values = self._settings_panel.values()
        if (
            values.include_timestamps
            and "txt" in values.output_formats
            and exporter.cadence_exceeds_duration(
                info.duration_s, values.timestamp_cadence_s
            )
        ):
            if not dialogs.prompt_cadence_exceeds_duration(
                self, info.duration_s, values.timestamp_cadence_s
            ):
                return
        device, compute_type = _split_device_key(values.device_key)
        transcriber = self._transcriber_factory(
            model=values.model, device=device, compute_type=compute_type
        )

        params = ChunkParams(
            min_silence_ms=values.min_silence_ms,
            silence_thresh_db=float(values.silence_threshold_dbfs),
            min_chunk_s=values.min_chunk_minutes * 60.0,
            max_chunk_s=values.max_chunk_minutes * 60.0,
            overlap_s=C.CHUNK_OVERLAP_SECONDS,
        )

        worker = TranscriptionWorker(
            audio_path=info.path,
            total_duration_s=info.duration_s,
            transcriber=transcriber,
            chunking_enabled=values.chunking_enabled,
            chunk_params=params,
            language=values.language,
            parent=self,
        )

        self._estimator = Estimator(info.duration_s)
        self._transcription_results = []

        self._progress_panel.clear_log()
        self._progress_panel.reset_progress()
        self._progress_panel.set_running(True)

        worker.chunk_started.connect(self._on_chunk_started)
        worker.chunk_completed.connect(self._on_chunk_completed)
        worker.progress.connect(self._on_transcription_progress)
        worker.log.connect(self._progress_panel.append_log)
        worker.completed.connect(self._on_transcription_completed)
        worker.cancelled.connect(self._on_transcription_cancelled)
        worker.failed.connect(self._on_transcription_failed)
        worker.finished.connect(worker.deleteLater)

        self._transcription_worker = worker
        worker.start()

    def _on_cancel_transcription(self) -> None:
        w = self._transcription_worker
        if w is None or not w.isRunning():
            return
        w.requestInterruption()
        self._progress_panel.append_log("Cancellation requested — finishing current chunk...")

    # --- worker signal handlers ----------------------------------------

    def _on_chunk_started(self, index: int, total: int) -> None:
        self._progress_panel.set_progress_text(f"chunk {index + 1}/{total} — %p%")

    def _on_chunk_completed(self, index: int) -> None:  # noqa: ARG002
        pass  # log message already added by the worker

    def _on_transcription_progress(
        self, percent: int, audio_done_s: float, wall_elapsed_s: float
    ) -> None:
        self._progress_panel.set_progress_percent(percent)
        if self._estimator is not None:
            self._estimator.update(audio_done_s, wall_elapsed_s)
            remaining = self._estimator.remaining()
            self._progress_panel.set_eta(_format_eta(remaining))

    def _on_transcription_completed(self, results: list) -> None:
        self._transcription_results = list(results)
        self._progress_panel.set_running(False)
        self._progress_panel.set_file_loaded(self._file_header.current_info is not None)
        self._progress_panel.append_log(
            f"Transcription complete ({len(results)} chunk(s))."
        )
        self._transcription_worker = None
        self._export_results(self._transcription_results, partial=False)

    def _on_transcription_cancelled(self, results: list) -> None:
        self._transcription_results = list(results)
        self._progress_panel.set_running(False)
        self._progress_panel.set_file_loaded(self._file_header.current_info is not None)
        self._progress_panel.append_log(
            f"Transcription cancelled — {len(results)} chunk(s) preserved."
        )
        self._transcription_worker = None

        if not results:
            return
        total_planned = max(len(results), 1)
        if dialogs.prompt_partial_output_on_cancel(self, len(results), total_planned):
            self._export_results(results, partial=True)

    def _on_transcription_failed(self, message: str, kind: str) -> None:
        self._progress_panel.set_running(False)
        self._progress_panel.set_file_loaded(self._file_header.current_info is not None)
        self._progress_panel.append_log(f"Transcription failed: {message}")
        self._transcription_worker = None

        retry = False
        if kind == ErrorKind.CUDA_OOM.value:
            retry = dialogs.prompt_oom_retry(self, detail=message)
        elif kind == ErrorKind.MODEL_DOWNLOAD.value:
            retry = dialogs.prompt_download_retry(self, detail=message)
        elif kind == ErrorKind.AUDIO_DECODE.value:
            dialogs.show_error(
                self,
                "Could not decode audio",
                "The audio file could not be decoded. It may be corrupted "
                "or in an unsupported format.",
                details=message,
            )
        elif kind == ErrorKind.DISK_FULL.value:
            dialogs.show_error(
                self,
                "Disk full",
                "The disk ran out of space during transcription. Free up "
                "space and try again.",
                details=message,
            )
        else:
            dialogs.show_error(self, "Transcription failed", message)

        if retry:
            self._on_start_transcription()

    # --- export --------------------------------------------------------

    def _export_results(self, results: list[ChunkResult], *, partial: bool) -> None:
        info = self._file_header.current_info
        if info is None or not results:
            return

        values = self._settings_panel.values()
        if not values.output_formats:
            self._progress_panel.append_log("No output formats selected — skipping export.")
            return

        segments = stitch(results, C.CHUNK_OVERLAP_SECONDS)
        if not segments:
            self._progress_panel.append_log("Stitcher produced no segments — nothing to export.")
            return

        out_dir = Path(values.output_dir) if values.output_dir else info.path.parent
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            dialogs.show_error(
                self,
                "Export failed",
                f"Could not create output directory: {out_dir}",
                details=str(exc),
            )
            return

        needed = disk_space.estimate_export_size(segments, values.output_formats)
        if not disk_space.has_sufficient_free_space(out_dir, needed):
            try:
                free = shutil.disk_usage(
                    str(out_dir if out_dir.exists() else out_dir.parent)
                ).free
            except OSError:
                free = 0
            if not dialogs.prompt_low_disk_space(
                self,
                needed_bytes=needed,
                free_bytes=free,
                output_dir=out_dir,
            ):
                self._progress_panel.append_log(
                    "Export cancelled — insufficient disk space."
                )
                return

        stem = info.path.stem + (".partial" if partial else "")
        writers = {
            "txt": lambda p: exporter.write_txt(
                segments,
                p,
                include_timestamps=values.include_timestamps,
                timestamp_cadence_s=values.timestamp_cadence_s,
            ),
            "srt": lambda p: exporter.write_srt(segments, p),
            "vtt": lambda p: exporter.write_vtt(segments, p),
            "json": lambda p: exporter.write_json(segments, p),
        }
        written: list[Path] = []
        for fmt in values.output_formats:
            writer = writers.get(fmt)
            if writer is None:
                continue
            out_path = out_dir / f"{stem}.{fmt}"
            try:
                writer(out_path)
            except OSError as exc:
                dialogs.show_error(
                    self,
                    "Export failed",
                    f"Could not write {out_path.name}",
                    details=str(exc),
                )
                return
            written.append(out_path)
            self._progress_panel.append_log(f"Wrote {out_path.name}")

        if written:
            dialogs.show_export_complete(self, out_dir, written)


# --- module-level helpers ---------------------------------------------------


def _default_transcriber_factory(*, model: str, device: str, compute_type: str) -> Transcriber:
    return Transcriber(model=model, device=device, compute_type=compute_type)


def _split_device_key(device_key: str) -> tuple[str, str]:
    """Translate the settings panel's device_key into faster-whisper args.

    ``"cpu"`` → ``("cpu", "int8")``; ``"cuda:N"`` → ``("cuda", "float16")``.
    Compute types are best-effort defaults; faster-whisper also accepts
    ``"auto"`` but explicit values make the choice visible in the log.
    """
    if device_key.startswith("cuda"):
        return "cuda", "float16"
    return "cpu", "int8"


def _format_eta(remaining_s: float | None) -> str:
    if remaining_s is None:
        return "—"
    remaining_s = max(0.0, float(remaining_s))
    if remaining_s < 1.0:
        return "<1s"
    total = int(round(remaining_s))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"~{h}h {m:02d}m"
    if m:
        return f"~{m}m {s:02d}s"
    return f"~{s}s"
