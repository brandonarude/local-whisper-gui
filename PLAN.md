# Local Whisper GUI — Implementation Plan

> Companion to `SPEC.md`. Each checkbox below is a single commit.
> Development is **TDD / red-green**: for every piece of pure logic, the
> failing-test commit lands first, then the commit that makes it pass. UI
> widgets use lighter smoke tests (single commit) because full PyQt TDD has
> too much ceremony for the value.

## TDD rules for this project

- **RED commits** add only tests (or fixtures). CI/`pytest` must fail on these
  commits — that failure is the point, it proves the test actually exercises
  the thing being built.
- **GREEN commits** add the minimum implementation that turns the previous
  commit's failures green. Refactoring beyond "make it pass" gets its own
  commit.
- **Core logic** (`src/core/**`, `src/utils/**` except `theme.py`): strict
  red/green pairs.
- **UI/integration** (`src/ui/**`, `src/workers/**`, `src/app.py`,
  `src/main.py`): single commit with a `pytest-qt` smoke test (instantiate,
  assert no exception, assert key widgets exist) where feasible.
- **Test fixtures** (tiny generated wavs, synthetic segment lists) live in
  `tests/fixtures/` and are committed with the RED they first serve.

---

## Phase 1 — Foundation

- [x] **01. Project scaffolding** — directory tree per §9; `requirements.txt`; `requirements-dev.txt` (pytest, pytest-qt, pytest-mock, numpy); `.gitignore`; stub `README.md`.
- [x] **02. Test infrastructure** — `pytest.ini` / `pyproject.toml` pytest config, `tests/conftest.py` with `qapp` fixture and `tiny_wav` generator (1-second 440Hz sine via numpy), smoke test that imports `src` and asserts `QApplication` can be created.
- [x] **03. App entry point** — `src/main.py` + `src/app.py`: launches an empty `QMainWindow` titled "Local Whisper GUI"; smoke test asserts window class instantiates.
- [x] **04. Constants module** — `src/utils/constants.py`: Whisper model list + VRAM hints, language list, chunking defaults, overlap seconds. Trivial assertion test (non-empty, English present, defaults in range).

## Phase 2 — Utilities (TDD pairs)

- [x] **05. RED: device detection tests** — `tests/test_device_detect.py` covers: CPU always present; CUDA-absent case (mock `torch.cuda.is_available() = False`) → only CPU; CUDA-present case (mock device count + name + vram) → GPU entry with expected fields.
- [x] **06. GREEN: device detection** — `src/utils/device_detect.py` returning `list[Device]`.
- [x] **07. RED: config persistence tests** — `tests/test_config.py`: round-trip theme / model / device / language / formats / geometry; unset keys return documented defaults; uses a temp `QSettings` scope so tests don't touch user settings.
- [x] **08. GREEN: config persistence** — `src/utils/config.py` wrapping `QSettings` with typed getters/setters.
- [x] **09. Theme manager** — `src/utils/theme.py` (apply light/dark/system palettes, detect via `QStyleHints.colorScheme()`); smoke test asserts `apply_theme` runs for each mode without raising.

## Phase 3 — Audio core (TDD pairs)

- [x] **10. RED: audio load/probe tests** — `tests/test_audio_processor.py::test_probe_*`: correct duration/sample-rate/channels/size for the `tiny_wav` fixture; missing-file raises; non-audio-file raises.
- [x] **11. GREEN: audio load/probe** — `src/core/audio_processor.py` probe via `pydub.utils.mediainfo`.
- [x] **12. RED: waveform extraction tests** — same test file: `generate_waveform_samples` returns exactly `target_points`, mono float array in `[-1, 1]`, monotonic time axis.
- [x] **13. GREEN: waveform extraction** — downsampled peak extraction with numpy.
- [x] **14. RED: chunking tests** — fixture: long wav of sine + inserted silence; asserts chunk count matches expected, no chunk shorter than `min_chunk`, none longer than `max_chunk`, consecutive chunks overlap by ~2s, full audio coverage (sum of non-overlap durations ≈ total).
- [x] **15. GREEN: chunking** — `chunk_audio()` using `pydub.silence.split_on_silence` + min/max enforcement + overlap per §6.

## Phase 4 — Core logic (TDD pairs)

- [x] **16. RED: exporter tests** — synthetic `[Segment]` input → asserts `.txt` (with and without timestamps), `.srt` parses back via a minimal SRT parser, `.vtt` starts with `WEBVTT`, `.json` contains word-level data when provided.
- [x] **17. GREEN: exporter** — `src/core/exporter.py`.
- [x] **18. RED: ETA estimator tests** — feed synthetic `(audio_secs_done, wall_secs_elapsed)` observations; assert `.remaining()` converges; zero-elapsed returns `None`; handles non-monotonic updates.
- [x] **19. GREEN: ETA estimator** — `src/core/estimator.py` sliding-window throughput.
- [x] **20. RED: stitcher tests** — the accuracy-critical suite (§3.7, §6): overlapping synthetic chunks with known ground-truth text → assert no dropped words, no duplicated words, timestamps offset correctly, low-confidence splice logged. Includes an adversarial case where overlap text is repeated phrasing.
- [x] **21. GREEN: stitcher** — `src/core/stitcher.py` using `difflib.SequenceMatcher` on tokenized overlap.
- [ ] **22. RED: transcriber wrapper tests** — patch `faster_whisper.WhisperModel`; assert lazy load, correct `(model, device, compute_type)` forwarding, segment normalization into our dataclass, cancellation cooperation via an injected `should_cancel` callable.
- [ ] **23. GREEN: transcriber wrapper** — `src/core/transcriber.py`.

## Phase 5 — UI (single commits with smoke tests)

- [ ] **24. Waveform worker** — `src/workers/waveform_worker.py` `QThread`; smoke test with `pytest-qt` `qtbot.waitSignal(samples_ready)` against the `tiny_wav` fixture.
- [ ] **25. Waveform widget** — `src/ui/waveform_widget.py` pyqtgraph wrapper; smoke test: instantiate, `set_samples`, `set_chunk_boundaries`, assert no exception.
- [ ] **26. File header + load flow** — "Load Audio File" button, file dialog, metadata labels, waveform wired via worker.
- [ ] **27. Settings panel** — `src/ui/settings_panel.py`: model / device / language (searchable) / output formats / timestamps toggle / output dir / chunking group.
- [ ] **28. Progress panel skeleton** — `src/ui/progress_panel.py`: Start/Cancel buttons (disabled until file loaded), progress bar, ETA label, log view.
- [ ] **29. Main window assembly** — compose all panels; status bar with device/model info.
- [ ] **30. Menu bar + theme toggle** — File, Settings (Theme, Pre-download Model, Clear Cache), Help; help `?` tooltips on Model/Device.
- [ ] **31. Dialogs** — `src/ui/dialogs.py`: >1h chunk recommendation, error dialog, partial-output-on-cancel prompt, missing-ffmpeg startup dialog.
- [ ] **32. Chunk preview on waveform** — when chunking toggle is on and a file is loaded, compute boundaries in the background and draw them.

## Phase 6 — Transcription integration

- [ ] **33. Transcription worker** — `src/workers/transcription_worker.py` `QThread`; `pytest-qt` smoke test with the mocked transcriber asserting signal order (`chunk_started` → `segment` → `chunk_completed` → `finished`).
- [ ] **34. Wire Start/Cancel to worker** — progress panel receives worker signals; cancel triggers `requestInterruption()` and preserves completed chunks.
- [ ] **35. Post-transcription export** — stitch → write selected formats → success notification with "Open folder"; on cancel, offer partial export.

## Phase 7 — Polish & robustness

- [ ] **36. Startup checks** — ffmpeg presence, faster-whisper import, CUDA probe (§5.2, §7); dialogs for each failure mode with "don't show again" for non-fatal warnings.
- [ ] **37. Error handling pass** — OOM retry suggestion, corrupted-file guard, disk-full pre-check before export, model-download retry. Add targeted RED/GREEN pairs for any new pure-logic helpers introduced here.
- [ ] **38. Settings persistence wire-up** — load on startup, save on close (geometry + all settings-panel values + theme + last output dir).
- [ ] **39. Model cache management** — Settings menu: "Pre-download model" (progress) and "Clear cached models" (confirm + size preview).

## Phase 8 — Packaging

- [ ] **40. PyInstaller Linux spec** — `build/build_linux.spec`; README notes ffmpeg as a system dep.
- [ ] **41. PyInstaller Windows spec** — `build/build_windows.spec`; bundle ffmpeg.exe; document CUDA DLL bundling vs. user-supplied toolkit.
- [ ] **42. README finalization** — install (user + dev), run-from-source, platform build steps, screenshots, §12 success-criteria checklist.

---

## Notes for future-me

- **App stays runnable after every commit.** If a feature needs more than one commit, stub the UI so launch still works.
- **RED commits must actually fail.** Run `pytest` before committing a RED and confirm the new tests are failing for the right reason (not an import error, not a typo).
- **GREEN commits must not change tests.** If a test needs to change, that's a separate commit with a rationale in the message.
- **Stitcher (20/21) is the accuracy-critical pair (§3.7, §6).** Don't merge 21 until 20's adversarial cases pass.
- **Out of scope for v0.1 (§10):** batch processing, in-app editing, mic recording, translation, diarization, macOS.
