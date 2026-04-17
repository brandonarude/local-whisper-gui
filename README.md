# Local Whisper GUI

A cross-platform desktop app for transcribing audio files locally with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). No cloud,
no API keys. Targets Windows 11 and Linux.

See `SPEC.md` for the full product spec and `PLAN.md` for the
commit-by-commit implementation history.

## Features

- Load any ffmpeg-compatible audio file; see its waveform and metadata.
- Silence-based chunking for long files (>1h auto-prompt) with a
  boundary preview on the waveform.
- faster-whisper backend with selectable model, compute device
  (CPU / CUDA), and language (auto-detect or 90+ explicit options).
- Exports to `.txt`, `.srt`, `.vtt`, and word-level `.json`.
- Real-time progress, ETA, and per-chunk log; cancel at any time and
  keep the partial transcript.
- Light/dark/system theme; settings persist across sessions.
- Model cache management (pre-download, clear) from the Settings menu.

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on `PATH` (or bundled, see Windows
  build notes below)
- (Optional) NVIDIA GPU + CUDA Toolkit for GPU transcription

## Install — end users

Pre-built releases will land on the project's Releases page once the
first tagged build ships. For now, run from source.

## Install — from source

```bash
git clone <repo-url> local-whisper-gui
cd local-whisper-gui
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Install — developers

```bash
pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes `pytest`, `pytest-qt`, `pytest-mock`,
and `pyinstaller` on top of the runtime deps.

## Run

```bash
python -m src.main
```

## Tests

```bash
QT_QPA_PLATFORM=offscreen pytest
```

`QT_QPA_PLATFORM=offscreen` keeps PyQt6 headless so the suite runs
without a display server (useful on WSL / CI). Drop it when debugging
widgets interactively.

## Build — Linux

```bash
pyinstaller build/build_linux.spec
```

Output: `dist/local-whisper-gui/`. ffmpeg is **not** bundled — install
it via your distro's package manager (`apt install ffmpeg`,
`dnf install ffmpeg`, etc.). CUDA support uses the user's existing
NVIDIA driver + CUDA Toolkit.

## Build — Windows

```bat
pyinstaller build\build_windows.spec
```

Output: `dist\local-whisper-gui\`.

- **ffmpeg:** drop `ffmpeg.exe` (and optionally `ffprobe.exe`) into
  `build\vendor\ffmpeg\` before building to bundle it. If that folder
  is empty, the frozen app falls back to whatever is on `PATH`.
- **CUDA DLLs:** two options, documented at the top of
  `build\build_windows.spec`. Bundle the required cuBLAS / cuDNN DLLs
  into `build\vendor\cuda\` for a self-contained GPU build, or leave
  it empty and require users to install the CUDA Toolkit themselves.
  CPU transcription works either way.

## Screenshots

_To be added once the first release ships; the `assets/` directory is
reserved for these._

## Success criteria (SPEC §12)

- [ ] Load any ffmpeg-compatible audio file and display its waveform
- [ ] Transcribe a short (<5 min) audio file with any model on CPU
      and GPU
- [ ] Transcribe a long (>1 hour) file with chunking, producing
      accurate stitched output
- [ ] Export to all four formats with correct timestamps
- [ ] Cancel mid-transcription and recover partial output
- [ ] Build standalone executables for Windows and Linux
- [ ] UI remains responsive during all long-running operations
