# Local Whisper GUI

A cross-platform desktop app for transcribing audio files locally with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). No cloud, no API
keys.

**Status:** pre-alpha / under construction. See `SPEC.md` for the product spec
and `PLAN.md` for the commit-by-commit implementation plan.

## Requirements

- Python 3.10+
- ffmpeg installed on `PATH`
- (Optional) NVIDIA GPU + CUDA for faster transcription

## Install (from source)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Run

```bash
python -m src.main
```

## Tests

```bash
pytest
```

## Build (Linux)

ffmpeg is **not** bundled into the Linux build — install it from your
distribution's package manager (`apt install ffmpeg`, `dnf install
ffmpeg`, etc.) before running the frozen app.

```bash
pyinstaller build/build_linux.spec
```

The output directory is `dist/local-whisper-gui/`. CUDA support requires
the user's existing NVIDIA driver + CUDA Toolkit install.
