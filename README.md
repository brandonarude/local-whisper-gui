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
