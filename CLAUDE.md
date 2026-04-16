# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This repo is **pre-implementation**. The only committed files are `SPEC.md`
(the product spec) and `PLAN.md` (the commit-by-commit implementation plan).
No source tree, dependencies, or tooling are in place yet. If you're asked to
make changes, you're almost certainly executing an item from `PLAN.md`.

Before doing anything else, read:

1. **`PLAN.md`** — the authoritative commit list. Each checkbox is one commit,
   in order. Find the next unchecked item and do that.
2. **`SPEC.md`** — feature requirements. Section numbers are referenced
   throughout `PLAN.md`; consult them when a plan item is ambiguous.

When you complete a commit, tick its checkbox in `PLAN.md` as part of that same
commit.

## Commit discipline (TDD red/green)

Development uses strict red/green pairs for all pure logic (everything under
`src/core/**` and `src/utils/**` except `theme.py`). The rules are spelled out
at the top of `PLAN.md`; the non-obvious ones:

- A **RED** commit adds only tests (and fixtures) and must actually fail
  `pytest` for the right reason — not an import error, not a typo. Run the
  tests before committing a RED to confirm.
- A **GREEN** commit adds the minimum implementation to turn the previous
  RED's failures green. It must not change any tests. If a test needs to
  change, that's a separate commit.
- UI work (`src/ui/**`, `src/workers/**`, `src/app.py`, `src/main.py`) is
  single-commit with `pytest-qt` smoke tests — not full TDD. This is
  deliberate; don't try to retrofit red/green onto widget instantiation.
- **The app must be launchable after every commit.** If a feature spans
  multiple commits, stub the UI so the rest still runs.

## Architecture notes worth loading up-front

These come from `SPEC.md` and matter across multiple files:

- **Chunk stitching is the accuracy-critical path** (§3.7, §6). Each chunk is
  transcribed with ~2s of audio overlap at the boundary; the stitcher uses
  `difflib.SequenceMatcher` on the overlap text to find a splice point,
  offsets timestamps by cumulative chunk durations, and must not drop or
  duplicate words. The adversarial cases in the stitcher test suite (PLAN
  commit 20) are the acceptance bar — don't merge the GREEN without them
  passing.
- **UI responsiveness is non-negotiable.** All long operations (waveform
  generation, chunking, transcription, model download) run in `QThread`
  workers under `src/workers/`. Nothing that touches ffmpeg or
  faster-whisper should live on the GUI thread.
- **faster-whisper models are lazy-loaded** inside the transcriber wrapper.
  First-use triggers a HuggingFace download; surface progress in the UI
  (§8.3).
- **Device detection runs at startup** and gates the GPU dropdown entry
  (§3.5, §5.2). CUDA absence is expected, not an error.
- **Settings persist via QSettings** (§5.5) — window geometry, last-used
  model/device/language, output formats and directory, chunking params,
  theme. Load on startup, save on close.

## Commands

Dev setup (one-time):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

Run tests:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/pytest
```

`QT_QPA_PLATFORM=offscreen` keeps PyQt6 headless so tests don't need a
display server (WSL, CI). Drop it when you want widgets to actually render
while debugging.

Run the app (once commit 03 lands):

```bash
.venv/bin/python -m src.main
```

**ffmpeg is a required system dependency** — if it's missing on the dev
machine, several tests (and the app) won't run. Install via your package
manager (`apt install ffmpeg`, etc.).

## Out of scope for v0.1

Per `SPEC.md` §10: batch processing, in-app transcript editing, mic
recording, translation, speaker diarization, cloud/API transcription, macOS
builds. Don't let scope creep in from related ideas.
