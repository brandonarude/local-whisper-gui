# Local Whisper GUI — Specification

> Version: 0.1 (Draft)
> Last updated: 2026-04-16

---

## 1. Overview

A cross-platform desktop GUI application for transcribing audio files using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) locally. The app
runs entirely on the user's machine — no cloud services, no API keys. It targets
Windows 11 and Linux, with standalone executables for both platforms.

---

## 2. Tech Stack

| Component          | Choice                        | Notes                                      |
|--------------------|-------------------------------|--------------------------------------------|
| Language           | Python 3.10+                  | Minimum version for faster-whisper support  |
| GUI Framework      | PyQt6                         | Cross-platform, native look                 |
| Whisper Backend    | faster-whisper                | CTranslate2-based, 4-6x faster than vanilla |
| Audio Processing   | ffmpeg (via pydub)            | Decoding, chunking, format support          |
| Waveform Rendering | numpy + pyqtgraph             | Lightweight, interactive                    |
| Packaging          | PyInstaller                   | .exe (Windows), AppImage (Linux)            |

### 2.1 Key Dependencies

```
faster-whisper
PyQt6
pydub
numpy
pyqtgraph
pyinstaller (build-time only)
```

System requirement: **ffmpeg** must be installed or bundled.

---

## 3. Features

### 3.1 Audio File Loading

- **Supported formats**: Anything ffmpeg can decode (mp3, wav, flac, m4a, ogg,
  wma, aac, opus, webm, etc.)
- File picker dialog (native OS dialog via PyQt6)
- On load:
  - Display file name, duration, sample rate, channels, and file size
  - Render a waveform visualization of the full audio
  - If duration > 1 hour, prompt the user with a recommendation to split the
    file into chunks (see §3.3)

### 3.2 Waveform Display

- Rendered using `pyqtgraph` embedded in the PyQt6 layout
- Shows amplitude over time for the full audio file
- Visual markers for chunk boundaries (when chunking is active)
- Lightweight downsampling for large files to keep rendering fast

### 3.3 Audio Chunking

- **Method**: Silence-based splitting via `pydub.silence.split_on_silence`
- **Auto-prompt threshold**: Files longer than **1 hour** trigger a
  recommendation to chunk. The user can accept or decline.
- **User controls**:
  - Minimum silence length (ms) — default: 700ms
  - Silence threshold (dBFS) — default: -40 dBFS
  - Minimum chunk length — default: 5 minutes (to avoid overly small segments)
  - Maximum chunk length — default: 30 minutes (safety cap)
- Chunks that exceed the maximum length after silence splitting are further
  split at the nearest silence within a tolerance window
- Chunk boundaries are shown on the waveform display
- Each chunk is transcribed independently, then results are stitched (see §3.7)

### 3.4 Whisper Model Selection

- **Available models**: tiny, base, small, medium, large-v2, large-v3, distil-large-v3
  (whatever faster-whisper supports at runtime)
- Dropdown selector showing model name and approximate VRAM/RAM requirements
- No pre-transcription time estimation per model — time estimates are provided
  during transcription only (see §3.8)

### 3.5 Compute Device Selection

- **Options**: CPU, CUDA (GPU)
- Dropdown selector
- **Help tooltip** (hover ? icon): explains the difference between CPU and GPU
  execution, when to choose each, and CUDA requirements
- Auto-detect available devices at startup:
  - If no CUDA-capable GPU is detected, GPU option is grayed out with an
    explanation
  - If CUDA is available, show GPU name and available VRAM

### 3.6 Language Selection

- **Searchable dropdown** (QComboBox with editable filter / completer)
- User can type to filter the language list
- **Default**: English
- "Auto-detect" option available at top of list
- Full list of Whisper-supported languages

### 3.7 Output Configuration

- **Output formats** (user selects one or more):
  - Plain text (.txt)
  - SRT subtitles (.srt)
  - VTT subtitles (.vtt)
  - JSON (.json) — includes word-level timestamps when available
- **Output directory**: folder picker, defaults to same directory as input file
- **Timestamps toggle**: include/exclude timestamps in plain text output
- **Chunk stitching**:
  - When audio was split into chunks, output is automatically stitched into a
    single continuous file per format
  - Timestamps are adjusted to be continuous across the full original file
    (chunk 2 timestamps offset by chunk 1 duration, etc.)
  - **Accuracy priority**: stitching logic must preserve exact transcription
    content — no words dropped or duplicated at chunk boundaries
  - Overlap strategy: chunks include a small overlap (~2 seconds of audio) at
    boundaries. The stitcher deduplicates overlapping text using
    sequence alignment to avoid duplication while preventing gaps.

### 3.8 Transcription Execution

- Runs in a **background QThread** — UI remains fully responsive
- **Progress bar**: shows overall progress (chunk N of M) and per-chunk progress
- **Estimated time to completion**: updated in real time based on elapsed time
  and chunks completed
- **Cancel button**: allows the user to stop transcription at any time. Already
  completed chunks are preserved and can be exported as partial output.
- Status log area showing per-chunk status messages (started, completed, errors)

---

## 4. UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Menu Bar: File | Settings (incl. Theme) | Help             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─── Audio File ────────────────────────────────────────┐  │
│  │  [Load Audio File]  filename.mp3  |  1h 23m  |  44.1k │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─── Waveform ──────────────────────────────────────────┐  │
│  │  ▁▂▃▅▇▅▃▂▁▁▂▅▇█▇▅▃▁▁▂▃▅▇▅▃▂▁▁▂▅▇█▇▅▃▁              │  │
│  │  |---- chunk 1 ----|---- chunk 2 ----|-- chunk 3 --|   │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─── Settings ──────────────────────────────────────────┐  │
│  │                                                       │  │
│  │  Model:    [large-v3        ▾]               [?]       │  │
│  │  Device:   [CUDA - RTX 3080 ▾]              [?]       │  │
│  │  Language: [English         ▾]                         │  │
│  │                                                       │  │
│  │  Output:   ☑ .txt  ☑ .srt  ☐ .vtt  ☐ .json           │  │
│  │  Timestamps in .txt: ☑                                │  │
│  │  Output dir: [/home/user/transcripts]  [Browse]       │  │
│  │                                                       │  │
│  │  Chunking: ☑ Split file (recommended for >1h)         │  │
│  │    Silence threshold: [-40 dBFS]  Min silence: [700ms]│  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─── Transcription ─────────────────────────────────────┐  │
│  │  [▶ Start Transcription]          [Cancel]            │  │
│  │                                                       │  │
│  │  Progress: ████████████░░░░░░░░  chunk 3/5  62%       │  │
│  │  ETA: ~4 minutes remaining                            │  │
│  │                                                       │  │
│  │  Log:                                                 │  │
│  │  ✓ Chunk 1/5 completed (0:00 - 18:32)                 │  │
│  │  ✓ Chunk 2/5 completed (18:32 - 35:10)                │  │
│  │  ⟳ Chunk 3/5 in progress (35:10 - 52:45)...           │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  Status bar: Ready | GPU: RTX 3080 (8GB) | Model loaded    │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Application Behavior

### 5.1 Theme

- **Light/dark mode toggle** available in Settings menu
- **Default**: Follow the system preference (detected via `QPalette` /
  `QStyleHints`). If system preference cannot be detected, default to light.
- User's choice is persisted across sessions

### 5.2 Startup

1. Detect available compute devices (CPU, CUDA GPUs)
2. Check for ffmpeg availability — show clear error if missing
3. Check for faster-whisper installation
4. Load last-used settings from config file (QSettings or JSON)
5. Apply saved theme (or detect system default on first launch)
6. Present main window

### 5.3 File Load Flow

1. User clicks "Load Audio File" → native file dialog opens
2. File is validated (ffmpeg probe for format/duration/metadata)
3. Waveform is generated asynchronously (background thread) with a loading
   indicator
4. File metadata displayed
5. If duration > 1 hour: prompt suggesting chunk splitting

### 5.4 Transcription Flow

1. User configures settings and clicks "Start Transcription"
2. If chunking enabled: audio is split (progress shown)
3. Transcription begins:
   - Each chunk (or whole file) is passed to faster-whisper
   - Progress bar and ETA update after each segment
   - Log entries appear in real time
4. On completion:
   - Chunks are stitched with adjusted timestamps
   - Output files are written to the selected directory
   - Success notification with link to output directory
5. On cancel:
   - Current chunk processing is interrupted
   - Completed chunks are offered as partial output
6. On error:
   - Error displayed in log area with details
   - Previously completed chunks preserved

### 5.5 Settings Persistence

- Save/restore via QSettings or a JSON config file in user's app data directory
- Persisted settings:
  - Theme preference (light/dark/system)
  - Last used model, device, language
  - Output format selections
  - Output directory
  - Chunking parameters
  - Window size and position

---

## 6. Chunk Stitching Strategy

Stitching is critical for accuracy. The approach:

1. **Overlap**: Each chunk includes ~2 seconds of audio from the end of the
   previous chunk
2. **Transcribe independently**: Each chunk goes through faster-whisper with
   no shared state
3. **Align overlap regions**: Use sequence matching (e.g., `difflib.SequenceMatcher`)
   on the overlapping text to find the best splice point
4. **Timestamp offset**: Add cumulative duration offset to all timestamps in
   chunks 2..N
5. **Merge**: Concatenate the non-overlapping portions of each chunk's output
6. **Validate**: Log any alignment issues (low confidence splices) so the user
   is aware

---

## 7. Error Handling

| Scenario                    | Behavior                                                  |
|-----------------------------|-----------------------------------------------------------|
| ffmpeg not found            | Startup warning with installation instructions            |
| CUDA not available          | GPU option disabled, CPU selected, tooltip explains why   |
| Corrupted audio file        | Error on load with message, no crash                      |
| Out of memory (GPU)         | Catch OOM, suggest smaller model or CPU, offer to retry   |
| Transcription failure       | Show error in log, preserve completed chunks              |
| Disk full on output         | Error message before/during write, no partial corrupt file|
| Model download failure      | Retry option, progress for model download                 |

---

## 8. Packaging & Distribution

### 8.1 Windows

- PyInstaller → single-folder or single-file .exe
- Bundle ffmpeg binary (or document as prerequisite)
- CUDA DLLs bundled for GPU support (or require user CUDA Toolkit install)
- Target: Windows 11 (may work on 10)

### 8.2 Linux

- PyInstaller → AppImage or single-folder distribution
- ffmpeg expected to be installed via package manager
- CUDA support requires user's existing NVIDIA driver + CUDA installation
- Target: Ubuntu 22.04+, Fedora 38+, and similar

### 8.3 Model Management

- faster-whisper models are downloaded on first use (~50MB to ~3GB depending on
  model size)
- Download progress shown in the UI
- Models cached in a standard location (default: `~/.cache/huggingface/`)
- Option in Settings to pre-download models or clear cache

---

## 9. Project Structure

```
local-whisper-gui/
├── src/
│   ├── main.py                  # Entry point
│   ├── app.py                   # QApplication setup
│   ├── ui/
│   │   ├── main_window.py       # Main window layout and logic
│   │   ├── waveform_widget.py   # Waveform display widget
│   │   ├── settings_panel.py    # Model/device/language/output controls
│   │   ├── progress_panel.py    # Progress bar, ETA, log
│   │   └── dialogs.py           # Chunk prompt, error dialogs, etc.
│   ├── core/
│   │   ├── transcriber.py       # faster-whisper wrapper
│   │   ├── audio_processor.py   # File loading, chunking, ffmpeg operations
│   │   ├── stitcher.py          # Chunk result stitching and timestamp adjustment
│   │   ├── estimator.py         # Runtime ETA calculation during transcription
│   │   └── exporter.py          # Output formatting (txt, srt, vtt, json)
│   ├── workers/
│   │   ├── transcription_worker.py  # QThread for transcription
│   │   ├── waveform_worker.py       # QThread for waveform generation
│   └── utils/
│       ├── config.py            # Settings persistence
│       ├── device_detect.py     # CUDA / CPU detection
│       ├── theme.py             # Light/dark theme management
│       └── constants.py         # Model info, languages, defaults
├── assets/
│   └── icons/                   # App icons, help icons
├── tests/
│   ├── test_transcriber.py
│   ├── test_audio_processor.py
│   ├── test_stitcher.py
│   └── test_exporter.py
├── build/
│   ├── build_windows.spec       # PyInstaller spec for Windows
│   └── build_linux.spec         # PyInstaller spec for Linux
├── requirements.txt
├── requirements-dev.txt
├── SPEC.md                      # This file
└── README.md
```

---

## 10. Out of Scope (for v0.1)

- File queue / batch processing
- In-app transcription editing
- Real-time / microphone recording
- Translation (whisper translate task)
- Speaker diarization
- Cloud/API-based transcription
- macOS support

---

## 11. Resolved Questions

1. **App theme**: Light/dark toggle, defaults to system preference (falls back
   to light). See §5.1.
2. **App name**: "Local Whisper GUI"
3. **Time estimation**: No pre-transcription benchmark. ETA is calculated at
   runtime during transcription based on elapsed time and progress.

---

## 12. Success Criteria

- [ ] Load any ffmpeg-compatible audio file and display its waveform
- [ ] Transcribe a short (<5 min) audio file with any model on CPU and GPU
- [ ] Transcribe a long (>1 hour) file with chunking, producing accurate
      stitched output
- [ ] Export to all four formats with correct timestamps
- [ ] Cancel mid-transcription and recover partial output
- [ ] Build standalone executables for Windows and Linux
- [ ] UI remains responsive during all long-running operations
