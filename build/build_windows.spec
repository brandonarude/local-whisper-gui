# PyInstaller spec for Windows builds of Local Whisper GUI.
#
# Usage (from repo root, with the dev venv active on Windows):
#
#     pyinstaller build\build_windows.spec
#
# Output lands in ``dist\local-whisper-gui\``.
#
# ffmpeg bundling
# ---------------
# Drop ``ffmpeg.exe`` (and ``ffprobe.exe`` if desired) into
# ``build\vendor\ffmpeg\`` before running PyInstaller. The spec picks up
# anything in that folder and ships it alongside the frozen app. Grab
# a static build from https://www.gyan.dev/ffmpeg/builds/ or similar.
# If the vendor folder is empty, the build still succeeds — pydub will
# then rely on ffmpeg being on ``PATH`` at runtime, which is fine for
# dev builds but not recommended for releases.
#
# CUDA DLL bundling
# -----------------
# Two options, pick one at build time:
#
# 1. **Bundle** (larger installer, no user setup): drop the required
#    CUDA runtime DLLs (``cublas64_*.dll``, ``cublasLt64_*.dll``,
#    ``cudnn_ops_infer64_*.dll``, ``cudnn_cnn_infer64_*.dll``, and their
#    dependencies) into ``build\vendor\cuda\``. Matching versions ship
#    with the CUDA Toolkit / cuDNN download — see ctranslate2's README
#    for the versions it's built against.
#
# 2. **User-supplied** (smaller installer): leave ``build\vendor\cuda\``
#    empty. Users must install the NVIDIA CUDA Toolkit themselves, and
#    the README should say so. CPU transcription works either way.

# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Run from repo root: ``pyinstaller build\build_windows.spec``.
PROJECT_ROOT = Path.cwd()
ENTRY = str(PROJECT_ROOT / "src" / "main.py")
ASSETS = str(PROJECT_ROOT / "assets")
VENDOR = PROJECT_ROOT / "build" / "vendor"

datas = []
binaries = []
hiddenimports = []
for pkg in ("faster_whisper", "ctranslate2", "av", "tokenizers", "huggingface_hub"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

hiddenimports += collect_submodules("pyqtgraph")
# audioop was removed from the stdlib in Python 3.13; pydub imports it
# unconditionally. audioop-lts ships a drop-in ``audioop`` module.
hiddenimports += ["audioop"]
datas += [(ASSETS, "assets")]

# Pick up any ffmpeg/ffprobe the packager dropped in build\vendor\ffmpeg\.
ffmpeg_dir = VENDOR / "ffmpeg"
if ffmpeg_dir.is_dir():
    for path in ffmpeg_dir.glob("*.exe"):
        binaries.append((str(path), "."))

# CUDA DLLs from build\vendor\cuda\ (optional — see header).
cuda_dir = VENDOR / "cuda"
if cuda_dir.is_dir():
    for path in cuda_dir.glob("*.dll"):
        binaries.append((str(path), "."))


a = Analysis(
    [ENTRY],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "pandas",
        "IPython",
        "notebook",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="local-whisper-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="local-whisper-gui",
)
