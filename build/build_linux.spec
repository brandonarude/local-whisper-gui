# PyInstaller spec for Linux builds of Local Whisper GUI.
#
# Usage (from repo root, with the dev venv active):
#
#     pyinstaller build/build_linux.spec
#
# Output lands in ``dist/local-whisper-gui/``. ffmpeg is expected to be
# installed system-wide (``apt install ffmpeg`` / ``dnf install ffmpeg``);
# it is not bundled into the Linux build. CUDA support relies on the
# user's NVIDIA driver + CUDA Toolkit — see README for details.

# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Run from repo root: ``pyinstaller build/build_linux.spec``.
PROJECT_ROOT = Path.cwd()
ENTRY = str(PROJECT_ROOT / "src" / "main.py")
ASSETS = str(PROJECT_ROOT / "assets")

# faster-whisper ships model metadata and links ctranslate2; av (PyAV) and
# tokenizers carry native libs. collect_all drags in submodules + data +
# binaries so they survive the freeze.
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
        # Heavy scientific deps pulled transitively that we never import.
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
