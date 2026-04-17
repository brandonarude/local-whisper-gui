"""Program entry point: `python -m src.main`."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _register_bundled_binaries() -> None:
    # In a PyInstaller onedir/onefile build, ffmpeg.exe/ffprobe.exe ship
    # alongside the frozen exe (see build/build_windows.spec). pydub resolves
    # them via shutil.which(), which only searches PATH — the exe's own
    # directory is not on PATH — so prepend it here before pydub imports.
    if not getattr(sys, "frozen", False):
        return
    bundle_dir = Path(sys.executable).resolve().parent
    os.environ["PATH"] = str(bundle_dir) + os.pathsep + os.environ.get("PATH", "")


_register_bundled_binaries()

from src.app import MainWindow, create_application, handle_startup_checks  # noqa: E402
from src.utils.config import Config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    app = create_application(argv if argv is not None else sys.argv)
    config = Config()
    handle_startup_checks(parent=None, config=config)
    window = MainWindow(config=config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
