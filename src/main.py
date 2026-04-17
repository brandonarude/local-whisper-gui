"""Program entry point: `python -m src.main`."""
from __future__ import annotations

import sys

from src.app import MainWindow, create_application, handle_startup_checks


def main(argv: list[str] | None = None) -> int:
    app = create_application(argv if argv is not None else sys.argv)
    handle_startup_checks(parent=None)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
