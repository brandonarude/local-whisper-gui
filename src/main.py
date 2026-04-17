"""Program entry point: `python -m src.main`."""
from __future__ import annotations

import sys

from src.app import MainWindow, create_application, handle_startup_checks
from src.utils.config import Config


def main(argv: list[str] | None = None) -> int:
    app = create_application(argv if argv is not None else sys.argv)
    config = Config()
    handle_startup_checks(parent=None, config=config)
    window = MainWindow(config=config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
