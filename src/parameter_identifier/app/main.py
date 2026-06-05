from __future__ import annotations

import sys
from pathlib import Path

from PyQt5 import QtGui, QtWidgets

from parameter_identifier.app.main_window import MainWindow


def _asset_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / name


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    icon_path = next(
        (
            path
            for path in (_asset_path("app_icon.ico"), _asset_path("app_icon.png"))
            if path.exists()
        ),
        None,
    )
    if icon_path is not None:
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))
    window = MainWindow()
    if icon_path is not None:
        window.setWindowIcon(QtGui.QIcon(str(icon_path)))
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
