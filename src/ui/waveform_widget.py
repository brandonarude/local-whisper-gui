"""Waveform display (SPEC §3.2).

Thin wrapper around :class:`pyqtgraph.PlotWidget` that lets the main window
push a :class:`~src.core.audio_processor.Waveform` (from the background
worker) and a list of chunk-boundary times (from the chunker preview, SPEC
§3.3). The widget owns no domain logic — upstream produces arrays, we
render them.

Axes are fixed: x in seconds, y in normalised amplitude [-1, 1]. Mouse
interaction (pan/zoom) is disabled because we only use this for overview,
not inspection.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6 import sip
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from src.core.audio_processor import Waveform


class WaveformWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot = pg.PlotWidget(background=None)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setYRange(-1.0, 1.0, padding=0)
        self._plot.getViewBox().setDefaultPadding(0)
        layout.addWidget(self._plot)

        self._curve: pg.PlotDataItem | None = None
        self._boundary_lines: list[pg.InfiniteLine] = []
        self._duration_s: float = 0.0

    def set_samples(self, waveform: Waveform) -> None:
        # Queued cross-thread slot: the widget may already be mid-teardown.
        if sip.isdeleted(self) or sip.isdeleted(self._plot):
            return
        times = np.asarray(waveform.times, dtype=np.float64)
        amps = np.asarray(waveform.amplitudes, dtype=np.float64)
        if self._curve is None:
            self._curve = self._plot.plot(
                times, amps, pen=pg.mkPen(color=(80, 140, 220), width=1)
            )
        else:
            self._curve.setData(times, amps)

        if times.size:
            self._duration_s = float(times[-1])
            try:
                self._plot.setXRange(0.0, self._duration_s, padding=0)
            except RuntimeError:
                # pyqtgraph internals (ViewBox.childGroup) can be torn down
                # between the check above and this call during widget teardown.
                return
        else:
            self._duration_s = 0.0

    def set_chunk_boundaries(self, boundaries_s: Sequence[float]) -> None:
        if sip.isdeleted(self) or sip.isdeleted(self._plot):
            return
        for line in self._boundary_lines:
            self._plot.removeItem(line)
        self._boundary_lines.clear()

        pen = pg.mkPen(color=QColor(220, 120, 80), width=1, style=Qt.PenStyle.DashLine)
        for t in boundaries_s:
            line = pg.InfiniteLine(pos=float(t), angle=90, movable=False, pen=pen)
            self._plot.addItem(line)
            self._boundary_lines.append(line)

    def clear(self) -> None:
        if self._curve is not None:
            self._plot.removeItem(self._curve)
            self._curve = None
        self.set_chunk_boundaries([])
        self._duration_s = 0.0
