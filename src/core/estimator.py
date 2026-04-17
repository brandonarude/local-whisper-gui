"""Runtime ETA estimator for transcription (SPEC §3.8).

The transcription worker pushes ``(audio_done_s, wall_elapsed_s)``
observations into an :class:`Estimator` as each chunk (or segment) reports
progress. :meth:`Estimator.remaining` returns the current best estimate of
the wall time left to finish the full audio.

Throughput is computed over a sliding window of the most recent observations
so that the ETA follows real changes in transcription rate (warm-up phase,
device switch, late-file slowdown) rather than being dragged by the
start-of-run average.

Robustness rules:

- Zero wall-time elapsed ⇒ ``None`` (can't divide by zero).
- Non-monotonic observations (audio or wall moving backwards) don't raise
  and never yield a negative ETA — we fall back to the most recent
  single-point ratio when the window delta is non-positive.
- Observations past the known total clamp remaining audio to 0.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class _Observation:
    audio_done_s: float
    wall_elapsed_s: float


class Estimator:
    def __init__(self, total_audio_s: float, *, window_size: int = 10) -> None:
        self._total = max(float(total_audio_s), 0.0)
        self._window: deque[_Observation] = deque(
            maxlen=max(int(window_size), 2)
        )

    def update(self, audio_done_s: float, wall_elapsed_s: float) -> None:
        self._window.append(
            _Observation(float(audio_done_s), float(wall_elapsed_s))
        )

    def remaining(self) -> float | None:
        if not self._window:
            return None

        latest = self._window[-1]
        audio_left = max(self._total - latest.audio_done_s, 0.0)
        if audio_left == 0.0:
            return 0.0

        # Prefer the window delta when it's well-defined (both positive).
        if len(self._window) >= 2:
            earliest = self._window[0]
            d_audio = latest.audio_done_s - earliest.audio_done_s
            d_wall = latest.wall_elapsed_s - earliest.wall_elapsed_s
            if d_wall > 0 and d_audio > 0:
                throughput = d_audio / d_wall
                return audio_left / throughput

        # Fall back to the latest absolute ratio. Handles single-observation
        # case and non-monotonic windows where the delta isn't usable.
        if latest.wall_elapsed_s > 0 and latest.audio_done_s > 0:
            throughput = latest.audio_done_s / latest.wall_elapsed_s
            return audio_left / throughput

        return None
