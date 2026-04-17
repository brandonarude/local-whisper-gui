"""Tests for src.core.estimator.

The ETA estimator takes (audio_seconds_transcribed, wall_seconds_elapsed)
observations during transcription and exposes ``.remaining()`` which returns
an estimate of the wall time left, in seconds, to finish the full audio (or
``None`` if there is not yet enough information to form an estimate).

SPEC §3.8: ETA is computed at runtime from elapsed time and progress; no
pre-transcription benchmarking. PLAN commit 19: sliding-window throughput —
older observations fall out of the window so the estimator tracks changes in
transcription rate (e.g., a model warm-up phase followed by steady state, or
a device switch).
"""
from __future__ import annotations

import pytest


# --- degenerate inputs -----------------------------------------------------


def test_no_observations_returns_none() -> None:
    from src.core.estimator import Estimator

    est = Estimator(total_audio_s=100.0)
    assert est.remaining() is None


def test_zero_wall_elapsed_returns_none() -> None:
    from src.core.estimator import Estimator

    est = Estimator(total_audio_s=100.0)
    est.update(audio_done_s=0.0, wall_elapsed_s=0.0)
    assert est.remaining() is None


def test_zero_wall_elapsed_with_progress_still_returns_none() -> None:
    # Defensive: even if audio_done > 0 with wall=0 (clock quantisation,
    # clearly nonsense), we can't divide by zero — return None.
    from src.core.estimator import Estimator

    est = Estimator(total_audio_s=100.0)
    est.update(audio_done_s=5.0, wall_elapsed_s=0.0)
    assert est.remaining() is None


# --- steady-state convergence ---------------------------------------------


def test_single_observation_yields_estimate() -> None:
    # throughput = 10 audio / 5 wall = 2×, remaining audio = 90, ETA = 45 s.
    from src.core.estimator import Estimator

    est = Estimator(total_audio_s=100.0)
    est.update(audio_done_s=10.0, wall_elapsed_s=5.0)
    assert est.remaining() == pytest.approx(45.0)


def test_remaining_converges_under_steady_throughput() -> None:
    """Stream of observations at throughput=2× should give ETA = (total-done)/2."""
    from src.core.estimator import Estimator

    est = Estimator(total_audio_s=100.0, window_size=5)
    for i in range(1, 11):
        est.update(audio_done_s=2.0 * i, wall_elapsed_s=float(i))
    # Last observation: done=20, wall=10 → throughput=2, remaining=(100-20)/2=40.
    assert est.remaining() == pytest.approx(40.0, abs=1e-6)


def test_sliding_window_tracks_rate_change() -> None:
    """When throughput changes mid-run, the window should follow the new rate.

    First 5 observations are at throughput=0.5× (slow), the next 5 at
    throughput=5× (fast). With window_size=5 the slow observations fall out
    of the window and ``.remaining()`` should reflect the fast rate.
    """
    from src.core.estimator import Estimator

    est = Estimator(total_audio_s=200.0, window_size=5)

    # Slow phase: wall 1..5, audio = 0.5 * wall.
    for i in range(1, 6):
        est.update(audio_done_s=0.5 * i, wall_elapsed_s=float(i))
    slow_eta = est.remaining()
    assert slow_eta is not None and slow_eta > 300.0  # slow → huge remaining

    # Fast phase: wall 6..10, audio jumps by 5 per wall second.
    audio = 2.5
    for i in range(6, 11):
        audio += 5.0
        est.update(audio_done_s=audio, wall_elapsed_s=float(i))

    # After fast phase, latest obs is (27.5, 10). The window holds wall 6..10
    # → Δaudio = 25, Δwall = 4, throughput = 6.25. ETA = (200-27.5)/6.25 ≈ 27.6.
    fast_eta = est.remaining()
    assert fast_eta is not None
    assert fast_eta == pytest.approx((200.0 - 27.5) / 6.25, abs=1e-3)
    assert fast_eta < slow_eta  # slow-phase ETA must have been larger


# --- non-monotonic / defensive --------------------------------------------


def test_non_monotonic_audio_does_not_raise() -> None:
    from src.core.estimator import Estimator

    est = Estimator(total_audio_s=100.0)
    est.update(audio_done_s=10.0, wall_elapsed_s=5.0)
    est.update(audio_done_s=8.0, wall_elapsed_s=7.0)  # audio moved backwards

    result = est.remaining()
    # Must produce a non-negative estimate or None — never a negative ETA.
    assert result is None or result >= 0.0


def test_non_monotonic_wall_does_not_raise() -> None:
    from src.core.estimator import Estimator

    est = Estimator(total_audio_s=100.0)
    est.update(audio_done_s=10.0, wall_elapsed_s=10.0)
    est.update(audio_done_s=20.0, wall_elapsed_s=5.0)  # wall moved backwards

    result = est.remaining()
    assert result is None or result >= 0.0


def test_remaining_nonnegative_when_past_total() -> None:
    """Observations beyond the known total clamp to 0, not negative."""
    from src.core.estimator import Estimator

    est = Estimator(total_audio_s=50.0)
    est.update(audio_done_s=60.0, wall_elapsed_s=30.0)
    assert est.remaining() == 0.0
