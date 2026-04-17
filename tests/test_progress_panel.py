"""Smoke tests for ProgressPanel (SPEC §3.8)."""
from __future__ import annotations


def test_progress_panel_starts_with_buttons_disabled(qtbot) -> None:
    from src.ui.progress_panel import ProgressPanel, TranscriptionState

    p = ProgressPanel()
    qtbot.addWidget(p)
    assert p.state() is TranscriptionState.IDLE_NO_FILE
    assert not p._start_button.isEnabled()
    assert not p._cancel_button.isEnabled()


def test_file_loaded_enables_start_not_cancel(qtbot) -> None:
    from src.ui.progress_panel import ProgressPanel, TranscriptionState

    p = ProgressPanel()
    qtbot.addWidget(p)
    p.set_file_loaded(True)
    assert p.state() is TranscriptionState.READY
    assert p._start_button.isEnabled()
    assert not p._cancel_button.isEnabled()


def test_running_enables_cancel_not_start(qtbot) -> None:
    from src.ui.progress_panel import ProgressPanel, TranscriptionState

    p = ProgressPanel()
    qtbot.addWidget(p)
    p.set_file_loaded(True)
    p.set_running(True)
    assert p.state() is TranscriptionState.RUNNING
    assert not p._start_button.isEnabled()
    assert p._cancel_button.isEnabled()


def test_start_click_emits_signal(qtbot) -> None:
    from src.ui.progress_panel import ProgressPanel

    p = ProgressPanel()
    qtbot.addWidget(p)
    p.set_file_loaded(True)
    with qtbot.waitSignal(p.start_clicked, timeout=1_000):
        p._start_button.click()


def test_cancel_click_emits_signal(qtbot) -> None:
    from src.ui.progress_panel import ProgressPanel

    p = ProgressPanel()
    qtbot.addWidget(p)
    p.set_file_loaded(True)
    p.set_running(True)
    with qtbot.waitSignal(p.cancel_clicked, timeout=1_000):
        p._cancel_button.click()


def test_progress_and_eta_and_log_updates(qtbot) -> None:
    from src.ui.progress_panel import ProgressPanel

    p = ProgressPanel()
    qtbot.addWidget(p)
    p.set_progress_percent(42)
    assert p._progress_bar.value() == 42
    p.set_progress_percent(-5)
    assert p._progress_bar.value() == 0
    p.set_progress_percent(999)
    assert p._progress_bar.value() == 100
    p.set_eta("~4 minutes")
    assert "4 minutes" in p._eta_label.text()
    p.append_log("chunk 1 done")
    assert "chunk 1 done" in p._log.toPlainText()
    p.clear_log()
    assert p._log.toPlainText() == ""
    p.reset_progress()
    assert p._progress_bar.value() == 0


def test_set_running_is_sticky_regardless_of_file_state(qtbot) -> None:
    from src.ui.progress_panel import ProgressPanel, TranscriptionState

    p = ProgressPanel()
    qtbot.addWidget(p)
    p.set_file_loaded(True)
    p.set_running(True)
    # A spurious set_file_loaded mid-run shouldn't flip buttons.
    p.set_file_loaded(True)
    assert p.state() is TranscriptionState.RUNNING
    p.set_running(False)
    assert p.state() is TranscriptionState.READY
