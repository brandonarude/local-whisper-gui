"""Tests for src.utils.disk_space (SPEC §7, disk-full pre-check).

Estimation is deliberately rough — the aim is to catch the catastrophic
"user tries to export 10 GB of JSON to a full USB stick" case, not to be
byte-accurate. Tests assert the estimate is monotonic in segment count,
format count, and word count, and that has_sufficient_free_space
consults shutil.disk_usage rather than actually probing the filesystem.
"""
from __future__ import annotations

from pathlib import Path


def _seg(words: int = 0):
    from src.core.exporter import Segment, Word

    return Segment(
        start=0.0,
        end=1.0,
        text="hello world " * max(words, 1),
        words=(
            tuple(
                Word(start=0.0, end=0.1, text=f"w{i}", probability=1.0)
                for i in range(words)
            )
            if words
            else None
        ),
    )


def test_estimate_export_size_scales_with_segments() -> None:
    from src.utils.disk_space import estimate_export_size

    one = estimate_export_size([_seg()], ["txt", "srt"])
    many = estimate_export_size([_seg() for _ in range(50)], ["txt", "srt"])
    assert many > one


def test_estimate_export_size_scales_with_formats() -> None:
    from src.utils.disk_space import estimate_export_size

    few_formats = estimate_export_size([_seg() for _ in range(10)], ["txt"])
    all_formats = estimate_export_size(
        [_seg() for _ in range(10)], ["txt", "srt", "vtt", "json"]
    )
    assert all_formats > few_formats


def test_estimate_export_size_json_accounts_for_words() -> None:
    from src.utils.disk_space import estimate_export_size

    no_words = estimate_export_size([_seg(words=0)], ["json"])
    with_words = estimate_export_size([_seg(words=20)], ["json"])
    assert with_words > no_words


def test_estimate_export_size_empty_segments_is_zero_ish() -> None:
    from src.utils.disk_space import estimate_export_size

    assert estimate_export_size([], ["txt", "srt", "vtt", "json"]) >= 0


def test_estimate_export_size_unknown_format_is_ignored() -> None:
    from src.utils.disk_space import estimate_export_size

    known = estimate_export_size([_seg()], ["txt"])
    assert estimate_export_size([_seg()], ["txt", "bogus"]) == known


def test_has_sufficient_free_space_true(tmp_path: Path, mocker) -> None:
    from src.utils import disk_space

    fake_usage = mocker.Mock(free=10_000_000_000)  # 10 GB
    mocker.patch.object(disk_space.shutil, "disk_usage", return_value=fake_usage)

    assert disk_space.has_sufficient_free_space(tmp_path, 1_000_000) is True


def test_has_sufficient_free_space_false(tmp_path: Path, mocker) -> None:
    from src.utils import disk_space

    fake_usage = mocker.Mock(free=100)
    mocker.patch.object(disk_space.shutil, "disk_usage", return_value=fake_usage)

    assert disk_space.has_sufficient_free_space(tmp_path, 1_000_000) is False


def test_has_sufficient_free_space_walks_up_to_existing_parent(
    tmp_path: Path, mocker
) -> None:
    """If the target dir doesn't exist yet, disk_usage must be called on the
    nearest existing ancestor — otherwise it raises FileNotFoundError."""
    from src.utils import disk_space

    missing = tmp_path / "not" / "yet" / "created"
    spy = mocker.patch.object(
        disk_space.shutil,
        "disk_usage",
        return_value=mocker.Mock(free=10**12),
    )

    disk_space.has_sufficient_free_space(missing, 1024)
    spy.assert_called_once()
    called_with = Path(spy.call_args.args[0])
    assert called_with.exists()


def test_has_sufficient_free_space_swallows_oserror(
    tmp_path: Path, mocker
) -> None:
    """If shutil.disk_usage blows up (permission denied, race), we must not
    block the export — return True and let the actual write surface the
    real error."""
    from src.utils import disk_space

    mocker.patch.object(
        disk_space.shutil, "disk_usage", side_effect=OSError("boom")
    )
    assert disk_space.has_sufficient_free_space(tmp_path, 1_000_000) is True
