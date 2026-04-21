"""Tests for src.core.exporter.

The exporter writes transcription results (a list of Segment / Word
dataclasses) into the four formats SPEC §3.7 requires:

- Plain text (.txt) with an optional timestamps toggle (SPEC §3.7).
- SRT subtitles (.srt) — 1-indexed blocks with ``HH:MM:SS,mmm`` cues.
- WebVTT (.vtt) — starts with ``WEBVTT``; dot-separated millisecond cues.
- JSON (.json) — structured dump including word-level timings whenever a
  segment carries them.

These tests pin the output *contract* rather than the exact formatting so the
exporter is free to evolve: SRT is validated by parsing it back with a minimal
regex-based parser, VTT by header + cue shape, TXT by presence/absence of
timestamp patterns, and JSON by round-tripping through ``json.loads``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


def _sample_segments():
    """Two-segment transcript; first has word-level timings, second does not."""
    from src.core.exporter import Segment, Word

    return [
        Segment(
            start=0.0,
            end=2.5,
            text="Hello world.",
            words=(
                Word(start=0.0, end=0.5, text="Hello", probability=0.95),
                Word(start=0.6, end=2.5, text="world.", probability=0.90),
            ),
        ),
        Segment(
            start=2.5,
            end=5.25,
            text="This is a test.",
            words=None,
        ),
    ]


# --- .txt ------------------------------------------------------------------


def test_txt_plain_contains_segment_text(tmp_path: Path) -> None:
    from src.core.exporter import write_txt

    out = tmp_path / "out.txt"
    write_txt(_sample_segments(), out, include_timestamps=False)

    content = out.read_text(encoding="utf-8")
    assert "Hello world." in content
    assert "This is a test." in content


def test_txt_plain_has_no_timestamps(tmp_path: Path) -> None:
    from src.core.exporter import write_txt

    out = tmp_path / "out.txt"
    write_txt(_sample_segments(), out, include_timestamps=False)

    content = out.read_text(encoding="utf-8")
    # Any H?H:MM or MM:SS pattern would be a timestamp leak.
    assert not re.search(r"\d{1,2}:\d{2}", content)


def test_txt_with_timestamps_includes_them(tmp_path: Path) -> None:
    from src.core.exporter import write_txt

    out = tmp_path / "out.txt"
    write_txt(_sample_segments(), out, include_timestamps=True)

    content = out.read_text(encoding="utf-8")
    assert "Hello world." in content
    assert re.search(r"\d{1,2}:\d{2}", content)


# --- .txt timestamp cadence -----------------------------------------------

_TS_LINE = re.compile(r"^\[(\d+):(\d{2})\]\s+(.*)$")


def _parse_txt_lines(content: str):
    """Return [(seconds, text), ...] for each timestamp line in ``content``."""
    out = []
    for line in content.splitlines():
        m = _TS_LINE.match(line)
        if m:
            out.append((int(m.group(1)) * 60 + int(m.group(2)), m.group(3)))
    return out


def _long_single_segment_with_words(duration_s: float = 120.0, step_s: float = 1.0):
    """One big segment spanning 0..duration_s with a word every ``step_s`` seconds."""
    from src.core.exporter import Segment, Word

    words = []
    t = 0.0
    i = 0
    while t < duration_s:
        words.append(Word(start=t, end=min(t + step_s, duration_s), text=f"w{i}", probability=1.0))
        t += step_s
        i += 1
    text = " ".join(w.text for w in words)
    return [Segment(start=0.0, end=duration_s, text=text, words=tuple(words))]


def _many_short_segments(total_s: float = 60.0, seg_s: float = 5.0):
    """Many 5-second segments with no word timings covering 0..total_s."""
    from src.core.exporter import Segment

    segs = []
    t = 0.0
    i = 0
    while t + seg_s <= total_s + 1e-9:
        segs.append(Segment(start=t, end=t + seg_s, text=f"part {i}.", words=None))
        t += seg_s
        i += 1
    return segs


def test_txt_timestamp_cadence_defaults_to_30s(tmp_path: Path) -> None:
    from src.core.exporter import write_txt

    out = tmp_path / "out.txt"
    write_txt(_long_single_segment_with_words(120.0), out, include_timestamps=True)

    content = out.read_text(encoding="utf-8")
    lines = _parse_txt_lines(content)
    # 120s audio at 30s cadence → timestamps at 0, 30, 60, 90.
    assert [secs for secs, _ in lines] == [0, 30, 60, 90]


def test_txt_timestamp_cadence_custom_interval(tmp_path: Path) -> None:
    from src.core.exporter import write_txt

    out = tmp_path / "out.txt"
    write_txt(
        _long_single_segment_with_words(60.0),
        out,
        include_timestamps=True,
        timestamp_cadence_s=15,
    )

    lines = _parse_txt_lines(out.read_text(encoding="utf-8"))
    assert [secs for secs, _ in lines] == [0, 15, 30, 45]


def test_txt_timestamp_cadence_groups_short_segments(tmp_path: Path) -> None:
    from src.core.exporter import write_txt

    out = tmp_path / "out.txt"
    write_txt(
        _many_short_segments(total_s=60.0, seg_s=5.0),
        out,
        include_timestamps=True,
        timestamp_cadence_s=30,
    )

    lines = _parse_txt_lines(out.read_text(encoding="utf-8"))
    # 12 short 5-second segments should collapse to two cadence groups.
    assert [secs for secs, _ in lines] == [0, 30]


def test_txt_timestamp_cadence_preserves_all_text(tmp_path: Path) -> None:
    from src.core.exporter import write_txt

    segs = _many_short_segments(total_s=60.0, seg_s=5.0)
    out = tmp_path / "out.txt"
    write_txt(segs, out, include_timestamps=True, timestamp_cadence_s=30)

    content = out.read_text(encoding="utf-8")
    for seg in segs:
        assert seg.text in content


def test_txt_timestamp_cadence_longer_than_audio_single_line(tmp_path: Path) -> None:
    from src.core.exporter import write_txt

    out = tmp_path / "out.txt"
    write_txt(
        _many_short_segments(total_s=20.0, seg_s=5.0),
        out,
        include_timestamps=True,
        timestamp_cadence_s=60,
    )

    lines = _parse_txt_lines(out.read_text(encoding="utf-8"))
    assert len(lines) == 1
    assert lines[0][0] == 0


def test_txt_timestamp_cadence_rejects_non_positive(tmp_path: Path) -> None:
    from src.core.exporter import write_txt

    out = tmp_path / "out.txt"
    with pytest.raises(ValueError):
        write_txt(
            _sample_segments(),
            out,
            include_timestamps=True,
            timestamp_cadence_s=0,
        )
    with pytest.raises(ValueError):
        write_txt(
            _sample_segments(),
            out,
            include_timestamps=True,
            timestamp_cadence_s=-5,
        )


def test_txt_timestamp_cadence_ignored_when_timestamps_disabled(tmp_path: Path) -> None:
    from src.core.exporter import write_txt

    out = tmp_path / "out.txt"
    # Passing a bogus cadence must not raise when timestamps are off.
    write_txt(
        _sample_segments(),
        out,
        include_timestamps=False,
        timestamp_cadence_s=0,
    )
    content = out.read_text(encoding="utf-8")
    assert not re.search(r"\[\d+:\d{2}\]", content)


def test_cadence_exceeds_duration_helper() -> None:
    from src.core.exporter import cadence_exceeds_duration

    assert cadence_exceeds_duration(20.0, 30) is True
    assert cadence_exceeds_duration(30.0, 30) is False
    assert cadence_exceeds_duration(120.0, 30) is False
    # Zero or unknown duration → treat as exceeded (caller should warn).
    assert cadence_exceeds_duration(0.0, 30) is True


# --- .srt ------------------------------------------------------------------


_SRT_CUE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+"
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


def _parse_srt(content: str):
    blocks = [b for b in re.split(r"\r?\n\r?\n", content.strip()) if b.strip()]
    parsed = []
    for block in blocks:
        lines = block.splitlines()
        assert lines, "SRT block must not be empty"
        index = int(lines[0].strip())
        match = _SRT_CUE.match(lines[1])
        assert match, f"Bad SRT cue line: {lines[1]!r}"
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(x) for x in match.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
        text = "\n".join(lines[2:]).strip()
        parsed.append((index, start, end, text))
    return parsed


def test_srt_has_one_block_per_segment(tmp_path: Path) -> None:
    from src.core.exporter import write_srt

    out = tmp_path / "out.srt"
    write_srt(_sample_segments(), out)

    parsed = _parse_srt(out.read_text(encoding="utf-8"))
    assert len(parsed) == 2


def test_srt_indexing_is_one_based_and_sequential(tmp_path: Path) -> None:
    from src.core.exporter import write_srt

    out = tmp_path / "out.srt"
    write_srt(_sample_segments(), out)

    parsed = _parse_srt(out.read_text(encoding="utf-8"))
    assert [p[0] for p in parsed] == [1, 2]


def test_srt_preserves_segment_timestamps(tmp_path: Path) -> None:
    from src.core.exporter import write_srt

    out = tmp_path / "out.srt"
    write_srt(_sample_segments(), out)

    parsed = _parse_srt(out.read_text(encoding="utf-8"))
    starts = [p[1] for p in parsed]
    ends = [p[2] for p in parsed]
    assert starts == pytest.approx([0.0, 2.5])
    assert ends == pytest.approx([2.5, 5.25])


def test_srt_preserves_segment_text(tmp_path: Path) -> None:
    from src.core.exporter import write_srt

    out = tmp_path / "out.srt"
    write_srt(_sample_segments(), out)

    parsed = _parse_srt(out.read_text(encoding="utf-8"))
    texts = [p[3] for p in parsed]
    assert texts == ["Hello world.", "This is a test."]


# --- .vtt ------------------------------------------------------------------


def test_vtt_starts_with_webvtt_header(tmp_path: Path) -> None:
    from src.core.exporter import write_vtt

    out = tmp_path / "out.vtt"
    write_vtt(_sample_segments(), out)

    content = out.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT")


def test_vtt_uses_dot_separated_millis(tmp_path: Path) -> None:
    from src.core.exporter import write_vtt

    out = tmp_path / "out.vtt"
    write_vtt(_sample_segments(), out)

    content = out.read_text(encoding="utf-8")
    # VTT cues are "HH:MM:SS.mmm --> HH:MM:SS.mmm" (dot, not comma).
    assert re.search(
        r"\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}",
        content,
    )
    # No SRT-style comma cue should leak in.
    assert not re.search(
        r"\d{2}:\d{2}:\d{2},\d{3}\s+-->", content
    )


def test_vtt_contains_all_segment_text(tmp_path: Path) -> None:
    from src.core.exporter import write_vtt

    out = tmp_path / "out.vtt"
    write_vtt(_sample_segments(), out)

    content = out.read_text(encoding="utf-8")
    assert "Hello world." in content
    assert "This is a test." in content


# --- .json -----------------------------------------------------------------


def test_json_contains_all_segments(tmp_path: Path) -> None:
    from src.core.exporter import write_json

    out = tmp_path / "out.json"
    write_json(_sample_segments(), out)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert "segments" in data
    assert len(data["segments"]) == 2


def test_json_segment_fields(tmp_path: Path) -> None:
    from src.core.exporter import write_json

    out = tmp_path / "out.json"
    write_json(_sample_segments(), out)

    data = json.loads(out.read_text(encoding="utf-8"))
    first = data["segments"][0]
    assert first["text"] == "Hello world."
    assert first["start"] == pytest.approx(0.0)
    assert first["end"] == pytest.approx(2.5)


def test_json_includes_word_level_data_when_present(tmp_path: Path) -> None:
    from src.core.exporter import write_json

    out = tmp_path / "out.json"
    write_json(_sample_segments(), out)

    data = json.loads(out.read_text(encoding="utf-8"))
    first = data["segments"][0]
    assert first.get("words"), "word-level data must be present when supplied"
    assert len(first["words"]) == 2
    word = first["words"][0]
    assert word["text"] == "Hello"
    assert word["start"] == pytest.approx(0.0)
    assert word["end"] == pytest.approx(0.5)
    assert word["probability"] == pytest.approx(0.95)


def test_json_omits_words_when_not_provided(tmp_path: Path) -> None:
    from src.core.exporter import write_json

    out = tmp_path / "out.json"
    write_json(_sample_segments(), out)

    data = json.loads(out.read_text(encoding="utf-8"))
    second = data["segments"][1]
    # Either the key is absent or explicitly null/empty — any of those means
    # "no word-level data" to downstream consumers.
    assert not second.get("words")


# --- empty input -----------------------------------------------------------


def test_writers_accept_empty_segment_list(tmp_path: Path) -> None:
    from src.core.exporter import write_json, write_srt, write_txt, write_vtt

    write_txt([], tmp_path / "e.txt", include_timestamps=False)
    write_srt([], tmp_path / "e.srt")
    write_vtt([], tmp_path / "e.vtt")
    write_json([], tmp_path / "e.json")

    assert (tmp_path / "e.txt").exists()
    assert (tmp_path / "e.srt").exists()
    assert (tmp_path / "e.vtt").read_text(encoding="utf-8").startswith("WEBVTT")
    assert json.loads((tmp_path / "e.json").read_text(encoding="utf-8")) == {
        "segments": []
    }
