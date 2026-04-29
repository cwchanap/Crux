import json
import shutil
from pathlib import Path
from unittest.mock import patch

from src.benchmark.prepare import (
    DRUM_AUDIO_FILENAMES,
    prepare_corpus,
    scan_raw_corpus,
)


def test_scan_raw_corpus_selects_highest_chart_and_allowed_drum_audio(tmp_path: Path):
    raw = tmp_path / "raw"
    song = raw / "Song One"
    song.mkdir(parents=True)
    (song / "bas.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song / "ext.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song / "mas.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song / DRUM_AUDIO_FILENAMES[0]).write_bytes(b"drums")

    result = scan_raw_corpus(raw)

    assert result.invalid_items == []
    assert len(result.valid_items) == 1
    assert result.valid_items[0].song_id == "Song One"
    assert result.valid_items[0].selected_chart.name == "mas.dtx"
    assert result.valid_items[0].selected_chart_level == "mas"
    assert result.valid_items[0].selected_audio.name == DRUM_AUDIO_FILENAMES[0]


def test_scan_raw_corpus_rejects_missing_allowed_drum_audio(tmp_path: Path):
    raw = tmp_path / "raw"
    song = raw / "Song Two"
    song.mkdir(parents=True)
    (song / "adv.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song / "bgm.ogg").write_bytes(b"mix")

    result = scan_raw_corpus(raw)

    assert result.valid_items == []
    assert result.invalid_items[0].reason == "missing allowed drum audio file"


def test_scan_raw_corpus_rejects_multiple_allowed_drum_audio_files(tmp_path: Path):
    raw = tmp_path / "raw"
    song = raw / "Song Three"
    song.mkdir(parents=True)
    (song / "mas.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song / "2 Drums.mp3").write_bytes(b"a")
    (song / "drum.mp3").write_bytes(b"b")

    result = scan_raw_corpus(raw)

    assert result.valid_items == []
    assert result.invalid_items[0].reason == "multiple allowed drum audio files"


def test_prepare_corpus_writes_parsed_dirs_and_reports(tmp_path: Path):
    raw = tmp_path / "raw"
    good = raw / "Good Song"
    bad = raw / "Bad Song"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    (good / "ext.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (good / "drum.mp3").write_bytes(b"drums")
    (bad / "bas.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    output = tmp_path / "parsed"

    result = prepare_corpus(raw, output)

    assert len(result.valid_items) == 1
    assert len(result.invalid_items) == 1
    assert (output / "charts" / "Good Song.dtx").exists()
    assert (output / "audio" / "Good Song.mp3").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    invalid = json.loads((output / "invalid.json").read_text())
    assert manifest["items"][0]["song_id"] == "Good Song"
    assert manifest["items"][0]["selected_chart_level"] == "ext"
    assert invalid["items"][0]["raw_folder"].endswith("Bad Song")


def test_prepare_corpus_preserves_txt_chart_suffix(tmp_path: Path):
    raw = tmp_path / "raw"
    song = raw / "TxtSong"
    song.mkdir(parents=True)
    (song / "ext.txt").write_text("#BPM: 120\n", encoding="utf-8")
    (song / "drum.mp3").write_bytes(b"drums")
    output = tmp_path / "parsed"

    result = prepare_corpus(raw, output)

    assert len(result.valid_items) == 1
    assert len(result.invalid_items) == 0
    assert (output / "charts" / "TxtSong.txt").exists()
    assert not (output / "charts" / "TxtSong.dtx").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    entry = manifest["items"][0]
    assert entry["selected_chart"] == "ext.txt"
    assert entry["parsed_chart_path"].endswith("TxtSong.txt")


def test_prepare_corpus_removes_stale_files_on_rerun(tmp_path: Path):
    """Re-running prepare_corpus into the same output dir must not leave stale files."""
    raw = tmp_path / "raw"
    output = tmp_path / "parsed"

    # First run: two songs
    song_a = raw / "SongA"
    song_a.mkdir(parents=True)
    (song_a / "ext.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song_a / "drum.mp3").write_bytes(b"drums-a")

    song_b = raw / "SongB"
    song_b.mkdir(parents=True)
    (song_b / "ext.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song_b / "drum.mp3").write_bytes(b"drums-b")

    prepare_corpus(raw, output)
    assert (output / "charts" / "SongA.dtx").exists()
    assert (output / "charts" / "SongB.dtx").exists()
    assert (output / "audio" / "SongA.mp3").exists()
    assert (output / "audio" / "SongB.mp3").exists()

    # Remove SongB from raw dir, re-run with only SongA
    shutil.rmtree(song_b)

    prepare_corpus(raw, output)

    # SongA still present; SongB must be gone
    assert (output / "charts" / "SongA.dtx").exists()
    assert (output / "audio" / "SongA.mp3").exists()
    assert not (output / "charts" / "SongB.dtx").exists()
    assert not (output / "audio" / "SongB.mp3").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["items"]) == 1
    assert manifest["items"][0]["song_id"] == "SongA"


def test_prepare_corpus_excludes_copy_failures_from_valid_items(tmp_path: Path):
    """Items that fail during file copy must not appear in valid_items or manifest."""
    raw = tmp_path / "raw"
    output = tmp_path / "parsed"

    song_a = raw / "SongA"
    song_a.mkdir(parents=True)
    (song_a / "ext.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song_a / "drum.mp3").write_bytes(b"drums-a")

    song_b = raw / "SongB"
    song_b.mkdir(parents=True)
    (song_b / "ext.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song_b / "drum.mp3").write_bytes(b"drums-b")

    original_copy2 = shutil.copy2
    call_count = 0

    def failing_copy2(src, dst, *, follow_symlinks=True):
        nonlocal call_count
        call_count += 1
        if "SongB" in str(src):
            raise OSError("disk full")
        return original_copy2(src, dst, follow_symlinks=follow_symlinks)

    with patch("src.benchmark.prepare.shutil.copy2", side_effect=failing_copy2):
        result = prepare_corpus(raw, output)

    assert len(result.valid_items) == 1
    assert result.valid_items[0].song_id == "SongA"
    assert len(result.invalid_items) == 1
    assert result.invalid_items[0].reason == "failed to copy corpus files"
    assert (output / "charts" / "SongA.dtx").exists()
    assert not (output / "charts" / "SongB.dtx").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["items"]) == 1
    assert manifest["items"][0]["song_id"] == "SongA"
