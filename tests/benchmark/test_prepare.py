import json
from pathlib import Path

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
