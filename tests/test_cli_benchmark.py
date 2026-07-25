import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from click.testing import CliRunner

import src.benchmark.r2_corpus_sync as r2_corpus_sync
from src.benchmark.r2_corpus_models import (
    MAX_SIMFILE_ID,
    OverallStatus,
    PublishedManifest,
    SyncCounters,
    SyncError,
    SyncOutcome,
    SyncRequest,
)
from src.benchmark.r2_corpus_sync import ProgressEvent
from src.cli import benchmark as benchmark_cli
from src.cli.main import main

runner = CliRunner()


def write_prediction(path: Path):
    midi = pretty_midi.PrettyMIDI()
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    midi.instruments.append(drums)
    midi.write(str(path))


def _write_render_song(song: Path, chart_name: str = "mas.dtx") -> None:
    song.mkdir(parents=True)
    (song / chart_name).write_text(
        "\n".join(["#BPM: 120", "#WAV01: drum.mp3", "#00111: 01"]),
        encoding="utf-8",
    )
    # Write a valid WAV container to the configured sample filename.
    sf.write(
        song / "drum.mp3",
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        8000,
        format="WAV",
        subtype="FLOAT",
    )


def test_prepare_corpus_command_writes_parsed_files(tmp_path: Path):
    raw = tmp_path / "raw"
    song = raw / "Soukyuu e no shouka"
    output = tmp_path / "parsed"
    song.mkdir(parents=True)
    (song / "adv.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song / "mas.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song / "2 Drums.mp3").write_bytes(b"drums")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "prepare-corpus",
            "--raw-dir",
            str(raw),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert (output / "charts" / "Soukyuu e no shouka.dtx").exists()
    assert (output / "audio" / "Soukyuu e no shouka.mp3").exists()
    assert (output / "manifest.json").exists()


def test_prepare_corpus_defaults_output_dir_from_raw_dir_name(tmp_path: Path, monkeypatch):
    raw = tmp_path / "Test DTX"
    song = raw / "Soukyuu e no shouka"
    expected_output = tmp_path / "artifacts" / "benchmark" / "Test DTX"
    monkeypatch.chdir(tmp_path)
    song.mkdir(parents=True)
    (song / "mas.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song / "drum.mp3").write_bytes(b"drums")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "prepare-corpus",
            "--raw-dir",
            str(raw),
        ],
    )

    assert result.exit_code == 0
    assert (expected_output / "charts" / "Soukyuu e no shouka.dtx").exists()
    assert (expected_output / "audio" / "Soukyuu e no shouka.mp3").exists()
    assert (expected_output / "manifest.json").exists()


def test_render_audio_command_writes_one_song_from_song_dir(tmp_path: Path):
    song = tmp_path / "Song Render"
    output = tmp_path / "rendered"
    _write_render_song(song)

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "render-audio",
            "--song-dir",
            str(song),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert (output / "audio" / "Song Render.wav").exists()
    assert (output / "renders" / "Song Render.wav").exists()
    assert (output / "manifest.json").exists()
    assert (output / "invalid.json").exists()


def test_render_audio_command_writes_batch_from_raw_dir(tmp_path: Path):
    raw = tmp_path / "raw"
    output = tmp_path / "rendered"
    _write_render_song(raw / "Song A")
    _write_render_song(raw / "Song B")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "render-audio",
            "--raw-dir",
            str(raw),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert (output / "audio" / "Song A.wav").exists()
    assert (output / "audio" / "Song B.wav").exists()
    assert (output / "renders" / "Song A.wav").exists()
    assert (output / "renders" / "Song B.wav").exists()
    assert (output / "manifest.json").exists()
    assert (output / "invalid.json").exists()


def test_render_audio_defaults_output_dir_from_input_dir_name(tmp_path: Path, monkeypatch):
    raw = tmp_path / "Test DTX"
    monkeypatch.chdir(tmp_path)
    _write_render_song(raw / "Song Default")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "render-audio",
            "--raw-dir",
            str(raw),
        ],
    )

    assert result.exit_code == 0
    assert (
        tmp_path / "artifacts" / "benchmark" / "Test DTX" / "audio" / "Song Default.wav"
    ).exists()


def test_score_midi_command_runs(tmp_path: Path):
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    output = tmp_path / "out"
    charts.mkdir()
    predictions.mkdir()
    (charts / "foo.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    write_prediction(predictions / "foo.mid")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "score-midi",
            "--charts-dir",
            str(charts),
            "--predictions-dir",
            str(predictions),
            "--output-dir",
            str(output),
            "--tolerance-ms",
            "50",
        ],
    )

    assert result.exit_code == 0
    assert (output / "summary.json").exists()


def test_score_midi_defaults_output_dir_from_run_name(tmp_path: Path, monkeypatch):
    charts = tmp_path / "parsed" / "charts"
    predictions = tmp_path / "parsed" / "predictions"
    expected_output = tmp_path / "artifacts" / "benchmark" / "soukyuu-stem-test"
    monkeypatch.chdir(tmp_path)
    charts.mkdir(parents=True)
    predictions.mkdir(parents=True)
    (charts / "foo.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    write_prediction(predictions / "foo.mid")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "score-midi",
            "--charts-dir",
            str(charts),
            "--predictions-dir",
            str(predictions),
            "--run-name",
            "soukyuu-stem-test",
            "--tolerance-ms",
            "50",
        ],
    )

    assert result.exit_code == 0
    assert (expected_output / "summary.json").exists()


def test_score_midi_fails_on_missing_prediction(tmp_path: Path):
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    output = tmp_path / "out"
    charts.mkdir()
    predictions.mkdir()
    (charts / "foo.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    (charts / "bar.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    write_prediction(predictions / "foo.mid")
    # bar.mid is missing

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "score-midi",
            "--charts-dir",
            str(charts),
            "--predictions-dir",
            str(predictions),
            "--output-dir",
            str(output),
            "--tolerance-ms",
            "50",
        ],
    )

    assert result.exit_code != 0
    assert "corpus validation failed" in result.output


def test_validate_corpus_reports_missing_prediction(tmp_path: Path):
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    charts.mkdir()
    predictions.mkdir()
    (charts / "foo.dtx").write_text("#BPM: 120\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "validate-corpus",
            "--charts-dir",
            str(charts),
            "--predictions-dir",
            str(predictions),
        ],
    )

    assert result.exit_code == 1
    assert "missing prediction MIDI" in result.output


def test_export_reference_midi_command_writes_files(tmp_path: Path):
    charts = tmp_path / "charts"
    output = tmp_path / "out"
    charts.mkdir()
    (charts / "foo.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "export-reference-midi",
            "--charts-dir",
            str(charts),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert (output / "foo.mid").exists()


def test_inspect_dtx_outputs_counts(tmp_path: Path):
    chart = tmp_path / "foo.dtx"
    chart.write_text("#BPM: 120\n#BPM01: 180\n#00008: 0100\n#00013: 0100\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["benchmark", "inspect-dtx", str(chart)])

    assert result.exit_code == 0
    assert "chart_id: foo" in result.output
    assert "events: 1" in result.output
    assert "bpm_events: 1" in result.output


def test_transcribe_and_score_cli_runs_and_reports_chart_count(tmp_path: Path, monkeypatch):
    from src.benchmark.models import ScoreSummary
    from src.benchmark.reports import ChartReport

    fake_reports = [ChartReport("foo", 50, "raw", ScoreSummary(1, 0, 0))]
    monkeypatch.setattr(
        "src.benchmark.runner.run_transcribe_and_score",
        lambda *args, **kwargs: fake_reports,
    )

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "transcribe-and-score",
            "--charts-dir",
            str(tmp_path),
            "--audio-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--tolerance-ms",
            "50",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "1 chart" in result.output


def make_sync_outcome(
    status: OverallStatus,
    tmp_path: Path,
    *,
    report: bool = True,
    counters: SyncCounters | None = None,
    errors: tuple[SyncError, ...] = (),
) -> SyncOutcome:
    exit_code = {
        "complete": 0,
        "partial": 1,
        "failed": 2,
        "dry_run_complete": 0,
        "dry_run_partial": 1,
    }[status]
    report_path = tmp_path / "report.json" if report else None
    if report_path is not None:
        report_path.write_text("{}\n", encoding="utf-8")
    manifest = None
    if status in {"complete", "partial"}:
        manifest_path = tmp_path / "manifest.jsonl"
        manifest_path.write_text("{}\n", encoding="utf-8")
        manifest = PublishedManifest(
            corpus_version=f"sha256:{'a' * 64}",
            manifest_sha256="b" * 64,
            relative_path="manifests/test.jsonl",
            path=manifest_path,
            latest_path=tmp_path / "latest.json",
        )
    return SyncOutcome(
        status,
        exit_code,
        report_path,
        manifest,
        errors,
        SyncCounters() if counters is None else counters,
    )


def test_sync_r2_corpus_help_lists_local_options_without_endpoint_flag():
    result = runner.invoke(main, ["benchmark", "sync-r2-corpus", "--help"])

    assert result.exit_code == 0
    assert "--cache-dir" in result.output
    assert "--output-dir" in result.output
    assert "--include-simfile-id" in result.output
    assert "--exclude-simfile-id" in result.output
    assert "--provenance-file" in result.output
    assert "--dry-run" in result.output
    assert "--profile" not in result.output
    assert "--endpoint" not in result.output


@pytest.mark.parametrize(
    ("outcome_status", "expected_exit", "exception_type"),
    [("complete", 0, None), ("partial", 1, SystemExit), ("failed", 2, SystemExit)],
)
def test_sync_r2_corpus_maps_outcome_to_explicit_exit(
    monkeypatch, tmp_path, outcome_status, expected_exit, exception_type
):
    monkeypatch.setattr(
        benchmark_cli,
        "sync_r2_corpus",
        lambda request, *, progress: make_sync_outcome(outcome_status, tmp_path),
    )

    result = runner.invoke(
        main,
        [
            "benchmark",
            "sync-r2-corpus",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == expected_exit
    if exception_type is None:
        assert result.exception is None
    else:
        assert type(result.exception) is exception_type


def test_sync_r2_corpus_builds_request_from_local_options(monkeypatch, tmp_path):
    captured: list[SyncRequest] = []

    def fake_sync(request: SyncRequest, *, progress):
        captured.append(request)
        return make_sync_outcome("dry_run_complete", tmp_path)

    monkeypatch.setattr(benchmark_cli, "sync_r2_corpus", fake_sync)
    explicit_provenance = tmp_path / "provenance.json"
    explicit_cache = tmp_path / "cache"
    explicit_output = tmp_path / "output"

    result = runner.invoke(
        main,
        [
            "benchmark",
            "sync-r2-corpus",
            "--cache-dir",
            str(explicit_cache),
            "--output-dir",
            str(explicit_output),
            "--provenance-file",
            str(explicit_provenance),
            "--include-simfile-id",
            "0",
            "--include-simfile-id",
            str(MAX_SIMFILE_ID),
            "--exclude-simfile-id",
            "0",
            "--exclude-simfile-id",
            "23",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert captured == [
        SyncRequest(
            cache_dir=explicit_cache,
            output_dir=explicit_output,
            include_simfile_ids=frozenset({0, MAX_SIMFILE_ID}),
            exclude_simfile_ids=frozenset({0, 23}),
            provenance_file=explicit_provenance,
            dry_run=True,
        )
    ]


def test_sync_r2_corpus_uses_local_default_paths_and_no_provenance(monkeypatch, tmp_path):
    captured: list[SyncRequest] = []

    def fake_sync(request: SyncRequest, *, progress):
        captured.append(request)
        return make_sync_outcome("complete", tmp_path)

    monkeypatch.setattr(benchmark_cli, "sync_r2_corpus", fake_sync)

    result = runner.invoke(main, ["benchmark", "sync-r2-corpus"])

    assert result.exit_code == 0
    assert captured == [
        SyncRequest(
            cache_dir=Path("artifacts/benchmark/r2-corpus/cache"),
            output_dir=Path("artifacts/benchmark/r2-corpus"),
            provenance_file=None,
        )
    ]


@pytest.mark.parametrize("invalid_id", ["-1", str(MAX_SIMFILE_ID + 1)])
def test_sync_r2_corpus_rejects_out_of_range_filters_before_sync(monkeypatch, invalid_id):
    def unexpected_sync(request: SyncRequest, *, progress):
        raise AssertionError("Click should reject invalid filter values before synchronization")

    monkeypatch.setattr(benchmark_cli, "sync_r2_corpus", unexpected_sync)

    result = runner.invoke(
        main,
        ["benchmark", "sync-r2-corpus", "--include-simfile-id", invalid_id],
    )

    assert result.exit_code == 2
    assert "Invalid value" in result.output


def test_sync_r2_corpus_emits_sanitized_progress_and_machine_summary(monkeypatch, tmp_path):
    endpoint = "https://account-secret.r2.cloudflarestorage.com"
    credential = "credential-secret"
    counters = SyncCounters(
        simfiles_discovered=3,
        simfiles_included=2,
        simfiles_excluded_by_filter=1,
        objects_listed=5,
        objects_selected=3,
        cache_hits=1,
        downloads_planned=2,
        downloads_completed=2,
        download_bytes_completed=99,
    )

    def fake_sync(request: SyncRequest, *, progress):
        progress(ProgressEvent("inventory", 1, 3, "inventory: 1/3 simfiles, 0 bytes."))
        return make_sync_outcome("complete", tmp_path, counters=counters)

    monkeypatch.setattr(benchmark_cli, "sync_r2_corpus", fake_sync)
    monkeypatch.setenv("CRUX_R2_ENDPOINT_URL", endpoint)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", credential)

    result = runner.invoke(main, ["benchmark", "sync-r2-corpus"])

    assert result.exit_code == 0
    assert result.stderr == "inventory: 1/3 simfiles, 0 bytes.\n"
    summary = json.loads(result.stdout)
    assert summary == {
        "corpus_version": f"sha256:{'a' * 64}",
        "counts": {
            "cache_hits": 1,
            "download_bytes_completed": 99,
            "download_bytes_planned": 0,
            "downloads_completed": 2,
            "downloads_failed": 0,
            "downloads_planned": 2,
            "objects_listed": 5,
            "objects_selected": 3,
            "simfiles_discovered": 3,
            "simfiles_empty": 0,
            "simfiles_excluded_by_filter": 1,
            "simfiles_included": 2,
        },
        "exit_code": 0,
        "manifest_published": True,
        "report_path": str(tmp_path / "report.json"),
        "status": "complete",
    }
    assert endpoint not in result.stdout
    assert endpoint not in result.stderr
    assert credential not in result.stdout
    assert credential not in result.stderr


def test_sync_r2_corpus_fatal_report_failure_uses_one_sanitized_stderr_line(monkeypatch, tmp_path):
    endpoint = "https://account-secret.r2.cloudflarestorage.com"
    credential = "credential-secret"
    failure = SyncError(
        "artifact", "artifact_write_failed", f"failed at {endpoint} with {credential}"
    )

    monkeypatch.setattr(
        benchmark_cli,
        "sync_r2_corpus",
        lambda request, *, progress: make_sync_outcome(
            "failed", tmp_path, report=False, errors=(failure,)
        ),
    )

    result = runner.invoke(main, ["benchmark", "sync-r2-corpus"])

    assert result.exit_code == 2
    assert result.stderr == "R2 synchronization failed before a report could be written.\n"
    summary = json.loads(result.stdout)
    assert summary["status"] == "failed"
    assert summary["report_path"] is None
    assert endpoint not in result.stdout
    assert endpoint not in result.stderr
    assert credential not in result.stdout
    assert credential not in result.stderr


def test_sync_r2_corpus_nonterminal_failed_progress_keeps_cli_fallback(monkeypatch, tmp_path):
    def fake_sync(request: SyncRequest, *, progress):
        progress(ProgressEvent("failed", 0, None, "failed phase started."))
        return make_sync_outcome("failed", tmp_path, report=False)

    monkeypatch.setattr(benchmark_cli, "sync_r2_corpus", fake_sync)

    result = runner.invoke(main, ["benchmark", "sync-r2-corpus"])

    assert result.exit_code == 2
    assert result.stderr == (
        "failed phase started.\nR2 synchronization failed before a report could be written.\n"
    )


def test_sync_r2_corpus_terminal_failed_progress_suppresses_cli_fallback(monkeypatch, tmp_path):
    def fake_sync(request: SyncRequest, *, progress):
        progress(ProgressEvent("failed", 1, 1, "failed synchronization outcome."))
        return make_sync_outcome("failed", tmp_path, report=False)

    monkeypatch.setattr(benchmark_cli, "sync_r2_corpus", fake_sync)

    result = runner.invoke(main, ["benchmark", "sync-r2-corpus"])

    assert result.exit_code == 2
    assert result.stderr == "failed synchronization outcome.\n"


def test_sync_r2_corpus_real_report_write_failure_keeps_one_final_stderr_line(
    monkeypatch, tmp_path
):
    class EmptyStore:
        def validate_bucket(self) -> None:
            return None

        def list_objects(self) -> tuple:
            return ()

    def real_sync(request: SyncRequest, *, progress):
        return r2_corpus_sync.sync_r2_corpus(
            request,
            environ={
                "CRUX_R2_ENDPOINT_URL": "https://account.example.invalid",
                "AWS_ACCESS_KEY_ID": "test-access-key",
                "AWS_SECRET_ACCESS_KEY": "test-secret-key",
            },
            dependency_check=lambda: None,
            store_factory=lambda _config: EmptyStore(),
            progress=progress,
        )

    def fail_report_write(*args, **kwargs):
        raise OSError("report write failed")

    monkeypatch.setattr(benchmark_cli, "sync_r2_corpus", real_sync)
    monkeypatch.setattr(r2_corpus_sync, "_publish_report_file", fail_report_write)

    result = runner.invoke(
        main,
        [
            "benchmark",
            "sync-r2-corpus",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 2
    assert result.stderr.count("failed synchronization outcome.\n") == 1
    assert "R2 synchronization failed before a report could be written." not in result.stderr
    assert json.loads(result.stdout)["report_path"] is None


def test_installed_sync_r2_corpus_help_is_silent_and_avoids_optional_imports(tmp_path):
    import_spies = tmp_path / "import_spies"
    import_spies.mkdir()
    for module_name in ("pretty_midi", "boto3"):
        (import_spies / f"{module_name}.py").write_text(
            f'raise RuntimeError("{module_name} must not be imported for sync help")\n',
            encoding="utf-8",
        )

    command = Path(sys.executable).with_name("crux")
    result = subprocess.run(
        [str(command), "benchmark", "sync-r2-corpus", "--help"],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(import_spies)},
        text=True,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert "sync-r2-corpus" in result.stdout


def test_sync_r2_corpus_dry_run_summary_states_no_manifest_was_published(monkeypatch, tmp_path):
    monkeypatch.setattr(
        benchmark_cli,
        "sync_r2_corpus",
        lambda request, *, progress: make_sync_outcome("dry_run_partial", tmp_path),
    )

    result = runner.invoke(main, ["benchmark", "sync-r2-corpus", "--dry-run"])

    assert result.exit_code == 1
    summary = json.loads(result.stdout)
    assert summary["status"] == "dry_run_partial"
    assert summary["manifest_published"] is False
