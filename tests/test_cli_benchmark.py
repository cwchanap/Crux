import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click
import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from click.testing import CliRunner

import src.benchmark.r2_corpus_sync as r2_corpus_sync
from src.benchmark.backend_registry import (
    HEURISTIC_BACKEND_ID,
    LEGACY_TF2_BACKEND_ID,
    OFFICIAL_BACKEND_ID,
    BackendRegistration,
    BackendRegistry,
)
from src.benchmark.backend_reports import OperationalReportPublicationError
from src.benchmark.backends import (
    BackendError,
    BackendVerification,
    PublishedArtifact,
    SmokeCheck,
    TensorCoverageCheck,
)
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
from src.benchmark.reference_chart_manifest import SelectionOutcome, SelectionRequest
from src.benchmark.reference_timing_manifest import (
    ReferenceTimingOutcome,
    ReferenceTimingRequest,
)
from src.benchmark.transcription import (
    TranscribeOneOutcome,
    TranscribeOneRequest,
    VerifyBackendOutcome,
    VerifyBackendRequest,
)
from src.cli import benchmark as benchmark_cli
from src.cli.main import main

runner = CliRunner()


def test_emit_backend_summary_renders_the_exact_four_key_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = io.BytesIO()
    monkeypatch.setattr(benchmark_cli.click, "get_binary_stream", lambda _: output)

    benchmark_cli._emit_backend_summary(
        status="failed",
        exit_code=2,
        report_path=tmp_path / "report.json",
        report_sha256="a" * 64,
    )

    assert output.getvalue() == (
        f'{{"exit_code":2,"report_path":"{tmp_path / "report.json"}",'
        f'"report_sha256":"{"a" * 64}","status":"failed"}}\n'
    ).encode("utf-8")


def test_verify_backend_help_lists_exact_options_and_defaults() -> None:
    result = runner.invoke(main, ["benchmark", "verify-backend", "--help"])

    assert result.exit_code == 0
    assert result.stderr == ""
    assert [line.strip() for line in result.stdout.splitlines() if line.startswith("  --")] == [
        f"--backend TEXT                [default: {OFFICIAL_BACKEND_ID}]",
        "--reports-root DIRECTORY      [default: artifacts/benchmark/backends]",
        "--allow-emulated-diagnostics",
    ]


@pytest.mark.parametrize(
    ("status", "exit_code"),
    (("verified", 0), ("environment_unsupported", 1), ("failed", 2)),
)
def test_verify_backend_emits_one_canonical_four_field_summary(
    status: str,
    exit_code: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[VerifyBackendRequest, object]] = []
    artifact = PublishedArtifact(
        role="verification_report",
        path=tmp_path / "verification.json",
        sha256="a" * 64,
    )

    def fake_run(
        request: VerifyBackendRequest,
        *,
        registry: object,
    ) -> VerifyBackendOutcome:
        captured.append((request, registry))
        return VerifyBackendOutcome(
            status=status,  # type: ignore[arg-type]
            exit_code=exit_code,  # type: ignore[arg-type]
            report_artifact=artifact,
        )

    monkeypatch.setattr("src.benchmark.transcription.run_verify_backend", fake_run)
    result = runner.invoke(
        main,
        [
            "benchmark",
            "verify-backend",
            "--backend",
            OFFICIAL_BACKEND_ID,
            "--reports-root",
            str(tmp_path / "reports"),
            "--allow-emulated-diagnostics",
        ],
    )

    assert result.exit_code == exit_code
    assert result.stderr_bytes == b""
    assert (
        result.stdout_bytes
        == (
            f'{{"exit_code":{exit_code},"report_path":"{artifact.path}",'
            f'"report_sha256":"{"a" * 64}","status":"{status}"}}\n'
        ).encode()
    )
    assert captured[0][0] == VerifyBackendRequest(
        backend_id=OFFICIAL_BACKEND_ID,
        reports_root=tmp_path / "reports",
        allow_emulated_diagnostics=True,
    )


def test_verify_backend_click_usage_error_occurs_before_report_or_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.benchmark.transcription.run_verify_backend",
        lambda *args, **kwargs: pytest.fail("usage error reached orchestration"),
    )

    result = runner.invoke(
        main,
        ["benchmark", "verify-backend", "--reports-root"],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "requires an argument" in result.stderr


@pytest.mark.parametrize(
    "backend_id",
    (LEGACY_TF2_BACKEND_ID, "unknown-backend"),
)
def test_verify_backend_rejects_legacy_and_unknown_at_click_parse_boundary(
    backend_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.benchmark.transcription.run_verify_backend",
        lambda *args, **kwargs: pytest.fail("invalid backend reached orchestration"),
    )
    reports_root = tmp_path / "reports"

    result = runner.invoke(
        main,
        [
            "benchmark",
            "verify-backend",
            "--backend",
            backend_id,
            "--reports-root",
            str(reports_root),
        ],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Invalid value for '--backend'" in result.stderr
    assert not reports_root.exists()


def test_verify_backend_report_publication_failure_is_sanitized_without_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe = f"signed URL https://example.invalid/?token=secret path={tmp_path}"

    def fail_run(*args: object, **kwargs: object) -> VerifyBackendOutcome:
        raise OperationalReportPublicationError(unsafe)

    monkeypatch.setattr("src.benchmark.transcription.run_verify_backend", fail_run)

    result = runner.invoke(main, ["benchmark", "verify-backend"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert (
        result.stderr == "report_publication_failed: Operational report could not be published.\n"
    )
    assert unsafe not in result.stderr
    assert str(tmp_path) not in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("status", "exit_code"),
    (("environment_unsupported", 1), ("failed", 2)),
)
def test_verify_backend_publishes_exact_phase_b_report_and_closes_once(
    status: str,
    exit_code: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verification = BackendVerification(
        status=status,  # type: ignore[arg-type]
        descriptor=None,
        max_input_audio_frames=None,
        backend_lock_sha256=None,
        runtime_lock_sha256=None,
        parameter_lock_sha256=None,
        seal_evidence_sha256=None,
        execution_attestation=None,
        tensor_coverage=TensorCoverageCheck(
            status="not_run",
            required_count=0,
            restored_count=0,
            non_inference_count=0,
            required_inventory_sha256=None,
            non_inference_inventory_sha256=None,
            report=None,
        ),
        smoke=SmokeCheck(
            status="not_run",
            audio_sha256=None,
            oracle_sha256=None,
            prediction=None,
        ),
        errors=(
            BackendError(
                code=(
                    "environment_unsupported"
                    if status == "environment_unsupported"
                    else "backend_process_died"
                ),
                message="Verification did not complete.",
            ),
        ),
    )

    class FakeBackend:
        close_count = 0

        def verify(self) -> BackendVerification:
            return verification

        def close(self) -> None:
            self.close_count += 1

    backend = FakeBackend()
    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        registrations={
            OFFICIAL_BACKEND_ID: BackendRegistration(
                backend_id=OFFICIAL_BACKEND_ID,
                seal_state="sealed",
                factory=lambda **_kwargs: backend,
            )
        },
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "src.benchmark.backend_registry.default_backend_registry",
        lambda: registry,
    )

    result = runner.invoke(
        main,
        ["benchmark", "verify-backend", "--allow-emulated-diagnostics"],
    )

    assert result.exit_code == exit_code
    summary = json.loads(result.stdout)
    report_path = Path(summary["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "crux.backend-verification-report/v1"
    assert report["report_type"] == "verification"
    assert report["status"] == status
    assert report["exit_code"] == exit_code
    assert report["errors"] == [
        {
            "code": verification.errors[0].code,
            "message": "Verification did not complete.",
        }
    ]
    assert backend.close_count == 1


@pytest.mark.parametrize("backend_id", (HEURISTIC_BACKEND_ID,))
def test_verify_backend_known_unavailable_factory_publishes_typed_failure_namespace(
    backend_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        main,
        ["benchmark", "verify-backend", "--backend", backend_id],
    )

    assert result.exit_code == 2
    assert result.stderr == ""
    summary = json.loads(result.stdout)
    report_path = Path(summary["report_path"])
    assert backend_id in report_path.parts
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["errors"] == [
        {
            "code": "backend_unavailable",
            "message": "Backend is unavailable.",
        }
    ]


def test_verify_backend_default_preseal_publishes_not_sealed_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(main, ["benchmark", "verify-backend"])

    assert result.exit_code == 1
    assert result.stderr == ""
    summary = json.loads(result.stdout)
    report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert report["status"] == "failed"
    assert report["exit_code"] == 1
    assert report["errors"] == [{"code": "backend_not_sealed", "message": "Backend is not sealed."}]


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


def test_transcribe_and_score_help_has_no_backend_option() -> None:
    result = CliRunner().invoke(main, ["benchmark", "transcribe-and-score", "--help"])

    assert result.exit_code == 0
    assert "--backend" not in result.output


def make_transcribe_one_outcome(
    tmp_path: Path,
    status: str = "complete",
    exit_code: int = 0,
    *,
    report_name: str = "report.json",
) -> TranscribeOneOutcome:
    return TranscribeOneOutcome(
        status=status,  # type: ignore[arg-type]
        exit_code=exit_code,  # type: ignore[arg-type]
        report_artifact=PublishedArtifact(
            role="execution_report",
            path=tmp_path / report_name,
            sha256="a" * 64,
        ),
    )


def direct_transcribe_one_args(tmp_path: Path) -> list[str]:
    return [
        "benchmark",
        "transcribe-one",
        "--audio",
        str(tmp_path / "audio.wav"),
        "--source-audio-id",
        "source-v1",
        "--input-view-id",
        "view-v1",
        "--output",
        str(tmp_path / "prediction.jsonl"),
    ]


def test_transcribe_one_help_lists_exact_options_and_reports_default() -> None:
    result = runner.invoke(main, ["benchmark", "transcribe-one", "--help"])

    assert result.exit_code == 0
    assert result.stderr == ""
    assert [line.strip() for line in result.stdout.splitlines() if line.startswith("  --")] == [
        "--backend TEXT",
        "--audio FILE                [required]",
        "--source-audio-id TEXT",
        "--input-view-id TEXT",
        "--input-view-manifest FILE",
        "--output FILE               [required]",
        "--midi-output FILE",
        "--reports-root DIRECTORY    [default: artifacts/benchmark/backends]",
    ]
    assert "--charts-dir" not in result.stdout
    assert "--predictions-dir" not in result.stdout
    assert "--tolerance-ms" not in result.stdout


def test_prepare_backend_help_lists_setup_only_options() -> None:
    result = runner.invoke(main, ["benchmark", "prepare-backend", "--help"])

    assert result.exit_code == 0
    assert result.stderr == ""
    assert [line.strip() for line in result.stdout.splitlines() if line.startswith("  --")] == [
        "--backend TEXT          [default: oaf]",
        "--download",
        "--archive FILE",
        "--cache-root DIRECTORY  [default: artifacts/benchmark/model-cache]",
    ]


def test_prepare_backend_emits_exact_ready_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "runtime.oaf_tf1.model.load_model_config",
        lambda: object(),
    )
    monkeypatch.setattr(
        "src.benchmark.checkpoint_acquisition.prepare_oaf_checkpoint",
        lambda config, cache_root, *, download, archive_path: tmp_path / "cache" / "checkpoint",
    )

    result = runner.invoke(
        main,
        [
            "benchmark",
            "prepare-backend",
            "--cache-root",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0
    assert result.stdout_bytes == (
        f'{{"backend":"oaf","checkpoint_path":"{tmp_path / "cache" / "checkpoint"}",'
        '"status":"ready"}\n'
    ).encode("utf-8")
    assert result.stderr_bytes == b""


def test_prepare_backend_rejects_mutually_exclusive_modes_as_click_usage(
    tmp_path: Path,
) -> None:
    result = runner.invoke(
        main,
        [
            "benchmark",
            "prepare-backend",
            "--download",
            "--archive",
            str(tmp_path / "checkpoint.zip"),
            "--cache-root",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Usage:" in result.stderr
    assert "mutually exclusive" in result.stderr


def test_prepare_backend_unknown_backend_emits_integrity_summary_after_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = runner.invoke(
        main,
        [
            "benchmark",
            "prepare-backend",
            "--backend",
            "unknown",
            "--cache-root",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 2
    assert result.stdout_bytes == (
        b'{"backend":"unknown","checkpoint_path":null,"status":"integrity_failed"}\n'
    )
    assert "backend_selection_invalid" in result.stderr
    assert str(tmp_path).encode() not in result.stderr_bytes


def test_transcribe_one_direct_mode_builds_complete_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[TranscribeOneRequest, object]] = []

    def fake_run(request: TranscribeOneRequest, *, registry: object) -> TranscribeOneOutcome:
        captured.append((request, registry))
        return make_transcribe_one_outcome(tmp_path)

    monkeypatch.setattr("src.benchmark.transcription.run_transcribe_one", fake_run)
    args = direct_transcribe_one_args(tmp_path)
    args.extend(
        [
            "--backend",
            HEURISTIC_BACKEND_ID,
            "--midi-output",
            str(tmp_path / "prediction.mid"),
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )

    result = runner.invoke(main, args)

    assert result.exit_code == 0
    assert len(captured) == 1
    request, registry = captured[0]
    assert request == TranscribeOneRequest(
        backend_id=HEURISTIC_BACKEND_ID,
        audio_path=tmp_path / "audio.wav",
        output_path=tmp_path / "prediction.jsonl",
        source_audio_id="source-v1",
        input_view_id="view-v1",
        input_view_manifest=None,
        midi_output_path=tmp_path / "prediction.mid",
        reports_root=tmp_path / "reports",
    )
    assert registry.default_backend_id == OFFICIAL_BACKEND_ID


def test_transcribe_one_derived_mode_uses_manifest_and_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[TranscribeOneRequest] = []

    def fake_run(request: TranscribeOneRequest, *, registry: object) -> TranscribeOneOutcome:
        del registry
        captured.append(request)
        return make_transcribe_one_outcome(tmp_path)

    monkeypatch.setattr("src.benchmark.transcription.run_transcribe_one", fake_run)

    result = runner.invoke(
        main,
        [
            "benchmark",
            "transcribe-one",
            "--audio",
            str(tmp_path / "derived.wav"),
            "--input-view-manifest",
            str(tmp_path / "input-view.json"),
            "--output",
            str(tmp_path / "derived.jsonl"),
        ],
    )

    assert result.exit_code == 0
    assert captured == [
        TranscribeOneRequest(
            backend_id=None,
            audio_path=tmp_path / "derived.wav",
            output_path=tmp_path / "derived.jsonl",
            source_audio_id=None,
            input_view_id=None,
            input_view_manifest=tmp_path / "input-view.json",
            midi_output_path=None,
            reports_root=Path("artifacts/benchmark/backends"),
        )
    ]


@pytest.mark.parametrize(
    "provenance_args",
    [
        [],
        ["--source-audio-id", "source-v1"],
        ["--input-view-id", "view-v1"],
        [
            "--source-audio-id",
            "source-v1",
            "--input-view-id",
            "view-v1",
            "--input-view-manifest",
            "input-view.json",
        ],
        [
            "--source-audio-id",
            "source-v1",
            "--input-view-manifest",
            "input-view.json",
        ],
        [
            "--input-view-id",
            "view-v1",
            "--input-view-manifest",
            "input-view.json",
        ],
    ],
)
def test_transcribe_one_rejects_missing_or_mixed_provenance_before_orchestration(
    provenance_args: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_run(*args: object, **kwargs: object) -> TranscribeOneOutcome:
        raise AssertionError("usage validation must happen before orchestration")

    monkeypatch.setattr("src.benchmark.transcription.run_transcribe_one", unexpected_run)
    reports_root = tmp_path / "reports"

    result = runner.invoke(
        main,
        [
            "benchmark",
            "transcribe-one",
            "--audio",
            str(tmp_path / "audio.wav"),
            "--output",
            str(tmp_path / "prediction.jsonl"),
            "--reports-root",
            str(reports_root),
            *provenance_args,
        ],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Usage:" in result.stderr
    assert "exactly one provenance mode" in result.stderr
    assert not reports_root.exists()


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [("complete", 0), ("partial", 1), ("failed", 2)],
)
def test_transcribe_one_emits_canonical_four_key_summary_for_each_outcome(
    status: str,
    exit_code: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_name = "報告.json"
    outcome = make_transcribe_one_outcome(
        tmp_path,
        status,
        exit_code,
        report_name=report_name,
    )
    monkeypatch.setattr(
        "src.benchmark.transcription.run_transcribe_one",
        lambda request, *, registry: outcome,
    )

    result = runner.invoke(main, direct_transcribe_one_args(tmp_path))

    report_path = tmp_path / report_name
    expected = (
        '{"exit_code":'
        f'{exit_code},"report_path":"{report_path}",'
        f'"report_sha256":"{"a" * 64}","status":"{status}"'
        "}\n"
    ).encode("utf-8")
    assert result.exit_code == exit_code
    assert result.stdout_bytes == expected
    assert result.stderr_bytes == b""
    assert set(json.loads(result.stdout)) == {
        "exit_code",
        "report_path",
        "report_sha256",
        "status",
    }
    assert result.stdout_bytes.count(b"\n") == 1
    assert b"Traceback" not in result.stdout_bytes
    assert b"Traceback" not in result.stderr_bytes


def test_transcribe_one_keeps_diagnostics_on_stderr_and_summary_on_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = make_transcribe_one_outcome(tmp_path)

    def fake_run(
        request: TranscribeOneRequest,
        *,
        registry: object,
    ) -> TranscribeOneOutcome:
        del request, registry
        click.echo("backend diagnostic", err=True)
        return outcome

    monkeypatch.setattr("src.benchmark.transcription.run_transcribe_one", fake_run)

    result = runner.invoke(main, direct_transcribe_one_args(tmp_path))

    assert result.exit_code == 0
    assert result.stderr == "backend diagnostic\n"
    assert "backend diagnostic" not in result.stdout
    assert len(result.stdout.splitlines()) == 1
    assert set(json.loads(result.stdout)) == {
        "exit_code",
        "report_path",
        "report_sha256",
        "status",
    }


def test_transcribe_one_report_publication_failure_has_only_stable_sanitized_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe = "signed URL https://example.invalid/?token=secret"

    def fail_run(*args: object, **kwargs: object) -> TranscribeOneOutcome:
        raise OperationalReportPublicationError(unsafe)

    monkeypatch.setattr("src.benchmark.transcription.run_transcribe_one", fail_run)

    result = runner.invoke(main, direct_transcribe_one_args(tmp_path))

    assert result.exit_code == 2
    assert result.stdout == ""
    assert (
        result.stderr == "report_publication_failed: Operational report could not be published.\n"
    )
    assert unsafe not in result.stderr
    assert "Traceback" not in result.stderr


def test_transcribe_one_unknown_backend_publishes_typed_failure_after_click_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        main,
        [
            *direct_transcribe_one_args(Path(".")),
            "--backend",
            "../../unknown",
        ],
    )

    assert result.exit_code == 2
    assert result.stderr == ""
    summary = json.loads(result.stdout)
    assert summary["status"] == "failed"
    assert summary["exit_code"] == 2
    assert "backend-unavailable" in Path(summary["report_path"]).parts
    report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
    assert report["errors"] == [
        {
            "code": "backend_unavailable",
            "message": "Backend is unavailable.",
        }
    ]


@pytest.mark.parametrize("backend_args", [["--backend", HEURISTIC_BACKEND_ID]])
def test_transcribe_one_known_unavailable_backend_publishes_typed_failure(
    backend_args: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        main,
        [
            *direct_transcribe_one_args(Path(".")),
            *backend_args,
        ],
    )

    assert result.exit_code == 2
    assert result.stderr == ""
    summary = json.loads(result.stdout)
    assert summary["status"] == "failed"
    assert HEURISTIC_BACKEND_ID in Path(summary["report_path"]).parts
    report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
    assert report["errors"][0]["code"] == "backend_unavailable"


def test_transcribe_one_default_preseal_publishes_not_sealed_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(main, [*direct_transcribe_one_args(Path("."))])

    assert result.exit_code == 1
    assert result.stderr == ""
    summary = json.loads(result.stdout)
    report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert report["status"] == "failed"
    assert report["exit_code"] == 1
    assert report["items"] == []
    assert report["errors"] == [{"code": "backend_not_sealed", "message": "Backend is not sealed."}]


def test_installed_transcribe_one_help_is_silent_and_avoids_backend_runtime_imports(
    tmp_path: Path,
) -> None:
    import_spies = tmp_path / "import_spies"
    import_spies.mkdir()
    for module_name in ("tensorflow", "librosa"):
        (import_spies / f"{module_name}.py").write_text(
            f'raise RuntimeError("{module_name} must not be imported for benchmark help")\n',
            encoding="utf-8",
        )

    command = Path(sys.executable).with_name("crux")
    for args in (
        ["benchmark", "--help"],
        ["benchmark", "transcribe-one", "--help"],
    ):
        result = subprocess.run(
            [str(command), *args],
            capture_output=True,
            check=False,
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(import_spies)},
            text=True,
        )

        assert result.returncode == 0
        assert result.stderr == ""
        assert "transcribe-one" in result.stdout


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


def test_select_reference_charts_builds_local_request_and_emits_manifest_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "r2-corpus" / "manifests" / "source.jsonl"
    output_dir = tmp_path / "reference-charts"
    published_path = output_dir / "manifests" / ("b" * 64 + ".jsonl")
    published = PublishedManifest(
        corpus_version="sha256:" + "a" * 64,
        manifest_sha256="b" * 64,
        relative_path="manifests/" + "b" * 64 + ".jsonl",
        path=published_path,
        latest_path=output_dir / "latest.json",
    )
    captured: list[SelectionRequest] = []

    def fake_select(request: SelectionRequest) -> SelectionOutcome:
        captured.append(request)
        return SelectionOutcome(
            status="partial",
            exit_code=1,
            manifest=published,
            selected_count=7,
            quarantined_count=1,
        )

    monkeypatch.setattr(
        "src.benchmark.reference_chart_manifest.select_reference_manifest",
        fake_select,
    )

    result = runner.invoke(
        main,
        [
            "benchmark",
            "select-reference-charts",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 1
    assert result.stderr_bytes == b""
    assert result.stdout_bytes == (
        f'{{"corpus_version":"sha256:{"a" * 64}","exit_code":1,'
        f'"manifest_path":"{published_path}","manifest_sha256":"{"b" * 64}",'
        '"quarantined_count":1,"selected_count":7,"status":"partial"}\n'
    ).encode("utf-8")
    assert captured == [
        SelectionRequest(
            manifest_path=manifest_path,
            cache_dir=tmp_path / "r2-corpus" / "cache",
            overrides_file=Path("config/benchmark-reference-chart-overrides.json"),
            output_dir=output_dir,
            default_overrides_missing_ok=True,
        )
    ]
    assert "report_path" not in json.loads(result.stdout)


def test_select_reference_charts_treats_an_explicit_override_file_as_mandatory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifests" / "source.jsonl"
    overrides_file = tmp_path / "overrides.json"
    captured: list[SelectionRequest] = []

    def fake_select(request: SelectionRequest) -> SelectionOutcome:
        captured.append(request)
        return SelectionOutcome(
            status="failed",
            exit_code=2,
            manifest=None,
            selected_count=0,
            quarantined_count=0,
        )

    monkeypatch.setattr(
        "src.benchmark.reference_chart_manifest.select_reference_manifest",
        fake_select,
    )

    result = runner.invoke(
        main,
        [
            "benchmark",
            "select-reference-charts",
            "--manifest",
            str(manifest_path),
            "--overrides-file",
            str(overrides_file),
        ],
    )

    assert result.exit_code == 2
    assert captured == [
        SelectionRequest(
            manifest_path=manifest_path,
            cache_dir=tmp_path / "cache",
            overrides_file=overrides_file,
            output_dir=Path("artifacts/benchmark/reference-charts"),
            default_overrides_missing_ok=False,
        )
    ]
    assert json.loads(result.stdout) == {
        "corpus_version": None,
        "exit_code": 2,
        "manifest_path": None,
        "manifest_sha256": None,
        "quarantined_count": 0,
        "selected_count": 0,
        "status": "failed",
    }


def test_installed_select_reference_charts_help_is_silent_and_avoids_optional_imports(
    tmp_path: Path,
) -> None:
    import_spies = tmp_path / "import_spies"
    import_spies.mkdir()
    for module_name in ("pretty_midi", "boto3"):
        (import_spies / f"{module_name}.py").write_text(
            f'raise RuntimeError("{module_name} must not be imported for selection help")\n',
            encoding="utf-8",
        )

    command = shutil.which("crux")
    if command is None:
        pytest.skip("crux executable is not installed on PATH")
    result = subprocess.run(
        [command, "benchmark", "select-reference-charts", "--help"],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(import_spies)},
        text=True,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert "--manifest" in result.stdout
    assert "--cache-dir" in result.stdout
    assert "--overrides-file" in result.stdout
    assert "--output-dir" in result.stdout


# ---------------------------------------------------------------------------
# build-reference-timing (HPA-323 Task 7)
# ---------------------------------------------------------------------------


def _make_timing_published(tmp_path: Path) -> PublishedManifest:
    published_path = tmp_path / "reference-timing" / "manifests" / ("b" * 64 + ".jsonl")
    published_path.parent.mkdir(parents=True, exist_ok=True)
    return PublishedManifest(
        corpus_version="sha256:" + "a" * 64,
        manifest_sha256="b" * 64,
        relative_path="manifests/" + "b" * 64 + ".jsonl",
        path=published_path,
        latest_path=tmp_path / "reference-timing" / "latest.json",
    )


def _make_timing_outcome(
    *,
    status: str,
    exit_code: int,
    tmp_path: Path,
    published: PublishedManifest | None = None,
    ready_count: int = 0,
    quarantined_count: int = 0,
    upstream_quarantined_count: int = 0,
    events_published: int = 0,
) -> ReferenceTimingOutcome:
    return ReferenceTimingOutcome(
        status=status,  # type: ignore[arg-type]
        exit_code=exit_code,  # type: ignore[arg-type]
        manifest=published,
        ready_count=ready_count,
        quarantined_count=quarantined_count,
        upstream_quarantined_count=upstream_quarantined_count,
        events_published=events_published,
    )


def test_build_reference_timing_help_lists_exact_options_and_defaults() -> None:
    result = runner.invoke(main, ["benchmark", "build-reference-timing", "--help"])

    assert result.exit_code == 0
    assert result.stderr == ""
    output = result.stdout
    # Exactly the three path options, in declaration order, with no overrides-file.
    option_lines = [line.strip() for line in output.splitlines() if line.startswith("  --")]
    assert [line.split()[0] for line in option_lines] == [
        "--manifest",
        "--cache-dir",
        "--output-dir",
    ]
    assert "--overrides-file" not in output
    # manifest and cache-dir are required; only output-dir carries a default.
    # Whitespace is normalized because the default value wraps in the terminal.
    normalized = " ".join(output.split())
    assert normalized.count("[required]") == 2
    assert "[default: artifacts/benchmark/reference-timing]" in normalized
    assert normalized.count("[default:") == 1


@pytest.mark.parametrize(
    "missing_args",
    [
        [],
        ["--cache-dir", "cache"],
        ["--manifest", "manifest.jsonl"],
    ],
)
def test_build_reference_timing_requires_manifest_and_cache_dir(
    missing_args: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # ``--cache-dir`` is intentionally required rather than inherited from
    # ``select-reference-charts``'s manifest-relative default
    # (``manifest.parent.parent / "cache"``).  For an HPA-322 manifest at
    # ``.../reference-charts/manifests/<sha>.jsonl`` that default resolves to
    # ``reference-charts/cache`` — NOT ``r2-corpus/cache`` where the verified
    # source-audio bodies live — so inheriting it would point the build away
    # from the audio cache.  This is enforced structurally by Click's
    # ``required=True`` on ``--cache-dir`` (exercised below), not by a separate
    # path-arithmetic unit test, since only the live CLI surface detects real
    # regressions in the option contract.
    monkeypatch.setattr(
        "src.benchmark.reference_timing_manifest.run_reference_timing",
        lambda *args, **kwargs: pytest.fail("missing required option reached orchestration"),
    )

    result = runner.invoke(main, ["benchmark", "build-reference-timing", *missing_args])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Missing option" in result.stderr


def test_build_reference_timing_defaults_only_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[ReferenceTimingRequest] = []
    manifest_path = tmp_path / "manifest.jsonl"
    cache_dir = tmp_path / "cache"

    def fake_run(request: ReferenceTimingRequest) -> ReferenceTimingOutcome:
        captured.append(request)
        return _make_timing_outcome(
            status="complete",
            exit_code=0,
            tmp_path=tmp_path,
            published=_make_timing_published(tmp_path),
            ready_count=1,
            events_published=1,
        )

    monkeypatch.setattr("src.benchmark.reference_timing_manifest.run_reference_timing", fake_run)

    result = runner.invoke(
        main,
        [
            "benchmark",
            "build-reference-timing",
            "--manifest",
            str(manifest_path),
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code == 0
    assert captured == [
        ReferenceTimingRequest(
            manifest_path=manifest_path,
            cache_dir=cache_dir,
            output_dir=Path("artifacts/benchmark/reference-timing"),
        )
    ]


@pytest.mark.parametrize(
    ("status", "exit_code", "exception_type"),
    [("complete", 0, None), ("partial", 1, SystemExit), ("failed", 2, SystemExit)],
)
def test_build_reference_timing_maps_outcome_to_explicit_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
    exit_code: int,
    exception_type: type[BaseException] | None,
) -> None:
    published = _make_timing_published(tmp_path) if exit_code != 2 else None
    outcome = _make_timing_outcome(
        status=status,
        exit_code=exit_code,
        tmp_path=tmp_path,
        published=published,
        ready_count=1 if exit_code == 0 else 0,
        quarantined_count=1 if exit_code == 1 else 0,
        upstream_quarantined_count=0,
        events_published=1 if exit_code == 0 else 0,
    )
    monkeypatch.setattr(
        "src.benchmark.reference_timing_manifest.run_reference_timing",
        lambda request: outcome,
    )

    result = runner.invoke(
        main,
        [
            "benchmark",
            "build-reference-timing",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == exit_code
    if exception_type is None:
        assert result.exception is None
    else:
        assert type(result.exception) is exception_type


def test_build_reference_timing_emits_sorted_canonical_summary_with_nine_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    published = _make_timing_published(tmp_path)
    outcome = _make_timing_outcome(
        status="partial",
        exit_code=1,
        tmp_path=tmp_path,
        published=published,
        ready_count=3,
        quarantined_count=1,
        upstream_quarantined_count=0,
        events_published=3,
    )
    monkeypatch.setattr(
        "src.benchmark.reference_timing_manifest.run_reference_timing",
        lambda request: outcome,
    )

    result = runner.invoke(
        main,
        [
            "benchmark",
            "build-reference-timing",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 1
    assert result.stderr_bytes == b""
    assert result.stdout_bytes == (
        '{"corpus_version":"sha256:' + "a" * 64 + '",'
        '"events_published":3,"exit_code":1,'
        f'"manifest_path":"{published.path}",'
        '"manifest_sha256":"' + "b" * 64 + '",'
        '"quarantined_count":1,"ready_count":3,'
        '"status":"partial","upstream_quarantined_count":0}\n'
    ).encode("utf-8")
    assert set(json.loads(result.stdout)) == {
        "status",
        "exit_code",
        "manifest_path",
        "manifest_sha256",
        "corpus_version",
        "ready_count",
        "quarantined_count",
        "upstream_quarantined_count",
        "events_published",
    }
    assert result.stdout_bytes.count(b"\n") == 1


def test_build_reference_timing_failed_outcome_nulls_manifest_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outcome = _make_timing_outcome(
        status="failed",
        exit_code=2,
        tmp_path=tmp_path,
        published=None,
    )
    monkeypatch.setattr(
        "src.benchmark.reference_timing_manifest.run_reference_timing",
        lambda request: outcome,
    )

    result = runner.invoke(
        main,
        [
            "benchmark",
            "build-reference-timing",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout) == {
        "status": "failed",
        "exit_code": 2,
        "manifest_path": None,
        "manifest_sha256": None,
        "corpus_version": None,
        "ready_count": 0,
        "quarantined_count": 0,
        "upstream_quarantined_count": 0,
        "events_published": 0,
    }


def test_build_reference_timing_summary_surfaces_upstream_quarantine_signal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An operator can see timing-stage quarantines despite a shared exit 1.

    ``quarantined_count=7`` with ``upstream_quarantined_count=5`` means two
    rows were quarantined by the timing stage itself, even though exit ``1`` is
    shared with the five upstream HPA-322 gaps.
    """
    published = _make_timing_published(tmp_path)
    outcome = _make_timing_outcome(
        status="partial",
        exit_code=1,
        tmp_path=tmp_path,
        published=published,
        ready_count=3,
        quarantined_count=7,
        upstream_quarantined_count=5,
        events_published=3,
    )
    monkeypatch.setattr(
        "src.benchmark.reference_timing_manifest.run_reference_timing",
        lambda request: outcome,
    )

    result = runner.invoke(
        main,
        [
            "benchmark",
            "build-reference-timing",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 1
    summary = json.loads(result.stdout)
    assert summary["quarantined_count"] == 7
    assert summary["upstream_quarantined_count"] == 5
    assert summary["quarantined_count"] - summary["upstream_quarantined_count"] == 2


def test_installed_build_reference_timing_help_is_silent_and_avoids_optional_imports(
    tmp_path: Path,
) -> None:
    """The complete-cache path must not eagerly import the optional R2 stack.

    ``--help`` never executes the lazy orchestration import inside the command
    body, so neither ``boto3`` nor ``pretty_midi`` may be imported.  The
    orchestration's no-R2 property (a complete cache never calls the dependency
    check / store factory) is already covered by the Task 6b acceptance suite.
    """
    import_spies = tmp_path / "import_spies"
    import_spies.mkdir()
    for module_name in ("pretty_midi", "boto3"):
        (import_spies / f"{module_name}.py").write_text(
            f'raise RuntimeError("{module_name} must not be imported for timing help")\n',
            encoding="utf-8",
        )

    command = shutil.which("crux")
    if command is None:
        pytest.skip("crux executable is not installed on PATH")
    result = subprocess.run(
        [command, "benchmark", "build-reference-timing", "--help"],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(import_spies)},
        text=True,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert "--manifest" in result.stdout
    assert "--cache-dir" in result.stdout
    assert "--output-dir" in result.stdout
