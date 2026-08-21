from __future__ import annotations

import json
import struct
from dataclasses import dataclass, replace
from pathlib import Path

import click
from click.testing import CliRunner

from src.benchmark.backend_identity import (
    OAF_BACKEND_ID,
    OAF_DESCRIPTOR_SCHEMA,
    build_descriptor,
    sha256_hex,
)
from src.benchmark.backends import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.oaf_corpus_run import OafCorpusRunOutcome
from src.benchmark.oaf_smoke_oracle import render_smoke_oracle
from src.benchmark.prediction_artifact import (
    publish_prediction_artifact,
    read_prediction_artifact,
)
from src.benchmark.r2_corpus_models import MAX_SIMFILE_ID, PublishedManifest
from src.benchmark.reviewed_subset import (
    FinalizeReviewedSubsetOutcome,
    PrepareReviewedSubsetOutcome,
    ScoreReviewedSubsetOutcome,
)
from src.cli.main import main


def _descriptor() -> object:
    payload = {
        "architecture_id": "magenta-oaf-model-tpu-drums-v1",
        "backend_id": OAF_BACKEND_ID,
        "descriptor_schema": OAF_DESCRIPTOR_SCHEMA,
        "model_id": "magenta-egmd-ckpt-569400-v1",
        "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
        "native_output_space_id": "magenta-oaf-midi88-a0-v1",
        "prediction_schema": "crux.drum-prediction-events/v2",
        "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
        "upstream_source_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
    }
    return build_descriptor(payload, frozenset(payload), OAF_DESCRIPTOR_SCHEMA)


def _audio(tmp_path: Path) -> CanonicalAudio:
    content = (
        struct.pack("<4sI4s", b"RIFF", 40, b"WAVE")
        + struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 44100, 88200, 2, 16)
        + struct.pack("<4sI", b"data", 4)
        + b"\x00\x00\x00\x00"
    )
    path = tmp_path / "canonical.wav"
    path.write_bytes(content)
    digest = sha256_hex(content)
    return CanonicalAudio(
        path, "oaf-smoke", digest, "oaf-smoke-v1", digest, len(content), 44100, 1, 2, 2
    )


def _prediction(tmp_path: Path) -> NativePrediction:
    audio = _audio(tmp_path)
    events = (
        NativeEvent(
            time_sec=0.5,
            native_class_id="midi_46",
            model_output_bin=25,
            native_midi_note=46,
            native_metadata={"upstream_8hit_group_id": "hihat"},
            confidence=0.75,
            velocity_midi=96,
        ),
        NativeEvent(
            time_sec=1.25,
            native_class_id="midi_75",
            model_output_bin=54,
            native_midi_note=75,
            native_metadata={"upstream_8hit_group_id": "sticks"},
            confidence=0.5,
            velocity_midi=80,
        ),
    )
    return NativePrediction(audio, _descriptor(), events)


@dataclass
class _FakeBackend:
    prediction: NativePrediction

    def descriptor(self):
        return self.prediction.descriptor

    def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
        return NativePrediction(audio, self.prediction.descriptor, self.prediction.events)

    def close(self) -> None:
        return None


class _FakeRegistry:
    def __init__(self, backend: _FakeBackend) -> None:
        self.backend = backend

    def create(self, backend_id: str | None = None, **kwargs: object) -> _FakeBackend:
        assert backend_id == "oaf"
        del kwargs
        return self.backend


class _ControllableClock:
    def __init__(self) -> None:
        self.calls = 0
        self.now = 10.0
        self.inference_stopped = False
        self.post_timer_operations: list[str] = []

    def perf_counter(self) -> float:
        self.calls += 1
        if self.calls == 1:
            return self.now
        if self.calls == 2:
            self.now = 12.0
            self.inference_stopped = True
            return self.now
        raise AssertionError("smoke benchmark timer should stop exactly once")

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_smoke_backend_times_only_backend_inference_and_publishes_v2(
    tmp_path: Path, monkeypatch
) -> None:
    output_root = tmp_path / "input"
    output_root.mkdir()
    prediction = _prediction(output_root)
    import src.cli.benchmark as benchmark_module

    fake_registry = _FakeRegistry(_FakeBackend(prediction))
    clock = _ControllableClock()
    original_map = benchmark_module.map_oaf_prediction
    original_publish = benchmark_module.publish_prediction_artifact

    def map_after_timer_stop(native_prediction: NativePrediction):
        assert clock.inference_stopped
        clock.post_timer_operations.append("mapping")
        clock.advance(100.0)
        return original_map(native_prediction)

    def publish_after_timer_stop(path: Path, mapped_prediction):
        assert clock.inference_stopped
        clock.post_timer_operations.append("publication")
        clock.advance(100.0)
        return original_publish(path, mapped_prediction)

    monkeypatch.setattr(benchmark_module, "default_backend_registry", lambda: fake_registry)
    monkeypatch.setattr(benchmark_module, "map_oaf_prediction", map_after_timer_stop)
    monkeypatch.setattr(benchmark_module, "publish_prediction_artifact", publish_after_timer_stop)
    monkeypatch.setattr(
        benchmark_module,
        "load_model_config",
        lambda: type(
            "Config",
            (),
            {
                "max_input_audio_frames": None,
                "checkpoint": type("Checkpoint", (), {"archive_sha256": "a" * 64})(),
            },
        )(),
    )
    monkeypatch.setattr(
        benchmark_module, "load_direct_audio", lambda *args, **kwargs: prediction.audio
    )
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)

    result = CliRunner().invoke(
        main,
        ["benchmark", "smoke-backend", "--backend", "oaf"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    summary = json.loads(result.output)
    assert summary["status"] == "ok"
    assert summary["mapped_event_count"] == 1
    assert summary["unmapped_event_count"] == 1
    assert summary["inference_elapsed_seconds"] == 2.0
    assert summary["real_time_factor"] == 2.0 / (
        prediction.audio.audio_frame_count / prediction.audio.sample_rate
    )
    assert clock.calls == 2
    assert clock.post_timer_operations == ["mapping", "publication"]
    assert summary["oracle_status"] == "not_checked"
    prediction_path = Path(summary["prediction_path"])
    assert prediction_path.exists()
    assert b'"schema":"crux.drum-prediction-events/v2"' in prediction_path.read_bytes()


def test_smoke_backend_matches_optional_oracle_and_rejects_missing_or_mismatching(
    tmp_path: Path, monkeypatch
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    prediction = _prediction(input_root)
    oracle_path = tmp_path / "oracle.json"
    oracle_path.write_bytes(render_smoke_oracle(map_oaf_prediction(prediction)[0]))

    import src.cli.benchmark as benchmark_module

    fake_registry = _FakeRegistry(_FakeBackend(prediction))
    monkeypatch.setattr(benchmark_module, "default_backend_registry", lambda: fake_registry)
    monkeypatch.setattr(
        benchmark_module,
        "load_model_config",
        lambda: type(
            "Config",
            (),
            {
                "max_input_audio_frames": None,
                "checkpoint": type("Checkpoint", (), {"archive_sha256": "a" * 64})(),
            },
        )(),
    )
    monkeypatch.setattr(
        benchmark_module, "load_direct_audio", lambda *args, **kwargs: prediction.audio
    )
    monkeypatch.setattr(benchmark_module.time, "perf_counter", iter((10.0, 12.0)).__next__)

    matching = CliRunner().invoke(
        main,
        ["benchmark", "smoke-backend", "--backend", "oaf", "--oracle", str(oracle_path)],
        catch_exceptions=False,
    )
    assert matching.exit_code == 0
    assert json.loads(matching.output)["oracle_status"] == "matched"

    missing = CliRunner().invoke(
        main,
        ["benchmark", "smoke-backend", "--backend", "oaf", "--oracle", str(tmp_path / "missing")],
    )
    assert missing.exit_code != 0

    oracle_path.write_bytes(oracle_path.read_bytes().replace(b"midi_46", b"midi_47"))
    monkeypatch.setattr(benchmark_module.time, "perf_counter", iter((10.0, 12.0)).__next__)
    mismatching = CliRunner().invoke(
        main,
        ["benchmark", "smoke-backend", "--backend", "oaf", "--oracle", str(oracle_path)],
    )
    assert mismatching.exit_code != 0


def test_smoke_backend_compares_oracle_to_canonical_published_prediction(
    tmp_path: Path, monkeypatch
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    base_prediction = _prediction(input_root)
    raw_prediction = replace(
        base_prediction,
        events=(
            replace(
                base_prediction.events[0],
                time_sec=0.123456789,
                confidence=0.987654321,
            ),
            *base_prediction.events[1:],
        ),
    )
    mapped_prediction = map_oaf_prediction(raw_prediction)[0]
    oracle_source_path = tmp_path / "oracle-source.jsonl"
    publish_prediction_artifact(oracle_source_path, mapped_prediction)
    canonical_prediction = read_prediction_artifact(oracle_source_path.read_bytes()).prediction
    oracle_path = tmp_path / "oracle.json"
    oracle_path.write_bytes(render_smoke_oracle(canonical_prediction))

    assert mapped_prediction.events[0].native.time_sec == 0.123456789
    assert canonical_prediction.events[0].native.time_sec == 0.123457
    assert mapped_prediction.events[0].native.confidence == 0.987654321
    assert canonical_prediction.events[0].native.confidence == 0.987654

    import src.cli.benchmark as benchmark_module

    fake_registry = _FakeRegistry(_FakeBackend(raw_prediction))
    monkeypatch.setattr(benchmark_module, "default_backend_registry", lambda: fake_registry)
    monkeypatch.setattr(
        benchmark_module,
        "load_model_config",
        lambda: type(
            "Config",
            (),
            {
                "max_input_audio_frames": None,
                "checkpoint": type("Checkpoint", (), {"archive_sha256": "a" * 64})(),
            },
        )(),
    )
    monkeypatch.setattr(
        benchmark_module, "load_direct_audio", lambda *args, **kwargs: raw_prediction.audio
    )
    monkeypatch.setattr(benchmark_module.time, "perf_counter", iter((10.0, 12.0)).__next__)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        ["benchmark", "smoke-backend", "--backend", "oaf", "--oracle", str(oracle_path)],
    )

    assert result.exit_code == 0
    summary = json.loads(result.output)
    assert summary["oracle_status"] == "matched"
    persisted = Path(summary["prediction_path"]).read_bytes()
    assert summary["prediction_sha256"] == sha256_hex(persisted)
    assert read_prediction_artifact(persisted).prediction == canonical_prediction


def test_run_oaf_corpus_builds_request_and_emits_quantized_canonical_summary(
    tmp_path: Path, monkeypatch
) -> None:
    import src.benchmark.oaf_corpus_run as runner_module

    manifest_path = tmp_path / "hpa324.jsonl"
    timing_manifest_path = tmp_path / "hpa323.jsonl"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    run_path = output_dir / "runs" / "run-123" / "run.json"
    reports_path = output_dir / "runs" / "run-123" / "reports"
    captured: list[object] = []

    def fake_run(request: object) -> OafCorpusRunOutcome:
        captured.append(request)
        return OafCorpusRunOutcome(
            overall_status="complete",
            exit_code=0,
            run_id="run-123",
            run_path=run_path,
            reports_path=reports_path,
            success_count=2,
            failed_count=0,
            skipped_count=0,
            quarantined_count=0,
            aggregate_rtf=1.23456789,
            projected_full_wall_time_sec=12.34567891,
        )

    monkeypatch.setattr(runner_module, "run_oaf_corpus", fake_run)
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "run-oaf-corpus",
            "--manifest",
            str(manifest_path),
            "--timing-manifest",
            str(timing_manifest_path),
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--include-simfile-id",
            "10",
            "--include-simfile-id",
            "20",
            "--exclude-simfile-id",
            "30",
            "--resume",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert len(captured) == 1
    request = captured[0]
    assert request.reference_manifest_path == manifest_path
    assert request.timing_manifest_path == timing_manifest_path
    assert request.cache_dir == cache_dir
    assert request.output_dir == output_dir
    assert request.include_simfile_ids == (10, 20)
    assert request.exclude_simfile_ids == (30,)
    assert request.resume is True

    summary = json.loads(result.output)
    assert set(summary) == {
        "aggregate_rtf",
        "exit_code",
        "failed_count",
        "projected_full_wall_time_sec",
        "quarantined_count",
        "reports_path",
        "run_id",
        "run_path",
        "skipped_count",
        "status",
        "success_count",
    }
    assert summary["status"] == "complete"
    assert summary["exit_code"] == 0
    assert summary["aggregate_rtf"] == 1.234568
    assert summary["projected_full_wall_time_sec"] == 12.345679
    assert summary["run_path"] == str(run_path)
    assert summary["reports_path"] == str(reports_path)


def test_run_oaf_corpus_preserves_domain_preflight_exit_code(tmp_path: Path, monkeypatch) -> None:
    import src.benchmark.oaf_corpus_run as runner_module

    def fake_run(_request: object) -> OafCorpusRunOutcome:
        return OafCorpusRunOutcome(
            overall_status="failed",
            exit_code=2,
            run_id=None,
            run_path=None,
            reports_path=None,
            success_count=0,
            failed_count=0,
            skipped_count=0,
            quarantined_count=0,
            aggregate_rtf=None,
            projected_full_wall_time_sec=None,
        )

    monkeypatch.setattr(runner_module, "run_oaf_corpus", fake_run)
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "run-oaf-corpus",
            "--manifest",
            str(tmp_path / "hpa324.jsonl"),
            "--timing-manifest",
            str(tmp_path / "hpa323.jsonl"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "output"),
            "--include-simfile-id",
            "10",
            "--include-simfile-id",
            "10",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 2
    assert "Usage:" not in result.output
    summary = json.loads(result.output)
    assert summary["status"] == "failed"
    assert summary["exit_code"] == 2
    assert summary["run_id"] is None
    assert summary["run_path"] is None
    assert summary["reports_path"] is None
    assert summary["success_count"] == 0
    assert summary["failed_count"] == 0
    assert summary["skipped_count"] == 0
    assert summary["quarantined_count"] == 0
    assert summary["aggregate_rtf"] is None
    assert summary["projected_full_wall_time_sec"] is None


def test_run_oaf_corpus_overlap_scope_is_canonical_fatal(tmp_path: Path, monkeypatch) -> None:
    import src.benchmark.oaf_corpus_run as runner_module

    def unexpected_run(_request: object) -> OafCorpusRunOutcome:
        raise AssertionError("domain runner must not run for invalid request scope")

    monkeypatch.setattr(runner_module, "run_oaf_corpus", unexpected_run)
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "run-oaf-corpus",
            "--manifest",
            str(tmp_path / "hpa324.jsonl"),
            "--timing-manifest",
            str(tmp_path / "hpa323.jsonl"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "output"),
            "--include-simfile-id",
            "10",
            "--exclude-simfile-id",
            "10",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.output)["exit_code"] == 2
    assert json.loads(result.output)["status"] == "failed"


def test_run_oaf_corpus_non_positive_scope_is_canonical_fatal(tmp_path: Path, monkeypatch) -> None:
    import src.benchmark.oaf_corpus_run as runner_module

    def unexpected_run(_request: object) -> OafCorpusRunOutcome:
        raise AssertionError("domain runner must not run for invalid request scope")

    monkeypatch.setattr(runner_module, "run_oaf_corpus", unexpected_run)
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "run-oaf-corpus",
            "--manifest",
            str(tmp_path / "hpa324.jsonl"),
            "--timing-manifest",
            str(tmp_path / "hpa323.jsonl"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "output"),
            "--include-simfile-id",
            "0",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.output)["exit_code"] == 2
    assert json.loads(result.output)["status"] == "failed"


def test_run_muscriptor_corpus_builds_frozen_request_and_emits_summary(
    tmp_path: Path, monkeypatch
) -> None:
    import src.benchmark.muscriptor_corpus_run as runner_module
    from src.benchmark.muscriptor_corpus_run import MuscriptorCorpusRunOutcome

    manifest_path = tmp_path / "hpa324.jsonl"
    timing_manifest_path = tmp_path / "hpa323.jsonl"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    run_path = output_dir / "runs" / "run-123" / "run.json"
    reports_path = output_dir / "runs" / "run-123" / "reports"
    captured: list[object] = []

    def fake_run(request: object) -> MuscriptorCorpusRunOutcome:
        captured.append(request)
        return MuscriptorCorpusRunOutcome(
            overall_status="complete",
            exit_code=0,
            run_id="run-123",
            run_path=run_path,
            reports_path=reports_path,
            success_count=2,
            failed_count=0,
            skipped_count=0,
            quarantined_count=0,
            aggregate_rtf=1.23456789,
            projected_full_wall_time_sec=12.34567891,
            peak_process_rss_bytes=123,
            device_peak_memory_bytes=None,
        )

    monkeypatch.setattr(runner_module, "run_muscriptor_corpus", fake_run)
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "run-muscriptor-corpus",
            "--manifest",
            str(manifest_path),
            "--timing-manifest",
            str(timing_manifest_path),
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--include-simfile-id",
            "10",
            "--include-simfile-id",
            "20",
            "--exclude-simfile-id",
            "30",
            "--resume",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert len(captured) == 1
    request = captured[0]
    assert request.reference_manifest_path == manifest_path
    assert request.timing_manifest_path == timing_manifest_path
    assert request.cache_dir == cache_dir
    assert request.output_dir == output_dir
    assert request.include_simfile_ids == (10, 20)
    assert request.exclude_simfile_ids == (30,)
    assert request.resume is True

    summary = json.loads(result.output)
    assert set(summary) == {
        "aggregate_rtf",
        "device_peak_memory_bytes",
        "exit_code",
        "failed_count",
        "peak_process_rss_bytes",
        "projected_full_wall_time_sec",
        "quarantined_count",
        "reports_path",
        "run_id",
        "run_path",
        "skipped_count",
        "status",
        "success_count",
    }
    assert summary["status"] == "complete"
    assert summary["exit_code"] == 0
    assert summary["aggregate_rtf"] == 1.234568
    assert summary["projected_full_wall_time_sec"] == 12.345679
    assert summary["peak_process_rss_bytes"] == 123
    assert summary["device_peak_memory_bytes"] is None
    assert summary["run_path"] == str(run_path)
    assert summary["reports_path"] == str(reports_path)


def test_score_muscriptor_reviewed_subset_command_builds_request_and_emits_summary(
    tmp_path: Path, monkeypatch
) -> None:
    import src.benchmark.reviewed_subset as reviewed_module

    run_path = tmp_path / "run.json"
    reference_manifest_path = tmp_path / "hpa324.jsonl"
    timing_manifest_path = tmp_path / "hpa323.jsonl"
    subset_manifest_path = tmp_path / "subset.jsonl"
    output_dir = tmp_path / "reports"
    for path in (
        run_path,
        reference_manifest_path,
        timing_manifest_path,
        subset_manifest_path,
    ):
        path.write_bytes(b"x")
    captured: list[object] = []

    def fake_score(request: object) -> ScoreReviewedSubsetOutcome:
        captured.append(request)
        return ScoreReviewedSubsetOutcome(
            exit_code=0,
            cohort_id="c" * 64,
            reports_path=output_dir,
            success_count=20,
            failed_count=0,
            skipped_count=0,
            quarantined_count=0,
        )

    monkeypatch.setattr(reviewed_module, "score_muscriptor_reviewed_subset", fake_score)
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "score-muscriptor-reviewed-subset",
            "--run",
            str(run_path),
            "--manifest",
            str(reference_manifest_path),
            "--timing-manifest",
            str(timing_manifest_path),
            "--subset-manifest",
            str(subset_manifest_path),
            "--output-dir",
            str(output_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert len(captured) == 1
    request = captured[0]
    assert request.run_path == run_path
    assert request.reference_manifest_path == reference_manifest_path
    assert request.timing_manifest_path == timing_manifest_path
    assert request.subset_manifest_path == subset_manifest_path
    assert request.output_dir == output_dir

    summary = json.loads(result.output)
    assert set(summary) == {
        "cohort_id",
        "exit_code",
        "failed_count",
        "quarantined_count",
        "reports_path",
        "skipped_count",
        "success_count",
    }
    assert summary["cohort_id"] == "c" * 64
    assert summary["success_count"] == 20
    assert summary["reports_path"] == str(output_dir)


def test_reviewed_subset_commands_declare_exact_options() -> None:
    expected_options = {
        "prepare-reviewed-subset": {
            "--manifest",
            "--timing-manifest",
            "--output-file",
            "--prior-ledger",
        },
        "finalize-reviewed-subset": {
            "--manifest",
            "--timing-manifest",
            "--review-file",
            "--output-dir",
            "--prior-ledger",
        },
        "score-oaf-reviewed-subset": {
            "--run",
            "--manifest",
            "--timing-manifest",
            "--subset-manifest",
            "--output-dir",
        },
        "run-muscriptor-corpus": {
            "--manifest",
            "--timing-manifest",
            "--cache-dir",
            "--output-dir",
            "--include-simfile-id",
            "--exclude-simfile-id",
            "--resume",
        },
        "score-muscriptor-reviewed-subset": {
            "--run",
            "--manifest",
            "--timing-manifest",
            "--subset-manifest",
            "--output-dir",
        },
        "compare-oaf-muscriptor": {
            "--oaf-run",
            "--muscriptor-run",
            "--manifest",
            "--timing-manifest",
            "--output-dir",
            "--subset-manifest",
        },
        "run-oaf-separation-pilot": {
            "--manifest",
            "--timing-manifest",
            "--subset-manifest",
            "--oaf-run",
            "--cache-dir",
            "--output-dir",
            "--spleeter-python",
            "--demucs-python",
            "--spleeter-model-root",
            "--demucs-model-root",
            "--resume",
        },
        "finalize-oaf-separation-pilot": {
            "--run",
            "--subset-manifest",
            "--output-manifest",
            "--decision",
            "--rationale",
        },
    }
    benchmark_group = main.commands["benchmark"]
    for command, expected in expected_options.items():
        declared = {
            opt
            for param in benchmark_group.commands[command].params
            if isinstance(param, click.Option)
            for opt in param.opts
            if opt.startswith("--")
        }
        message = f"{command} declared options {sorted(declared)}"
        message += f" != expected {sorted(expected)}"
        assert declared == expected, message
        # The help output must agree with the introspected option set and must
        # not advertise any unsupported selector such as --model-path.
        result = CliRunner().invoke(main, ["benchmark", command, "--help"])
        assert result.exit_code == 0
        for option in expected:
            assert option in result.output
        assert "--model-path" not in result.output
        for selector in (
            "--seed",
            "--count",
            "--threshold",
            "--model",
            "--backend",
            "--model-size",
            "--device",
            "--dtype",
            "--temperature",
            "--beam",
            "--instrument",
            "--map-version",
            "--model-path",
            "--model-id",
            "--model-lock",
            "--qc",
            "--qc-policy",
            "--tolerance",
            "--tolerance-ms",
            "--full-corpus",
            "--third-separator",
            "--third-separator-python",
            "--cost-rate",
            "--cost-per-minute",
            "--dollar-cost-rate",
        ):
            assert selector not in result.output


def test_separation_pilot_commands_keep_domain_imports_lazy() -> None:
    import subprocess
    import sys

    probe = """
import json
import sys

from click.testing import CliRunner

from src.cli.main import main

for command in ("run-oaf-separation-pilot", "finalize-oaf-separation-pilot"):
    result = CliRunner().invoke(main, ["benchmark", command, "--help"])
    if result.exit_code != 0:
        raise SystemExit(result.output)

print(json.dumps({
    "pilot_imported": "src.benchmark.separation_pilot" in sys.modules,
    "handoff_imported": "src.benchmark.separation_handoff" in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {"pilot_imported": False, "handoff_imported": False}


def test_run_oaf_separation_pilot_command_builds_request_and_propagates_partial_exit(
    tmp_path: Path, monkeypatch
) -> None:
    import src.benchmark.separation_pilot as pilot_module
    import src.cli.benchmark as benchmark_module

    manifest = tmp_path / "reference.jsonl"
    timing_manifest = tmp_path / "timing.jsonl"
    subset_manifest = tmp_path / "subset.jsonl"
    oaf_run = tmp_path / "oaf" / "run.json"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    spleeter_python = tmp_path / "spleeter-python"
    demucs_python = tmp_path / "demucs-python"
    spleeter_model_root = tmp_path / "spleeter-model-root"
    demucs_model_root = tmp_path / "demucs-model-root"
    for path in (manifest, timing_manifest, subset_manifest, spleeter_python, demucs_python):
        path.write_bytes(b"fixture")
    oaf_run.parent.mkdir()
    oaf_run.write_bytes(b"fixture")
    cache_dir.mkdir()
    output_dir.mkdir()
    spleeter_model_root.mkdir()
    demucs_model_root.mkdir()
    captured: list[object] = []
    run_path = output_dir / "runs" / "pilot" / "run.json"
    monkeypatch.setattr(benchmark_module, "_current_crux_commit", lambda: "a" * 40)

    def fake_run(request: object) -> pilot_module.OafSeparationPilotOutcome:
        captured.append(request)
        return pilot_module.OafSeparationPilotOutcome(
            overall_status="partial",
            exit_code=1,
            run_id="oaf-separation-1234567890abcdef",
            run_path=run_path,
            reports_path=run_path.parent / "reports",
            full_mix_reports_path=run_path.parent / "full-mix-reports",
            success_count=3,
            failed_count=1,
            skipped_count=2,
            quarantined_count=4,
            failure_code=None,
        )

    monkeypatch.setattr(pilot_module, "run_oaf_separation_pilot", fake_run)
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "run-oaf-separation-pilot",
            "--manifest",
            str(manifest),
            "--timing-manifest",
            str(timing_manifest),
            "--subset-manifest",
            str(subset_manifest),
            "--oaf-run",
            str(oaf_run),
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--spleeter-python",
            str(spleeter_python),
            "--demucs-python",
            str(demucs_python),
            "--spleeter-model-root",
            str(spleeter_model_root),
            "--demucs-model-root",
            str(demucs_model_root),
            "--resume",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert len(captured) == 1
    request = captured[0]
    assert request.reference_manifest_path == manifest
    assert request.timing_manifest_path == timing_manifest
    assert request.subset_manifest_path == subset_manifest
    assert request.oaf_run_path == oaf_run
    assert request.cache_dir == cache_dir
    assert request.output_dir == output_dir
    assert request.spleeter_python == spleeter_python
    assert request.demucs_python == demucs_python
    assert request.spleeter_model_root == spleeter_model_root
    assert request.demucs_model_root == demucs_model_root
    assert request.resume is True
    assert request.crux_commit == "a" * 40
    assert json.loads(result.output) == {
        "exit_code": 1,
        "failed_count": 1,
        "failure_code": None,
        "full_mix_reports_path": str(run_path.parent / "full-mix-reports"),
        "quarantined_count": 4,
        "reports_path": str(run_path.parent / "reports"),
        "run_id": "oaf-separation-1234567890abcdef",
        "run_path": str(run_path),
        "skipped_count": 2,
        "status": "partial",
        "success_count": 3,
    }


def test_current_crux_commit_accepts_canonical_git_revision(
    monkeypatch,
) -> None:
    import src.cli.benchmark as benchmark_module

    class Result:
        stdout = "a" * 40 + "\n"

    monkeypatch.setattr(benchmark_module.subprocess, "run", lambda *_args, **_kwargs: Result())

    assert benchmark_module._current_crux_commit() == "a" * 40


def test_run_oaf_separation_pilot_command_preserves_complete_and_fatal_exit_codes(
    tmp_path: Path, monkeypatch
) -> None:
    import src.benchmark.separation_pilot as pilot_module
    import src.cli.benchmark as benchmark_module

    monkeypatch.setattr(benchmark_module, "_current_crux_commit", lambda: "a" * 40)
    outcomes = [
        pilot_module.OafSeparationPilotOutcome(
            overall_status="complete",
            exit_code=0,
            run_id="oaf-separation-1234567890abcdef",
            run_path=tmp_path / "complete.json",
            reports_path=None,
            full_mix_reports_path=None,
            success_count=20,
            failed_count=0,
            skipped_count=0,
            quarantined_count=0,
            failure_code=None,
        ),
        pilot_module.OafSeparationPilotOutcome(
            overall_status="failed",
            exit_code=2,
            run_id=None,
            run_path=None,
            reports_path=None,
            full_mix_reports_path=None,
            success_count=0,
            failed_count=0,
            skipped_count=0,
            quarantined_count=0,
            failure_code="separator_environment_mismatch",
        ),
    ]

    def fake_run(_request: object) -> pilot_module.OafSeparationPilotOutcome:
        return outcomes.pop(0)

    monkeypatch.setattr(pilot_module, "run_oaf_separation_pilot", fake_run)
    args = [
        "benchmark",
        "run-oaf-separation-pilot",
        "--manifest",
        str(tmp_path / "reference.jsonl"),
        "--timing-manifest",
        str(tmp_path / "timing.jsonl"),
        "--subset-manifest",
        str(tmp_path / "subset.jsonl"),
        "--oaf-run",
        str(tmp_path / "oaf-run.json"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--output-dir",
        str(tmp_path / "output"),
        "--spleeter-python",
        str(tmp_path / "spleeter-python"),
        "--demucs-python",
        str(tmp_path / "demucs-python"),
        "--spleeter-model-root",
        str(tmp_path / "spleeter-model-root"),
        "--demucs-model-root",
        str(tmp_path / "demucs-model-root"),
    ]

    complete = CliRunner().invoke(main, args, catch_exceptions=False)
    fatal = CliRunner().invoke(main, args, catch_exceptions=False)

    assert complete.exit_code == 0
    assert json.loads(complete.output)["exit_code"] == 0
    assert json.loads(complete.output)["failure_code"] is None
    assert fatal.exit_code == 2
    fatal_payload = json.loads(fatal.output)
    assert fatal_payload == {
        "exit_code": 2,
        "failed_count": 0,
        "failure_code": "separator_environment_mismatch",
        "full_mix_reports_path": None,
        "quarantined_count": 0,
        "reports_path": None,
        "run_id": None,
        "run_path": None,
        "skipped_count": 0,
        "status": "failed",
        "success_count": 0,
    }
    assert fatal_payload["failure_code"] in {
        "separator_lock_companion_mismatch",
        "separator_interpreter_mismatch",
        "separator_environment_mismatch",
        "separator_model_root_invalid",
        "separator_environment_probe_failed",
    }


def test_run_oaf_separation_pilot_command_sanitizes_native_fatal_detail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import src.benchmark.separation_pilot as pilot_module
    import src.cli.benchmark as benchmark_module
    from src.benchmark.separators import SeparatorExecutionError
    from tests.benchmark.reviewed_subset_fixtures import build_reviewed_subset_oaf_fixture
    from tests.benchmark.test_separation_pilot import _subset_path, _task6_seams

    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20, failed_count=0)
    subset_manifest = _subset_path(tmp_path, fixture)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    del calls

    private_runtime_path = (tmp_path / "private-runtime" / "spleeter" / "python").resolve()

    def failing_attest(*_args: object, **_kwargs: object) -> object:
        raise SeparatorExecutionError(
            "separator_environment_mismatch",
            f"private runtime detail: {private_runtime_path}",
        )

    monkeypatch.setattr(pilot_module, "attest_separator_runtime", failing_attest)
    monkeypatch.setattr(benchmark_module, "_current_crux_commit", lambda: "a" * 40)
    args = [
        "benchmark",
        "run-oaf-separation-pilot",
        "--manifest",
        str(fixture.reference_manifest_path),
        "--timing-manifest",
        str(fixture.timing_manifest_path),
        "--subset-manifest",
        str(subset_manifest),
        "--oaf-run",
        str(fixture.run_path),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--output-dir",
        str(tmp_path / "separation-output"),
        "--spleeter-python",
        str(private_runtime_path),
        "--demucs-python",
        str(tmp_path / "private-runtime" / "demucs" / "python"),
        "--spleeter-model-root",
        str(tmp_path / "private-runtime" / "spleeter-model-root"),
        "--demucs-model-root",
        str(tmp_path / "private-runtime" / "demucs-model-root"),
    ]

    result = CliRunner().invoke(main, args, catch_exceptions=False)

    assert result.exit_code == 2
    assert json.loads(result.output) == {
        "exit_code": 2,
        "failed_count": 0,
        "failure_code": "separator_environment_mismatch",
        "full_mix_reports_path": None,
        "quarantined_count": 0,
        "reports_path": None,
        "run_id": None,
        "run_path": None,
        "skipped_count": 0,
        "status": "failed",
        "success_count": 0,
    }
    assert str(private_runtime_path) not in result.output
    assert "private runtime detail" not in result.output


def test_finalize_oaf_separation_pilot_command_publishes_manifest_summary(
    tmp_path: Path, monkeypatch
) -> None:
    import src.benchmark.separation_handoff as handoff_module

    run_path = tmp_path / "run.json"
    subset_manifest = tmp_path / "subset.jsonl"
    output_manifest = tmp_path / "handoff.jsonl"
    run_path.write_bytes(b"fixture")
    subset_manifest.write_bytes(b"fixture")
    published_path = tmp_path / "manifests" / ("a" * 64 + ".jsonl")
    published = type(
        "Published",
        (),
        {"path": published_path, "manifest_sha256": "a" * 64},
    )()
    captured: list[object] = []

    def fake_finalize(request: object) -> handoff_module.FinalizeSeparationPilotOutcome:
        captured.append(request)
        return handoff_module.FinalizeSeparationPilotOutcome(exit_code=0, manifest=published)

    monkeypatch.setattr(handoff_module, "finalize_separation_pilot", fake_finalize)
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "finalize-oaf-separation-pilot",
            "--run",
            str(run_path),
            "--subset-manifest",
            str(subset_manifest),
            "--output-manifest",
            str(output_manifest),
            "--decision",
            "use_htdemucs",
            "--rationale",
            "The retained evidence supports the reviewed HTDemucs choice.",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert len(captured) == 1
    request = captured[0]
    assert request.run_path == run_path
    assert request.subset_manifest_path == subset_manifest
    assert request.output_manifest == output_manifest
    assert request.decision == "use_htdemucs"
    assert request.rationale == "The retained evidence supports the reviewed HTDemucs choice."
    assert json.loads(result.output) == {
        "exit_code": 0,
        "manifest_path": str(published_path),
        "manifest_sha256": "a" * 64,
        "failure_reason": None,
    }


def test_finalize_oaf_separation_pilot_command_rejects_unpublished_success(
    tmp_path: Path, monkeypatch
) -> None:
    import src.benchmark.separation_handoff as handoff_module

    run_path = tmp_path / "run.json"
    subset_manifest = tmp_path / "subset.jsonl"
    output_manifest = tmp_path / "handoff.jsonl"
    run_path.write_bytes(b"fixture")
    subset_manifest.write_bytes(b"fixture")

    def fake_finalize(_request: object) -> handoff_module.FinalizeSeparationPilotOutcome:
        return handoff_module.FinalizeSeparationPilotOutcome(exit_code=0, manifest=None)

    monkeypatch.setattr(handoff_module, "finalize_separation_pilot", fake_finalize)
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "finalize-oaf-separation-pilot",
            "--run",
            str(run_path),
            "--subset-manifest",
            str(subset_manifest),
            "--output-manifest",
            str(output_manifest),
            "--decision",
            "keep_full_mix",
            "--rationale",
            "No immutable decision artifact was published.",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["exit_code"] == 2
    assert payload["manifest_path"] is None
    assert payload["manifest_sha256"] is None
    assert payload["failure_reason"] is None


def test_prepare_reviewed_subset_command_builds_request_and_emits_canonical_summary(
    tmp_path: Path, monkeypatch
) -> None:
    import src.benchmark.reviewed_subset as reviewed_module

    reference_manifest_path = tmp_path / "hpa324.jsonl"
    timing_manifest_path = tmp_path / "hpa323.jsonl"
    prior_ledger_path = tmp_path / "prior.csv"
    output_file = tmp_path / "prepared.csv"
    for path in (reference_manifest_path, timing_manifest_path, prior_ledger_path):
        path.write_bytes(b"x")
    captured: list[object] = []

    def fake_prepare(request: object) -> PrepareReviewedSubsetOutcome:
        captured.append(request)
        return PrepareReviewedSubsetOutcome(
            exit_code=0,
            output_file=output_file,
            candidate_count=30,
            carried_include_count=4,
            replacement_count=26,
        )

    monkeypatch.setattr(reviewed_module, "prepare_reviewed_subset", fake_prepare)
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "prepare-reviewed-subset",
            "--manifest",
            str(reference_manifest_path),
            "--timing-manifest",
            str(timing_manifest_path),
            "--output-file",
            str(output_file),
            "--prior-ledger",
            str(prior_ledger_path),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert len(captured) == 1
    request = captured[0]
    assert request.reference_manifest_path == reference_manifest_path
    assert request.timing_manifest_path == timing_manifest_path
    assert request.output_file == output_file
    assert request.prior_ledger_path == prior_ledger_path

    summary = json.loads(result.output)
    assert set(summary) == {
        "candidate_count",
        "carried_include_count",
        "exit_code",
        "output_file",
        "replacement_count",
    }
    assert summary["exit_code"] == 0
    assert summary["candidate_count"] == 30
    assert summary["carried_include_count"] == 4
    assert summary["replacement_count"] == 26
    assert summary["output_file"] == str(output_file)

    captured.clear()
    omitted = CliRunner().invoke(
        main,
        [
            "benchmark",
            "prepare-reviewed-subset",
            "--manifest",
            str(reference_manifest_path),
            "--timing-manifest",
            str(timing_manifest_path),
            "--output-file",
            str(output_file),
        ],
        catch_exceptions=False,
    )
    assert omitted.exit_code == 0
    assert len(captured) == 1
    assert captured[0].prior_ledger_path is None


def test_finalize_reviewed_subset_command_builds_request_and_emits_canonical_summary(
    tmp_path: Path, monkeypatch
) -> None:
    import src.benchmark.reviewed_subset as reviewed_module

    reference_manifest_path = tmp_path / "hpa324.jsonl"
    timing_manifest_path = tmp_path / "hpa323.jsonl"
    review_file = tmp_path / "review.csv"
    output_dir = tmp_path / "subset"
    for path in (reference_manifest_path, timing_manifest_path, review_file):
        path.write_bytes(b"x")
    manifest = PublishedManifest(
        corpus_version="sha256:" + "a" * 64,
        manifest_sha256="b" * 64,
        relative_path="manifests/" + "b" * 64 + ".jsonl",
        path=output_dir / "manifests" / ("b" * 64 + ".jsonl"),
        latest_path=output_dir / "latest.json",
    )
    review_ledger_path = output_dir / "review-ledger.csv"
    captured: list[object] = []

    def fake_finalize(request: object) -> FinalizeReviewedSubsetOutcome:
        captured.append(request)
        return FinalizeReviewedSubsetOutcome(
            exit_code=0,
            manifest=manifest,
            review_ledger_path=review_ledger_path,
            included_count=20,
            excluded_count=10,
        )

    monkeypatch.setattr(reviewed_module, "finalize_reviewed_subset", fake_finalize)
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "finalize-reviewed-subset",
            "--manifest",
            str(reference_manifest_path),
            "--timing-manifest",
            str(timing_manifest_path),
            "--review-file",
            str(review_file),
            "--output-dir",
            str(output_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert len(captured) == 1
    request = captured[0]
    assert request.reference_manifest_path == reference_manifest_path
    assert request.timing_manifest_path == timing_manifest_path
    assert request.review_file == review_file
    assert request.output_dir == output_dir
    assert request.prior_ledger_path is None

    summary = json.loads(result.output)
    assert set(summary) == {
        "excluded_count",
        "exit_code",
        "included_count",
        "manifest_path",
        "review_ledger_path",
    }
    assert summary["manifest_path"] == str(manifest.path)
    assert summary["review_ledger_path"] == str(review_ledger_path)
    assert summary["included_count"] == 20
    assert summary["excluded_count"] == 10


def test_score_oaf_reviewed_subset_command_builds_request_and_emits_canonical_summary(
    tmp_path: Path, monkeypatch
) -> None:
    import src.benchmark.reviewed_subset as reviewed_module

    run_path = tmp_path / "run.json"
    reference_manifest_path = tmp_path / "hpa324.jsonl"
    timing_manifest_path = tmp_path / "hpa323.jsonl"
    subset_manifest_path = tmp_path / "subset.jsonl"
    output_dir = tmp_path / "reports"
    for path in (
        run_path,
        reference_manifest_path,
        timing_manifest_path,
        subset_manifest_path,
    ):
        path.write_bytes(b"x")
    captured: list[object] = []

    def fake_score(request: object) -> ScoreReviewedSubsetOutcome:
        captured.append(request)
        return ScoreReviewedSubsetOutcome(
            exit_code=0,
            cohort_id="c" * 64,
            reports_path=output_dir,
            success_count=20,
            failed_count=0,
            skipped_count=0,
            quarantined_count=0,
        )

    monkeypatch.setattr(reviewed_module, "score_oaf_reviewed_subset", fake_score)
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "score-oaf-reviewed-subset",
            "--run",
            str(run_path),
            "--manifest",
            str(reference_manifest_path),
            "--timing-manifest",
            str(timing_manifest_path),
            "--subset-manifest",
            str(subset_manifest_path),
            "--output-dir",
            str(output_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert len(captured) == 1
    request = captured[0]
    assert request.run_path == run_path
    assert request.reference_manifest_path == reference_manifest_path
    assert request.timing_manifest_path == timing_manifest_path
    assert request.subset_manifest_path == subset_manifest_path
    assert request.output_dir == output_dir

    summary = json.loads(result.output)
    assert set(summary) == {
        "cohort_id",
        "exit_code",
        "failed_count",
        "quarantined_count",
        "reports_path",
        "skipped_count",
        "success_count",
    }
    assert summary["cohort_id"] == "c" * 64
    assert summary["success_count"] == 20
    assert summary["failed_count"] == 0
    assert summary["skipped_count"] == 0
    assert summary["quarantined_count"] == 0
    assert summary["reports_path"] == str(output_dir)


def test_reviewed_subset_commands_propagate_domain_exit_code(tmp_path: Path, monkeypatch) -> None:
    import src.benchmark.reviewed_subset as reviewed_module

    reference_manifest_path = tmp_path / "hpa324.jsonl"
    timing_manifest_path = tmp_path / "hpa323.jsonl"
    review_file = tmp_path / "review.csv"
    run_path = tmp_path / "run.json"
    subset_manifest_path = tmp_path / "subset.jsonl"
    for path in (
        reference_manifest_path,
        timing_manifest_path,
        review_file,
        run_path,
        subset_manifest_path,
    ):
        path.write_bytes(b"x")

    def fake_prepare(_request: object) -> PrepareReviewedSubsetOutcome:
        return PrepareReviewedSubsetOutcome(
            exit_code=2,
            output_file=None,
            candidate_count=0,
            carried_include_count=0,
            replacement_count=0,
        )

    def fake_finalize(_request: object) -> FinalizeReviewedSubsetOutcome:
        return FinalizeReviewedSubsetOutcome(
            exit_code=2,
            manifest=None,
            review_ledger_path=None,
            included_count=0,
            excluded_count=0,
        )

    def fake_score(_request: object) -> ScoreReviewedSubsetOutcome:
        return ScoreReviewedSubsetOutcome(
            exit_code=2,
            cohort_id=None,
            reports_path=None,
            success_count=0,
            failed_count=0,
            skipped_count=0,
            quarantined_count=0,
        )

    monkeypatch.setattr(reviewed_module, "prepare_reviewed_subset", fake_prepare)
    monkeypatch.setattr(reviewed_module, "finalize_reviewed_subset", fake_finalize)
    monkeypatch.setattr(reviewed_module, "score_oaf_reviewed_subset", fake_score)

    prepare = CliRunner().invoke(
        main,
        [
            "benchmark",
            "prepare-reviewed-subset",
            "--manifest",
            str(reference_manifest_path),
            "--timing-manifest",
            str(timing_manifest_path),
            "--output-file",
            str(tmp_path / "prepared.csv"),
        ],
        catch_exceptions=False,
    )
    assert prepare.exit_code == 2
    assert json.loads(prepare.output)["exit_code"] == 2

    finalize = CliRunner().invoke(
        main,
        [
            "benchmark",
            "finalize-reviewed-subset",
            "--manifest",
            str(reference_manifest_path),
            "--timing-manifest",
            str(timing_manifest_path),
            "--review-file",
            str(review_file),
            "--output-dir",
            str(tmp_path / "subset"),
        ],
        catch_exceptions=False,
    )
    assert finalize.exit_code == 2
    assert json.loads(finalize.output)["exit_code"] == 2

    score = CliRunner().invoke(
        main,
        [
            "benchmark",
            "score-oaf-reviewed-subset",
            "--run",
            str(run_path),
            "--manifest",
            str(reference_manifest_path),
            "--timing-manifest",
            str(timing_manifest_path),
            "--subset-manifest",
            str(subset_manifest_path),
            "--output-dir",
            str(tmp_path / "reports"),
        ],
        catch_exceptions=False,
    )
    assert score.exit_code == 2
    assert json.loads(score.output)["exit_code"] == 2


def test_run_oaf_corpus_rejects_out_of_range_id_as_click_error(tmp_path: Path, monkeypatch) -> None:
    import src.benchmark.oaf_corpus_run as runner_module

    def unexpected_run(_request: object) -> OafCorpusRunOutcome:
        raise AssertionError("domain runner must not run for a syntax-range violation")

    monkeypatch.setattr(runner_module, "run_oaf_corpus", unexpected_run)
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "run-oaf-corpus",
            "--manifest",
            str(tmp_path / "hpa324.jsonl"),
            "--timing-manifest",
            str(tmp_path / "hpa323.jsonl"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "output"),
            "--include-simfile-id",
            str(MAX_SIMFILE_ID + 1),
        ],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--include-simfile-id'" in result.output


def test_compare_oaf_muscriptor_command_builds_request_and_emits_summary(
    tmp_path: Path, monkeypatch
) -> None:
    import src.benchmark.muscriptor_comparison as comparison_module

    oaf_run = tmp_path / "oaf" / "run.json"
    muscriptor_run = tmp_path / "muscriptor" / "run.json"
    manifest = tmp_path / "hpa324.jsonl"
    timing_manifest = tmp_path / "hpa323.jsonl"
    output_dir = tmp_path / "comparison"
    for path in (oaf_run, muscriptor_run, manifest, timing_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    captured: list[object] = []

    def fake_compare(request: object):
        captured.append(request)
        return type(
            "Outcome",
            (),
            {
                "exit_code": 0,
                "output_dir": output_dir,
                "pairable_success_count": 7,
                "paired_song_count": 14,
                "paired_class_count": 28,
            },
        )()

    monkeypatch.setattr(comparison_module, "compare_oaf_muscriptor", fake_compare)
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "compare-oaf-muscriptor",
            "--oaf-run",
            str(oaf_run),
            "--muscriptor-run",
            str(muscriptor_run),
            "--manifest",
            str(manifest),
            "--timing-manifest",
            str(timing_manifest),
            "--output-dir",
            str(output_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert len(captured) == 1
    request = captured[0]
    assert request.oaf_run_path == oaf_run
    assert request.muscriptor_run_path == muscriptor_run
    assert request.reference_manifest_path == manifest
    assert request.timing_manifest_path == timing_manifest
    assert request.output_dir == output_dir
    assert request.subset_manifest_path is None

    summary = json.loads(result.output)
    assert summary == {
        "exit_code": 0,
        "output_dir": str(output_dir),
        "paired_class_count": 28,
        "paired_song_count": 14,
        "pairable_success_count": 7,
    }


def _idm_pilot_cli_args(tmp_path: Path) -> list[str]:
    return [
        "benchmark",
        "run-idm-pilot",
        "--handoff",
        str(tmp_path / "handoff.jsonl"),
        "--manifest",
        str(tmp_path / "reference.jsonl"),
        "--timing-manifest",
        str(tmp_path / "timing.jsonl"),
        "--separation-artifact-root",
        str(tmp_path / "separation-artifacts"),
        "--stem-cache-root",
        str(tmp_path / "stem-cache"),
        "--output-dir",
        str(tmp_path / "output"),
        "--runtime-python",
        str(tmp_path / "idm-python"),
        "--model-lock",
        str(tmp_path / "model.json"),
        "--model-root",
        str(tmp_path / "model-root"),
        "--resume",
    ]


def test_run_idm_pilot_builds_request_compares_and_emits_canonical_summary(
    tmp_path: Path, monkeypatch
) -> None:
    import src.benchmark.idm_comparison as comparison_module
    import src.benchmark.idm_pilot_run as pilot_module
    import src.cli.benchmark as benchmark_module

    output_dir = tmp_path / "output"
    run_path = output_dir / "runs" / "idm-run" / "run.json"
    reports_path = run_path.parent / "reports"
    comparison_path = output_dir / "comparison"
    captured: list[object] = []

    def fake_run(request: object) -> pilot_module.IdmPilotRunOutcome:
        captured.append(request)
        return pilot_module.IdmPilotRunOutcome(
            overall_status="complete",
            exit_code=0,
            run_id="idm-run",
            run_path=run_path,
            reports_path=reports_path,
            success_count=12,
            failed_count=0,
            skipped_count=0,
            quarantined_count=0,
            native_failure_counts=(),
        )

    comparison_requests: list[object] = []

    def fake_compare(request: object) -> Path:
        comparison_requests.append(request)
        return comparison_path

    monkeypatch.setattr(benchmark_module, "_current_crux_commit", lambda: "a" * 40)
    monkeypatch.setattr(pilot_module, "run_idm_pilot", fake_run)
    monkeypatch.setattr(comparison_module, "compare_oaf_idm", fake_compare)

    result = CliRunner().invoke(main, _idm_pilot_cli_args(tmp_path), catch_exceptions=False)

    assert result.exit_code == 0
    assert len(captured) == 1
    request = captured[0]
    assert request.separation_handoff_path == tmp_path / "handoff.jsonl"
    assert request.reference_manifest_path == tmp_path / "reference.jsonl"
    assert request.timing_manifest_path == tmp_path / "timing.jsonl"
    assert request.separation_artifact_root == tmp_path / "separation-artifacts"
    assert request.stem_cache_root == tmp_path / "stem-cache"
    assert request.output_dir == output_dir
    assert request.runtime_python == tmp_path / "idm-python"
    assert request.model_lock_path == tmp_path / "model.json"
    assert request.model_root == tmp_path / "model-root"
    assert request.resume is True
    assert request.crux_commit == "a" * 40
    assert len(comparison_requests) == 1
    comparison_request = comparison_requests[0]
    assert comparison_request.run_path == run_path
    assert comparison_request.output_dir == comparison_path

    summary = json.loads(result.output)
    assert summary["status"] == "complete"
    assert summary["exit_code"] == 0
    assert summary["run_id"] == "idm-run"
    assert summary["comparison_path"] == str(comparison_path)
    assert summary["smoke"] is None
    assert result.output.encode() == benchmark_module.canonical_json_bytes(
        summary, trailing_newline=True
    )


def test_run_idm_pilot_routes_explicit_smoke_group_to_separate_request(
    tmp_path: Path, monkeypatch
) -> None:
    import src.benchmark.idm_comparison as comparison_module
    import src.benchmark.idm_pilot_run as pilot_module
    import src.cli.benchmark as benchmark_module

    output_dir = tmp_path / "output"
    run_path = output_dir / "runs" / "idm-run" / "run.json"
    pilot_requests: list[object] = []
    smoke_requests: list[object] = []

    def fake_run(request: object) -> pilot_module.IdmPilotRunOutcome:
        pilot_requests.append(request)
        return pilot_module.IdmPilotRunOutcome(
            overall_status="complete",
            exit_code=0,
            run_id="idm-run",
            run_path=run_path,
            reports_path=run_path.parent / "reports",
            success_count=12,
            failed_count=0,
            skipped_count=0,
            quarantined_count=0,
        )

    def fake_smoke(request: object) -> pilot_module.IdmFullMixSmokeOutcome:
        smoke_requests.append(request)
        return pilot_module.IdmFullMixSmokeOutcome(
            overall_status="complete",
            exit_code=0,
            run_id="smoke-run",
            run_path=output_dir / "full-mix-smoke" / "run.json",
            reports_path=output_dir / "full-mix-smoke" / "reports",
            success_count=5,
            failed_count=0,
            skipped_count=0,
            quarantined_count=0,
        )

    monkeypatch.setattr(benchmark_module, "_current_crux_commit", lambda: "a" * 40)
    monkeypatch.setattr(pilot_module, "run_idm_pilot", fake_run)
    monkeypatch.setattr(pilot_module, "run_idm_full_mix_smoke", fake_smoke)
    monkeypatch.setattr(
        comparison_module, "compare_oaf_idm", lambda _request: output_dir / "comparison"
    )

    args = _idm_pilot_cli_args(tmp_path) + [
        "--smoke-manifest",
        str(tmp_path / "smoke.json"),
        "--source-cache-dir",
        str(tmp_path / "source-cache"),
    ]
    result = CliRunner().invoke(main, args, catch_exceptions=False)

    assert result.exit_code == 0
    assert len(pilot_requests) == 1
    assert not hasattr(pilot_requests[0], "source_cache_dir")
    assert len(smoke_requests) == 1
    smoke_request = smoke_requests[0]
    assert smoke_request.separation_handoff_path == tmp_path / "handoff.jsonl"
    assert smoke_request.reference_manifest_path == tmp_path / "reference.jsonl"
    assert smoke_request.timing_manifest_path == tmp_path / "timing.jsonl"
    assert smoke_request.smoke_manifest_path == tmp_path / "smoke.json"
    assert smoke_request.source_cache_dir == tmp_path / "source-cache"
    assert smoke_request.output_dir == output_dir
    assert smoke_request.model_lock_path == tmp_path / "model.json"
    assert smoke_request.model_root == tmp_path / "model-root"
    assert smoke_request.runtime_python == tmp_path / "idm-python"
    assert smoke_request.crux_commit == "a" * 40
    assert json.loads(result.output)["smoke"]["run_id"] == "smoke-run"
