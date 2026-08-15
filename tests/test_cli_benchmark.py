from __future__ import annotations

import json
import struct
from dataclasses import dataclass, replace
from pathlib import Path

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


def test_reviewed_subset_commands_declare_exact_options() -> None:
    expected_options = {
        "prepare-reviewed-subset": (
            "--manifest",
            "--timing-manifest",
            "--output-file",
            "--prior-ledger",
        ),
        "finalize-reviewed-subset": (
            "--manifest",
            "--timing-manifest",
            "--review-file",
            "--output-dir",
            "--prior-ledger",
        ),
        "score-oaf-reviewed-subset": (
            "--run",
            "--manifest",
            "--timing-manifest",
            "--subset-manifest",
            "--output-dir",
        ),
    }
    for command, options in expected_options.items():
        result = CliRunner().invoke(main, ["benchmark", command, "--help"])
        assert result.exit_code == 0
        for option in options:
            assert option in result.output
        for selector in ("--seed", "--count", "--threshold", "--model", "--backend"):
            assert selector not in result.output


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
