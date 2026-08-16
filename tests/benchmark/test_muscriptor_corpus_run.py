from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.benchmark.backend_identity import BackendDescriptor, canonical_json_bytes
from src.benchmark.backends.base import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.backends.muscriptor import MuscriptorBackendError
from src.benchmark.corpus_cache import ResolvedSourceAudio
from src.benchmark.muscriptor_corpus_run import (
    MUSCRIPTOR_ADAPTER_REVISION,
    MUSCRIPTOR_BACKEND_ERROR_POLICY,
    MUSCRIPTOR_CANONICALIZATION_REVISION,
    MUSCRIPTOR_CORPUS_RUN_SCHEMA,
    MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID,
    MUSCRIPTOR_INFERENCE_CONFIG_SCHEMA,
    MUSCRIPTOR_PREDICTION_MAP_ID,
    MuscriptorCorpusRunRequest,
    _expected_muscriptor_descriptor,
    _project_runtime,
    build_inference_config,
    build_run_id,
    inference_config_sha256,
    parse_muscriptor_corpus_run,
    render_muscriptor_corpus_run,
    run_muscriptor_corpus,
)
from src.benchmark.muscriptor_model import (
    MUSCRIPTOR_RELEASE_COMMIT,
    MuscriptorModelLock,
    model_lock_payload,
)
from src.benchmark.reference_set import ReferenceMappingResult, map_reference_events
from src.benchmark.reference_set_manifest import (
    LoadedReferenceSetManifest,
    LoadedReferenceSetRow,
    ReferenceSetRowView,
)
from src.benchmark.reference_timing import NativeReferenceEvent
from src.benchmark.reference_timing_manifest import LoadedReferenceTimingManifest

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
CHECKPOINT_REVISION = "d" * 40
COMMIT = "e" * 40
WEIGHTS = b"fake safetensors"
CONFIG = b'{"model":"fake"}'


def _lock() -> MuscriptorModelLock:
    checkpoint_sha = hashlib.sha256(WEIGHTS).hexdigest()
    config_sha = hashlib.sha256(CONFIG).hexdigest()
    return MuscriptorModelLock(
        package_name="muscriptor",
        package_version="0.3.0",
        upstream_source_commit=MUSCRIPTOR_RELEASE_COMMIT,
        code_license="MIT",
        weight_license="CC BY-NC 4.0",
        checkpoint_variant="medium",
        checkpoint_repo_id="MuScriptor/muscriptor-medium",
        checkpoint_revision=CHECKPOINT_REVISION,
        checkpoint_filename="model.safetensors",
        checkpoint_sha256=checkpoint_sha,
        checkpoint_byte_length=len(WEIGHTS),
        checkpoint_config_filename="config.json",
        checkpoint_config_sha256=config_sha,
        checkpoint_config_byte_length=len(CONFIG),
        model_id=f"muscriptor-medium-{CHECKPOINT_REVISION[:12]}-{checkpoint_sha[:12]}",
        device="cpu",
        dtype="float32",
        input_sample_rate_hz=16000,
        chunk_duration_sec=5.0,
        use_sampling=False,
        temperature=1.0,
        cfg_coef=1.0,
        instruments=("drums",),
        batch_size=1,
        no_eos_is_ok=True,
        beam_size=1,
        prelude_forcing=True,
        native_output_space_id="muscriptor-drums-midi128-v1",
        native_metadata_schema_id="muscriptor-note-start-metadata-v1",
        training_data_map_id="muscriptor-training-data-v0.3.0",
    )


def _write_lock(tmp_path: Path) -> tuple[Path, Path, MuscriptorModelLock]:
    lock = _lock()
    lock_path = tmp_path / "model.json"
    lock_path.write_bytes(canonical_json_bytes(model_lock_payload(lock), trailing_newline=True))
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / lock.checkpoint_filename).write_bytes(WEIGHTS)
    (checkpoint_dir / lock.checkpoint_config_filename).write_bytes(CONFIG)
    return lock_path, checkpoint_dir, lock


def _manifests() -> tuple[LoadedReferenceSetManifest, LoadedReferenceTimingManifest]:
    rows = tuple(
        LoadedReferenceSetRow(
            source_row={
                "selected_chart_key": f"{simfile_id}/chart.dtx",
                "selected_chart_content_hash": SHA_A,
                "source_audio_key": f"{simfile_id}/audio.wav",
                "source_audio_content_hash": SHA_A,
                "source_endpoint_sha256": SHA_A,
                "source_bucket": "simfile-dtx",
            },
            view=ReferenceSetRowView(
                simfile_id=simfile_id,
                eligibility_status="eligible",
                eligibility_reason_codes=(),
                eligibility_warnings=(),
                mapped_event_count=1,
                common_scored_event_count=1,
                ignored_event_count=0,
                unmapped_event_count=0,
                duplicate_common_event_count=0,
            ),
        )
        for simfile_id in (10, 20, 30)
    )
    return (
        LoadedReferenceSetManifest(
            manifest_sha256=SHA_A,
            corpus_version="reference-v1",
            source_reference_timing_manifest_sha256=SHA_B,
            source_reference_timing_version="timing-v1",
            rows=rows,
        ),
        LoadedReferenceTimingManifest(
            manifest_sha256=SHA_B,
            corpus_version="timing-v1",
            rows=(),
        ),
    )


def _mapping(simfile_id: int) -> ReferenceMappingResult:
    return map_reference_events(
        (
            NativeReferenceEvent(
                simfile_id=simfile_id,
                selected_chart_key=f"{simfile_id}/chart.dtx",
                selected_chart_content_hash=SHA_A,
                source_audio_key=f"{simfile_id}/audio.wav",
                source_audio_content_hash=SHA_A,
                source_order=0,
                measure=1,
                position=0.0,
                lane_id="13",
                note_id="kick-0",
                chart_time_sec=0.25,
                audio_time_sec=0.25,
            ),
        )
    )


def _request(tmp_path: Path, *, resume: bool = False) -> MuscriptorCorpusRunRequest:
    return MuscriptorCorpusRunRequest(
        reference_manifest_path=tmp_path / "reference.jsonl",
        timing_manifest_path=tmp_path / "timing.jsonl",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        resume=resume,
        crux_commit=COMMIT,
    )


def _install_seams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    input_hash: str = SHA_B,
) -> tuple[MuscriptorModelLock, BackendDescriptor, list[tuple[int, bool]], list[int]]:
    import src.benchmark.muscriptor_corpus_run as run_module

    lock_path, checkpoint_dir, lock = _write_lock(tmp_path)
    reference_manifest, timing_manifest = _manifests()
    descriptor = _expected_muscriptor_descriptor(lock)
    monkeypatch.setattr(run_module, "_model_lock_path", lambda: lock_path)
    monkeypatch.setattr(run_module, "_checkpoint_dir", lambda: checkpoint_dir)
    monkeypatch.setattr(run_module, "load_reference_set_manifest", lambda _: reference_manifest)
    monkeypatch.setattr(run_module, "load_reference_timing_manifest", lambda _: timing_manifest)
    monkeypatch.setattr(
        run_module,
        "preflight_reference_mappings",
        lambda *_args, **_kwargs: {simfile_id: _mapping(simfile_id) for simfile_id in (10, 20, 30)},
    )
    resolve_calls: list[tuple[int, bool]] = []

    def resolve(source, *_args, **kwargs):
        simfile_id = int(source["source_audio_key"].split("/", 1)[0])
        resolve_calls.append((simfile_id, kwargs["load_body"]))
        return ResolvedSourceAudio(
            path=tmp_path / f"{simfile_id}.wav",
            source_audio_id=source["source_audio_key"],
            source_audio_sha256=SHA_A,
            duration_sec=1.0,
            content=b"source bytes" if kwargs["load_body"] else None,
        )

    monkeypatch.setattr(run_module, "resolve_source_audio", resolve)
    materialize_calls: list[int] = []

    def materialize(source, output_path, *, input_root, input_view_id, max_input_audio_frames):
        del input_root, input_view_id, max_input_audio_frames
        simfile_id = int(source.source_audio_id.split("/", 1)[0])
        materialize_calls.append(simfile_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"canonical wav")
        return CanonicalAudio(
            path=output_path,
            source_audio_id=source.source_audio_id,
            source_audio_sha256=source.source_audio_sha256,
            input_view_id=MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID,
            input_audio_sha256=input_hash,
            byte_length=46,
            sample_rate=44100,
            channel_count=1,
            sample_width_bytes=2,
            audio_frame_count=1,
        )

    monkeypatch.setattr(run_module, "materialize_full_mix_audio", materialize)
    return lock, descriptor, resolve_calls, materialize_calls


def _prediction(audio: CanonicalAudio, descriptor: BackendDescriptor) -> NativePrediction:
    return NativePrediction(
        audio=audio,
        descriptor=descriptor,
        events=(
            NativeEvent(
                time_sec=0.25,
                native_class_id="drums:midi_36",
                model_output_bin=None,
                native_midi_note=36,
                native_metadata={"instrument_group": "drums"},
                confidence=None,
                velocity_midi=None,
            ),
        ),
    )


def test_constants_request_and_identity_bind_all_frozen_inputs() -> None:
    assert MUSCRIPTOR_CORPUS_RUN_SCHEMA == "crux.muscriptor-corpus-run/v1"
    assert MUSCRIPTOR_INFERENCE_CONFIG_SCHEMA == "crux.muscriptor-inference-config/v1"
    assert MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID == "crux.oaf-full-mix-mono44k1-pcm16/v1"
    assert MUSCRIPTOR_ADAPTER_REVISION == "crux.muscriptor-adapter/v1"
    assert MUSCRIPTOR_CANONICALIZATION_REVISION == ("librosa-soxr-hq-mono44k1-soundfile-pcm16/v1")

    request = MuscriptorCorpusRunRequest(
        reference_manifest_path=Path("reference.jsonl"),
        timing_manifest_path=Path("timing.jsonl"),
        cache_dir=Path("cache"),
        output_dir=Path("output"),
        include_simfile_ids=(30, 10, 10),
        exclude_simfile_ids=(40,),
        crux_commit=COMMIT,
    )
    assert request.include_simfile_ids == (10, 30)
    assert request.exclude_simfile_ids == (40,)

    common = dict(
        reference_manifest_sha256=SHA_A,
        reference_manifest_version="reference-v1",
        reference_timing_manifest_sha256=SHA_B,
        reference_timing_version="timing-v1",
        model_lock_sha256=SHA_C,
        checkpoint_revision="d" * 40,
        checkpoint_sha256=SHA_A,
        backend_descriptor_sha256=SHA_B,
        inference_config_sha256=SHA_C,
        crux_commit=COMMIT,
        include_simfile_ids=request.include_simfile_ids,
        exclude_simfile_ids=request.exclude_simfile_ids,
    )
    first = build_run_id(**common)
    assert first.startswith("muscriptor-")
    assert first == build_run_id(**common)
    for field, value in (
        ("reference_manifest_sha256", SHA_B),
        ("reference_manifest_version", "reference-v2"),
        ("reference_timing_manifest_sha256", SHA_C),
        ("reference_timing_version", "timing-v2"),
        ("model_lock_sha256", SHA_A),
        ("checkpoint_revision", "f" * 40),
        ("checkpoint_sha256", SHA_B),
        ("backend_descriptor_sha256", SHA_C),
        ("inference_config_sha256", SHA_A),
        ("crux_commit", "1" * 40),
        ("include_simfile_ids", (10,)),
        ("exclude_simfile_ids", (41,)),
    ):
        assert first != build_run_id(**{**common, field: value}), field


def test_inference_config_contains_frozen_lock_semantics_and_hashes() -> None:
    lock = _lock()
    descriptor = _expected_muscriptor_descriptor(lock)
    payload = build_inference_config(lock, descriptor, SHA_A)
    assert set(payload) == {
        "schema",
        "backend_descriptor_sha256",
        "model_lock_sha256",
        "checkpoint_revision",
        "checkpoint_sha256",
        "adapter_revision",
        "prediction_map_version",
        "input_view_id",
        "canonicalization_revision",
        "device",
        "dtype",
        "input_sample_rate_hz",
        "chunk_duration_sec",
        "use_sampling",
        "temperature",
        "cfg_coef",
        "instruments",
        "batch_size",
        "no_eos_is_ok",
        "beam_size",
        "prelude_forcing",
    }
    assert payload["schema"] == MUSCRIPTOR_INFERENCE_CONFIG_SCHEMA
    assert payload["prediction_map_version"] == MUSCRIPTOR_PREDICTION_MAP_ID
    assert payload["input_view_id"] == MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID
    assert payload["checkpoint_revision"] == CHECKPOINT_REVISION
    assert inference_config_sha256(payload) == inference_config_sha256(dict(payload))
    assert inference_config_sha256({**payload, "dtype": "float16"}) != inference_config_sha256(
        payload
    )


def test_snapshot_sorts_unique_items_and_reconciles_complete_counts() -> None:
    snapshot = {
        "schema": MUSCRIPTOR_CORPUS_RUN_SCHEMA,
        "run_id": "muscriptor-" + SHA_A[:16],
        "exclude_simfile_ids": [20],
        "overall_status": "complete",
        "success_count": 1,
        "failed_count": 0,
        "skipped_count": 1,
        "quarantined_count": 0,
        "completed_at": "2026-08-14T00:00:00+00:00",
        "items": [
            {"simfile_id": 20, "execution_disposition": "skipped"},
            {"simfile_id": 10, "execution_disposition": "inferred", "rtf": 0.5},
        ],
    }
    parsed = parse_muscriptor_corpus_run(render_muscriptor_corpus_run(snapshot))
    assert [row["simfile_id"] for row in parsed["items"]] == [10, 20]
    with pytest.raises(ValueError, match="count"):
        render_muscriptor_corpus_run({**snapshot, "success_count": 2})


def test_project_runtime_projects_only_when_duration_coverage_is_complete() -> None:
    runtime = _project_runtime(
        (
            {
                "simfile_id": 10,
                "execution_disposition": "inferred",
                "wall_time_sec": 2.0,
                "source_duration_sec": 4.0,
            },
        ),
        eligible_audio_durations=(4.0, None),
    )
    assert runtime["aggregate_rtf"] == pytest.approx(0.5)
    assert runtime["projected_full_wall_time_sec"] is None
    assert runtime["eligible_audio_duration_coverage_count"] == 1


def test_run_resolves_sources_in_two_passes_reuses_one_backend_and_persists_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, descriptor, resolve_calls, materialize_calls = _install_seams(monkeypatch, tmp_path)
    factory_calls: list[dict[str, object]] = []
    transcribe_calls: list[int] = []
    close_calls: list[bool] = []

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            transcribe_calls.append(int(audio.source_audio_id.split("/", 1)[0]))
            return _prediction(audio, descriptor)

        def close(self) -> None:
            close_calls.append(True)

    def factory(**kwargs: object) -> HealthyBackend:
        factory_calls.append(kwargs)
        return HealthyBackend()

    outcome = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=factory,
        perf_counter=iter((0.0, 0.25, 1.0, 1.5, 2.0, 2.25)).__next__,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert outcome.overall_status == "complete"
    assert outcome.exit_code == 0
    assert transcribe_calls == [10, 20, 30]
    assert materialize_calls == [10, 20, 30]
    assert resolve_calls == [
        (10, False),
        (20, False),
        (30, False),
        (10, True),
        (20, True),
        (30, True),
    ]
    assert close_calls == [True]
    assert factory_calls == [
        {
            "checkpoint_dir": tmp_path / "checkpoint",
            "model_lock_path": tmp_path / "model.json",
            "device": lock.device,
            "dtype": lock.dtype,
            "descriptor": descriptor,
        }
    ]
    assert outcome.run_path is not None
    snapshot = parse_muscriptor_corpus_run(outcome.run_path.read_bytes())
    assert snapshot["input_view_id"] == MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID
    assert snapshot["device"] == "cpu"
    assert snapshot["dtype"] == "float32"
    assert snapshot["peak_process_rss_bytes"] is not None
    assert snapshot["device_peak_memory_bytes"] is None
    for row in snapshot["items"]:
        assert {
            "source_audio_id",
            "source_audio_sha256",
            "source_duration_sec",
            "prediction_path",
            "prediction_artifact_sha256",
            "wall_time_sec",
            "rtf",
        } <= set(row)
        assert "input_audio_sha256" in row


def test_item_local_inference_failure_continues_to_later_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)
    calls: list[int] = []

    class Backend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            simfile_id = int(audio.source_audio_id.split("/", 1)[0])
            calls.append(simfile_id)
            if simfile_id == 20:
                raise MuscriptorBackendError("song failed", code="inference_failed")
            return _prediction(audio, descriptor)

        def close(self) -> None:
            return None

    outcome = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: Backend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.exit_code == 1
    assert calls == [10, 20, 30]
    assert outcome.run_path is not None
    rows = {
        row["simfile_id"]: row
        for row in parse_muscriptor_corpus_run(outcome.run_path.read_bytes())["items"]
    }
    assert rows[20]["runner_failure_code"] == "inference_failed"
    assert rows[30]["execution_disposition"] == "inferred"


def test_poisoned_model_state_stops_later_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)
    calls: list[int] = []

    class Backend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            simfile_id = int(audio.source_audio_id.split("/", 1)[0])
            calls.append(simfile_id)
            if simfile_id == 20:
                raise MuscriptorBackendError("state poisoned", code="backend_closed")
            return _prediction(audio, descriptor)

        def close(self) -> None:
            return None

    outcome = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: Backend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.exit_code == 1
    assert calls == [10, 20]
    assert outcome.run_path is not None
    rows = {
        row["simfile_id"]: row
        for row in parse_muscriptor_corpus_run(outcome.run_path.read_bytes())["items"]
    }
    assert rows[20]["runner_failure_code"] == "worker_protocol_failed"
    assert rows[30]["runner_failure_code"] == "worker_protocol_failed"


def test_fatal_scope_preflight_does_not_construct_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_seams(monkeypatch, tmp_path)
    calls: list[bool] = []

    def factory(**_: object):
        calls.append(True)
        raise AssertionError("fatal preflight must precede backend construction")

    outcome = run_muscriptor_corpus(
        replace(_request(tmp_path), include_simfile_ids=(999,)),
        backend_factory=factory,
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2
    assert calls == []


def test_resume_rejects_edited_input_evidence_without_constructing_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, descriptor, _, _ = _install_seams(monkeypatch, tmp_path, input_hash=SHA_B)

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return _prediction(audio, descriptor)

        def close(self) -> None:
            return None

    first = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert first.exit_code == 0
    assert first.run_path is not None
    original = {
        row["simfile_id"]: (
            tmp_path / "output" / row["prediction_path"],
            (tmp_path / "output" / row["prediction_path"]).read_bytes(),
        )
        for row in parse_muscriptor_corpus_run(first.run_path.read_bytes())["items"]
    }

    # Replace the materializer with a valid domain value carrying a changed input hash.
    import src.benchmark.muscriptor_corpus_run as run_module

    def changed_materialize(source, output_path, **kwargs):
        del kwargs
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"edited canonical wav")
        return CanonicalAudio(
            path=output_path,
            source_audio_id=source.source_audio_id,
            source_audio_sha256=source.source_audio_sha256,
            input_view_id=MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID,
            input_audio_sha256=SHA_C,
            byte_length=46,
            sample_rate=44100,
            channel_count=1,
            sample_width_bytes=2,
            audio_frame_count=1,
        )

    monkeypatch.setattr(run_module, "materialize_full_mix_audio", changed_materialize)
    factory_calls: list[bool] = []

    def should_not_construct(**_: object):
        factory_calls.append(True)
        raise AssertionError("edited resume evidence must not infer")

    resumed = run_muscriptor_corpus(
        _request(tmp_path, resume=True),
        backend_factory=should_not_construct,
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert resumed.exit_code == 1
    assert factory_calls == []
    assert resumed.run_path is not None
    rows = parse_muscriptor_corpus_run(resumed.run_path.read_bytes())["items"]
    assert all(row["runner_failure_code"] == "prediction_artifact_invalid" for row in rows)
    for path, content in original.values():
        assert path.read_bytes() == content


def test_backend_error_policy_keeps_item_local_and_poison_paths_closed() -> None:
    assert MUSCRIPTOR_BACKEND_ERROR_POLICY["inference_failed"] == (
        "inference_failed",
        "item_local",
    )
    assert MUSCRIPTOR_BACKEND_ERROR_POLICY["backend_closed"] == (
        "worker_protocol_failed",
        "poison",
    )
