"""Targeted branch-coverage tests for ``src.benchmark.oaf_corpus_run``.

These tests exercise the validation, type-guard, and defensive error paths
that are not reached by the happy-path acceptance suite.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from runtime.oaf_tf1.model import load_model_config
from src.benchmark.backend_identity import (
    BackendDescriptor,
    StrictJsonError,
    canonical_json_bytes,
)
from src.benchmark.backends.base import CanonicalAudio, NativePrediction
from src.benchmark.cohort_scoring import CohortIdentity
from src.benchmark.corpus_cache import (
    CacheIndexStore,
    ResolvedSourceAudio,
    _remote_from_source_mapping,
    _source_audio_parts,
    resolve_source_audio,
)
from src.benchmark.corpus_manifest import ManifestRowView
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.oaf_corpus_run import (
    OAF_CORPUS_RUN_SCHEMA,
    OAF_FULL_MIX_INPUT_VIEW_ID,
    OAF_INFERENCE_CONFIG_SCHEMA,
    OAF_PREDICTION_MAP_ID,
    OafCorpusRunOutcome,
    OafCorpusRunRequest,
    _bounded_close_error,
    _bounded_error,
    _cohort_identity_from_snapshot,
    _cohort_item_from_run_row,
    _expected_oaf_descriptor,
    _fatal_outcome,
    _finalize_scoring_and_outcome,
    _item_counts,
    _model_lock_path,
    _normalize_scope,
    _normalize_snapshot_value,
    _prediction_artifact_matches,
    _prediction_artifact_matches_run_row,
    _prediction_relative_path,
    _project_runtime,
    _read_existing_prediction,
    _remove_temporary_input,
    _source_failure_code,
    _timestamp,
    _utc_now,
    _validate_scope,
    _validate_snapshot,
    build_inference_config,
    classify_oaf_backend_error,
    compute_model_lock_sha256,
    inference_config_sha256,
    parse_oaf_corpus_run,
    render_oaf_corpus_run,
    write_oaf_corpus_run,
)
from src.benchmark.prediction_artifact import (
    MappedPrediction,
    PredictionArtifact,
    prediction_path,
    read_prediction_artifact,
    render_prediction_artifact,
)
from src.benchmark.r2_corpus_models import (
    ProvenanceRecord,
    RemoteObject,
    SimfileInventory,
)
from src.benchmark.reference_set import map_reference_events
from src.benchmark.reference_set_manifest import (
    LoadedReferenceSetManifest,
    LoadedReferenceSetRow,
    ReferenceSetRowView,
    preflight_reference_mappings,
)
from src.benchmark.reference_timing import NativeReferenceEvent
from src.benchmark.reference_timing_manifest import (
    LoadedReferenceTimingManifest,
    LoadedReferenceTimingRow,
    ReferenceTimingRowView,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
COMMIT = "d" * 40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _descriptor() -> BackendDescriptor:
    return _expected_oaf_descriptor(load_model_config())


def _cohort_identity() -> CohortIdentity:
    return CohortIdentity(
        cohort_id="oaf-run",
        reference_manifest_sha256=SHA_A,
        reference_timing_version="sha256:" + SHA_B,
        taxonomy_version="crux.drum-taxonomy/v1",
        lane_map_version="crux.dtx-lane-map/v1",
        backend_id="crux.oaf",
        model_id="model",
        model_lock_sha256=SHA_C,
        backend_descriptor_sha256=SHA_A,
        prediction_map_version=OAF_PREDICTION_MAP_ID,
        input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
    )


def _reference_mapping(simfile_id: int = 10) -> object:
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
                chart_time_sec=0.5,
                audio_time_sec=0.5,
            ),
        )
    )


def _canonical_audio(source_audio_id: str = "10/audio.wav") -> CanonicalAudio:
    return CanonicalAudio(
        path=Path("/tmp/x.wav"),
        source_audio_id=source_audio_id,
        source_audio_sha256=SHA_A,
        input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
        input_audio_sha256=SHA_B,
        byte_length=88244,
        sample_rate=44100,
        channel_count=1,
        sample_width_bytes=2,
        audio_frame_count=44100,
    )


def _resolved_source(
    *, source_audio_id: str = "10/audio.wav", source_audio_sha256: str = SHA_A
) -> ResolvedSourceAudio:
    return ResolvedSourceAudio(
        path=Path("/tmp/s.wav"),
        source_audio_id=source_audio_id,
        source_audio_sha256=source_audio_sha256,
        duration_sec=1.0,
    )


def _native_prediction(audio: CanonicalAudio, descriptor: BackendDescriptor) -> NativePrediction:
    return NativePrediction(audio=audio, descriptor=descriptor, events=())


def _mapped_prediction(
    audio: CanonicalAudio | None = None,
    descriptor: BackendDescriptor | None = None,
) -> MappedPrediction:
    audio = audio or _canonical_audio()
    descriptor = descriptor or _descriptor()
    mapped, _ = map_oaf_prediction(_native_prediction(audio, descriptor))
    return mapped


def _prediction_artifact(
    audio: CanonicalAudio | None = None,
    descriptor: BackendDescriptor | None = None,
) -> PredictionArtifact:
    return read_prediction_artifact(
        render_prediction_artifact(_mapped_prediction(audio, descriptor))
    )


def _source_wav_bytes(*, sample_rate: int = 44100, frames: int = 44100) -> bytes:
    output = BytesIO()
    sf.write(
        output,
        np.zeros((frames, 1), dtype=np.float32),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return output.getvalue()


def _source_wav_bytes_with_value(
    value: float, *, sample_rate: int = 44100, frames: int = 44100
) -> bytes:
    """A valid WAV filled with one sample value (same byte length as zeros)."""
    output = BytesIO()
    sf.write(
        output,
        np.full((frames, 1), value, dtype=np.float32),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return output.getvalue()


def _cache_body(cache_dir: Path, content: bytes) -> tuple[str, Path]:
    from hashlib import sha256

    digest = sha256(content).hexdigest()
    path = cache_dir / "sha256" / digest[:2] / digest
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return digest, path


def _source_remote(
    *,
    key: str = "42/bgm.wav",
    content: bytes | None = None,
    cache_status: str = "verified",
    digest: str | None = None,
    cache_path: str | None = None,
) -> RemoteObject:
    content = content or _source_wav_bytes()
    return RemoteObject(
        key=key,
        size=len(content),
        etag='"etag-42"',
        etag_is_weak=False,
        last_modified=datetime(2026, 8, 14, tzinfo=timezone.utc),
        content_type="audio/wav",
        cache_status=cache_status,  # type: ignore[arg-type]
        sha256=digest,
        cache_path=cache_path,
    )


# ---------------------------------------------------------------------------
# OafCorpusRunRequest __post_init__ validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["reference_manifest_path", "timing_manifest_path", "cache_dir", "output_dir"]
)
def test_request_rejects_non_path_fields(field: str) -> None:
    kwargs = dict(
        reference_manifest_path=Path("r.jsonl"),
        timing_manifest_path=Path("t.jsonl"),
        cache_dir=Path("cache"),
        output_dir=Path("output"),
    )
    kwargs[field] = "not-a-path"
    with pytest.raises(TypeError, match=f"{field} must be a Path"):
        OafCorpusRunRequest(**kwargs)


def test_request_rejects_non_bool_resume() -> None:
    with pytest.raises(TypeError, match="resume must be a bool"):
        OafCorpusRunRequest(
            reference_manifest_path=Path("r.jsonl"),
            timing_manifest_path=Path("t.jsonl"),
            cache_dir=Path("cache"),
            output_dir=Path("output"),
            resume="yes",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# OafCorpusRunOutcome __post_init__ validation
# ---------------------------------------------------------------------------


def _valid_outcome_kwargs() -> dict[str, object]:
    return dict(
        overall_status="complete",
        exit_code=0,
        run_id="oaf-" + SHA_A[:16],
        run_path=Path("run.json"),
        reports_path=None,
        success_count=1,
        failed_count=0,
        skipped_count=0,
        quarantined_count=0,
        aggregate_rtf=1.25,
        projected_full_wall_time_sec=3600.0,
    )


def test_outcome_rejects_invalid_status() -> None:
    kwargs = _valid_outcome_kwargs()
    kwargs["overall_status"] = "bogus"
    with pytest.raises(ValueError, match="overall_status"):
        OafCorpusRunOutcome(**kwargs)


@pytest.mark.parametrize("exit_code", [3, -1, True])
def test_outcome_rejects_invalid_exit_code(exit_code: object) -> None:
    kwargs = _valid_outcome_kwargs()
    kwargs["exit_code"] = exit_code
    with pytest.raises(ValueError, match="exit_code"):
        OafCorpusRunOutcome(**kwargs)


def test_outcome_rejects_empty_run_id() -> None:
    kwargs = _valid_outcome_kwargs()
    kwargs["run_id"] = ""
    with pytest.raises(ValueError, match="run_id"):
        OafCorpusRunOutcome(**kwargs)


@pytest.mark.parametrize("field", ["run_path", "reports_path"])
def test_outcome_rejects_non_path_optional_fields(field: str) -> None:
    kwargs = _valid_outcome_kwargs()
    kwargs[field] = "not-a-path"
    with pytest.raises(TypeError, match=f"{field} must be a Path or None"):
        OafCorpusRunOutcome(**kwargs)


@pytest.mark.parametrize(
    "field", ["success_count", "failed_count", "skipped_count", "quarantined_count"]
)
def test_outcome_rejects_negative_or_bool_counts(field: str) -> None:
    kwargs = _valid_outcome_kwargs()
    kwargs[field] = -1
    with pytest.raises(ValueError, match=field):
        OafCorpusRunOutcome(**kwargs)
    kwargs[field] = True
    with pytest.raises(ValueError, match=field):
        OafCorpusRunOutcome(**kwargs)


@pytest.mark.parametrize("field", ["aggregate_rtf", "projected_full_wall_time_sec"])
def test_outcome_rejects_non_float_runtime_fields(field: str) -> None:
    kwargs = _valid_outcome_kwargs()
    kwargs[field] = "not-a-float"
    with pytest.raises(TypeError, match=f"{field} must be a float or None"):
        OafCorpusRunOutcome(**kwargs)


@pytest.mark.parametrize("field", ["aggregate_rtf", "projected_full_wall_time_sec"])
def test_outcome_rejects_non_finite_runtime_fields(field: str) -> None:
    kwargs = _valid_outcome_kwargs()
    kwargs[field] = float("inf")
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        OafCorpusRunOutcome(**kwargs)


# ---------------------------------------------------------------------------
# compute_model_lock_sha256 / build_inference_config / inference_config_sha256
# ---------------------------------------------------------------------------


def test_compute_model_lock_sha256_rejects_non_path() -> None:
    with pytest.raises(TypeError, match="model lock path must be a Path"):
        compute_model_lock_sha256("not-a-path")  # type: ignore[arg-type]


def test_build_inference_config_rejects_non_model_config() -> None:
    with pytest.raises(TypeError, match="config must be OafModelConfig"):
        build_inference_config("not-config", _descriptor(), SHA_B)  # type: ignore[arg-type]


def test_build_inference_config_rejects_non_descriptor() -> None:
    with pytest.raises(TypeError, match="descriptor must be BackendDescriptor"):
        build_inference_config(load_model_config(), "not-descriptor", SHA_B)  # type: ignore[arg-type]


def test_inference_config_sha256_rejects_wrong_key_set() -> None:
    with pytest.raises(StrictJsonError, match="exact key set"):
        inference_config_sha256({"schema": OAF_INFERENCE_CONFIG_SCHEMA})


def test_inference_config_sha256_rejects_non_string_values() -> None:
    payload = build_inference_config(load_model_config(), _descriptor(), SHA_B)
    payload["adapter_revision"] = 123  # type: ignore[assignment]
    with pytest.raises(StrictJsonError, match="nonempty string"):
        inference_config_sha256(payload)


def test_inference_config_sha256_rejects_wrong_schema() -> None:
    payload = build_inference_config(load_model_config(), _descriptor(), SHA_B)
    payload["schema"] = "wrong/v1"
    with pytest.raises(StrictJsonError, match="schema must be"):
        inference_config_sha256(payload)


# ---------------------------------------------------------------------------
# _normalize_scope / _validate_scope
# ---------------------------------------------------------------------------


def test_normalize_scope_rejects_string_iterable() -> None:
    with pytest.raises(ValueError, match="integer simfile IDs"):
        _normalize_scope("10", ())  # type: ignore[arg-type]


def test_normalize_scope_rejects_non_iterable() -> None:
    with pytest.raises(ValueError, match="integer simfile IDs"):
        _normalize_scope(10, ())  # type: ignore[arg-type]


def test_normalize_scope_rejects_non_positive_ids() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _normalize_scope((0,), ())


def test_normalize_scope_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        _normalize_scope((10,), (10,))


def test_validate_scope_rejects_non_iterable_loaded_ids() -> None:
    with pytest.raises(ValueError, match="loaded manifest IDs must be iterable"):
        _validate_scope((), (), 10)  # type: ignore[arg-type]


def test_validate_scope_rejects_non_positive_loaded_ids() -> None:
    with pytest.raises(ValueError, match="loaded manifest IDs must be positive"):
        _validate_scope((), (), {0})


# ---------------------------------------------------------------------------
# prediction_path
# ---------------------------------------------------------------------------


def test_prediction_path_rejects_non_path_output_dir() -> None:
    with pytest.raises(TypeError, match="output_dir must be a Path"):
        prediction_path(
            "output",
            simfile_id=10,
            source_audio_sha256=SHA_A,
            backend_descriptor_sha256=SHA_B,
            inference_config_sha256=SHA_C,
        )  # type: ignore[arg-type]


def test_prediction_path_rejects_non_positive_simfile_id() -> None:
    with pytest.raises(ValueError, match="simfile_id must be a positive integer"):
        prediction_path(
            Path("output"),
            simfile_id=0,
            source_audio_sha256=SHA_A,
            backend_descriptor_sha256=SHA_B,
            inference_config_sha256=SHA_C,
        )


# ---------------------------------------------------------------------------
# _remote_from_source_mapping
# ---------------------------------------------------------------------------


def test_remote_from_source_mapping_rejects_non_list_objects() -> None:
    with pytest.raises(ValueError, match="object inventory"):
        _remote_from_source_mapping({"objects": "not-a-list"}, source_audio_key="42/bgm.wav")


def test_remote_from_source_mapping_rejects_missing_key() -> None:
    with pytest.raises(ValueError, match="absent from the source inventory"):
        _remote_from_source_mapping(
            {"objects": [{"key": "other.wav"}]},
            source_audio_key="42/bgm.wav",
        )


def test_remote_from_source_mapping_rejects_invalid_object() -> None:
    with pytest.raises(ValueError, match="invalid audio object"):
        _remote_from_source_mapping(
            {"objects": [{"key": "42/bgm.wav", "size": "bad"}]},
            source_audio_key="42/bgm.wav",
        )


def test_remote_from_source_mapping_builds_remote() -> None:
    remote = _remote_from_source_mapping(
        {
            "objects": [
                {
                    "key": "42/bgm.wav",
                    "size": 100,
                    "etag": '"etag"',
                    "etag_is_weak": False,
                    "last_modified": "2026-08-14T00:00:00Z",
                    "content_type": "audio/wav",
                    "cache_status": "verified",
                    "sha256": SHA_A,
                    "cache_path": "sha256/aa/" + SHA_A,
                }
            ]
        },
        source_audio_key="42/bgm.wav",
    )
    assert remote.key == "42/bgm.wav"
    assert remote.sha256 == SHA_A


# ---------------------------------------------------------------------------
# _source_audio_parts
# ---------------------------------------------------------------------------


def _manifest_row_view(key: str = "42/bgm.wav") -> ManifestRowView:
    inv = SimfileInventory(
        simfile_id=42,
        object_prefix="42/",
        objects=(_source_remote(key=key),),
        sync_status="complete",
    )
    return ManifestRowView(
        inventory=inv,
        provenance=ProvenanceRecord(),
        corpus_version="sha256:" + SHA_C,
        cache_profile="default",
        source_endpoint_sha256=SHA_A,
        source_bucket="simfile-dtx",
        source_discovery_method="r2_list_objects_v2",
    )


def test_source_audio_parts_from_manifest_row_view() -> None:
    view = _manifest_row_view()
    remote, endpoint, bucket, expected = _source_audio_parts(
        view,
        source_audio_key="42/bgm.wav",
        source_audio_content_hash=SHA_A,
        source_endpoint_sha256=None,
        source_bucket=None,
    )
    assert remote.key == "42/bgm.wav"
    assert endpoint == SHA_A
    assert bucket == "simfile-dtx"
    assert expected == SHA_A


def test_source_audio_parts_manifest_row_view_missing_key() -> None:
    view = _manifest_row_view()
    with pytest.raises(ValueError, match="absent from the source inventory"):
        _source_audio_parts(
            view,
            source_audio_key="missing.wav",
            source_audio_content_hash=SHA_A,
            source_endpoint_sha256=None,
            source_bucket=None,
        )


def test_source_audio_parts_manifest_row_view_requires_key() -> None:
    view = _manifest_row_view()
    with pytest.raises(ValueError, match="source_audio_key is required"):
        _source_audio_parts(
            view,
            source_audio_key=None,
            source_audio_content_hash=SHA_A,
            source_endpoint_sha256=None,
            source_bucket=None,
        )


def test_source_audio_parts_from_mapping() -> None:
    source = {
        "source_audio_key": "42/bgm.wav",
        "source_endpoint_sha256": SHA_A,
        "source_bucket": "simfile-dtx",
        "objects": [
            {
                "key": "42/bgm.wav",
                "size": 100,
                "etag": '"etag"',
                "etag_is_weak": False,
                "last_modified": "2026-08-14T00:00:00Z",
                "content_type": "audio/wav",
                "cache_status": "verified",
                "sha256": SHA_A,
                "cache_path": "sha256/aa/" + SHA_A,
            }
        ],
    }
    remote, endpoint, bucket, expected = _source_audio_parts(
        source,
        source_audio_key=None,
        source_audio_content_hash=SHA_A,
        source_endpoint_sha256=None,
        source_bucket=None,
    )
    assert remote.key == "42/bgm.wav"
    assert endpoint == SHA_A


def test_source_audio_parts_mapping_requires_key() -> None:
    with pytest.raises(ValueError, match="source_audio_key is required"):
        _source_audio_parts(
            {},
            source_audio_key=None,
            source_audio_content_hash=SHA_A,
            source_endpoint_sha256=None,
            source_bucket=None,
        )


def test_source_audio_parts_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError, match="source must be"):
        _source_audio_parts(
            42,
            source_audio_key="k",
            source_audio_content_hash=SHA_A,
            source_endpoint_sha256=SHA_A,
            source_bucket="b",
        )  # type: ignore[arg-type]


def test_source_audio_parts_requires_endpoint() -> None:
    remote = _source_remote()
    with pytest.raises(ValueError, match="source_endpoint_sha256 is required"):
        _source_audio_parts(
            remote,
            source_audio_key=None,
            source_audio_content_hash=SHA_A,
            source_endpoint_sha256=None,
            source_bucket=None,
        )


def test_source_audio_parts_requires_bucket() -> None:
    remote = _source_remote()
    with pytest.raises(ValueError, match="source_bucket is required"):
        _source_audio_parts(
            remote,
            source_audio_key=None,
            source_audio_content_hash=SHA_A,
            source_endpoint_sha256=SHA_A,
            source_bucket=None,
        )


def test_source_audio_parts_requires_content_hash() -> None:
    remote = _source_remote()
    with pytest.raises(ValueError, match="source_audio_content_hash is required"):
        _source_audio_parts(
            remote,
            source_audio_key=None,
            source_audio_content_hash=None,
            source_endpoint_sha256=SHA_A,
            source_bucket="b",
        )


# ---------------------------------------------------------------------------
# resolve_source_audio
# ---------------------------------------------------------------------------


def test_resolve_source_audio_rejects_non_path_cache_dir() -> None:
    with pytest.raises(TypeError, match="cache_dir must be a Path"):
        resolve_source_audio(
            _source_remote(),
            "not-a-path",  # type: ignore[arg-type]
            source_audio_key="42/bgm.wav",
            source_audio_content_hash=SHA_A,
            source_endpoint_sha256=SHA_A,
            source_bucket="b",
        )


def test_resolve_source_audio_rejects_disagreeing_index_args(tmp_path: Path) -> None:
    idx_a = CacheIndexStore(tmp_path, {})
    idx_b = CacheIndexStore(tmp_path, {})
    with pytest.raises(ValueError, match="cache index arguments disagree"):
        resolve_source_audio(
            _source_remote(),
            tmp_path,
            idx_a,
            index=idx_b,
            source_audio_key="42/bgm.wav",
            source_audio_content_hash=SHA_A,
            source_endpoint_sha256=SHA_A,
            source_bucket="b",
        )


# ---------------------------------------------------------------------------
# materialize_full_mix_audio
# ---------------------------------------------------------------------------


def test_materialize_rejects_non_resolved_source(tmp_path: Path) -> None:
    from src.benchmark.input_view import materialize_full_mix_audio

    with pytest.raises(TypeError, match="source_audio must be ResolvedSourceAudio"):
        materialize_full_mix_audio(
            "not-source",  # type: ignore[arg-type]
            tmp_path / "out.wav",
            input_root=tmp_path,
            input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
            max_input_audio_frames=None,
        )


def test_materialize_rejects_non_path_args(tmp_path: Path) -> None:
    from src.benchmark.input_view import materialize_full_mix_audio

    source = ResolvedSourceAudio(
        path=tmp_path / "s.wav",
        source_audio_id="42/bgm.wav",
        source_audio_sha256=SHA_A,
        duration_sec=1.0,
    )
    with pytest.raises(TypeError, match="output_path and input_root must be Paths"):
        materialize_full_mix_audio(
            source,
            "out",  # type: ignore[arg-type]
            input_root=tmp_path,
            input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
            max_input_audio_frames=None,
        )


def test_materialize_rejects_non_model_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.benchmark import oaf_corpus_run as run_module

    calls: list[object] = []

    def materialize(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("neutral materializer must not receive an invalid config")

    monkeypatch.setattr(run_module, "materialize_full_mix_audio", materialize)
    source = ResolvedSourceAudio(
        path=tmp_path / "s.wav",
        source_audio_id="42/bgm.wav",
        source_audio_sha256=SHA_A,
        duration_sec=1.0,
    )
    with pytest.raises(TypeError, match="config must be OafModelConfig"):
        run_module._materialize_oaf_full_mix(
            source,
            tmp_path / "out.wav",
            input_root=tmp_path,
            config="bad",  # type: ignore[arg-type]
        )

    assert calls == []


def test_materialize_rejects_output_outside_input_root(tmp_path: Path) -> None:
    from src.benchmark.input_view import materialize_full_mix_audio

    source = ResolvedSourceAudio(
        path=tmp_path / "s.wav",
        source_audio_id="42/bgm.wav",
        source_audio_sha256=SHA_A,
        duration_sec=1.0,
    )
    with pytest.raises(ValueError, match="beneath input_root"):
        materialize_full_mix_audio(
            source,
            tmp_path / "elsewhere" / "out.wav",
            input_root=tmp_path / "inputs",
            input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
            max_input_audio_frames=None,
        )


# ---------------------------------------------------------------------------
# preflight_reference_mappings
# ---------------------------------------------------------------------------


def _preflight_fixture(tmp_path: Path):
    """Build minimal reference + timing manifests for preflight tests."""
    event = NativeReferenceEvent(
        simfile_id=42,
        selected_chart_key="42/real.dtx",
        selected_chart_content_hash=SHA_A,
        source_audio_key="42/bgm.wav",
        source_audio_content_hash=SHA_B,
        source_order=0,
        measure=1,
        position=0.0,
        lane_id="11",
        note_id="01",
        chart_time_sec=1.0,
        audio_time_sec=0.5,
    )
    from hashlib import sha256

    from src.benchmark.reference_timing import render_reference_events

    artifact = (
        tmp_path / "events" / (sha256(render_reference_events((event,))).hexdigest() + ".jsonl")
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(render_reference_events((event,)))
    relative_artifact = artifact.relative_to(tmp_path).as_posix()
    source_row = {
        "selected_chart_key": event.selected_chart_key,
        "selected_chart_content_hash": event.selected_chart_content_hash,
        "source_audio_key": event.source_audio_key,
        "source_audio_content_hash": event.source_audio_content_hash,
        "reference_events_cache_path": relative_artifact,
        "source_endpoint_sha256": SHA_A,
        "source_bucket": "simfile-dtx",
    }
    timing_view = ReferenceTimingRowView(
        simfile_id=42,
        corpus_version="sha256:" + SHA_C,
        timing_status="ready",
        timing_reason_codes=(),
        timing_warnings=(),
        reference_events_cache_path=relative_artifact,
        source_audio_key="42/bgm.wav",
        source_audio_content_hash=SHA_B,
    )
    timing_row = LoadedReferenceTimingRow(source_row=source_row, view=timing_view)
    timing_manifest = LoadedReferenceTimingManifest(
        manifest_sha256=SHA_A,
        corpus_version="sha256:" + SHA_C,
        rows=(timing_row,),
    )
    set_view = ReferenceSetRowView(
        simfile_id=42,
        eligibility_status="eligible",
        eligibility_reason_codes=(),
        eligibility_warnings=(),
        mapped_event_count=1,
        common_scored_event_count=1,
        ignored_event_count=0,
        unmapped_event_count=0,
        duplicate_common_event_count=0,
    )
    set_row = LoadedReferenceSetRow(source_row=source_row, view=set_view)
    set_manifest = LoadedReferenceSetManifest(
        manifest_sha256=SHA_B,
        corpus_version="sha256:" + SHA_C,
        source_reference_timing_manifest_sha256=SHA_A,
        source_reference_timing_version="sha256:" + SHA_C,
        rows=(set_row,),
    )
    return tmp_path, set_manifest, timing_manifest, artifact


def test_preflight_rejects_non_reference_manifest(tmp_path: Path) -> None:
    _, _, timing_manifest, _ = _preflight_fixture(tmp_path)
    with pytest.raises(TypeError, match="reference_manifest must be"):
        preflight_reference_mappings("bad", timing_manifest, timing_output_root=tmp_path)  # type: ignore[arg-type]


def test_preflight_rejects_non_timing_manifest(tmp_path: Path) -> None:
    _, set_manifest, _, _ = _preflight_fixture(tmp_path)
    with pytest.raises(TypeError, match="timing_manifest must be"):
        preflight_reference_mappings(set_manifest, "bad", timing_output_root=tmp_path)  # type: ignore[arg-type]


def test_preflight_rejects_non_path_output_root(tmp_path: Path) -> None:
    _, set_manifest, timing_manifest, _ = _preflight_fixture(tmp_path)
    with pytest.raises(TypeError, match="timing_output_root must be a Path"):
        preflight_reference_mappings(set_manifest, timing_manifest, timing_output_root="bad")  # type: ignore[arg-type]


def test_preflight_rejects_lineage_mismatch(tmp_path: Path) -> None:
    _, set_manifest, timing_manifest, _ = _preflight_fixture(tmp_path)
    timing_manifest = replace(timing_manifest, manifest_sha256=SHA_C)
    with pytest.raises(ValueError, match="lineage"):
        preflight_reference_mappings(set_manifest, timing_manifest, timing_output_root=tmp_path)


def test_preflight_non_eligible_without_timing_row_is_none(tmp_path: Path) -> None:
    _, set_manifest, timing_manifest, _ = _preflight_fixture(tmp_path)
    quarantined_view = replace(
        set_manifest.rows[0].view,
        eligibility_status="quarantined",
        eligibility_reason_codes=("unclassified_reference_lane",),
    )
    set_manifest = replace(
        set_manifest,
        rows=(replace(set_manifest.rows[0], view=quarantined_view),),
    )
    timing_manifest = replace(timing_manifest, rows=())
    mappings = preflight_reference_mappings(
        set_manifest, timing_manifest, timing_output_root=tmp_path
    )
    assert mappings == {42: None}


def test_preflight_eligible_without_timing_row_is_fatal(tmp_path: Path) -> None:
    _, set_manifest, timing_manifest, _ = _preflight_fixture(tmp_path)
    timing_manifest = replace(timing_manifest, rows=())
    with pytest.raises(ValueError, match="eligible reference timing row is unavailable"):
        preflight_reference_mappings(set_manifest, timing_manifest, timing_output_root=tmp_path)


def test_preflight_non_eligible_corrupt_artifact_is_none(tmp_path: Path) -> None:
    """A non-eligible row with a corrupt artifact and a ready timing row maps to None (line 686)."""
    root, set_manifest, timing_manifest, artifact = _preflight_fixture(tmp_path)
    artifact.write_bytes(b"corrupt")
    # Use a reason code that is NOT in the early-exit set, so the code falls
    # through to the try/except at line 677 and hits line 686.
    quarantined_view = replace(
        set_manifest.rows[0].view,
        eligibility_status="quarantined",
        eligibility_reason_codes=("unclassified_reference_lane",),
    )
    set_manifest = replace(
        set_manifest,
        rows=(replace(set_manifest.rows[0], view=quarantined_view),),
    )
    mappings = preflight_reference_mappings(set_manifest, timing_manifest, timing_output_root=root)
    assert mappings == {42: None}


# ---------------------------------------------------------------------------
# classify_oaf_backend_error
# ---------------------------------------------------------------------------


def test_classify_oaf_backend_error_rejects_non_str_code() -> None:
    assert classify_oaf_backend_error(123) == ("worker_protocol_failed", "poison")


# ---------------------------------------------------------------------------
# _normalize_snapshot_value
# ---------------------------------------------------------------------------


def test_normalize_snapshot_value_rejects_non_str_keys() -> None:
    with pytest.raises(StrictJsonError, match="keys must be strings"):
        _normalize_snapshot_value({1: "value"})  # type: ignore[dict-item]


def test_normalize_snapshot_value_rejects_unsupported_type() -> None:
    with pytest.raises(StrictJsonError, match="unsupported run snapshot value"):
        _normalize_snapshot_value(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _validate_snapshot
# ---------------------------------------------------------------------------


def _valid_snapshot() -> dict[str, object]:
    return {
        "schema": OAF_CORPUS_RUN_SCHEMA,
        "run_id": "oaf-" + SHA_A[:16],
        "exclude_simfile_ids": [20],
        "overall_status": "complete",
        "success_count": 1,
        "failed_count": 0,
        "skipped_count": 1,
        "quarantined_count": 0,
        "completed_at": "2026-08-14T00:00:00+00:00",
        "items": [
            {"simfile_id": 20, "execution_disposition": "skipped"},
            {
                "simfile_id": 10,
                "execution_disposition": "inferred",
                "wall_time_sec": 1.25,
                "rtf": 0.5,
            },
        ],
    }


def test_validate_snapshot_rejects_non_array_scope() -> None:
    snap = _valid_snapshot()
    snap["include_simfile_ids"] = "not-a-list"
    with pytest.raises(StrictJsonError, match="scope must be arrays"):
        _validate_snapshot(snap)


def test_validate_snapshot_rejects_non_array_items() -> None:
    snap = _valid_snapshot()
    snap["items"] = "not-a-list"
    with pytest.raises(StrictJsonError, match="items must be an array"):
        _validate_snapshot(snap)


def test_validate_snapshot_rejects_non_object_item() -> None:
    snap = _valid_snapshot()
    snap["items"] = ["not-an-object"]
    with pytest.raises(StrictJsonError, match="item must be an object"):
        _validate_snapshot(snap)


def test_validate_snapshot_rejects_invalid_simfile_id() -> None:
    snap = _valid_snapshot()
    snap["items"] = [{"simfile_id": "bad", "execution_disposition": "skipped"}]
    with pytest.raises(StrictJsonError, match="simfile_id is invalid"):
        _validate_snapshot(snap)


def test_validate_snapshot_rejects_duplicate_simfile_ids() -> None:
    snap = _valid_snapshot()
    snap["items"] = [
        {"simfile_id": 10, "execution_disposition": "skipped"},
        {"simfile_id": 10, "execution_disposition": "skipped"},
    ]
    snap["exclude_simfile_ids"] = [10]
    with pytest.raises(StrictJsonError, match="unique simfile IDs"):
        _validate_snapshot(snap)


def test_validate_snapshot_rejects_invalid_disposition() -> None:
    snap = _valid_snapshot()
    snap["items"] = [{"simfile_id": 10, "execution_disposition": "bogus"}]
    snap["exclude_simfile_ids"] = [10]
    with pytest.raises(StrictJsonError, match="disposition is invalid"):
        _validate_snapshot(snap)


def test_validate_snapshot_rejects_skipped_without_exclusion() -> None:
    snap = _valid_snapshot()
    snap["exclude_simfile_ids"] = []
    snap["items"] = [{"simfile_id": 10, "execution_disposition": "skipped"}]
    with pytest.raises(StrictJsonError, match="skipped item requires explicit filter exclusion"):
        _validate_snapshot(snap)


def test_validate_snapshot_rejects_invalid_overall_status() -> None:
    snap = _valid_snapshot()
    snap["overall_status"] = "bogus"
    with pytest.raises(StrictJsonError, match="overall_status is invalid"):
        _validate_snapshot(snap)


def test_validate_snapshot_complete_requires_counts() -> None:
    snap = _valid_snapshot()
    del snap["success_count"]
    with pytest.raises(StrictJsonError, match="requires all counts"):
        _validate_snapshot(snap)


def test_validate_snapshot_complete_requires_dispositions() -> None:
    snap = _valid_snapshot()
    snap["items"] = [{"simfile_id": 10}, {"simfile_id": 20, "execution_disposition": "skipped"}]
    with pytest.raises(StrictJsonError, match="requires item dispositions"):
        _validate_snapshot(snap)


def test_validate_snapshot_rejects_negative_counts() -> None:
    snap = _valid_snapshot()
    snap["success_count"] = -1
    with pytest.raises(StrictJsonError, match="counts must be nonnegative"):
        _validate_snapshot(snap)


def test_validate_snapshot_rejects_mismatched_counts() -> None:
    snap = _valid_snapshot()
    snap["success_count"] = 99
    with pytest.raises(StrictJsonError, match="counts do not reconcile"):
        _validate_snapshot(snap)


def test_validate_snapshot_rejects_completed_at_for_non_complete() -> None:
    snap = _valid_snapshot()
    snap["overall_status"] = "partial"
    with pytest.raises(StrictJsonError, match="completed timestamp"):
        _validate_snapshot(snap)


# ---------------------------------------------------------------------------
# render_oaf_corpus_run / parse_oaf_corpus_run / write_oaf_corpus_run
# ---------------------------------------------------------------------------


def test_render_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="run snapshot must be a mapping"):
        render_oaf_corpus_run("not-a-mapping")  # type: ignore[arg-type]


def test_parse_rejects_non_object() -> None:
    with pytest.raises(StrictJsonError, match="run snapshot must be an object"):
        parse_oaf_corpus_run(canonical_json_bytes([1, 2, 3]))


def test_parse_rejects_run_id_mismatch() -> None:
    content = render_oaf_corpus_run(_valid_snapshot())
    with pytest.raises(StrictJsonError, match="run_id does not match"):
        parse_oaf_corpus_run(content, expected_run_id="oaf-different")


def test_write_rejects_non_path() -> None:
    with pytest.raises(TypeError, match="run_path must be a Path"):
        write_oaf_corpus_run("not-a-path", _valid_snapshot())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _timestamp / _bounded_error / _bounded_close_error
# ---------------------------------------------------------------------------


def test_timestamp_rejects_non_datetime() -> None:
    with pytest.raises(TypeError, match="clock must return a datetime"):
        _timestamp("not-a-datetime")  # type: ignore[arg-type]


def test_timestamp_handles_naive_datetime() -> None:
    result = _timestamp(datetime(2026, 8, 14, 12, 0, 0))
    assert result == "2026-08-14T12:00:00+00:00"


def test_utc_now_returns_timezone_aware_datetime() -> None:
    result = _utc_now()
    assert isinstance(result, datetime)
    assert result.tzinfo is not None
    assert result.utcoffset() == timezone.utc.utcoffset(None)


def test_bounded_error_falls_back_to_type_name() -> None:
    assert _bounded_error(ValueError("")) == "ValueError"


def test_bounded_close_error_defaults_code_for_non_str() -> None:
    error = ValueError("fail")
    result = _bounded_close_error(error)
    assert result["code"] == "worker_close_failed"
    assert result["message"] == "fail"


def test_bounded_close_error_handles_non_exception() -> None:
    result = _bounded_close_error("just a string")
    assert result["code"] == "worker_close_failed"


# ---------------------------------------------------------------------------
# _source_failure_code
# ---------------------------------------------------------------------------


def test_source_failure_code_classifies_decode_error() -> None:
    assert _source_failure_code(OSError("cannot decode wav header")) == "source_audio_decode_failed"
    assert _source_failure_code(RuntimeError("invalid wav")) == "source_audio_decode_failed"


def test_source_failure_code_defaults_to_unavailable() -> None:
    assert _source_failure_code(ValueError("some other error")) == "source_audio_unavailable"


# ---------------------------------------------------------------------------
# _remove_temporary_input
# ---------------------------------------------------------------------------


def test_remove_temporary_input_ignores_outside_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"data")
    _remove_temporary_input(outside, tmp_path / "inputs")
    assert outside.exists()


def test_remove_temporary_input_swallows_unlink_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "inputs" / "10" / "full-mix.wav"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"data")

    def fail_unlink(self, *args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    _remove_temporary_input(target, tmp_path / "inputs")  # should not raise


# ---------------------------------------------------------------------------
# _read_existing_prediction
# ---------------------------------------------------------------------------


def test_read_existing_prediction_returns_false_on_missing(tmp_path: Path) -> None:
    exists, content = _read_existing_prediction(tmp_path / "missing.jsonl")
    assert exists is False
    assert content is None


def test_read_existing_prediction_returns_true_none_on_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "pred.jsonl"
    target.write_bytes(b"data")

    def fail_read(*_args, **_kwargs):
        raise OSError("io error")

    monkeypatch.setattr("src.benchmark.oaf_corpus_run.read_regular_file_no_follow", fail_read)
    exists, content = _read_existing_prediction(target)
    assert exists is True
    assert content is None


# ---------------------------------------------------------------------------
# _prediction_artifact_matches
# ---------------------------------------------------------------------------


def test_prediction_artifact_matches_rejects_descriptor_sha_mismatch() -> None:
    descriptor = _descriptor()
    artifact = _prediction_artifact(descriptor=descriptor)
    source = _resolved_source()
    wrong_descriptor = replace(descriptor, sha256=SHA_C)
    assert not _prediction_artifact_matches(
        artifact, source=source, audio=_canonical_audio(), descriptor=wrong_descriptor
    )


def test_prediction_artifact_matches_rejects_descriptor_payload_mismatch() -> None:
    descriptor = _descriptor()
    artifact = _prediction_artifact(descriptor=descriptor)
    source = _resolved_source()
    wrong_descriptor = replace(descriptor, payload={**descriptor.payload, "model_id": "other"})
    assert not _prediction_artifact_matches(
        artifact, source=source, audio=_canonical_audio(), descriptor=wrong_descriptor
    )


def test_prediction_artifact_matches_rejects_source_audio_id_mismatch() -> None:
    descriptor = _descriptor()
    audio = _canonical_audio(source_audio_id="10/audio.wav")
    artifact = _prediction_artifact(audio=audio, descriptor=descriptor)
    source = _resolved_source(source_audio_id="99/audio.wav")
    assert not _prediction_artifact_matches(
        artifact, source=source, audio=audio, descriptor=descriptor
    )


def test_prediction_artifact_matches_rejects_source_sha_mismatch() -> None:
    descriptor = _descriptor()
    audio = _canonical_audio()
    artifact = _prediction_artifact(audio=audio, descriptor=descriptor)
    source = _resolved_source(source_audio_sha256=SHA_C)
    assert not _prediction_artifact_matches(
        artifact, source=source, audio=audio, descriptor=descriptor
    )


def test_prediction_artifact_matches_rejects_input_view_mismatch() -> None:
    descriptor = _descriptor()
    audio = _canonical_audio()
    artifact = _prediction_artifact(audio=audio, descriptor=descriptor)
    source = _resolved_source()
    wrong_audio = replace(audio, input_view_id="other/v1")
    assert not _prediction_artifact_matches(
        artifact, source=source, audio=wrong_audio, descriptor=descriptor
    )


def test_prediction_artifact_matches_rejects_input_sha_mismatch() -> None:
    descriptor = _descriptor()
    audio = _canonical_audio()
    artifact = _prediction_artifact(audio=audio, descriptor=descriptor)
    source = _resolved_source()
    wrong_audio = replace(audio, input_audio_sha256=SHA_C)
    assert not _prediction_artifact_matches(
        artifact, source=source, audio=wrong_audio, descriptor=descriptor
    )


def test_prediction_artifact_matches_rejects_audio_source_id_mismatch() -> None:
    """The artifact's source_audio_id must match both ``source`` and ``audio``."""
    descriptor = _descriptor()
    audio = _canonical_audio(source_audio_id="10/audio.wav")
    artifact = _prediction_artifact(audio=audio, descriptor=descriptor)
    source = _resolved_source()
    mismatched_audio = replace(audio, source_audio_id="99/audio.wav")
    assert not _prediction_artifact_matches(
        artifact, source=source, audio=mismatched_audio, descriptor=descriptor
    )


def test_prediction_artifact_matches_rejects_audio_source_sha_mismatch() -> None:
    """The artifact's source_audio_sha256 must match both ``source`` and ``audio``."""
    descriptor = _descriptor()
    audio = _canonical_audio()
    artifact = _prediction_artifact(audio=audio, descriptor=descriptor)
    source = _resolved_source()
    mismatched_audio = replace(audio, source_audio_sha256=SHA_C)
    assert not _prediction_artifact_matches(
        artifact, source=source, audio=mismatched_audio, descriptor=descriptor
    )


# ---------------------------------------------------------------------------
# _prediction_artifact_matches_run_row
# ---------------------------------------------------------------------------


def test_prediction_artifact_matches_run_row_rejects_non_artifact() -> None:
    assert not _prediction_artifact_matches_run_row("not-artifact", {"simfile_id": 10})  # type: ignore[arg-type]


def test_prediction_artifact_matches_run_row_rejects_non_mapping_row() -> None:
    artifact = _prediction_artifact()
    assert not _prediction_artifact_matches_run_row(artifact, "not-a-mapping")  # type: ignore[arg-type]


def test_prediction_artifact_matches_run_row_rejects_missing_fields() -> None:
    artifact = _prediction_artifact()
    assert not _prediction_artifact_matches_run_row(artifact, {"simfile_id": 10})


# ---------------------------------------------------------------------------
# _cohort_item_from_run_row
# ---------------------------------------------------------------------------


def test_cohort_item_from_run_row_rejects_non_identity() -> None:
    with pytest.raises(TypeError, match="identity must be CohortIdentity"):
        _cohort_item_from_run_row("bad", {"simfile_id": 10}, None, output_dir=Path("output"))  # type: ignore[arg-type]


def test_cohort_item_from_run_row_rejects_non_mapping_row() -> None:
    with pytest.raises(TypeError, match="run row must be a mapping"):
        _cohort_item_from_run_row(_cohort_identity(), "bad", None, output_dir=Path("output"))  # type: ignore[arg-type]


def test_cohort_item_from_run_row_rejects_non_path_output_dir() -> None:
    with pytest.raises(TypeError, match="output_dir must be a Path"):
        _cohort_item_from_run_row(_cohort_identity(), {"simfile_id": 10}, None, output_dir="bad")  # type: ignore[arg-type]


def test_cohort_item_from_run_row_rejects_invalid_simfile_id() -> None:
    with pytest.raises(ValueError, match="simfile_id is invalid"):
        _cohort_item_from_run_row(
            _cohort_identity(), {"simfile_id": True}, None, output_dir=Path("output")
        )


def test_cohort_item_from_run_row_rejects_empty_simfile_id() -> None:
    with pytest.raises(ValueError, match="simfile_id is invalid"):
        _cohort_item_from_run_row(
            _cohort_identity(), {"simfile_id": ""}, None, output_dir=Path("output")
        )


def test_cohort_item_from_run_row_inferred_missing_prediction_path() -> None:
    item = _cohort_item_from_run_row(
        _cohort_identity(),
        {"simfile_id": 10, "execution_disposition": "inferred"},
        _reference_mapping(10),
        output_dir=Path("output"),
    )
    assert item.status == "failed"
    assert item.failure_reason == "prediction_missing"


def test_cohort_item_from_run_row_inferred_file_not_found(tmp_path: Path) -> None:
    item = _cohort_item_from_run_row(
        _cohort_identity(),
        {"simfile_id": 10, "execution_disposition": "inferred", "prediction_path": "missing.jsonl"},
        _reference_mapping(10),
        output_dir=tmp_path,
    )
    assert item.status == "failed"
    assert item.failure_reason == "prediction_missing"


def test_cohort_item_from_run_row_inferred_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "pred.jsonl"
    target.write_bytes(b"data")

    def fail_read(*_args, **_kwargs):
        raise OSError("io error")

    monkeypatch.setattr("src.benchmark.oaf_corpus_run.read_regular_file_no_follow", fail_read)
    item = _cohort_item_from_run_row(
        _cohort_identity(),
        {"simfile_id": 10, "execution_disposition": "inferred", "prediction_path": "pred.jsonl"},
        _reference_mapping(10),
        output_dir=tmp_path,
    )
    assert item.status == "failed"
    assert item.failure_reason == "prediction_artifact_invalid"


def test_cohort_item_from_run_row_inferred_mapping_none_with_artifact(tmp_path: Path) -> None:
    descriptor = _descriptor()
    audio = _canonical_audio()
    artifact = _prediction_artifact(audio=audio, descriptor=descriptor)
    target = tmp_path / "pred.jsonl"
    target.write_bytes(
        render_prediction_artifact(_mapped_prediction(audio=audio, descriptor=descriptor))
    )
    item = _cohort_item_from_run_row(
        _cohort_identity(),
        {
            "simfile_id": 10,
            "execution_disposition": "inferred",
            "prediction_path": "pred.jsonl",
            "prediction_artifact_sha256": artifact.artifact_sha256,
            "source_audio_id": "10/audio.wav",
            "source_audio_sha256": SHA_A,
            "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
            "input_audio_sha256": SHA_B,
        },
        None,
        output_dir=tmp_path,
    )
    assert item.status == "failed"
    assert item.failure_reason == "prediction_artifact_invalid"


def test_cohort_item_from_run_row_inferred_corrupt_artifact(tmp_path: Path) -> None:
    target = tmp_path / "pred.jsonl"
    target.write_bytes(b"corrupt artifact content")
    item = _cohort_item_from_run_row(
        _cohort_identity(),
        {
            "simfile_id": 10,
            "execution_disposition": "inferred",
            "prediction_path": "pred.jsonl",
        },
        _reference_mapping(10),
        output_dir=tmp_path,
    )
    assert item.status == "failed"
    assert item.failure_reason == "prediction_artifact_invalid"


# ---------------------------------------------------------------------------
# _prediction_relative_path
# ---------------------------------------------------------------------------


def test_prediction_relative_path_fallback_for_outside_path(tmp_path: Path) -> None:
    outside = Path("/absolute/elsewhere/pred.jsonl")
    result = _prediction_relative_path(outside, tmp_path / "output")
    assert result == outside.as_posix()


# ---------------------------------------------------------------------------
# _cohort_identity_from_snapshot
# ---------------------------------------------------------------------------


def test_cohort_identity_from_snapshot_rejects_missing_descriptor() -> None:
    with pytest.raises(ValueError, match="backend descriptor is missing"):
        _cohort_identity_from_snapshot({"run_id": "oaf-x"})


def test_cohort_identity_from_snapshot_rejects_incomplete_identity() -> None:
    with pytest.raises(ValueError, match="complete cohort identity"):
        _cohort_identity_from_snapshot({"backend_descriptor": {"backend_id": "crux.oaf"}})


# ---------------------------------------------------------------------------
# _item_counts
# ---------------------------------------------------------------------------


def test_item_counts_classifies_all_statuses() -> None:
    from src.benchmark.cohort_scoring import CohortCoverage, CohortItem

    def _item(status: str) -> CohortItem:
        return CohortItem(
            simfile_id="10",
            status=status,  # type: ignore[arg-type]
            reference_events=(),
            prediction_events=None,
            coverage=CohortCoverage(
                reference_native_event_count=0,
                reference_common_event_count=0,
                reference_ignored_event_count=0,
                reference_unmapped_event_count=0,
                reference_duplicate_collapsed_count=0,
                prediction_native_event_count=None,
                prediction_mapped_event_count=None,
                prediction_unmapped_event_count=None,
                prediction_native_class_counts=(),
            ),
            warnings=(),
            artifact_identity=None,
            reference_artifact=None,
            prediction_artifact=None,
        )

    counts = _item_counts(
        [
            _item("success"),
            _item("failed"),
            _item("skipped"),
            _item("quarantined"),
        ]
    )
    assert counts == {
        "success_count": 1,
        "failed_count": 1,
        "skipped_count": 1,
        "quarantined_count": 1,
    }


# ---------------------------------------------------------------------------
# _finalize_scoring_and_outcome
# ---------------------------------------------------------------------------


def test_finalize_scoring_returns_fatal_outcome_for_non_list_items(tmp_path: Path) -> None:
    snapshot = {"items": "not-a-list", "overall_status": "complete"}
    outcome = _finalize_scoring_and_outcome(
        snapshot,
        run_id="oaf-test",
        run_path=tmp_path / "run.json",
        reports_path=tmp_path / "reports",
        aggregate_rtf=None,
        projected_full_wall_time_sec=None,
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_finalize_scoring_defaults_unknown_status_to_partial(tmp_path: Path) -> None:
    snapshot = {
        "items": [],
        "overall_status": "bogus",
        "backend_descriptor": {"backend_id": "crux.oaf"},
        "run_id": "oaf-test",
        "reference_manifest_sha256": SHA_A,
        "reference_timing_version": "sha256:" + SHA_B,
        "model_id": "model",
        "model_lock_sha256": SHA_C,
        "backend_descriptor_sha256": SHA_A,
        "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
    }
    outcome = _finalize_scoring_and_outcome(
        snapshot,
        run_id="oaf-test",
        run_path=tmp_path / "run.json",
        reports_path=tmp_path / "reports",
        aggregate_rtf=None,
        projected_full_wall_time_sec=None,
    )
    assert outcome.overall_status == "partial"
    assert outcome.exit_code == 1


# ---------------------------------------------------------------------------
# _fatal_outcome
# ---------------------------------------------------------------------------


def test_fatal_outcome_is_failed_exit_two() -> None:
    outcome = _fatal_outcome()
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2
    assert outcome.run_id is None


# ---------------------------------------------------------------------------
# Remaining unit-level branches
# ---------------------------------------------------------------------------


def test_require_hash_rejects_non_string() -> None:
    from src.benchmark.oaf_corpus_run import _require_hash

    with pytest.raises(StrictJsonError, match="must be lowercase SHA-256"):
        _require_hash(123, "test_field")


def test_validate_snapshot_rejects_non_string_run_id() -> None:
    snap = _valid_snapshot()
    snap["run_id"] = 123
    with pytest.raises(StrictJsonError, match="run_id must be a nonempty string"):
        _validate_snapshot(snap)


def test_render_rejects_non_object_normalized() -> None:
    """render_oaf_corpus_run rejects a non-Mapping input."""
    with pytest.raises(TypeError, match="run snapshot must be a mapping"):
        render_oaf_corpus_run([1, 2, 3])  # type: ignore[arg-type]


def test_project_runtime_skips_non_finite_duration() -> None:
    runtime = _project_runtime(
        (),
        eligible_audio_durations=(float("inf"),),
    )
    assert runtime["eligible_audio_duration_sec"] is None
    assert runtime["eligible_audio_duration_coverage_count"] == 0


def test_prediction_artifact_matches_rejects_non_oaf_input_view() -> None:
    """Artifact whose input_view_id is not OAF_FULL_MIX_INPUT_VIEW_ID is rejected."""
    descriptor = _descriptor()
    audio = _canonical_audio()
    mapped = _mapped_prediction(audio=audio, descriptor=descriptor)
    wrong = replace(mapped, audio=replace(audio, input_view_id="other/v1"))
    artifact = read_prediction_artifact(render_prediction_artifact(wrong))
    source = _resolved_source()
    assert not _prediction_artifact_matches(
        artifact, source=source, audio=audio, descriptor=descriptor
    )


def test_cohort_item_from_run_row_falls_back_on_non_list_warnings() -> None:
    item = _cohort_item_from_run_row(
        _cohort_identity(),
        {
            "simfile_id": 10,
            "execution_disposition": "skipped",
            "eligibility_warnings": "not-a-list",
        },
        _reference_mapping(10),
        output_dir=Path("output"),
    )
    assert item.warnings == ()


def test_source_audio_parts_remote_requires_key() -> None:
    """A RemoteObject with an empty key requires source_audio_key (line 506)."""
    remote = replace(_source_remote(), key="")
    with pytest.raises(ValueError, match="source_audio_key is required"):
        _source_audio_parts(
            remote,
            source_audio_key=None,
            source_audio_content_hash=SHA_A,
            source_endpoint_sha256=SHA_A,
            source_bucket="b",
        )


def test_finalize_scoring_complete_with_failures_becomes_partial(tmp_path: Path) -> None:
    """A 'complete' snapshot with failed items is downgraded to partial (line 1481)."""
    snapshot = {
        "items": [
            {"simfile_id": 10, "execution_disposition": "inferred"},
        ],
        "overall_status": "complete",
        "backend_descriptor": {
            "backend_id": "crux.oaf",
            "model_id": "model",
            "architecture_id": "arch",
            "backend_descriptor_sha256": SHA_A,
            "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
            "prediction_map_version": OAF_PREDICTION_MAP_ID,
        },
        "run_id": "oaf-test",
        "reference_manifest_sha256": SHA_A,
        "reference_timing_version": "sha256:" + SHA_B,
        "model_id": "model",
        "model_lock_sha256": SHA_C,
        "backend_descriptor_sha256": SHA_A,
        "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
    }
    outcome = _finalize_scoring_and_outcome(
        snapshot,
        run_id="oaf-test",
        run_path=tmp_path / "run.json",
        reports_path=tmp_path / "reports",
        aggregate_rtf=None,
        projected_full_wall_time_sec=None,
        mappings={10: _reference_mapping(10)},
        output_dir=tmp_path,
    )
    # The inferred item has no prediction artifact → failed → partial
    assert outcome.overall_status == "partial"
    assert outcome.exit_code == 1


def test_bounded_close_error_handles_non_str_code() -> None:
    """A non-string code attribute falls back to the default (line 896)."""

    class WeirdError(Exception):
        code = 123  # type: ignore[assignment]

    result = _bounded_close_error(WeirdError("fail"))
    assert result["code"] == "worker_close_failed"


# ---------------------------------------------------------------------------
# run_oaf_corpus orchestration error paths
# ---------------------------------------------------------------------------


def _install_run_seams(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Install deterministic manifest/source/materializer seams for lifecycle tests."""
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.reference_set_manifest import (
        LoadedReferenceSetManifest,
        LoadedReferenceSetRow,
        ReferenceSetRowView,
    )
    from src.benchmark.reference_timing_manifest import (
        LoadedReferenceTimingManifest,
    )

    rows = tuple(
        LoadedReferenceSetRow(
            source_row={
                "selected_chart_key": f"{sid}/chart.dtx",
                "selected_chart_content_hash": SHA_A,
                "source_audio_key": f"{sid}/audio.wav",
                "source_audio_content_hash": SHA_A,
                "source_endpoint_sha256": SHA_A,
                "source_bucket": "simfile-dtx",
            },
            view=ReferenceSetRowView(
                simfile_id=sid,
                eligibility_status="eligible",
                eligibility_reason_codes=(),
                eligibility_warnings=(),
                mapped_event_count=0,
                common_scored_event_count=0,
                ignored_event_count=0,
                unmapped_event_count=0,
                duplicate_common_event_count=0,
            ),
        )
        for sid in (10,)
    )
    reference_manifest = LoadedReferenceSetManifest(
        manifest_sha256=SHA_A,
        corpus_version="sha256:" + SHA_B,
        source_reference_timing_manifest_sha256=SHA_B,
        source_reference_timing_version="sha256:" + SHA_A,
        rows=rows,
    )
    timing_manifest = LoadedReferenceTimingManifest(
        manifest_sha256=SHA_B,
        corpus_version="sha256:" + SHA_A,
        rows=(),
    )
    monkeypatch.setattr(run_module, "load_reference_set_manifest", lambda _: reference_manifest)
    monkeypatch.setattr(run_module, "load_reference_timing_manifest", lambda _: timing_manifest)
    monkeypatch.setattr(
        run_module,
        "preflight_reference_mappings",
        lambda *_a, **_kw: {10: _reference_mapping(10)},
    )
    monkeypatch.setattr(
        run_module,
        "resolve_source_audio",
        lambda *_a, **_kw: ResolvedSourceAudio(
            path=tmp_path / "source.wav",
            source_audio_id="10/audio.wav",
            source_audio_sha256=SHA_A,
            duration_sec=1.0,
        ),
    )
    descriptor = _expected_oaf_descriptor(load_model_config())

    def materialize(source, output_path, *, input_root, config):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"canonical wav")
        return _canonical_audio(source_audio_id=source.source_audio_id)

    monkeypatch.setattr(run_module, "_materialize_oaf_full_mix", materialize)
    return descriptor


def _real_prediction_path(tmp_path: Path, descriptor: BackendDescriptor) -> Path:
    """Compute the actual prediction path used by run_oaf_corpus."""
    config = load_model_config()
    lock_sha = compute_model_lock_sha256(_model_lock_path())
    inf_sha = inference_config_sha256(build_inference_config(config, descriptor, lock_sha))
    return prediction_path(
        tmp_path / "output",
        simfile_id=10,
        source_audio_sha256=SHA_A,
        backend_descriptor_sha256=descriptor.sha256,
        inference_config_sha256=inf_sha,
    )


def _request(tmp_path: Path, **kwargs) -> OafCorpusRunRequest:
    return OafCorpusRunRequest(
        reference_manifest_path=tmp_path / "reference.jsonl",
        timing_manifest_path=tmp_path / "timing.jsonl",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        **kwargs,
    )


class _HealthyBackend:
    def __init__(self, descriptor: BackendDescriptor) -> None:
        self._descriptor = descriptor

    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
        return NativePrediction(audio=audio, descriptor=self._descriptor, events=())

    def close(self) -> None:
        return None


def test_run_oaf_corpus_rejects_non_request() -> None:
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    with pytest.raises(TypeError, match="request must be OafCorpusRunRequest"):
        run_oaf_corpus("not-a-request")  # type: ignore[arg-type]


def test_run_oaf_corpus_clock_failure_returns_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)

    def bad_clock():
        raise RuntimeError("clock broken")

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(_descriptor()),
        perf_counter=lambda: 0.0,
        clock=bad_clock,
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_run_oaf_corpus_resume_parse_failure_returns_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    # First run to create a valid run.json
    first = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert first.exit_code == 0
    assert first.run_path is not None

    # Corrupt the run.json so resume parsing fails
    first.run_path.write_bytes(b"corrupt content\n")

    outcome = run_oaf_corpus(
        _request(tmp_path, resume=True),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_run_oaf_corpus_cache_index_load_failure_returns_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    def fail_cache_load(_path):
        raise OSError("cache unavailable")

    monkeypatch.setattr(run_module.CacheIndexStore, "load", fail_cache_load)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_run_oaf_corpus_source_resolve_failure_marks_item_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    def fail_resolve(*_a, **_kw):
        raise OSError("source audio unavailable")

    monkeypatch.setattr(run_module, "resolve_source_audio", fail_resolve)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.exit_code == 1
    assert outcome.failed_count == 1


def test_run_oaf_corpus_materialize_failure_marks_item_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    def fail_materialize(*_a, **_kw):
        raise OSError("materialization failed")

    monkeypatch.setattr(run_module, "_materialize_oaf_full_mix", fail_materialize)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.exit_code == 1
    assert outcome.failed_count == 1


def test_run_oaf_corpus_rejects_source_body_mutated_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache body mutated after the bulk preflight must be rejected at inference.

    The bulk pass resolves each source with load_body=False (verified path only).
    The per-item materialization step re-pins the body with load_body=True so the
    bytes inferred by librosa are guaranteed to match the HPA-323 digest recorded
    in the run row. If the cache body is replaced between preflight and inference,
    the re-pin must fail the item rather than silently infer different bytes.
    """
    from hashlib import sha256

    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.oaf_corpus_run import run_oaf_corpus
    from src.benchmark.reference_set_manifest import (
        LoadedReferenceSetManifest,
        LoadedReferenceSetRow,
        ReferenceSetRowView,
    )
    from src.benchmark.reference_timing import NativeReferenceEvent
    from src.benchmark.reference_timing_manifest import LoadedReferenceTimingManifest

    # Real verified cache body for simfile 10's source audio.
    original_body = _source_wav_bytes(frames=44100)
    audio_digest = sha256(original_body).hexdigest()
    cache_dir = tmp_path / "cache"
    cache_body_path = cache_dir / "sha256" / audio_digest[:2] / audio_digest
    cache_body_path.parent.mkdir(parents=True)
    cache_body_path.write_bytes(original_body)

    # A different valid WAV of identical byte length: same size, different digest.
    # Without the per-item re-pin, librosa would load this and infer wrong bytes
    # while the run row still claims the original HPA-323 digest.
    corrupt_body = _source_wav_bytes_with_value(1.0, frames=44100)
    assert len(corrupt_body) == len(original_body)
    assert sha256(corrupt_body).hexdigest() != audio_digest

    source_row = {
        "selected_chart_key": "10/chart.dtx",
        "selected_chart_content_hash": SHA_A,
        "source_audio_key": "10/audio.wav",
        "source_audio_content_hash": audio_digest,
        "source_endpoint_sha256": SHA_A,
        "source_bucket": "simfile-dtx",
        "objects": [
            {
                "key": "10/audio.wav",
                "size": len(original_body),
                "etag": '"etag-audio"',
                "etag_is_weak": False,
                "last_modified": "2026-08-14T00:00:00Z",
                "content_type": "audio/wav",
                "cache_status": "verified",
                "sha256": audio_digest,
                "cache_path": f"sha256/{audio_digest[:2]}/{audio_digest}",
            }
        ],
    }
    reference_manifest = LoadedReferenceSetManifest(
        manifest_sha256=SHA_A,
        corpus_version="sha256:" + SHA_B,
        source_reference_timing_manifest_sha256=SHA_B,
        source_reference_timing_version="sha256:" + SHA_A,
        rows=(
            LoadedReferenceSetRow(
                source_row=source_row,
                view=ReferenceSetRowView(
                    simfile_id=10,
                    eligibility_status="eligible",
                    eligibility_reason_codes=(),
                    eligibility_warnings=(),
                    mapped_event_count=0,
                    common_scored_event_count=0,
                    ignored_event_count=0,
                    unmapped_event_count=0,
                    duplicate_common_event_count=0,
                ),
            ),
        ),
    )
    timing_manifest = LoadedReferenceTimingManifest(
        manifest_sha256=SHA_B,
        corpus_version="sha256:" + SHA_A,
        rows=(),
    )
    monkeypatch.setattr(run_module, "load_reference_set_manifest", lambda _: reference_manifest)
    monkeypatch.setattr(run_module, "load_reference_timing_manifest", lambda _: timing_manifest)
    monkeypatch.setattr(
        run_module,
        "preflight_reference_mappings",
        lambda *_a, **_kw: {
            10: map_reference_events(
                (
                    NativeReferenceEvent(
                        simfile_id=10,
                        selected_chart_key="10/chart.dtx",
                        selected_chart_content_hash=SHA_A,
                        source_audio_key="10/audio.wav",
                        source_audio_content_hash=audio_digest,
                        source_order=0,
                        measure=1,
                        position=0.0,
                        lane_id="13",
                        note_id="kick-0",
                        chart_time_sec=0.5,
                        audio_time_sec=0.5,
                    ),
                )
            )
        },
    )

    # Wrap the real resolver: after the bulk preflight (load_body=False) succeeds,
    # replace the cache body so the per-item re-pin (load_body=True) sees mutated
    # bytes. _materialize_oaf_full_mix is left real so a missing re-pin would
    # actually infer the wrong bytes against the original digest.
    real_resolve = run_module.resolve_source_audio
    corrupted = False

    def resolve_wrapper(*args, **kwargs):
        nonlocal corrupted
        resolved = real_resolve(*args, **kwargs)
        if not kwargs.get("load_body", True) and not corrupted:
            cache_body_path.write_bytes(corrupt_body)
            corrupted = True
        return resolved

    monkeypatch.setattr(run_module, "resolve_source_audio", resolve_wrapper)

    descriptor = _descriptor()
    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    # The item must be failed at the source-verification step, not inferred.
    assert outcome.exit_code == 1
    assert outcome.failed_count == 1
    assert outcome.run_path is not None
    snapshot = parse_oaf_corpus_run(outcome.run_path.read_bytes())
    (item,) = snapshot["items"]
    assert item["execution_disposition"] == "failed"
    assert item["runner_failure_code"] == "source_audio_unavailable"

    # No prediction artifact may exist: wrong bytes were never inferred.
    config = load_model_config()
    lock_sha = compute_model_lock_sha256(_model_lock_path())
    inf_sha = inference_config_sha256(build_inference_config(config, descriptor, lock_sha))
    prediction_target = prediction_path(
        tmp_path / "output",
        simfile_id=10,
        source_audio_sha256=audio_digest,
        backend_descriptor_sha256=descriptor.sha256,
        inference_config_sha256=inf_sha,
    )
    assert not prediction_target.exists()

    # The canonical input must not have been materialized from the mutated body.
    canonical_path = outcome.run_path.parent / "inputs" / "10" / "full-mix.wav"
    assert not canonical_path.exists()


def test_run_oaf_corpus_backend_descriptor_type_invalid_is_fatal_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    class BadDescriptorBackend:
        def descriptor(self) -> object:
            return "not-a-descriptor"

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

        def close(self) -> None:
            return None

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: BadDescriptorBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_run_oaf_corpus_backend_descriptor_identity_changed_is_fatal_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()
    changed_descriptor = replace(descriptor, payload={**descriptor.payload, "backend_id": "other"})

    class ChangedDescriptorBackend:
        def descriptor(self) -> BackendDescriptor:
            return changed_descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

        def close(self) -> None:
            return None

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: ChangedDescriptorBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_run_oaf_corpus_native_prediction_invalid_type_is_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    class BadPredictionBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> object:
            return "not-a-prediction"

        def close(self) -> None:
            return None

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: BadPredictionBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.exit_code == 1
    assert outcome.failed_count == 1


def test_run_oaf_corpus_prediction_publish_failure_marks_item_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.artifact_io import ArtifactPublicationError
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    def fail_publish(*_a, **_kw):
        raise ArtifactPublicationError("publish failed")

    monkeypatch.setattr(run_module, "publish_prediction_artifact", fail_publish)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.exit_code == 1
    assert outcome.failed_count == 1


def test_run_oaf_corpus_base_exception_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A BaseException (e.g. KeyboardInterrupt) during materialization propagates."""
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    def fail_materialize(*_a, **_kw):
        raise KeyboardInterrupt("unexpected interrupt")

    monkeypatch.setattr(run_module, "_materialize_oaf_full_mix", fail_materialize)

    with pytest.raises(KeyboardInterrupt):
        run_oaf_corpus(
            _request(tmp_path),
            backend_factory=lambda **_: _HealthyBackend(descriptor),
            perf_counter=lambda: 0.0,
            clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
        )


def test_run_oaf_corpus_close_base_exception_is_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A BaseException during close is captured and re-raised after scoring."""
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    class CrashCloseBackend(_HealthyBackend):
        def close(self) -> None:
            raise KeyboardInterrupt("close crash")

    with pytest.raises(KeyboardInterrupt, match="close crash"):
        run_oaf_corpus(
            _request(tmp_path),
            backend_factory=lambda **_: CrashCloseBackend(descriptor),
            perf_counter=lambda: 0.0,
            clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
        )


def test_run_oaf_corpus_final_snapshot_parse_failure_returns_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    real_checkpoint = run_module._write_snapshot_checkpoint

    def corrupt_final_checkpoint(run_path, header, items, **kwargs):
        real_checkpoint(run_path, header, items, **kwargs)
        # Corrupt the run.json after the final checkpoint write
        if kwargs.get("overall_status") is not None:
            run_path.write_bytes(b"corrupt\n")

    monkeypatch.setattr(run_module, "_write_snapshot_checkpoint", corrupt_final_checkpoint)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_run_oaf_corpus_non_resume_existing_prediction_is_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-existing prediction artifact without resume marks the item failed as a conflict."""
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    # Pre-create the prediction artifact at the expected path
    audio = _canonical_audio(source_audio_id="10/audio.wav")
    mapped = _mapped_prediction(audio=audio, descriptor=descriptor)
    artifact_bytes = render_prediction_artifact(mapped)
    pred_path = _real_prediction_path(tmp_path, descriptor)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_bytes(artifact_bytes)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.failed_count == 1


def test_run_oaf_corpus_resume_existing_unreadable_prediction_is_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume with an unreadable prediction artifact marks the item failed."""
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    # First run to create a valid run.json with a prediction
    first = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert first.exit_code == 0

    # Make the prediction artifact unreadable via OS error
    pred_path = _real_prediction_path(tmp_path, descriptor)
    assert pred_path.exists()

    real_read = run_module.read_regular_file_no_follow

    def fail_read_for_prediction(path):
        if path == pred_path:
            raise OSError("io error")
        return real_read(path)

    monkeypatch.setattr(run_module, "read_regular_file_no_follow", fail_read_for_prediction)

    outcome = run_oaf_corpus(
        _request(tmp_path, resume=True),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.failed_count == 1


def test_run_oaf_corpus_resume_existing_corrupt_prediction_is_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume with a corrupt prediction artifact marks the item failed."""
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    # First run to create a valid run.json with a prediction
    first = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert first.exit_code == 0

    # Corrupt the prediction artifact
    pred_path = _real_prediction_path(tmp_path, descriptor)
    pred_path.write_bytes(b"corrupt artifact")

    outcome = run_oaf_corpus(
        _request(tmp_path, resume=True),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.failed_count == 1


def test_run_oaf_corpus_resume_artifact_mismatch_run_evidence_is_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume with an artifact that doesn't match persisted run evidence is failed."""
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    # First run to create a valid run.json with a prediction
    first = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert first.exit_code == 0

    # Replace the prediction artifact with a valid one for a different source
    audio = _canonical_audio(source_audio_id="99/audio.wav")
    mapped = _mapped_prediction(audio=audio, descriptor=descriptor)
    pred_path = _real_prediction_path(tmp_path, descriptor)
    pred_path.write_bytes(render_prediction_artifact(mapped))

    outcome = run_oaf_corpus(
        _request(tmp_path, resume=True),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.failed_count == 1


def test_run_oaf_corpus_checkpoint_failure_during_canonical_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint write failure after source resolution is fatal."""
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    real_checkpoint = run_module._write_snapshot_checkpoint

    def fail_after_source_resolution(run_path, header, items, **kwargs):
        items_list = list(items)
        has_source_sha = any("source_audio_sha256" in item for item in items_list)
        has_disposition = any("execution_disposition" in item for item in items_list)
        is_final = kwargs.get("overall_status") is not None
        if has_source_sha and not has_disposition and not is_final:
            raise OSError("checkpoint write failed")
        return real_checkpoint(run_path, header, items_list, **kwargs)

    monkeypatch.setattr(run_module, "_write_snapshot_checkpoint", fail_after_source_resolution)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_run_oaf_corpus_initial_checkpoint_failure_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint write failure during the initial snapshot is fatal."""
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    def fail_checkpoint(*_args, **_kwargs):
        raise OSError("checkpoint write failed")

    monkeypatch.setattr(run_module, "_write_snapshot_checkpoint", fail_checkpoint)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_run_oaf_corpus_source_checkpoint_failure_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint write failure after prediction handling is fatal."""
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    real_checkpoint = run_module._write_snapshot_checkpoint

    def fail_after_prediction_handling(run_path, header, items, **kwargs):
        items_list = list(items)
        has_disposition = any("execution_disposition" in item for item in items_list)
        is_final = kwargs.get("overall_status") is not None
        if has_disposition and not is_final:
            raise OSError("checkpoint write failed")
        return real_checkpoint(run_path, header, items_list, **kwargs)

    monkeypatch.setattr(run_module, "_write_snapshot_checkpoint", fail_after_prediction_handling)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_run_oaf_corpus_final_checkpoint_failure_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint write failure during the final checkpoint is fatal."""
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    real_checkpoint = run_module._write_snapshot_checkpoint

    def fail_final_checkpoint(run_path, header, items, **kwargs):
        if kwargs.get("overall_status") is not None:
            raise OSError("checkpoint write failed")
        return real_checkpoint(run_path, header, items, **kwargs)

    monkeypatch.setattr(run_module, "_write_snapshot_checkpoint", fail_final_checkpoint)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_run_oaf_corpus_checkpoint_failure_during_canonical_failure_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint failure during canonical input failure handling is fatal."""
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    def fail_materialize(*_a, **_kw):
        raise OSError("materialization failed")

    monkeypatch.setattr(run_module, "_materialize_oaf_full_mix", fail_materialize)

    real_checkpoint = run_module._write_snapshot_checkpoint

    def fail_after_canonical_failure(run_path, header, items, **kwargs):
        items_list = list(items)
        has_disposition = any("execution_disposition" in item for item in items_list)
        is_final = kwargs.get("overall_status") is not None
        if has_disposition and not is_final:
            raise OSError("checkpoint write failed")
        return real_checkpoint(run_path, header, items_list, **kwargs)

    monkeypatch.setattr(run_module, "_write_snapshot_checkpoint", fail_after_canonical_failure)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_run_oaf_corpus_clock_failure_during_completion_is_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clock failure during completion timestamp downgrades to failed."""
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    call_count = 0

    def clock():
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise RuntimeError("clock broke during completion")
        return datetime(2026, 8, 14, tzinfo=timezone.utc)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=clock,
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_run_oaf_corpus_scoring_failure_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scoring failure returns a fatal outcome."""
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    def fail_score(*_a, **_kw):
        raise RuntimeError("scoring crashed")

    monkeypatch.setattr(run_module, "score_cohort", fail_score)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_run_oaf_corpus_worker_start_failure_is_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker start failure (non-OafBackendError) is wrapped and poisons the run."""
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)

    def crash_backend_factory(**_kwargs):
        raise OSError("worker subprocess failed to start")

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=crash_backend_factory,
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.failed_count == 1


def test_run_oaf_corpus_generic_exception_during_prediction_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generic Exception during prediction handling is fatal."""
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    class WeirdError(Exception):
        pass

    class CrashBackend(_HealthyBackend):
        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            raise WeirdError("unexpected error")

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: CrashBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_run_oaf_corpus_prediction_artifact_invalid_exception_is_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A TypeError during prediction publishing marks the item failed."""
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    def fail_publish(*_a, **_kw):
        raise TypeError("publish crashed")

    monkeypatch.setattr(run_module, "publish_prediction_artifact", fail_publish)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: _HealthyBackend(descriptor),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.failed_count == 1


def test_run_oaf_corpus_unexpected_exception_during_scoring_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected exception during scoring with interrupted_error re-raises."""
    import src.benchmark.oaf_corpus_run as run_module
    from src.benchmark.oaf_corpus_run import run_oaf_corpus

    _install_run_seams(monkeypatch, tmp_path)
    descriptor = _descriptor()

    class CrashCloseBackend(_HealthyBackend):
        def close(self) -> None:
            raise KeyboardInterrupt("close crash")

    def fail_score(*_a, **_kw):
        raise RuntimeError("scoring crashed after close crash")

    monkeypatch.setattr(run_module, "score_cohort", fail_score)

    with pytest.raises(KeyboardInterrupt, match="close crash"):
        run_oaf_corpus(
            _request(tmp_path),
            backend_factory=lambda **_: CrashCloseBackend(descriptor),
            perf_counter=lambda: 0.0,
            clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
        )
