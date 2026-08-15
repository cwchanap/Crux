from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
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
    strict_json_loads,
)
from src.benchmark.backends.base import CanonicalAudio
from src.benchmark.backends.oaf import OAF_ADAPTER_REVISION
from src.benchmark.cohort_scoring import (
    COHORT_FAILURE_REASONS,
    CohortIdentity,
    validate_cohort_items,
)
from src.benchmark.corpus_cache import CacheIndexEntry, CacheIndexStore
from src.benchmark.corpus_manifest import render_manifest
from src.benchmark.oaf_corpus_run import (
    OAF_BACKEND_ERROR_POLICY,
    OAF_CANONICALIZATION_REVISION,
    OAF_CORPUS_REQUEST_TIMEOUT_SECONDS,
    OAF_CORPUS_RUN_SCHEMA,
    OAF_FULL_MIX_INPUT_VIEW_ID,
    OAF_INFERENCE_CONFIG_SCHEMA,
    OAF_WORKER_CLOSE_TIMEOUT_SECONDS,
    RUNNER_FAILURE_TO_COHORT_REASON,
    OafCorpusRunOutcome,
    OafCorpusRunRequest,
    ResolvedSourceAudio,
    _cohort_item_from_run_row,
    _materialize_oaf_full_mix,
    _project_runtime,
    _resolve_source_audio,
    _validate_scope,
    build_inference_config,
    build_run_id,
    classify_oaf_backend_error,
    compute_model_lock_sha256,
    inference_config_sha256,
    parse_oaf_corpus_run,
    prediction_path,
    render_oaf_corpus_run,
    write_oaf_corpus_run,
)
from src.benchmark.r2_corpus_models import RemoteObject
from src.benchmark.reference_set import ReferenceMappingResult, map_reference_events
from src.benchmark.reference_set_manifest import (
    LoadedReferenceSetManifest,
    LoadedReferenceSetRow,
    ReferenceSetRequest,
    ReferenceSetRowView,
    load_reference_set_manifest,
    preflight_reference_mappings,
    run_reference_set,
)
from src.benchmark.reference_timing import NativeReferenceEvent, render_reference_events
from src.benchmark.reference_timing_manifest import (
    LoadedReferenceTimingManifest,
    LoadedReferenceTimingRow,
    ReferenceTimingRowView,
    load_reference_timing_manifest,
)
from src.benchmark.taxonomy import OAF_PREDICTION_MAP_ID

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
COMMIT = "d" * 40


def test_constants_and_public_dataclasses_are_frozen_contracts() -> None:
    assert OAF_CORPUS_RUN_SCHEMA == "crux.oaf-corpus-run/v1"
    assert OAF_INFERENCE_CONFIG_SCHEMA == "crux.oaf-inference-config/v1"
    assert OAF_FULL_MIX_INPUT_VIEW_ID == "crux.oaf-full-mix-mono44k1-pcm16/v1"
    assert OAF_CANONICALIZATION_REVISION == "librosa-soxr-hq-mono44k1-soundfile-pcm16/v1"
    assert OAF_CORPUS_REQUEST_TIMEOUT_SECONDS == 3600.0
    assert OAF_WORKER_CLOSE_TIMEOUT_SECONDS == 30.0

    request = OafCorpusRunRequest(
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
    outcome = OafCorpusRunOutcome(
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
    assert outcome.success_count == 1


def test_project_runtime_excludes_resume_without_retained_timing() -> None:
    runtime = _project_runtime(
        (
            {
                "simfile_id": 10,
                "execution_disposition": "inferred",
                "wall_time_sec": 2.0,
                "source_duration_sec": 4.0,
            },
            {"simfile_id": 20, "execution_disposition": "resumed"},
        ),
        eligible_audio_durations=(4.0, 6.0),
    )

    assert runtime["measured_wall_time_sec"] == pytest.approx(2.0)
    assert runtime["measured_audio_duration_sec"] == pytest.approx(4.0)
    assert runtime["aggregate_rtf"] == pytest.approx(0.5)
    assert runtime["projected_full_wall_time_sec"] == pytest.approx(5.0)
    assert runtime["eligible_audio_duration_sec"] == pytest.approx(10.0)
    assert runtime["eligible_audio_duration_coverage_count"] == 2


def test_project_runtime_reports_missing_duration_without_projection() -> None:
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
    assert runtime["eligible_audio_duration_sec"] is None
    assert runtime["eligible_audio_duration_coverage_count"] == 1
    assert runtime["eligible_audio_duration_total_count"] == 2


def test_project_runtime_counts_zero_elapsed_inference() -> None:
    runtime = _project_runtime(
        (
            {
                "simfile_id": 10,
                "execution_disposition": "inferred",
                "wall_time_sec": 0.0,
                "source_duration_sec": 4.0,
            },
        ),
        eligible_audio_durations=(4.0,),
    )

    assert runtime["measured_wall_time_sec"] == pytest.approx(0.0)
    assert runtime["measured_audio_duration_sec"] == pytest.approx(4.0)
    assert runtime["aggregate_rtf"] == pytest.approx(0.0)
    assert runtime["projected_full_wall_time_sec"] == pytest.approx(0.0)


def _cohort_test_identity() -> CohortIdentity:
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


def _cohort_test_mapping() -> ReferenceMappingResult:
    return map_reference_events(
        (
            NativeReferenceEvent(
                simfile_id=10,
                selected_chart_key="10/chart.dtx",
                selected_chart_content_hash=SHA_A,
                source_audio_key="10/audio.wav",
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


def test_cohort_item_from_run_row_pins_non_success_reference_and_coverage_shapes() -> None:
    identity = _cohort_test_identity()
    mapping = _cohort_test_mapping()

    failed = _cohort_item_from_run_row(
        identity,
        {
            "simfile_id": 10,
            "execution_disposition": "failed",
            "runner_failure_code": "inference_failed",
        },
        mapping,
        output_dir=Path("output"),
    )
    assert failed.status == "failed"
    assert failed.failure_reason == "inference_failed"
    assert failed.reference_events
    validate_cohort_items(identity, (failed,))

    skipped = _cohort_item_from_run_row(
        identity,
        {
            "simfile_id": 10,
            "execution_disposition": "skipped",
            "runner_failure_code": "explicitly_skipped",
        },
        mapping,
        output_dir=Path("output"),
    )
    assert skipped.status == "skipped"
    assert skipped.failure_reason == "explicitly_skipped"
    validate_cohort_items(identity, (skipped,))

    quarantined = _cohort_item_from_run_row(
        identity,
        {
            "simfile_id": 10,
            "execution_disposition": "quarantined",
            "runner_failure_code": "reference_quarantined",
        },
        mapping,
        output_dir=Path("output"),
    )
    assert quarantined.status == "quarantined"
    assert quarantined.failure_reason == "reference_quarantined"
    validate_cohort_items(identity, (quarantined,))

    unavailable = _cohort_item_from_run_row(
        identity,
        {
            "simfile_id": 10,
            "execution_disposition": "quarantined",
            "runner_failure_code": "reference_quarantined",
        },
        None,
        output_dir=Path("output"),
    )
    assert unavailable.reference_events == ()
    assert unavailable.coverage.reference_native_event_count == 0
    assert unavailable.coverage.reference_common_event_count == 0
    validate_cohort_items(identity, (unavailable,))


def test_cohort_item_from_run_row_adapts_missing_prediction_to_failed() -> None:
    item = _cohort_item_from_run_row(
        _cohort_test_identity(),
        {"simfile_id": 10, "execution_disposition": "inferred"},
        None,
        output_dir=Path("output"),
    )

    assert item.status == "failed"
    assert item.failure_reason == "prediction_missing"
    validate_cohort_items(_cohort_test_identity(), (item,))


def test_resolved_source_audio_preserves_authoritative_identity() -> None:
    resolved = ResolvedSourceAudio(
        path=Path("cache/sha256/aa/" + SHA_A),
        source_audio_id="42/audio.wav",
        source_audio_sha256=SHA_A,
        duration_sec=12.5,
    )

    assert resolved.path == Path("cache/sha256/aa/" + SHA_A)
    assert resolved.source_audio_id == "42/audio.wav"
    assert resolved.source_audio_sha256 == SHA_A
    assert resolved.duration_sec == 12.5


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


def _cache_body(cache_dir: Path, content: bytes) -> tuple[str, Path]:
    digest = sha256(content).hexdigest()
    path = cache_dir / "sha256" / digest[:2] / digest
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return digest, path


def _source_remote(
    *,
    key: str = "42/bgm.wav",
    content: bytes,
    cache_status: str = "verified",
    digest: str | None = None,
    cache_path: str | None = None,
) -> RemoteObject:
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


def test_resolve_source_audio_uses_carried_verified_remote_and_probes_duration(
    tmp_path: Path,
) -> None:
    content = _source_wav_bytes(frames=22050)
    digest, path = _cache_body(tmp_path, content)
    remote = _source_remote(
        content=content,
        digest=digest,
        cache_path=path.relative_to(tmp_path).as_posix(),
    )

    resolved = _resolve_source_audio(
        remote,
        tmp_path,
        CacheIndexStore(tmp_path, {}),
        source_endpoint_sha256=SHA_B,
        source_bucket="simfile-dtx",
        source_audio_content_hash=digest,
    )

    assert resolved.path == path
    assert resolved.source_audio_id == "42/bgm.wav"
    assert resolved.source_audio_sha256 == digest
    assert resolved.duration_sec == pytest.approx(0.5)


def test_resolve_source_audio_requires_hpa323_timing_hash(
    tmp_path: Path,
) -> None:
    content = _source_wav_bytes()
    digest, path = _cache_body(tmp_path, content)
    remote = _source_remote(
        content=content,
        digest=digest,
        cache_path=path.relative_to(tmp_path).as_posix(),
    )

    with pytest.raises(ValueError, match="source_audio_content_hash is required"):
        _resolve_source_audio(
            remote,
            tmp_path,
            CacheIndexStore(tmp_path, {}),
            source_endpoint_sha256=SHA_B,
            source_bucket="simfile-dtx",
        )


def test_resolve_source_audio_rehydrates_matching_stale_cache_index_entry(
    tmp_path: Path,
) -> None:
    content = _source_wav_bytes(frames=44100)
    digest, path = _cache_body(tmp_path, content)
    remote = _source_remote(content=content, cache_status="not_selected")
    entry = CacheIndexEntry(
        source_endpoint_sha256=SHA_B,
        bucket="simfile-dtx",
        key=remote.key,
        etag=remote.etag,
        etag_is_weak=remote.etag_is_weak,
        size=remote.size,
        last_modified="2026-08-14T00:00:00Z",
        sha256=digest,
        cache_path=path.relative_to(tmp_path).as_posix(),
    )

    resolved = _resolve_source_audio(
        remote,
        tmp_path,
        CacheIndexStore(tmp_path, {(SHA_B, "simfile-dtx", remote.key): entry}),
        source_endpoint_sha256=SHA_B,
        source_bucket="simfile-dtx",
        source_audio_content_hash=digest,
    )

    assert resolved.path == path
    assert resolved.source_audio_sha256 == digest
    assert resolved.duration_sec == pytest.approx(1.0)


def test_resolve_source_audio_rejects_changed_remote_identity(tmp_path: Path) -> None:
    content = _source_wav_bytes()
    digest, path = _cache_body(tmp_path, content)
    remote = _source_remote(content=content, cache_status="not_selected")
    entry = CacheIndexEntry(
        source_endpoint_sha256=SHA_B,
        bucket="simfile-dtx",
        key=remote.key,
        etag='"different-etag"',
        etag_is_weak=False,
        size=remote.size,
        last_modified="2026-08-14T00:00:00Z",
        sha256=digest,
        cache_path=path.relative_to(tmp_path).as_posix(),
    )

    with pytest.raises(ValueError, match="verified source audio unavailable"):
        _resolve_source_audio(
            remote,
            tmp_path,
            CacheIndexStore(tmp_path, {(SHA_B, "simfile-dtx", remote.key): entry}),
            source_endpoint_sha256=SHA_B,
            source_bucket="simfile-dtx",
            source_audio_content_hash=digest,
        )


@pytest.mark.parametrize("body_state", ["missing", "corrupt"])
def test_resolve_source_audio_rejects_missing_or_corrupt_cache_body(
    tmp_path: Path,
    body_state: str,
) -> None:
    content = _source_wav_bytes()
    digest = sha256(content).hexdigest()
    path = tmp_path / "sha256" / digest[:2] / digest
    path.parent.mkdir(parents=True)
    if body_state == "corrupt":
        path.write_bytes(b"not the source body")
    remote = _source_remote(content=content, cache_status="not_selected")
    entry = CacheIndexEntry(
        source_endpoint_sha256=SHA_B,
        bucket="simfile-dtx",
        key=remote.key,
        etag=remote.etag,
        etag_is_weak=remote.etag_is_weak,
        size=len(content),
        last_modified="2026-08-14T00:00:00Z",
        sha256=digest,
        cache_path=path.relative_to(tmp_path).as_posix(),
    )

    with pytest.raises(ValueError, match="verified source audio unavailable"):
        _resolve_source_audio(
            remote,
            tmp_path,
            CacheIndexStore(tmp_path, {(SHA_B, "simfile-dtx", remote.key): entry}),
            source_endpoint_sha256=SHA_B,
            source_bucket="simfile-dtx",
            source_audio_content_hash=digest,
        )


def test_preflight_published_eligible_artifact_deletion_is_fatal(tmp_path: Path) -> None:
    timing_path, reference_path, event_path = _published_reference_preflight_fixture(tmp_path)
    timing_manifest = load_reference_timing_manifest(timing_path)
    reference_manifest = load_reference_set_manifest(reference_path)
    timing_output_root = timing_path.parent.parent
    simfile_id = reference_manifest.rows[0].view.simfile_id

    before_deletion = preflight_reference_mappings(
        reference_manifest,
        timing_manifest,
        timing_output_root=timing_output_root,
    )
    assert isinstance(before_deletion[simfile_id], ReferenceMappingResult)

    event_path.unlink()

    with pytest.raises(ValueError, match="eligible reference event artifact invalid"):
        preflight_reference_mappings(
            reference_manifest,
            timing_manifest,
            timing_output_root=timing_output_root,
        )


def test_resolve_source_audio_rejects_digest_mismatch_against_timing_manifest(
    tmp_path: Path,
) -> None:
    content = _source_wav_bytes()
    digest, path = _cache_body(tmp_path, content)
    remote = _source_remote(
        content=content,
        digest=digest,
        cache_path=path.relative_to(tmp_path).as_posix(),
    )

    with pytest.raises(ValueError, match="digest does not match"):
        _resolve_source_audio(
            remote,
            tmp_path,
            CacheIndexStore(tmp_path, {}),
            source_endpoint_sha256=SHA_B,
            source_bucket="simfile-dtx",
            source_audio_content_hash=SHA_C,
        )


def test_resolve_source_audio_duration_probe_failure_is_item_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _source_wav_bytes()
    digest, path = _cache_body(tmp_path, content)
    remote = _source_remote(
        content=content,
        digest=digest,
        cache_path=path.relative_to(tmp_path).as_posix(),
    )
    import src.benchmark.oaf_corpus_run as run_module

    def fail_probe(_: Path):
        raise OSError("unreadable source audio")

    monkeypatch.setattr(run_module, "inspect_source_audio", fail_probe)

    with pytest.raises(OSError, match="unreadable source audio"):
        _resolve_source_audio(
            remote,
            tmp_path,
            CacheIndexStore(tmp_path, {}),
            source_endpoint_sha256=SHA_B,
            source_bucket="simfile-dtx",
            source_audio_content_hash=digest,
        )

    assert RUNNER_FAILURE_TO_COHORT_REASON["source_audio_decode_failed"] == "inference_failed"


def test_materialize_oaf_full_mix_uses_pinned_resampling_and_canonical_wav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.wav"
    sf.write(
        source_path,
        np.zeros((22050, 2), dtype=np.float32),
        22050,
        format="WAV",
        subtype="PCM_16",
    )
    source_digest = sha256(source_path.read_bytes()).hexdigest()
    source = ResolvedSourceAudio(
        path=source_path,
        source_audio_id="42/bgm.wav",
        source_audio_sha256=source_digest,
        duration_sec=1.0,
    )
    input_root = tmp_path / "inputs"
    output_path = input_root / "42" / "full-mix.wav"
    config = load_model_config()
    import src.benchmark.oaf_corpus_run as run_module

    original_load = run_module.librosa.load
    original_write = run_module.soundfile.write
    load_calls: list[tuple[Path, dict[str, object]]] = []
    write_calls: list[tuple[Path, int, dict[str, object]]] = []

    def wrapped_load(path: Path, **kwargs: object):
        load_calls.append((path, kwargs))
        return original_load(path, **kwargs)

    def wrapped_write(path: Path, data: object, samplerate: int, **kwargs: object) -> None:
        write_calls.append((path, samplerate, kwargs))
        original_write(path, data, samplerate, **kwargs)

    monkeypatch.setattr(run_module.librosa, "load", wrapped_load)
    monkeypatch.setattr(run_module.soundfile, "write", wrapped_write)

    audio = _materialize_oaf_full_mix(
        source,
        output_path=output_path,
        input_root=input_root,
        config=config,
    )

    assert isinstance(audio, CanonicalAudio)
    assert load_calls == [
        (
            source_path,
            {"sr": 44100, "mono": True, "res_type": "soxr_hq"},
        )
    ]
    assert write_calls == [
        (
            output_path,
            44100,
            {"format": "WAV", "subtype": "PCM_16"},
        )
    ]
    assert audio.path == output_path
    assert audio.source_audio_id == source.source_audio_id
    assert audio.source_audio_sha256 == source_digest
    assert audio.input_audio_sha256 == sha256(output_path.read_bytes()).hexdigest()
    assert audio.sample_rate == 44100
    assert audio.channel_count == 1
    assert audio.audio_frame_count == 44100


def test_materialize_uses_verified_source_bytes_after_cache_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_content = _source_wav_bytes(frames=22050)
    digest, source_path = _cache_body(tmp_path, original_content)
    remote = _source_remote(
        content=original_content,
        digest=digest,
        cache_path=source_path.relative_to(tmp_path).as_posix(),
    )
    resolved = _resolve_source_audio(
        remote,
        tmp_path,
        CacheIndexStore(tmp_path, {}),
        source_endpoint_sha256=SHA_B,
        source_bucket="simfile-dtx",
        source_audio_content_hash=digest,
    )
    source_path.write_bytes(_source_wav_bytes(frames=44100))
    assert getattr(resolved, "content", None) == original_content

    import src.benchmark.oaf_corpus_run as run_module

    original_load = run_module.librosa.load
    load_inputs: list[object] = []

    def wrapped_load(source: object, **kwargs: object):
        load_inputs.append(source)
        if hasattr(source, "read"):
            source.seek(0)
            assert source.read() == original_content
            source.seek(0)
        else:
            assert source.read_bytes() == original_content  # type: ignore[union-attr]
        return original_load(source, **kwargs)

    monkeypatch.setattr(run_module.librosa, "load", wrapped_load)
    output_path = tmp_path / "inputs" / "42" / "full-mix.wav"
    audio = _materialize_oaf_full_mix(
        resolved,
        output_path=output_path,
        input_root=tmp_path / "inputs",
        config=load_model_config(),
    )

    assert isinstance(audio, CanonicalAudio)
    assert len(load_inputs) == 1
    assert hasattr(load_inputs[0], "read")


def _reference_preflight_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    LoadedReferenceSetManifest,
    LoadedReferenceTimingManifest,
    Path,
]:
    source_audio_key = "42/bgm.wav"
    event = NativeReferenceEvent(
        simfile_id=42,
        selected_chart_key="42/real.dtx",
        selected_chart_content_hash=SHA_A,
        source_audio_key=source_audio_key,
        source_audio_content_hash=SHA_B,
        source_order=0,
        measure=1,
        position=0.0,
        lane_id="11",
        note_id="01",
        chart_time_sec=1.0,
        audio_time_sec=0.5,
    )
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
    }
    timing_view = ReferenceTimingRowView(
        simfile_id=42,
        corpus_version="sha256:" + SHA_C,
        timing_status="ready",
        timing_reason_codes=(),
        timing_warnings=(),
        reference_events_cache_path=relative_artifact,
        source_audio_key=source_audio_key,
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


def _published_reference_preflight_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    golden_path = Path(__file__).parent / "schema_goldens/crux.reference-timing-manifest-v1.jsonl"
    ready_row = json.loads(golden_path.read_text(encoding="utf-8").splitlines()[0])
    event = NativeReferenceEvent(
        simfile_id=ready_row["simfile_id"],
        selected_chart_key=ready_row["selected_chart_key"],
        selected_chart_content_hash=ready_row["selected_chart_content_hash"],
        source_audio_key=ready_row["source_audio_key"],
        source_audio_content_hash=ready_row["source_audio_content_hash"],
        source_order=0,
        measure=1,
        position=0.0,
        lane_id="13",
        note_id="01",
        chart_time_sec=1.0,
        audio_time_sec=0.5,
    )
    event_content = render_reference_events((event,))
    event_hash = sha256(event_content).hexdigest()
    ready_row["reference_events_cache_path"] = f"events/{event_hash}.jsonl"
    rendered_timing = render_manifest(
        ({key: value for key, value in ready_row.items() if key != "corpus_version"},)
    )
    timing_path = tmp_path / "timing" / "manifests" / f"{rendered_timing.manifest_sha256}.jsonl"
    timing_path.parent.mkdir(parents=True)
    timing_path.write_bytes(rendered_timing.content)
    event_path = timing_path.parent.parent / ready_row["reference_events_cache_path"]
    event_path.parent.mkdir(parents=True)
    event_path.write_bytes(event_content)

    reference_outcome = run_reference_set(
        ReferenceSetRequest(timing_path, tmp_path / "reference-set")
    )
    assert reference_outcome.manifest is not None
    return timing_path, reference_outcome.manifest.path, event_path


def test_preflight_published_eligible_artifact_corruption_is_fatal(tmp_path: Path) -> None:
    timing_path, reference_path, event_path = _published_reference_preflight_fixture(tmp_path)
    loaded_timing = load_reference_timing_manifest(timing_path)
    loaded_reference = load_reference_set_manifest(reference_path)
    event_path.write_bytes(b"corrupt event artifact")

    with pytest.raises(ValueError, match="eligible reference event artifact invalid"):
        preflight_reference_mappings(
            loaded_reference,
            loaded_timing,
            timing_output_root=timing_path.parent.parent,
        )


def test_preflight_reference_mappings_reconstructs_eligible_artifact(
    tmp_path: Path,
) -> None:
    root, reference_manifest, timing_manifest, _ = _reference_preflight_fixture(tmp_path)

    mappings = preflight_reference_mappings(
        reference_manifest,
        timing_manifest,
        timing_output_root=root,
    )

    assert isinstance(mappings[42], ReferenceMappingResult)
    assert mappings[42] is not None
    assert len(mappings[42].common_events) == 1


@pytest.mark.parametrize(
    ("field", "reference_value", "timing_value"),
    [
        ("source_endpoint_sha256", SHA_A, SHA_B),
        ("source_bucket", "simfile-dtx", "other-bucket"),
    ],
)
def test_preflight_binds_complete_remote_source_identity_lineage(
    tmp_path: Path,
    field: str,
    reference_value: str,
    timing_value: str,
) -> None:
    root, reference_manifest, timing_manifest, _ = _reference_preflight_fixture(tmp_path)
    reference_source = {
        **reference_manifest.rows[0].source_row,
        "source_endpoint_sha256": SHA_A,
        "source_bucket": "simfile-dtx",
    }
    timing_source = {
        **timing_manifest.rows[0].source_row,
        "source_endpoint_sha256": SHA_A,
        "source_bucket": "simfile-dtx",
    }
    reference_source[field] = reference_value
    timing_source[field] = timing_value
    reference_manifest = replace(
        reference_manifest,
        rows=(replace(reference_manifest.rows[0], source_row=reference_source),),
    )
    timing_manifest = replace(
        timing_manifest,
        rows=(replace(timing_manifest.rows[0], source_row=timing_source),),
    )

    with pytest.raises(ValueError, match="eligible reference identity"):
        preflight_reference_mappings(
            reference_manifest,
            timing_manifest,
            timing_output_root=root,
        )


def test_preflight_reference_mappings_fails_before_inference_for_corrupt_eligible_artifact(
    tmp_path: Path,
) -> None:
    root, reference_manifest, timing_manifest, artifact = _reference_preflight_fixture(tmp_path)
    artifact.write_bytes(b"corrupt event artifact")

    with pytest.raises(ValueError, match="eligible reference event artifact invalid"):
        preflight_reference_mappings(
            reference_manifest,
            timing_manifest,
            timing_output_root=root,
        )


@pytest.mark.parametrize(
    "reason", ["upstream_reference_unavailable", "reference_event_artifact_invalid"]
)
def test_preflight_reference_mappings_quarantined_missing_artifact_is_not_fatal(
    tmp_path: Path,
    reason: str,
) -> None:
    root, reference_manifest, timing_manifest, artifact = _reference_preflight_fixture(tmp_path)
    artifact.unlink()
    source_row = reference_manifest.rows[0].source_row
    quarantine_view = ReferenceSetRowView(
        simfile_id=42,
        eligibility_status="quarantined",
        eligibility_reason_codes=(reason,),  # type: ignore[arg-type]
        eligibility_warnings=(),
        mapped_event_count=0,
        common_scored_event_count=0,
        ignored_event_count=0,
        unmapped_event_count=0,
        duplicate_common_event_count=0,
    )
    quarantine_manifest = LoadedReferenceSetManifest(
        manifest_sha256=SHA_B,
        corpus_version=reference_manifest.corpus_version,
        source_reference_timing_manifest_sha256=reference_manifest.source_reference_timing_manifest_sha256,
        source_reference_timing_version=reference_manifest.source_reference_timing_version,
        rows=(LoadedReferenceSetRow(source_row=source_row, view=quarantine_view),),
    )

    mappings = preflight_reference_mappings(
        quarantine_manifest,
        timing_manifest,
        timing_output_root=root,
    )

    assert mappings == {42: None}


def test_model_lock_hash_is_exact_file_sha256(tmp_path: Path) -> None:
    model_lock = tmp_path / "model.json"
    model_lock.write_bytes(b'{"schema":"test"}\n')

    assert compute_model_lock_sha256(model_lock) == (
        "b3c34c00a38e16700dcf90b345fbd4820bac96cbb0137d459d7856bf88fbcc3e"
    )


def test_inference_config_has_exact_keys_and_deterministic_hash() -> None:
    config = load_model_config()
    descriptor = BackendDescriptor(payload={}, sha256=SHA_A)
    payload = build_inference_config(config, descriptor, SHA_B)

    assert set(payload) == {
        "schema",
        "backend_descriptor_sha256",
        "model_lock_sha256",
        "checkpoint_archive_sha256",
        "adapter_revision",
        "prediction_map_version",
        "input_view_id",
        "canonicalization_revision",
    }
    assert payload == {
        "schema": OAF_INFERENCE_CONFIG_SCHEMA,
        "backend_descriptor_sha256": SHA_A,
        "model_lock_sha256": SHA_B,
        "checkpoint_archive_sha256": config.checkpoint.archive_sha256,
        "adapter_revision": OAF_ADAPTER_REVISION,
        "prediction_map_version": OAF_PREDICTION_MAP_ID,
        "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
        "canonicalization_revision": OAF_CANONICALIZATION_REVISION,
    }
    first_hash = inference_config_sha256(payload)
    assert first_hash == inference_config_sha256(dict(payload))
    assert first_hash != inference_config_sha256({**payload, "input_view_id": "other/v1"})
    assert first_hash != inference_config_sha256({**payload, "adapter_revision": "other/v1"})
    assert first_hash != inference_config_sha256({**payload, "checkpoint_archive_sha256": SHA_C})
    assert first_hash != inference_config_sha256({**payload, "backend_descriptor_sha256": SHA_C})


def test_run_id_is_deterministic_and_binds_reference_and_scope() -> None:
    common = dict(
        reference_manifest_sha256=SHA_A,
        reference_timing_manifest_sha256=SHA_B,
        backend_descriptor_sha256=SHA_C,
        model_lock_sha256=SHA_A,
        checkpoint_archive_sha256=SHA_B,
        inference_config_sha256=SHA_C,
        include_simfile_ids=(30, 10, 10),
        exclude_simfile_ids=(40,),
    )
    first = build_run_id(**common)
    assert first == build_run_id(**common)
    assert first.startswith("oaf-")
    assert first != build_run_id(**{**common, "reference_manifest_sha256": SHA_B})
    assert first != build_run_id(**{**common, "include_simfile_ids": (10, 20)})
    assert first != build_run_id(**{**common, "exclude_simfile_ids": (41,)})


def test_prediction_path_is_source_keyed_and_reference_independent(tmp_path: Path) -> None:
    path = prediction_path(
        tmp_path,
        simfile_id=10,
        source_audio_sha256=SHA_A,
        backend_descriptor_sha256=SHA_B,
        inference_config_sha256=SHA_C,
    )

    assert path == (tmp_path / "predictions" / "10" / SHA_A / SHA_B / f"{SHA_C}.jsonl")
    assert "input_audio_sha256" not in str(path)
    assert "reference_manifest" not in str(path)


def test_scope_preflight_normalizes_ids_and_rejects_unknown_or_overlap() -> None:
    assert _validate_scope((10, 10), (), {10, 20, 30}) == ((10,), ())
    assert _validate_scope((10,), (), {10, 20, 30}) == ((10,), ())
    with pytest.raises(ValueError, match="unknown include"):
        _validate_scope((99,), (), {10, 20, 30})
    with pytest.raises(ValueError, match="unknown exclude"):
        _validate_scope((), (99,), {10, 20, 30})
    with pytest.raises(ValueError, match="overlap"):
        _validate_scope((10,), (10,), {10, 20, 30})


def test_invalid_identity_and_scope_values_fail_closed() -> None:
    config = load_model_config()
    with pytest.raises(ValueError):
        build_inference_config(config, BackendDescriptor(payload={}, sha256="bad"), SHA_B)
    with pytest.raises(ValueError):
        build_run_id(
            reference_manifest_sha256="bad",
            reference_timing_manifest_sha256=SHA_B,
            backend_descriptor_sha256=SHA_C,
            model_lock_sha256=SHA_A,
            checkpoint_archive_sha256=SHA_B,
            inference_config_sha256=SHA_C,
        )
    with pytest.raises(ValueError, match="commit"):
        OafCorpusRunRequest(
            reference_manifest_path=Path("reference.jsonl"),
            timing_manifest_path=Path("timing.jsonl"),
            cache_dir=Path("cache"),
            output_dir=Path("output"),
            crux_commit="not-a-commit",
        )
    with pytest.raises(ValueError):
        _validate_scope((True,), (), {1})


def test_backend_error_policy_is_closed_and_unknown_errors_poison() -> None:
    assert OAF_BACKEND_ERROR_POLICY == {
        "inference_failed": ("inference_failed", "item_local"),
        "invalid_request": ("inference_failed", "item_local"),
        "input_path_invalid": ("canonical_input_failed", "item_local"),
        "native_event_invalid": ("inference_failed", "item_local"),
        "worker_error": ("worker_protocol_failed", "poison"),
        "worker_start_failed": ("backend_unavailable", "poison"),
        "worker_ready_invalid": ("backend_unavailable", "poison"),
        "worker_identity_invalid": ("backend_unavailable", "poison"),
        "worker_response_invalid": ("worker_protocol_failed", "poison"),
        "backend_closed": ("worker_protocol_failed", "poison"),
        "descriptor_invalid": (None, "fatal_preflight"),
        "worker_close_failed": (None, "finalization"),
    }
    assert classify_oaf_backend_error("future_code") == (
        "worker_protocol_failed",
        "poison",
    )
    assert set(RUNNER_FAILURE_TO_COHORT_REASON.values()) <= COHORT_FAILURE_REASONS
    assert RUNNER_FAILURE_TO_COHORT_REASON == {
        "source_audio_unavailable": "inference_failed",
        "source_audio_decode_failed": "inference_failed",
        "canonical_input_failed": "inference_failed",
        "backend_unavailable": "backend_unavailable",
        "worker_protocol_failed": "backend_unavailable",
        "inference_failed": "inference_failed",
        "prediction_artifact_invalid": "prediction_artifact_invalid",
        "prediction_output_conflict": "prediction_artifact_invalid",
        "prediction_publish_failed": "prediction_artifact_invalid",
        "prediction_missing": "prediction_missing",
        "explicitly_skipped": "explicitly_skipped",
    }


def _snapshot() -> dict[str, object]:
    return {
        "schema": OAF_CORPUS_RUN_SCHEMA,
        "run_id": "oaf-" + SHA_A[:16],
        "reference_manifest_sha256": SHA_A,
        "request_timeout_seconds": 3600.0,
        "exclude_simfile_ids": [20],
        "overall_status": "complete",
        "success_count": 1,
        "failed_count": 0,
        "skipped_count": 1,
        "quarantined_count": 0,
        "completed_at": "2026-08-14T00:00:00+00:00",
        "items": [
            {
                "simfile_id": 20,
                "execution_disposition": "skipped",
            },
            {
                "simfile_id": 10,
                "execution_disposition": "inferred",
                "wall_time_sec": 1.25,
                "rtf": 0.5,
            },
        ],
    }


def test_render_normalizes_floats_sorts_items_and_round_trips() -> None:
    content = render_oaf_corpus_run(_snapshot())
    parsed = parse_oaf_corpus_run(content)

    assert parsed["request_timeout_seconds"] == Decimal("3600")
    assert parsed["items"][0]["wall_time_sec"] == Decimal("1.25")
    assert [item["simfile_id"] for item in parsed["items"]] == [10, 20]
    assert render_oaf_corpus_run(parsed) == content


def test_render_quantizes_runtime_projection_fields() -> None:
    snapshot = _snapshot()
    snapshot.update(
        {
            "measured_wall_time_sec": 1.23456789,
            "measured_audio_duration_sec": 2.34567891,
            "aggregate_rtf": 0.52500001,
            "eligible_audio_duration_sec": 10.98765432,
            "projected_full_wall_time_sec": 5.76543219,
        }
    )

    parsed = parse_oaf_corpus_run(render_oaf_corpus_run(snapshot))

    assert parsed["measured_wall_time_sec"] == Decimal("1.234568")
    assert parsed["measured_audio_duration_sec"] == Decimal("2.345679")
    assert parsed["aggregate_rtf"] == Decimal("0.525")
    assert parsed["eligible_audio_duration_sec"] == Decimal("10.987654")
    assert parsed["projected_full_wall_time_sec"] == Decimal("5.765432")


def test_parse_rejects_canonical_but_semantically_unsorted_items() -> None:
    rendered = render_oaf_corpus_run(_snapshot())
    canonical_snapshot = strict_json_loads(rendered)
    assert isinstance(canonical_snapshot, dict)
    items = canonical_snapshot["items"]
    assert isinstance(items, list)
    canonical_snapshot["items"] = list(reversed(items))
    semantically_unsorted = canonical_json_bytes(canonical_snapshot)

    with pytest.raises(StrictJsonError, match="semantically canonical"):
        parse_oaf_corpus_run(semantically_unsorted)


def test_canonical_json_rejects_direct_float_injection() -> None:
    with pytest.raises(StrictJsonError, match="unsupported JSON value"):
        canonical_json_bytes({"wall_time_sec": 1.25})


def test_snapshot_validates_schema_counts_skips_and_completion() -> None:
    invalid_schema = _snapshot()
    invalid_schema["schema"] = "other/v1"
    with pytest.raises(ValueError, match="schema"):
        render_oaf_corpus_run(invalid_schema)

    invalid_counts = _snapshot()
    invalid_counts["success_count"] = 2
    with pytest.raises(ValueError, match="count"):
        render_oaf_corpus_run(invalid_counts)

    invalid_skip = _snapshot()
    invalid_skip["exclude_simfile_ids"] = []
    with pytest.raises(ValueError, match="skipped"):
        render_oaf_corpus_run(invalid_skip)

    incomplete = _snapshot()
    incomplete["overall_status"] = "partial"
    incomplete["completed_at"] = "2026-08-14T00:00:00+00:00"
    with pytest.raises(ValueError, match="completed"):
        render_oaf_corpus_run(incomplete)


@pytest.mark.parametrize("missing_field", ["execution_disposition", "failed_count"])
def test_complete_snapshot_requires_complete_item_accounting(missing_field: str) -> None:
    incomplete = _snapshot()
    if missing_field == "execution_disposition":
        del incomplete["items"][0][missing_field]
    else:
        del incomplete[missing_field]

    with pytest.raises(ValueError, match="complete|count|disposition"):
        render_oaf_corpus_run(incomplete)


def test_complete_snapshot_rejects_null_item_disposition() -> None:
    incomplete = _snapshot()
    incomplete["exclude_simfile_ids"] = []
    incomplete["items"] = [{"simfile_id": 10, "execution_disposition": None}]
    incomplete["success_count"] = 0
    incomplete["failed_count"] = 0
    incomplete["skipped_count"] = 0
    incomplete["quarantined_count"] = 0

    with pytest.raises(ValueError, match="disposition"):
        render_oaf_corpus_run(incomplete)


def test_snapshot_writer_delegates_to_atomic_replace(tmp_path: Path, monkeypatch) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    run_path = tmp_path / "runs" / "run.json"
    run_path.parent.mkdir()
    captured: dict[str, object] = {}

    def fake_atomic_replace(path: Path, content: bytes) -> None:
        captured["path"] = path
        captured["content"] = content

    monkeypatch.setattr(run_module, "atomic_replace_bytes", fake_atomic_replace)
    write_oaf_corpus_run(run_path, _snapshot())

    assert captured == {"path": run_path, "content": render_oaf_corpus_run(_snapshot())}
