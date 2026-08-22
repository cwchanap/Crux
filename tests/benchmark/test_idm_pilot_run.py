from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.benchmark.artifact_io import ArtifactPublicationError
from src.benchmark.backend_identity import canonical_json_bytes, strict_json_loads
from src.benchmark.backends.base import CanonicalAudio
from src.benchmark.cohort_scoring import COHORT_FAILURE_REASONS
from src.benchmark.corpus_cache import ResolvedSourceAudio
from src.benchmark.idm_pilot_run import (
    IDM_FAILURE_TO_COHORT_REASON,
    IDM_FULL_MIX_INPUT_VIEW_ID,
    IDM_PILOT_RUN_SCHEMA,
    IDM_SMOKE_CASE_ORDER,
    IDM_SMOKE_SCHEMA,
    IDM_STEM_INPUT_VIEW_ID,
    IdmFullMixSmokeRequest,
    IdmPilotRunRequest,
    IdmSmokeCase,
    IdmSmokeManifestError,
    build_idm_inference_config,
    build_run_id,
    classify_idm_backend_error,
    idm_inference_config_sha256,
    materialize_idm_full_mix_audio,
    parse_idm_smoke_manifest,
    render_idm_smoke_manifest,
    select_idm_smoke_cases,
)
from src.benchmark.separation_handoff import LoadedSeparationPilotManifest
from tests.benchmark.idm_pilot_fixtures import IDM_MODEL_LOCK_PATH


def _smoke_handoff(*rows: dict[str, object]) -> LoadedSeparationPilotManifest:
    return LoadedSeparationPilotManifest(
        manifest_sha256="a" * 64,
        corpus_version="sha256:" + "b" * 64,
        rows=tuple(rows),
    )


def _smoke_row(
    simfile_id: int,
    duration: float,
    event_count: int,
    *,
    status: str = "success",
) -> dict[str, object]:
    return {
        "simfile_id": simfile_id,
        "source_duration_sec": duration,
        "htdemucs": {
            "status": status,
            "input": {"path": f"inputs/{simfile_id}.wav"},
        },
        "common_reference_event_count": event_count,
    }


def _smoke_mappings(rows: tuple[dict[str, object], ...]) -> dict[int, object]:
    return {
        int(row["simfile_id"]): SimpleNamespace(
            common_events=tuple(range(int(row["common_reference_event_count"])))
        )
        for row in rows
    }


def test_fixed_failure_mapping_is_closed_and_preserves_upstream_stem_semantics() -> None:
    assert set(IDM_FAILURE_TO_COHORT_REASON.values()) <= COHORT_FAILURE_REASONS
    assert IDM_FAILURE_TO_COHORT_REASON["upstream_stem_unavailable"] == "inference_failed"


def test_idm_pilot_request_has_only_explicit_handoff_and_runtime_inputs() -> None:
    names = {field.name for field in fields(IdmPilotRunRequest)}
    assert "source_cache_dir" not in names
    request = IdmPilotRunRequest(
        separation_handoff_path=Path("handoff.jsonl"),
        reference_manifest_path=Path("reference.jsonl"),
        timing_manifest_path=Path("timing.jsonl"),
        separation_artifact_root=Path("separation"),
        stem_cache_root=Path("stems"),
        output_dir=Path("output"),
        model_lock_path=Path("model.json"),
        model_root=Path("model"),
        runtime_python=Path("python"),
    )
    assert request.resume is False


def test_run_identity_binds_schema_lineage_backend_config_view_and_commit() -> None:
    kwargs = {
        "handoff_manifest_sha256": "a" * 64,
        "handoff_manifest_version": "sha256:" + "b" * 64,
        "reference_manifest_sha256": "c" * 64,
        "reference_manifest_version": "sha256:" + "d" * 64,
        "timing_manifest_sha256": "e" * 64,
        "timing_manifest_version": "sha256:" + "f" * 64,
        "backend_descriptor_sha256": "1" * 64,
        "model_lock_sha256": "2" * 64,
        "inference_config_sha256": "3" * 64,
        "input_view_id": IDM_STEM_INPUT_VIEW_ID,
        "crux_commit": "4" * 40,
    }
    first = build_run_id(**kwargs)
    second = build_run_id(**{**kwargs, "inference_config_sha256": "5" * 64})
    assert first.startswith("idm-")
    assert first != second
    assert IDM_PILOT_RUN_SCHEMA == "crux.idm-stem-pilot-run/v1"


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("handoff_manifest_sha256", "6" * 64),
        ("reference_manifest_sha256", "7" * 64),
        ("timing_manifest_sha256", "8" * 64),
        ("crux_commit", "9" * 40),
    ),
)
def test_run_identity_changes_for_each_immutable_lineage_identity(field: str, changed: str) -> None:
    kwargs = {
        "handoff_manifest_sha256": "a" * 64,
        "handoff_manifest_version": "sha256:" + "b" * 64,
        "reference_manifest_sha256": "c" * 64,
        "reference_manifest_version": "sha256:" + "d" * 64,
        "timing_manifest_sha256": "e" * 64,
        "timing_manifest_version": "sha256:" + "f" * 64,
        "backend_descriptor_sha256": "1" * 64,
        "model_lock_sha256": "2" * 64,
        "inference_config_sha256": "3" * 64,
        "input_view_id": IDM_STEM_INPUT_VIEW_ID,
        "crux_commit": "4" * 40,
    }
    assert build_run_id(**kwargs) != build_run_id(**{**kwargs, field: changed})


def test_timeout_is_part_of_the_idm_inference_identity() -> None:
    default = build_idm_inference_config("1" * 64, "2" * 64)
    changed = build_idm_inference_config("1" * 64, "2" * 64, timeout_seconds=1799)
    assert idm_inference_config_sha256(default) != idm_inference_config_sha256(changed)


def test_unknown_backend_codes_poison_the_persistent_worker() -> None:
    assert classify_idm_backend_error("future_worker_code") == (
        "worker_protocol_failed",
        "poison",
    )


@pytest.mark.parametrize(
    "field",
    [
        "separation_handoff_path",
        "reference_manifest_path",
        "timing_manifest_path",
        "separation_artifact_root",
        "stem_cache_root",
        "output_dir",
        "model_lock_path",
        "model_root",
        "runtime_python",
    ],
)
def test_idm_pilot_request_rejects_non_path_values(field: str) -> None:
    kwargs = {
        "separation_handoff_path": Path("handoff.jsonl"),
        "reference_manifest_path": Path("reference.jsonl"),
        "timing_manifest_path": Path("timing.jsonl"),
        "separation_artifact_root": Path("separation"),
        "stem_cache_root": Path("stems"),
        "output_dir": Path("output"),
        "model_lock_path": Path("model.json"),
        "model_root": Path("model"),
        "runtime_python": Path("python"),
    }
    kwargs[field] = "not-a-path"
    with pytest.raises(TypeError):
        IdmPilotRunRequest(**kwargs)  # type: ignore[arg-type]


def test_smoke_selection_uses_binding_order_and_lowest_id_ties() -> None:
    rows = (
        _smoke_row(1, 10, 5),
        _smoke_row(2, 10, 4),
        _smoke_row(3, 30, 6),
        _smoke_row(4, 30, 1),
        _smoke_row(5, 20, 2),
        _smoke_row(6, 20, 3),
        _smoke_row(7, 25, 3),
    )

    selected = select_idm_smoke_cases(_smoke_handoff(*rows), _smoke_mappings(rows))

    assert tuple(case.reason for case in selected) == IDM_SMOKE_CASE_ORDER
    assert tuple(case.simfile_id for case in selected) == (1, 3, 4, 2, 5)
    assert tuple(case.simfile_id for case in selected) == tuple(
        case.simfile_id
        for case in select_idm_smoke_cases(_smoke_handoff(*reversed(rows)), _smoke_mappings(rows))
    )


@pytest.mark.parametrize(
    "cases",
    [
        [
            {"reason": reason, "simfile_id": index}
            for index, reason in enumerate(IDM_SMOKE_CASE_ORDER, start=1)
        ][::-1],
        [{"reason": reason, "simfile_id": 1} for reason in IDM_SMOKE_CASE_ORDER],
        [
            {"reason": reason, "simfile_id": 0 if index == 1 else index}
            for index, reason in enumerate(IDM_SMOKE_CASE_ORDER, start=1)
        ],
        [
            {"reason": reason, "simfile_id": 99 if index == 1 else index}
            for index, reason in enumerate(IDM_SMOKE_CASE_ORDER, start=1)
        ],
    ],
)
def test_smoke_manifest_rejects_wrong_order_duplicate_nonpositive_or_foreign_ids(
    cases: list[dict[str, object]],
) -> None:
    rows = tuple(_smoke_row(index, float(index), index) for index in range(1, 6))
    handoff = _smoke_handoff(*rows)
    content = canonical_json_bytes(
        {"cases": cases, "schema": IDM_SMOKE_SCHEMA}, trailing_newline=True
    )

    with pytest.raises(IdmSmokeManifestError):
        parse_idm_smoke_manifest(content, handoff=handoff)


def test_smoke_manifest_round_trip_is_canonical_and_handoff_bound() -> None:
    rows = tuple(_smoke_row(index, float(index), index) for index in range(1, 6))
    handoff = _smoke_handoff(*rows)
    cases = tuple(
        IdmSmokeCase(reason=reason, simfile_id=index)
        for index, reason in enumerate(IDM_SMOKE_CASE_ORDER, start=1)
    )

    content = render_idm_smoke_manifest(cases)
    manifest = parse_idm_smoke_manifest(content, handoff=handoff)

    assert manifest.schema == IDM_SMOKE_SCHEMA
    assert manifest.cases == cases
    assert content.endswith(b"\n")


def test_full_mix_smoke_request_owns_source_cache_and_full_mix_view() -> None:
    fields_by_name = {field.name for field in fields(IdmFullMixSmokeRequest)}
    assert "source_cache_dir" in fields_by_name
    assert "source_cache_dir" not in {field.name for field in fields(IdmPilotRunRequest)}
    assert IDM_FULL_MIX_INPUT_VIEW_ID == "crux.oaf-full-mix-mono44k1-pcm16/v1"
    assert IDM_FULL_MIX_INPUT_VIEW_ID != IDM_STEM_INPUT_VIEW_ID


def test_full_mix_materializer_uses_historical_view_and_neutral_rail(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    source = ResolvedSourceAudio(
        path=tmp_path / "source.wav",
        source_audio_id="source",
        source_audio_sha256="a" * 64,
        duration_sec=1.0,
        content=b"source",
    )

    def materialize(source: object, output: Path, **kwargs: object) -> object:
        calls.append({"source": source, "output": output, **kwargs})
        return CanonicalAudio(
            path=output,
            source_audio_id="source",
            source_audio_sha256="a" * 64,
            input_view_id=IDM_FULL_MIX_INPUT_VIEW_ID,
            input_audio_sha256="b" * 64,
            byte_length=44,
            sample_rate=44100,
            channel_count=1,
            sample_width_bytes=2,
            audio_frame_count=1,
        )

    monkeypatch.setattr("src.benchmark.idm_pilot_run.materialize_full_mix_audio", materialize)
    result = materialize_idm_full_mix_audio(
        source,
        tmp_path / "inputs" / "7" / "full-mix.wav",
        input_root=tmp_path / "inputs",
    )

    assert isinstance(result, CanonicalAudio)
    assert calls[0]["input_view_id"] == IDM_FULL_MIX_INPUT_VIEW_ID
    assert calls[0]["max_input_audio_frames"] is None


def test_full_mix_smoke_uses_a_separate_report_namespace() -> None:
    # The offline seam must remain explicit: smoke output cannot be written
    # below the primary ``runs``/``reports`` namespace or selected for the
    # headline comparison.
    import src.benchmark.idm_pilot_run as run_module

    assert run_module.IDM_FULL_MIX_SMOKE_DIRNAME == "full-mix-smoke"
    assert run_module.IDM_FULL_MIX_SMOKE_REPORT_DIRNAME == "reports"
    assert run_module.IDM_FULL_MIX_SMOKE_DIRNAME != "runs"


def test_full_mix_smoke_rejects_output_alias_before_loading_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    import src.benchmark.idm_pilot_run as run_module

    request = IdmFullMixSmokeRequest(
        separation_handoff_path=tmp_path / "handoff.jsonl",
        reference_manifest_path=tmp_path / "reference.jsonl",
        timing_manifest_path=tmp_path / "timing.jsonl",
        smoke_manifest_path=tmp_path / "smoke.json",
        source_cache_dir=tmp_path / "shared",
        output_dir=tmp_path / "shared",
        model_lock_path=tmp_path / "model.json",
        model_root=tmp_path / "model",
        runtime_python=tmp_path / "python",
    )
    monkeypatch.setattr(
        run_module,
        "load_separation_pilot_manifest",
        lambda *_args: pytest.fail("aliased output must fail before loading the handoff"),
    )

    outcome = run_module.run_idm_full_mix_smoke(request)

    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


class _FullMixSmokeEnvironment:
    """One isolated offline full-mix smoke setup with per-test counters."""

    def __init__(self, monkeypatch, tmp_path: Path) -> None:
        import src.benchmark.idm_pilot_run as run_module
        from src.benchmark.backends.base import NativeEvent, NativePrediction
        from src.benchmark.idm_model import load_idm_model_lock
        from tests.benchmark.idm_pilot_fixtures import (
            CRUX_COMMIT,
            SHA_B,
            canonical_wav,
            loaded_reference_manifests,
            sha256,
        )

        self.monkeypatch = monkeypatch
        self.run_module = run_module
        ids = tuple(range(20, 40))
        reference, timing, mappings = loaded_reference_manifests(
            ids=ids,
            reference_sha256="1" * 64,
            reference_version="sha256:" + "2" * 64,
            timing_sha256="3" * 64,
            timing_version="sha256:" + "4" * 64,
        )
        rows = tuple(
            {
                "simfile_id": simfile_id,
                "source_audio_id": f"{simfile_id}/audio.wav",
                "source_audio_sha256": SHA_B,
                "source_duration_sec": 1,
                "htdemucs": {"status": "success"},
            }
            for simfile_id in ids
        )
        handoff = LoadedSeparationPilotManifest(
            manifest_sha256="8" * 64,
            corpus_version="sha256:" + "9" * 64,
            rows=rows,
        )
        smoke_path = tmp_path / "smoke.json"
        smoke_path.write_bytes(
            render_idm_smoke_manifest(
                [
                    {"reason": reason, "simfile_id": simfile_id}
                    for reason, simfile_id in zip(IDM_SMOKE_CASE_ORDER, ids[:5], strict=True)
                ]
            )
        )
        wav = canonical_wav()

        def resolve_source(source_row, *_args, **_kwargs):
            return ResolvedSourceAudio(
                path=tmp_path / "source.wav",
                source_audio_id=source_row["source_audio_key"],
                source_audio_sha256=source_row["source_audio_content_hash"],
                duration_sec=1.0,
                content=wav,
            )

        lock = load_idm_model_lock(IDM_MODEL_LOCK_PATH)
        descriptor = run_module.descriptor_for_lock(lock)
        self.descriptor = descriptor
        input_sha = sha256(wav)

        monkeypatch.setattr(run_module, "load_separation_pilot_manifest", lambda _: handoff)
        monkeypatch.setattr(run_module, "load_reference_set_manifest", lambda _: reference)
        monkeypatch.setattr(run_module, "load_reference_timing_manifest", lambda _: timing)
        monkeypatch.setattr(run_module, "_validate_lineage", lambda *_args: None)
        monkeypatch.setattr(
            run_module, "preflight_reference_mappings", lambda *_args, **_kwargs: mappings
        )
        monkeypatch.setattr(run_module, "load_idm_model_lock", lambda _: lock)
        monkeypatch.setattr(run_module, "resolve_source_audio", resolve_source)

        self.materialize_calls: list[Path] = []

        def materialize(source_audio, output_path, *, input_root):
            self.materialize_calls.append(output_path)
            return CanonicalAudio(
                path=output_path,
                source_audio_id=source_audio.source_audio_id,
                source_audio_sha256=source_audio.source_audio_sha256,
                input_view_id=IDM_FULL_MIX_INPUT_VIEW_ID,
                input_audio_sha256=input_sha,
                byte_length=len(wav),
                sample_rate=44100,
                channel_count=1,
                sample_width_bytes=2,
                audio_frame_count=32,
            )

        monkeypatch.setattr(run_module, "materialize_idm_full_mix_audio", materialize)
        self.factory_calls: list[dict[str, object]] = []

        class FakeBackend:
            def descriptor(self):
                return descriptor

            def transcribe(self, audio):
                return NativePrediction(
                    audio=audio,
                    descriptor=descriptor,
                    events=(
                        NativeEvent(
                            time_sec=0.25,
                            native_class_id="KD",
                            model_output_bin=4,
                            native_midi_note=None,
                            native_metadata={"frame_index": "43", "native_velocity": "1"},
                            confidence=0.9,
                            velocity_midi=64,
                        ),
                    ),
                )

            def close(self):
                return None

        self.FakeBackend = FakeBackend
        self.request = IdmFullMixSmokeRequest(
            separation_handoff_path=tmp_path / "handoff.jsonl",
            reference_manifest_path=tmp_path / "reference.jsonl",
            timing_manifest_path=tmp_path / "timing.jsonl",
            smoke_manifest_path=smoke_path,
            source_cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
            model_lock_path=tmp_path / "model.json",
            model_root=tmp_path / "model",
            runtime_python=tmp_path / "python",
            crux_commit=CRUX_COMMIT,
        )
        self.request.model_lock_path.write_bytes(IDM_MODEL_LOCK_PATH.read_bytes())

    def backend_factory(self, **kwargs: object):
        self.factory_calls.append(kwargs)
        return self.FakeBackend()

    def run(self, request: IdmFullMixSmokeRequest | None = None):
        from src.benchmark.idm_pilot_run import run_idm_full_mix_smoke

        return run_idm_full_mix_smoke(
            self.request if request is None else request,
            backend_factory=self.backend_factory,
            perf_counter=lambda: 1.0,
        )


@pytest.fixture
def full_mix_smoke(monkeypatch, tmp_path: Path) -> _FullMixSmokeEnvironment:
    return _FullMixSmokeEnvironment(monkeypatch, tmp_path)


def test_full_mix_smoke_materializes_infers_and_reports_in_its_own_namespace(
    full_mix_smoke: _FullMixSmokeEnvironment,
) -> None:
    outcome = full_mix_smoke.run()

    assert outcome.overall_status == "complete"
    assert outcome.success_count == 5
    assert outcome.reports_path is not None
    assert "full-mix-smoke" in outcome.reports_path.as_posix()
    assert outcome.reports_path.name == "reports"
    assert (outcome.reports_path / "summary.json").exists()
    assert not (full_mix_smoke.request.output_dir / "runs").exists()
    assert len(full_mix_smoke.factory_calls) == 1
    assert full_mix_smoke.factory_calls[0]["input_root"] == outcome.run_path.parent / "inputs"
    assert len(full_mix_smoke.materialize_calls) == 5


def test_full_mix_smoke_rejects_preseeded_symlinked_run_directory(
    full_mix_smoke: _FullMixSmokeEnvironment, tmp_path: Path
) -> None:
    outcome = full_mix_smoke.run()
    assert outcome.run_id is not None

    preseeded_request = replace(full_mix_smoke.request, output_dir=tmp_path / "preseeded-output")
    preseeded_run_dir = preseeded_request.output_dir / "full-mix-smoke" / "runs" / outcome.run_id
    preseeded_outside = tmp_path / "preseeded-outside"
    preseeded_outside.mkdir()
    preseeded_sentinel = preseeded_outside / "sentinel"
    preseeded_sentinel.write_bytes(b"must remain untouched")
    preseeded_run_dir.mkdir(parents=True)
    (preseeded_run_dir / "predictions").symlink_to(preseeded_outside, target_is_directory=True)

    preseeded = full_mix_smoke.run(preseeded_request)

    assert preseeded.overall_status == "failed"
    assert preseeded.exit_code == 2
    assert not (preseeded_run_dir / "run.json").exists()
    assert preseeded_sentinel.read_bytes() == b"must remain untouched"
    assert list(preseeded_outside.iterdir()) == [preseeded_sentinel]
    assert len(full_mix_smoke.factory_calls) == 1
    assert len(full_mix_smoke.materialize_calls) == 5


def test_full_mix_smoke_records_partial_publication_failure(
    full_mix_smoke: _FullMixSmokeEnvironment, tmp_path: Path
) -> None:
    publication_request = replace(
        full_mix_smoke.request, output_dir=tmp_path / "publication-output"
    )
    run_module = full_mix_smoke.run_module
    real_publish = run_module.publish_prediction_artifact
    publish_calls: list[Path] = []

    def fail_first_publish(target: Path, mapped: object) -> object:
        publish_calls.append(target)
        if len(publish_calls) == 1:
            raise ArtifactPublicationError("simulated publication failure")
        return real_publish(target, mapped)

    full_mix_smoke.monkeypatch.setattr(
        run_module, "publish_prediction_artifact", fail_first_publish
    )
    publication = full_mix_smoke.run(publication_request)

    assert publication.overall_status == "partial"
    assert publication.success_count == 4
    assert publication.failed_count == 1
    assert len(full_mix_smoke.factory_calls) == 1
    assert len(full_mix_smoke.materialize_calls) == 5
    assert publication.run_path is not None
    publication_snapshot = strict_json_loads(
        publication.run_path.read_bytes()[:-1], require_canonical=True
    )
    publication_items = publication_snapshot["items"]
    assert publication_items[0]["native_failure_code"] == "prediction_publish_failed"
    assert all(item["execution_disposition"] == "inferred" for item in publication_items[1:])
    assert all(
        item.get("native_failure_code") != "worker_protocol_failed" for item in publication_items
    )


def _guard_against_duplicate_run_rewrites(
    full_mix_smoke: _FullMixSmokeEnvironment,
) -> dict[str, list[object]]:
    run_module = full_mix_smoke.run_module
    observed: dict[str, list[object]] = {
        "mapping_calls": [],
        "cache_load_calls": [],
        "write_calls": [],
    }

    def mapping_must_not_run(*args: object, **kwargs: object) -> object:
        observed["mapping_calls"].append((args, kwargs))
        raise AssertionError("duplicate-run rejection must precede mapping reconstruction")

    def cache_load_must_not_run(cls: object, cache_dir: object) -> object:
        del cls
        observed["cache_load_calls"].append(cache_dir)
        raise AssertionError("duplicate-run rejection must precede cache loading")

    def snapshot_write_must_not_run(path: Path, *_args: object, **_kwargs: object) -> None:
        observed["write_calls"].append(path)
        raise AssertionError("duplicate-run rejection must precede snapshot writes")

    full_mix_smoke.monkeypatch.setattr(
        run_module, "preflight_reference_mappings", mapping_must_not_run
    )
    full_mix_smoke.monkeypatch.setattr(
        run_module.CacheIndexStore,
        "load",
        classmethod(cache_load_must_not_run),
    )
    full_mix_smoke.monkeypatch.setattr(
        run_module, "_write_full_mix_smoke_snapshot", snapshot_write_must_not_run
    )
    return observed


def test_full_mix_smoke_rejects_repeated_complete_run_without_rewrites(
    full_mix_smoke: _FullMixSmokeEnvironment,
) -> None:
    outcome = full_mix_smoke.run()
    assert outcome.run_path is not None
    complete_snapshot = outcome.run_path.read_bytes()
    materialize_before_repeat = len(full_mix_smoke.materialize_calls)

    observed = _guard_against_duplicate_run_rewrites(full_mix_smoke)
    repeated = full_mix_smoke.run()

    assert repeated.overall_status == "failed"
    assert repeated.exit_code == 2
    assert outcome.run_path.read_bytes() == complete_snapshot
    assert len(full_mix_smoke.factory_calls) == 1
    assert len(full_mix_smoke.materialize_calls) == materialize_before_repeat
    assert observed["mapping_calls"] == []
    assert observed["cache_load_calls"] == []
    assert observed["write_calls"] == []


def test_full_mix_smoke_rejects_interrupted_snapshot_without_rewrites(
    full_mix_smoke: _FullMixSmokeEnvironment, tmp_path: Path
) -> None:
    outcome = full_mix_smoke.run()
    assert outcome.run_id is not None
    materialize_before_repeat = len(full_mix_smoke.materialize_calls)

    interrupted_request = replace(
        full_mix_smoke.request, output_dir=tmp_path / "interrupted-output"
    )
    interrupted_path = (
        interrupted_request.output_dir / "full-mix-smoke" / "runs" / outcome.run_id / "run.json"
    )
    interrupted_snapshot = b'{"interrupted":true}\n'
    interrupted_path.parent.mkdir(parents=True)
    interrupted_path.write_bytes(interrupted_snapshot)

    observed = _guard_against_duplicate_run_rewrites(full_mix_smoke)
    interrupted = full_mix_smoke.run(interrupted_request)

    assert interrupted.overall_status == "failed"
    assert interrupted.exit_code == 2
    assert interrupted_path.read_bytes() == interrupted_snapshot
    assert len(full_mix_smoke.factory_calls) == 1
    assert len(full_mix_smoke.materialize_calls) == materialize_before_repeat
    assert observed["mapping_calls"] == []
    assert observed["cache_load_calls"] == []
    assert observed["write_calls"] == []


@pytest.mark.parametrize("component", ("full-mix-smoke", "runs", "inputs", "reports"))
def test_full_mix_smoke_rejects_derived_symlink_without_outside_writes(
    monkeypatch, tmp_path: Path, component: str
) -> None:
    import src.benchmark.idm_pilot_run as run_module
    from src.benchmark.idm_model import load_idm_model_lock
    from tests.benchmark.idm_pilot_fixtures import (
        CRUX_COMMIT,
        SHA_B,
        loaded_reference_manifests,
    )

    ids = tuple(range(20, 25))
    reference, timing, mappings = loaded_reference_manifests(
        ids=ids,
        reference_sha256="1" * 64,
        reference_version="sha256:" + "2" * 64,
        timing_sha256="3" * 64,
        timing_version="sha256:" + "4" * 64,
    )
    handoff = LoadedSeparationPilotManifest(
        manifest_sha256="8" * 64,
        corpus_version="sha256:" + "9" * 64,
        rows=tuple(
            {
                "simfile_id": simfile_id,
                "source_audio_id": f"{simfile_id}/audio.wav",
                "source_audio_sha256": SHA_B,
                "source_duration_sec": 1,
                "htdemucs": {"status": "success"},
            }
            for simfile_id in ids
        ),
    )
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_bytes(
        render_idm_smoke_manifest(
            [
                {"reason": reason, "simfile_id": simfile_id}
                for reason, simfile_id in zip(IDM_SMOKE_CASE_ORDER, ids, strict=True)
            ]
        )
    )
    lock = load_idm_model_lock(IDM_MODEL_LOCK_PATH)
    request = IdmFullMixSmokeRequest(
        separation_handoff_path=tmp_path / "handoff.jsonl",
        reference_manifest_path=tmp_path / "reference.jsonl",
        timing_manifest_path=tmp_path / "timing.jsonl",
        smoke_manifest_path=smoke_path,
        source_cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        model_lock_path=tmp_path / "model.json",
        model_root=tmp_path / "model",
        runtime_python=tmp_path / "python",
        crux_commit=CRUX_COMMIT,
    )
    request.model_lock_path.write_bytes(IDM_MODEL_LOCK_PATH.read_bytes())
    monkeypatch.setattr(run_module, "load_separation_pilot_manifest", lambda _: handoff)
    monkeypatch.setattr(run_module, "load_reference_set_manifest", lambda _: reference)
    monkeypatch.setattr(run_module, "load_reference_timing_manifest", lambda _: timing)
    monkeypatch.setattr(run_module, "_validate_lineage", lambda *_args: None)
    monkeypatch.setattr(
        run_module, "preflight_reference_mappings", lambda *_args, **_kwargs: mappings
    )
    monkeypatch.setattr(run_module, "load_idm_model_lock", lambda _: lock)
    monkeypatch.setattr(run_module, "_smoke_run_id", lambda **_kwargs: "fixed-run")
    monkeypatch.setattr(
        run_module,
        "IdmBackend",
        lambda **_kwargs: pytest.fail("derived symlink must fail before backend creation"),
    )

    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"must remain untouched")
    namespace = request.output_dir / "full-mix-smoke"
    if component == "full-mix-smoke":
        request.output_dir.mkdir()
        namespace.symlink_to(outside, target_is_directory=True)
    elif component == "runs":
        namespace.mkdir(parents=True)
        (namespace / "runs").symlink_to(outside, target_is_directory=True)
    else:
        run_dir = namespace / "runs" / "fixed-run"
        run_dir.mkdir(parents=True)
        (run_dir / component).symlink_to(outside, target_is_directory=True)

    outcome = run_module.run_idm_full_mix_smoke(request)

    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2
    assert sentinel.read_bytes() == b"must remain untouched"
    assert list(outside.iterdir()) == [sentinel]
