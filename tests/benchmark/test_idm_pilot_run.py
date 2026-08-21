from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.benchmark.backend_identity import canonical_json_bytes
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


def test_full_mix_smoke_uses_a_separate_report_namespace(monkeypatch, tmp_path: Path) -> None:
    # The offline seam must remain explicit: smoke output cannot be written
    # below the primary ``runs``/``reports`` namespace or selected for the
    # headline comparison.
    import src.benchmark.idm_pilot_run as run_module

    assert run_module.IDM_FULL_MIX_SMOKE_DIRNAME == "full-mix-smoke"
    assert run_module.IDM_FULL_MIX_SMOKE_REPORT_DIRNAME == "reports"
    assert run_module.IDM_FULL_MIX_SMOKE_DIRNAME not in "runs/idm-stem/reports"


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


def test_full_mix_smoke_materializes_infers_scores_and_reports_separately(
    monkeypatch, tmp_path: Path
) -> None:
    import src.benchmark.idm_pilot_run as run_module
    from src.benchmark.backends.base import NativeEvent, NativePrediction
    from src.benchmark.idm_model import load_idm_model_lock
    from src.benchmark.idm_pilot_run import run_idm_full_mix_smoke
    from tests.benchmark.idm_pilot_fixtures import (
        CRUX_COMMIT,
        SHA_B,
        canonical_wav,
        loaded_reference_manifests,
        sha256,
    )

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

    lock = load_idm_model_lock(Path("runtime/idm/model.json"))
    descriptor = run_module.descriptor_for_lock(lock)
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

    def materialize(source_audio, output_path, *, input_root):
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
    factory_calls: list[dict[str, object]] = []

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

    def backend_factory(**kwargs: object) -> FakeBackend:
        factory_calls.append(kwargs)
        return FakeBackend()

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
    request.model_lock_path.write_bytes(Path("runtime/idm/model.json").read_bytes())
    outcome = run_idm_full_mix_smoke(
        request,
        backend_factory=backend_factory,
        perf_counter=lambda: 1.0,
    )

    assert outcome.overall_status == "complete"
    assert outcome.success_count == 5
    assert outcome.reports_path is not None
    assert "full-mix-smoke" in outcome.reports_path.as_posix()
    assert outcome.reports_path.name == "reports"
    assert (outcome.reports_path / "summary.json").exists()
    assert not (tmp_path / "output" / "runs").exists()
    assert len(factory_calls) == 1
    assert factory_calls[0]["input_root"] == outcome.run_path.parent / "inputs"
