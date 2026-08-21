from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from src.benchmark.backends.base import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.backends.idm import IdmBackendError
from src.benchmark.idm_model import load_idm_model_lock
from src.benchmark.idm_pilot_run import (
    IdmPilotRunRequest,
    parse_idm_pilot_run,
    run_idm_pilot,
)
from src.benchmark.reference_set_manifest import LoadedReferenceSetManifest
from src.benchmark.separation_handoff import (
    LoadedSeparationPilotManifest,
    SeparationHandoffError,
    load_separation_pilot_manifest,
)
from src.benchmark.taxonomy import OAF_PREDICTION_MAP_ID
from tests.benchmark.idm_pilot_fixtures import (
    CRUX_COMMIT,
    SHA_B,
    SHA_C,
    canonical_wav,
    loaded_reference_manifests,
    oaf_artifact,
    oaf_descriptor,
    sha256,
    write_actual_handoff,
)


def test_idm_pilot_acceptance_entry_point_is_available(tmp_path: Path) -> None:
    request = IdmPilotRunRequest(
        separation_handoff_path=tmp_path / "handoff.jsonl",
        reference_manifest_path=tmp_path / "reference.jsonl",
        timing_manifest_path=tmp_path / "timing.jsonl",
        separation_artifact_root=tmp_path / "separation",
        stem_cache_root=tmp_path / "stems",
        output_dir=tmp_path / "output",
        model_lock_path=tmp_path / "model.json",
        model_root=tmp_path / "model",
        runtime_python=tmp_path / "python",
    )
    assert callable(run_idm_pilot)
    assert request.output_dir == tmp_path / "output"


def _synthetic_handoff(
    tmp_path: Path,
) -> tuple[LoadedSeparationPilotManifest, LoadedReferenceSetManifest, object, dict[int, object]]:
    ids = tuple(range(20, 40))
    reference_sha = "1" * 64
    reference_version = "sha256:" + "2" * 64
    timing_sha = "3" * 64
    timing_version = "sha256:" + "4" * 64
    reference, timing, mappings = loaded_reference_manifests(
        ids=ids,
        reference_sha256=reference_sha,
        reference_version=reference_version,
        timing_sha256=timing_sha,
        timing_version=timing_version,
    )
    separation_root = tmp_path / "separation"
    stem_root = tmp_path / "stems"
    wav = canonical_wav()
    input_sha = sha256(wav)
    descriptor = oaf_descriptor()
    rows: list[dict[str, object]] = []
    for simfile_id in ids:
        source_id = f"{simfile_id}/audio.wav"
        input_relative = f"inputs/{simfile_id}/htdemucs.wav"
        stem_relative = f"derived/stems/{simfile_id}/drums.wav"
        input_path = separation_root / input_relative
        stem_path = stem_root / stem_relative
        input_path.parent.mkdir(parents=True, exist_ok=True)
        stem_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(wav)
        stem_path.write_bytes(wav)
        audio = CanonicalAudio(
            input_path,
            source_id,
            SHA_B,
            "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
            input_sha,
            len(wav),
            44100,
            1,
            2,
            32,
        )
        oaf_relative = f"predictions/{simfile_id}.jsonl"
        oaf_path = separation_root / oaf_relative
        oaf_content = oaf_artifact(audio)
        oaf_path.parent.mkdir(parents=True, exist_ok=True)
        oaf_path.write_bytes(oaf_content)
        successful = simfile_id < 38
        htdemucs: dict[str, object] = {
            "status": "resumed" if successful else "separation_failed",
            "failure_code": None if successful else "separator_timeout",
            "separator_lock_sha256": SHA_C,
            "input_view_id": "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
            "stem": (
                {
                    "path": stem_relative,
                    "sha256": sha256(wav),
                    "source_audio_sha256": SHA_B,
                    "separator_lock_sha256": SHA_C,
                }
                if successful
                else None
            ),
            "input": (
                {
                    "path": input_relative,
                    "input_view_id": "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
                    "input_audio_sha256": input_sha,
                    "source_audio_id": source_id,
                    "source_audio_sha256": SHA_B,
                }
                if successful
                else None
            ),
            "prediction": (
                {
                    "path": oaf_relative,
                    "artifact_sha256": sha256(oaf_content),
                    "source_audio_id": source_id,
                    "source_audio_sha256": SHA_B,
                    "input_view_id": "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
                    "input_audio_sha256": input_sha,
                }
                if successful
                else None
            ),
        }
        rows.append(
            {
                "schema_version": "crux.oaf-separation-pilot/v1",
                "separation_run_id": "oaf-separation-1234567890abcdef",
                "simfile_id": simfile_id,
                "source_row_sha256": SHA_C,
                "reference_manifest_sha256": reference_sha,
                "reference_manifest_version": reference_version,
                "reference_timing_manifest_sha256": timing_sha,
                "reference_timing_version": timing_version,
                "source_audio_id": source_id,
                "source_audio_sha256": SHA_B,
                "source_duration_sec": 1,
                "oaf_model_id": descriptor.payload["model_id"],
                "oaf_backend_descriptor_sha256": descriptor.sha256,
                "oaf_model_lock_sha256": "5" * 64,
                "oaf_checkpoint_archive_sha256": "6" * 64,
                "oaf_adapter_revision": "oaf-adapter-v1",
                "oaf_canonicalization_revision": "oaf-canonicalization-v1",
                "oaf_inference_config_sha256": "7" * 64,
                "oaf_prediction_map_version": OAF_PREDICTION_MAP_ID,
                "crux_commit": CRUX_COMMIT,
                "full_mix": {},
                "spleeter": {},
                "htdemucs": htdemucs,
                "comparison_artifacts": {},
                "decision": "use_htdemucs",
                "rationale": "synthetic immutable fixture",
            }
        )
    handoff = LoadedSeparationPilotManifest(
        manifest_sha256="8" * 64,
        corpus_version="sha256:" + "9" * 64,
        rows=tuple(rows),
    )
    return handoff, reference, timing, mappings


def _install_synthetic_seams(monkeypatch, reference, timing, mappings):
    import src.benchmark.idm_pilot_run as runner

    monkeypatch.setattr(runner, "load_reference_set_manifest", lambda _: reference)
    monkeypatch.setattr(runner, "load_reference_timing_manifest", lambda _: timing)
    monkeypatch.setattr(runner, "preflight_reference_mappings", lambda *_args, **_kwargs: mappings)
    return runner


def _request(
    tmp_path: Path,
    *,
    resume: bool = False,
    crux_commit: str = CRUX_COMMIT,
) -> IdmPilotRunRequest:
    return IdmPilotRunRequest(
        separation_handoff_path=tmp_path / "handoff.jsonl",
        reference_manifest_path=tmp_path / "reference.jsonl",
        timing_manifest_path=tmp_path / "manifests" / "timing.jsonl",
        separation_artifact_root=tmp_path / "separation",
        stem_cache_root=tmp_path / "stems",
        output_dir=tmp_path / "output",
        model_lock_path=Path("runtime/idm/model.json"),
        model_root=tmp_path / "model",
        runtime_python=tmp_path / "python",
        resume=resume,
        crux_commit=crux_commit,
    )


def _healthy_backend_factory(descriptor, calls: list[int]):
    class FakeBackend:
        def descriptor(self):
            return descriptor

        def transcribe(self, audio):
            calls.append(int(audio.source_audio_id.split("/")[0]))
            return NativePrediction(
                audio,
                descriptor,
                (
                    NativeEvent(
                        0.25,
                        "KD",
                        4,
                        None,
                        {"frame_index": "43", "native_velocity": "1"},
                        0.9,
                        64,
                    ),
                ),
            )

        def close(self):
            return None

    return lambda **_: FakeBackend()


def _run_first_synthetic(tmp_path: Path, monkeypatch):
    handoff, reference, timing, mappings = _synthetic_handoff(tmp_path)
    request = _request(tmp_path)
    write_actual_handoff(request.separation_handoff_path, handoff)
    runner = _install_synthetic_seams(monkeypatch, reference, timing, mappings)
    loaded_handoff = runner.load_separation_pilot_manifest(request.separation_handoff_path)
    descriptor = runner.descriptor_for_lock(load_idm_model_lock(Path("runtime/idm/model.json")))
    calls: list[int] = []
    outcome = run_idm_pilot(
        request,
        backend_factory=_healthy_backend_factory(descriptor, calls),
        perf_counter=lambda: 1.0,
    )
    assert outcome.success_count == 18
    assert len(calls) == 18
    return runner, loaded_handoff, request, calls, outcome


def test_synthetic_immutable_handoff_runs_both_complete_populations_and_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    handoff, reference, timing, mappings = _synthetic_handoff(tmp_path)
    request = _request(tmp_path)
    write_actual_handoff(request.separation_handoff_path, handoff)
    runner = _install_synthetic_seams(monkeypatch, reference, timing, mappings)
    lock = load_idm_model_lock(Path("runtime/idm/model.json"))
    descriptor = runner.descriptor_for_lock(lock)
    calls: list[int] = []

    class FakeBackend:
        def descriptor(self):
            return descriptor

        def transcribe(self, audio):
            calls.append(int(audio.source_audio_id.split("/")[0]))
            return NativePrediction(
                audio,
                descriptor,
                (
                    NativeEvent(
                        0.25,
                        "KD",
                        4,
                        None,
                        {"frame_index": "43", "native_velocity": "1"},
                        0.9,
                        64,
                    ),
                ),
            )

        def close(self):
            return None

    outcome = run_idm_pilot(
        request, backend_factory=lambda **_: FakeBackend(), perf_counter=lambda: 1.0
    )
    assert outcome.overall_status == "partial"
    assert outcome.success_count == 18
    assert outcome.failed_count == 2
    assert len(calls) == 18
    assert outcome.run_path is not None and outcome.run_path.exists()
    assert outcome.reports_path is not None
    assert (outcome.reports_path / "oaf" / "summary.json").exists()
    assert (outcome.reports_path / "idm" / "summary.json").exists()

    calls.clear()
    resumed = run_idm_pilot(
        request.__class__(**{**request.__dict__, "resume": True}),
        backend_factory=lambda **_: FakeBackend(),
        perf_counter=lambda: 1.0,
    )
    assert resumed.success_count == 18
    assert resumed.failed_count == 2
    assert calls == []


def test_interrupted_resume_preserves_unvisited_exact_ledger_items(
    tmp_path: Path, monkeypatch
) -> None:
    runner, _handoff, request, calls, first = _run_first_synthetic(tmp_path, monkeypatch)
    assert first.run_path is not None
    calls.clear()
    original_checkpoint = runner._write_checkpoint
    checkpoint_calls = 0

    def interrupt_after_first_row(*args, **kwargs):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        original_checkpoint(*args, **kwargs)
        if checkpoint_calls == 2:
            raise KeyboardInterrupt("deterministic row-checkpoint interruption")

    monkeypatch.setattr(runner, "_write_checkpoint", interrupt_after_first_row)
    with pytest.raises(KeyboardInterrupt, match="deterministic row-checkpoint interruption"):
        run_idm_pilot(
            replace(request, resume=True),
            backend_factory=lambda **_: pytest.fail("valid artifacts must not invoke IDM"),
            perf_counter=lambda: 1.0,
        )

    interrupted = parse_idm_pilot_run(first.run_path.read_bytes())
    interrupted_rows = {row["simfile_id"]: row for row in interrupted["items"]}
    assert interrupted_rows[20]["execution_disposition"] == "resumed"
    assert interrupted_rows[21]["execution_disposition"] == "inferred"
    assert interrupted_rows[21]["prediction_artifact_sha256"]

    monkeypatch.setattr(runner, "_write_checkpoint", original_checkpoint)
    resumed = run_idm_pilot(
        replace(request, resume=True),
        backend_factory=lambda **_: pytest.fail("preserved artifacts must not invoke IDM"),
        perf_counter=lambda: 1.0,
    )

    assert resumed.success_count == 18
    assert resumed.failed_count == 2
    assert calls == []


@pytest.mark.parametrize("identity", ("handoff", "reference", "timing", "commit"))
def test_changed_identity_does_not_resume_shared_prediction_artifacts(
    tmp_path: Path, monkeypatch, identity: str
) -> None:
    runner, handoff, request, calls, first = _run_first_synthetic(tmp_path, monkeypatch)
    assert first.run_id is not None
    calls.clear()

    if identity == "handoff":
        changed_handoff = replace(handoff, manifest_sha256="2" * 64)
        monkeypatch.setattr(runner, "load_separation_pilot_manifest", lambda _: changed_handoff)
        changed = replace(request, resume=True)
    elif identity == "reference":
        changed_handoff = replace(
            handoff,
            rows=tuple({**row, "reference_manifest_sha256": "2" * 64} for row in handoff.rows),
        )
        changed_reference = replace(
            runner.load_reference_set_manifest(request.reference_manifest_path),
            manifest_sha256="2" * 64,
        )
        monkeypatch.setattr(runner, "load_separation_pilot_manifest", lambda _: changed_handoff)
        monkeypatch.setattr(runner, "load_reference_set_manifest", lambda _: changed_reference)
        changed = replace(request, resume=True)
    elif identity == "timing":
        changed_handoff = replace(
            handoff,
            rows=tuple(
                {**row, "reference_timing_manifest_sha256": "2" * 64} for row in handoff.rows
            ),
        )
        changed_timing = replace(
            runner.load_reference_timing_manifest(request.timing_manifest_path),
            manifest_sha256="2" * 64,
        )
        monkeypatch.setattr(runner, "load_separation_pilot_manifest", lambda _: changed_handoff)
        monkeypatch.setattr(runner, "load_reference_timing_manifest", lambda _: changed_timing)
        changed = replace(request, resume=True)
    else:
        changed = replace(request, resume=True, crux_commit="2" * 40)
    resumed = run_idm_pilot(
        changed,
        backend_factory=lambda **_: pytest.fail("cross-run artifact must not invoke IDM"),
        perf_counter=lambda: 1.0,
    )

    assert resumed.overall_status == "partial"
    assert resumed.success_count == 0
    assert resumed.failed_count == 20
    assert calls == []
    assert resumed.run_path is not None
    snapshot = parse_idm_pilot_run(resumed.run_path.read_bytes())
    rows = {row["simfile_id"]: row for row in snapshot["items"]}
    assert rows[20]["native_failure_code"] == "prediction_output_conflict"
    assert rows[20]["execution_disposition"] == "failed"
    assert resumed.run_id != first.run_id


def test_resume_rejects_changed_retained_wav_without_rerunning_upstream(
    tmp_path: Path, monkeypatch
) -> None:
    _runner, _handoff, request, calls, _first = _run_first_synthetic(tmp_path, monkeypatch)
    calls.clear()
    changed_path = tmp_path / "separation" / "inputs" / "20" / "htdemucs.wav"
    changed_path.write_bytes(canonical_wav(sample=1))

    resumed = run_idm_pilot(
        replace(request, resume=True),
        backend_factory=lambda **_: pytest.fail("changed retained WAV must not rerun IDM"),
        perf_counter=lambda: 1.0,
    )

    assert resumed.success_count == 17
    assert resumed.failed_count == 3
    assert calls == []
    assert resumed.run_path is not None
    rows = {
        row["simfile_id"]: row
        for row in parse_idm_pilot_run(resumed.run_path.read_bytes())["items"]
    }
    assert rows[20]["native_failure_code"] == "retained_input_invalid"
    assert rows[20]["cohort_failure_reason"] == "prediction_artifact_invalid"


@pytest.mark.parametrize("mismatch", ("source", "input"))
def test_resume_rejects_oaf_header_source_or_input_identity_mismatch_without_rerun(
    tmp_path: Path, monkeypatch, mismatch: str
) -> None:
    _runner, handoff, request, calls, _first = _run_first_synthetic(tmp_path, monkeypatch)
    calls.clear()
    row = handoff.rows[0]
    prediction = row["htdemucs"]["prediction"]
    original_path = tmp_path / "separation" / prediction["path"]
    original = row["htdemucs"]["input"]
    bad_audio = CanonicalAudio(
        original_path,
        "changed/audio.wav" if mismatch == "source" else row["source_audio_id"],
        row["source_audio_sha256"],
        original["input_view_id"],
        SHA_C if mismatch == "input" else original["input_audio_sha256"],
        len(canonical_wav()),
        44100,
        1,
        2,
        32,
    )
    bad_content = oaf_artifact(bad_audio)
    original_path.write_bytes(bad_content)
    prediction["artifact_sha256"] = sha256(bad_content)

    resumed = run_idm_pilot(
        replace(request, resume=True),
        backend_factory=lambda **_: pytest.fail("changed OaF evidence must not rerun IDM"),
        perf_counter=lambda: 1.0,
    )

    assert resumed.success_count == 17
    assert resumed.failed_count == 3
    assert calls == []
    assert resumed.run_path is not None
    rows = {
        row["simfile_id"]: row
        for row in parse_idm_pilot_run(resumed.run_path.read_bytes())["items"]
    }
    assert rows[20]["native_failure_code"] == "retained_oaf_prediction_invalid"


def test_resume_rejects_idm_artifact_hash_mismatch_without_rerun(
    tmp_path: Path, monkeypatch
) -> None:
    _runner, _handoff, request, calls, first = _run_first_synthetic(tmp_path, monkeypatch)
    calls.clear()
    assert first.run_path is not None
    first_snapshot = parse_idm_pilot_run(first.run_path.read_bytes())
    prediction_path = tmp_path / "output" / first_snapshot["items"][0]["prediction_path"]
    prediction_path.write_bytes(b"tampered prediction bytes")

    resumed = run_idm_pilot(
        replace(request, resume=True),
        backend_factory=lambda **_: pytest.fail("changed IDM artifact must not rerun"),
        perf_counter=lambda: 1.0,
    )

    assert resumed.success_count == 17
    assert resumed.failed_count == 3
    assert calls == []
    assert resumed.run_path is not None
    rows = {
        row["simfile_id"]: row
        for row in parse_idm_pilot_run(resumed.run_path.read_bytes())["items"]
    }
    assert rows[20]["native_failure_code"] == "prediction_artifact_invalid"


def test_retained_path_escape_is_rejected_by_real_handoff_loader(tmp_path: Path) -> None:
    handoff, _reference, _timing, _mappings = _synthetic_handoff(tmp_path)
    request = _request(tmp_path)
    escaped = deepcopy(handoff.rows[0])
    escaped_view = deepcopy(escaped["htdemucs"])
    escaped_input = deepcopy(escaped_view["input"])
    escaped_input["path"] = "../outside.wav"
    escaped_view["input"] = escaped_input
    escaped["htdemucs"] = escaped_view
    escaped_handoff = replace(handoff, rows=(escaped, *handoff.rows[1:]))
    write_actual_handoff(request.separation_handoff_path, escaped_handoff)

    with pytest.raises(SeparationHandoffError, match="path is invalid"):
        load_separation_pilot_manifest(request.separation_handoff_path)


def test_retained_symlink_is_rejected_through_real_handoff_loader(
    tmp_path: Path, monkeypatch
) -> None:
    _runner, _handoff, request, calls, _first = _run_first_synthetic(tmp_path, monkeypatch)
    calls.clear()
    input_path = tmp_path / "separation" / "inputs" / "20" / "htdemucs.wav"
    outside = tmp_path / "outside.wav"
    outside.write_bytes(canonical_wav())
    input_path.unlink()
    input_path.symlink_to(outside)

    resumed = run_idm_pilot(
        replace(request, resume=True),
        backend_factory=lambda **_: pytest.fail("symlink evidence must not rerun IDM"),
        perf_counter=lambda: 1.0,
    )

    assert resumed.success_count == 17
    assert resumed.failed_count == 3
    assert calls == []
    assert resumed.run_path is not None
    rows = {
        row["simfile_id"]: row
        for row in parse_idm_pilot_run(resumed.run_path.read_bytes())["items"]
    }
    assert rows[20]["native_failure_code"] == "retained_input_invalid"


def test_poisoned_backend_is_not_restarted_and_remaining_rows_are_protocol_failed(
    tmp_path: Path, monkeypatch
) -> None:
    handoff, reference, timing, mappings = _synthetic_handoff(tmp_path)
    request = _request(tmp_path)
    write_actual_handoff(request.separation_handoff_path, handoff)
    import src.benchmark.idm_pilot_run as runner

    _install_synthetic_seams(monkeypatch, reference, timing, mappings)
    descriptor = runner.descriptor_for_lock(load_idm_model_lock(Path("runtime/idm/model.json")))
    transcribe_calls: list[int] = []
    factory_calls: list[int] = []

    class PoisonBackend:
        def descriptor(self):
            return descriptor

        def transcribe(self, audio):
            transcribe_calls.append(int(audio.source_audio_id.split("/")[0]))
            raise IdmBackendError("worker protocol failed", code="worker_protocol_failed")

        def close(self):
            return None

    def factory(**_kwargs):
        factory_calls.append(1)
        return PoisonBackend()

    outcome = run_idm_pilot(_request(tmp_path), backend_factory=factory, perf_counter=lambda: 1.0)

    assert outcome.success_count == 0
    assert outcome.failed_count == 20
    assert factory_calls == [1]
    assert transcribe_calls == [20]
    assert outcome.run_path is not None
    rows = {
        row["simfile_id"]: row
        for row in parse_idm_pilot_run(outcome.run_path.read_bytes())["items"]
    }
    assert rows[20]["native_failure_code"] == "worker_protocol_failed"
    assert rows[21]["native_failure_code"] == "worker_protocol_failed"
    assert rows[38]["native_failure_code"] == "upstream_stem_unavailable"
