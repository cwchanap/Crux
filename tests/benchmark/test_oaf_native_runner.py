from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.oaf_tf1 import calibration_entrypoint
from src.benchmark.backend_identity import canonical_json_bytes
from tools.hpa320 import oaf_candidate_builder, oaf_native_runner
from tools.hpa320.seal_oaf_backend import MeasurementRow, SealError


def _native_event() -> dict[str, object]:
    return {
        "confidence_binary64": "3fe0000000000000",
        "frame_index": 0,
        "model_output_bin": 15,
        "native_class_id": "midi_36",
        "native_midi_note": 36,
        "time_sec_binary64": "0000000000000000",
        "upstream_8hit_group_id": "kick",
        "velocity_midi": 100,
    }


def test_native_runner_derives_the_locked_ten_second_fixture() -> None:
    repository = Path(__file__).parents[2]
    source = (repository / "tests/fixtures/oaf_tf1_smoke/canonical.wav").read_bytes()

    content = oaf_native_runner._derive_fixture(source, 441_000)

    assert len(content) == 882_044
    assert (
        hashlib.sha256(content).hexdigest()
        == "17a326ecfd1789bf2757dd82646326ffaaff9781574fe41077e804ab8cbb555b"
    )


def test_native_runner_materializes_canonical_smoke_without_derivation(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[2]
    source = (repository / "tests/fixtures/oaf_tf1_smoke/canonical.wav").read_bytes()
    runner = object.__new__(oaf_native_runner.NativeCalibrationRunner)
    runner._repository_root = repository
    runner._input_root = tmp_path
    runner._maximum = 26_214_378

    relative, digest, frame_count = runner._materialize_canonical_smoke()

    assert relative == "fixtures/canonical-smoke.wav"
    assert (tmp_path / relative).read_bytes() == source
    assert digest == hashlib.sha256(source).hexdigest()
    assert frame_count == (len(source) - 44) // 2


def test_native_runner_requires_typed_pre_inference_rejection() -> None:
    accepted = {
        "audio_sha256": "a" * 64,
        "inference_call_count_after": 4,
        "inference_call_count_before": 3,
        "native_events": [_native_event()],
        "prediction_sha256": hashlib.sha256(
            (
                b'[{"confidence_binary64":"3fe0000000000000","frame_index":0,'
                b'"model_output_bin":15,"native_class_id":"midi_36",'
                b'"native_midi_note":36,"time_sec_binary64":"0000000000000000",'
                b'"upstream_8hit_group_id":"kick","velocity_midi":100}]'
            )
        ).hexdigest(),
        "rejected_before_inference": False,
        "request_id": "request-1",
        "type": "calibration_probe",
    }
    oaf_native_runner._validate_response(
        accepted,
        request_type="calibration_probe",
        request_id="request-1",
        audio_sha256="a" * 64,
    )

    rejected = {
        **accepted,
        "inference_call_count_after": 3,
        "native_events": [],
        "prediction_sha256": None,
        "rejected_before_inference": True,
    }
    oaf_native_runner._validate_response(
        rejected,
        request_type="calibration_probe",
        request_id="request-1",
        audio_sha256="a" * 64,
    )

    rejected["inference_call_count_after"] = 4
    with pytest.raises(SealError, match="after inference"):
        oaf_native_runner._validate_response(
            rejected,
            request_type="calibration_probe",
            request_id="request-1",
            audio_sha256="a" * 64,
        )


def test_completed_measurement_row_uses_final_monitor_and_stderr_peaks() -> None:
    container = object.__new__(oaf_native_runner._CalibrationContainer)
    container._monitor = SimpleNamespace(
        peak_cpu_millis=1_500,
        peak_pid_count=8,
        peak_rss_bytes=2_000,
        peak_shm_bytes=3_000,
        peak_tmp_bytes=4_000,
    )
    container._stderr = oaf_native_runner._Diagnostics(100)
    container._stderr.consume(b"diagnostic-without-newline")
    row = MeasurementRow(
        input_audio_sha256="a" * 64,
        input_frame_count=1,
        repetition=1,
        process_instance_id="process",
        inference_call_count_before=0,
        inference_call_count_after=1,
        peak_cpu_millis=1,
        peak_rss_bytes=1,
        peak_tmp_bytes=1,
        peak_shm_bytes=1,
        peak_pid_count=1,
        startup_millis=1,
        request_millis=1,
        stdout_max_line_bytes=1,
        stderr_max_line_bytes=0,
        exit_code=0,
        signal=None,
        oom_killed=False,
        prediction_sha256="b" * 64,
    )

    completed = container.observed_row(row)

    assert completed.peak_cpu_millis == 1_500
    assert completed.peak_pid_count == 8
    assert completed.peak_rss_bytes == 2_000
    assert completed.peak_shm_bytes == 3_000
    assert completed.peak_tmp_bytes == 4_000
    assert completed.stderr_max_line_bytes == len(b"diagnostic-without-newline")


def test_native_runner_rejects_malformed_native_event() -> None:
    event = _native_event()
    event.pop("velocity_midi")
    response = {
        "audio_sha256": "a" * 64,
        "inference_call_count_after": 1,
        "inference_call_count_before": 0,
        "native_events": [event],
        "prediction_sha256": hashlib.sha256(
            (
                b'[{"confidence_binary64":"3fe0000000000000","frame_index":0,'
                b'"model_output_bin":15,"native_class_id":"midi_36",'
                b'"native_midi_note":36,"time_sec_binary64":"0000000000000000",'
                b'"upstream_8hit_group_id":"kick"}]'
            )
        ).hexdigest(),
        "rejected_before_inference": False,
        "request_id": "request-1",
        "type": "calibration_probe",
    }

    with pytest.raises(SealError, match="native event"):
        oaf_native_runner._validate_response(
            response,
            request_type="calibration_probe",
            request_id="request-1",
            audio_sha256="a" * 64,
        )


def test_native_container_command_uses_only_authenticated_config_digest(
    tmp_path: Path,
) -> None:
    for name in (
        "bootstrap-request.json",
        "bootstrap-evidence.json",
        "checkpoint-evidence.json",
        "base-evidence.json",
        "runtime-digest.txt",
    ):
        (tmp_path / name).write_bytes(b"fixture")
    for name in ("model", "input", "candidate"):
        (tmp_path / name).mkdir()
    request = SimpleNamespace(
        runtime_uid=65_532,
        runtime_gid=65_532,
        payload={"resource_ceiling": calibration_entrypoint.EXPECTED_RESOURCE_CEILING},
    )
    digest = "sha256:" + "a" * 64
    bootstrap = SimpleNamespace(payload={"runtime_image_config_digest": digest})

    command = oaf_native_runner._docker_create_command(
        name="fixture",
        bootstrap_request=request,
        bootstrap=bootstrap,
        bootstrap_request_path=tmp_path / "bootstrap-request.json",
        bootstrap_evidence_path=tmp_path / "bootstrap-evidence.json",
        checkpoint_evidence_path=tmp_path / "checkpoint-evidence.json",
        base_system_evidence_path=tmp_path / "base-evidence.json",
        model_cache=tmp_path / "model",
        input_root=tmp_path / "input",
        runtime_digest_file=tmp_path / "runtime-digest.txt",
        candidate_evidence_root=tmp_path / "candidate",
        checkpoint_request_path=tmp_path / "checkpoint-request.json",
        base_system_request_path=tmp_path / "base-request.json",
    )

    assert digest in command
    assert not any(value.startswith("python:") for value in command)
    assert "--network" in command
    assert "none" in command
    assert "--read-only" in command
    assert "--cap-drop" in command
    assert "ALL" in command
    assert "no-new-privileges" in command
    assert "--privileged" not in command
    assert "PYTHONCOERCECLOCALE=0" in command


def test_native_ready_requires_every_authenticated_repeated_identity() -> None:
    request = SimpleNamespace(
        sha256="1" * 64,
        payload={
            "runner_source_manifest_sha256": "2" * 64,
            "upstream_source_manifest_sha256": "3" * 64,
        },
    )
    bootstrap = SimpleNamespace(payload={"runtime_image_config_digest": "sha256:" + "4" * 64})
    ready = {
        "base_system_package_evidence_sha256": "5" * 64,
        "calibration_bootstrap_request_sha256": "1" * 64,
        "checkpoint_acquisition_evidence_sha256": "6" * 64,
        "checkpoint_inventory_sha256": "7" * 64,
        "non_inference_count": 52,
        "non_inference_inventory_sha256": "8" * 64,
        "process_instance_id": "a" * 12,
        "protocol_schema": "crux.oaf-calibration-runner/v1",
        "required_inference_count": 78,
        "required_inference_inventory_sha256": "9" * 64,
        "restored_inference_count": 78,
        "runner_source_manifest_sha256": "2" * 64,
        "runtime_image_config_digest": "sha256:" + "4" * 64,
        "tensorflow_abi": "cp37-cp37m-manylinux2010_x86_64",
        "tensorflow_build": "v1.15.5-0-g590d6eef7e",
        "type": "ready",
        "upstream_source_manifest_sha256": "3" * 64,
    }
    oaf_native_runner._validate_ready(
        ready,
        bootstrap_request=request,
        bootstrap=bootstrap,
        checkpoint_evidence_sha256="6" * 64,
        base_system_evidence_sha256="5" * 64,
        container_id="a" * 64,
    )

    ready["tensorflow_build"] = "v1.15.5-local"
    with pytest.raises(SealError, match="ready response"):
        oaf_native_runner._validate_ready(
            ready,
            bootstrap_request=request,
            bootstrap=bootstrap,
            checkpoint_evidence_sha256="6" * 64,
            base_system_evidence_sha256="5" * 64,
            container_id="a" * 64,
        )


def test_tensor_evidence_must_match_the_ready_inventory_hashes() -> None:
    checkpoint = [{"name": f"checkpoint-{index}"} for index in range(130)]
    required = [{"name": f"required-{index}"} for index in range(78)]
    non_inference = [{"name": f"non-inference-{index}"} for index in range(52)]
    payload = {
        "active_predict_dropout": False,
        "checkpoint_inventory": checkpoint,
        "non_inference_inventory": non_inference,
        "note_sequence_byte_parity": True,
        "required_inference_inventory": required,
        "schema": "crux.oaf-tensor-coverage/v1",
        "uninitialized_required": [],
    }
    ready = {
        "checkpoint_inventory_sha256": hashlib.sha256(canonical_json_bytes(checkpoint)).hexdigest(),
        "non_inference_inventory_sha256": hashlib.sha256(
            canonical_json_bytes(non_inference)
        ).hexdigest(),
        "required_inference_inventory_sha256": hashlib.sha256(
            canonical_json_bytes(required)
        ).hexdigest(),
    }

    oaf_native_runner._validate_tensor_coverage_against_ready(payload, ready)

    payload["required_inference_inventory"] = list(reversed(required))
    with pytest.raises(SealError, match="ready handshake"):
        oaf_native_runner._validate_tensor_coverage_against_ready(payload, ready)


def test_candidate_builder_reproduces_real_runtime_distribution_inventory() -> None:
    repository = Path(__file__).parents[2]
    wheelhouse = repository / "runtime/oaf_tf1/wheelhouse/runtime"
    if not wheelhouse.is_dir():
        pytest.skip("verified runtime wheelhouse is an external authenticated input")

    distributions = oaf_candidate_builder._runtime_distributions(repository)

    tensorflow = next(row for row in distributions if row["name"] == "tensorflow")
    assert tensorflow == {
        "filename": "tensorflow-1.15.5-cp37-cp37m-manylinux2010_x86_64.whl",
        "name": "tensorflow",
        "sha256": "29831dda98d668067de75403b2fca0d06a2f026ef6f217fa2ca873c20b4ee4d3",
        "version": "1.15.5",
    }
