from __future__ import annotations

import builtins
import hashlib
import io
import json
import stat
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.oaf_tf1 import calibration_entrypoint, entrypoint
from runtime.oaf_tf1.calibration_protocol import (
    CalibrationProtocolFailure,
    read_verified_calibration_wav,
    validate_calibration_request,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
    )


def _canonical_wav(sample_frames: int = 4) -> bytes:
    samples = b"\x00\x00" * sample_frames
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(samples))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 44100, 88200, 2, 16)
        + b"data"
        + struct.pack("<I", len(samples))
        + samples
    )


def _request(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "audio_frame_count": 4,
        "audio_path": "fixture.wav",
        "audio_sha256": "a" * 64,
        "max_input_audio_frames": 8,
        "request_id": "request-1",
        "type": "measure",
    }
    payload.update(overrides)
    return payload


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_canonical(path: Path, payload: object) -> bytes:
    content = _canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def _calibration_authorities(tmp_path: Path) -> dict[str, Path]:
    runner_root = tmp_path / "image"
    runner_file = runner_root / "runtime/oaf_tf1/runner.py"
    upstream_root = tmp_path / "upstream"
    upstream_file = upstream_root / "magenta/model.py"
    runner_file.parent.mkdir(parents=True)
    upstream_file.parent.mkdir(parents=True)
    runner_file.write_bytes(b"runner\n")
    upstream_file.write_bytes(b"upstream\n")

    runner_manifest_path = tmp_path / "runner-source-manifest.json"
    runner_manifest = {
        "covered_roots": ["runtime/oaf_tf1"],
        "files": [
            {
                "byte_length": len(runner_file.read_bytes()),
                "path": "runtime/oaf_tf1/runner.py",
                "sha256": _sha256(runner_file.read_bytes()),
            }
        ],
        "schema": "crux.oaf-runner-source-manifest/v1",
    }
    runner_manifest_content = _write_canonical(runner_manifest_path, runner_manifest)

    upstream_manifest_path = tmp_path / "upstream-source-manifest.json"
    upstream_manifest = {
        "covered_roots": ["magenta"],
        "files": [
            {
                "license": "Apache-2.0",
                "path": "magenta/model.py",
                "sha256": _sha256(upstream_file.read_bytes()),
            }
        ],
        "schema": "crux.oaf-upstream-source-manifest/v1",
        "upstream_commit": "a" * 40,
        "upstream_repository": "https://example.invalid/upstream.git",
    }
    upstream_manifest_content = _write_canonical(upstream_manifest_path, upstream_manifest)

    checkpoint_components = [
        {"name": "model.ckpt-569400.data-00000-of-00001", "sha256": "", "size": 4},
        {"name": "model.ckpt-569400.index", "sha256": "", "size": 5},
        {"name": "model.ckpt-569400.meta", "sha256": "", "size": 4},
    ]
    model_cache = tmp_path / "model"
    model_cache.mkdir()
    contents = (b"data", b"index", b"meta")
    assert len(checkpoint_components) == len(contents)
    for row, content in zip(checkpoint_components, contents):
        row["sha256"] = _sha256(content)
        (model_cache / str(row["name"])).write_bytes(content)
    archive = {"name": "checkpoint.zip", "sha256": "1" * 64, "size": 1}
    archive_members = [
        {"name": "checkpoint", "role": "pointer", "sha256": "2" * 64, "size": 1},
        *[{**row, "role": "published_component"} for row in checkpoint_components],
    ]
    checkpoint_request_path = tmp_path / "checkpoint-request.json"
    checkpoint_request = {
        "archive": archive,
        "archive_members": archive_members,
        "backend_id": "magenta-egmd-tf1-94529798-8hit-v1",
        "checkpoint_url": "https://example.invalid/checkpoint.zip",
        "published_component_names": [row["name"] for row in checkpoint_components],
        "schema": "crux.oaf-checkpoint-acquisition-request/v1",
    }
    checkpoint_request_content = _write_canonical(checkpoint_request_path, checkpoint_request)
    artifact_set_sha256 = _sha256(
        json.dumps(
            checkpoint_components,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    checkpoint_evidence_path = tmp_path / "checkpoint-evidence.json"
    checkpoint_evidence = {
        "acquisition_mode": "cache_verify",
        "archive": archive,
        "archive_members": archive_members,
        "cache_path": "model",
        "model_artifact_set_sha256": artifact_set_sha256,
        "published_components": checkpoint_components,
        "request_sha256": _sha256(checkpoint_request_content),
        "schema": "crux.oaf-checkpoint-acquisition-evidence/v1",
    }
    _write_canonical(checkpoint_evidence_path, checkpoint_evidence)

    base_request_path = tmp_path / "base-request.json"
    base_request = {
        "additional_system_packages": [],
        "base_image": "python:3.7.17-slim-bullseye",
        "base_image_archive_keyring_sha256": "3" * 64,
        "base_image_manifest_digest": "sha256:" + "4" * 64,
        "platform": "linux/amd64",
        "required_probes": [
            "base_python_version",
            "runtime_python_version",
            "runtime_tensorflow_version",
            "runtime_smoke",
        ],
        "schema": "crux.oaf-base-system-package-request/v1",
    }
    base_request_content = _write_canonical(base_request_path, base_request)
    inventory = [{"architecture": "amd64", "name": "base-files", "version": "1"}]
    host_payload = {
        "api_record_sha256": "5" * 64,
        "approved_labels": ["Linux", "X64"],
        "host_numeric_fingerprint": {
            "architecture": "x86_64",
            "cpu_family": "6",
            "cpu_model": "1",
            "cpu_stepping": "1",
            "cpu_vendor_id": "GenuineIntel",
        },
        "job_id": 1,
        "run_url": "https://github.com/acme/crux/actions/runs/2/job/1",
        "runner_arch": "X64",
        "runner_os": "Linux",
        "workflow_commit": "6" * 40,
    }
    host_evidence = {
        "kind": "github_hosted",
        "official_execution_allowed": True,
        "payload": host_payload,
        "sha256": _sha256(
            json.dumps(
                host_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ),
    }
    base_evidence_path = tmp_path / "base-evidence.json"
    base_evidence = {
        "additional_system_packages": [],
        "base_image_archive_keyring_sha256": base_request["base_image_archive_keyring_sha256"],
        "base_image_manifest_digest": base_request["base_image_manifest_digest"],
        "native_host_evidence": host_evidence,
        "package_inventory": inventory,
        "package_inventory_sha256": _sha256(
            json.dumps(
                inventory,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            + b"\n"
        ),
        "probes": [{"name": name, "value": "passed"} for name in base_request["required_probes"]],
        "request_sha256": _sha256(base_request_content),
        "schema": "crux.oaf-base-system-package-evidence/v1",
    }
    _write_canonical(base_evidence_path, base_evidence)

    image_build = calibration_entrypoint.EXPECTED_IMAGE_BUILD
    distribution_manifest_path = tmp_path / "distribution-build-manifest.json"
    distribution_manifest_path.write_bytes(b"distribution\n")
    instrumentation_patch_path = tmp_path / "capture-emitted-frame.patch"
    instrumentation_patch_path.write_bytes(b"patch\n")
    bootstrap_request_path = tmp_path / "calibration-bootstrap-request.json"
    bootstrap_request = {
        "backend_id": "magenta-egmd-tf1-94529798-8hit-v1",
        "base_image_manifest_digest": base_request["base_image_manifest_digest"],
        "base_system_package_request_sha256": _sha256(base_request_content),
        "build_context_manifest_sha256": "7" * 64,
        "checkpoint_acquisition_request_sha256": _sha256(checkpoint_request_content),
        "container_restrictions": {
            "drop_capabilities": ["ALL"],
            "network": "none",
            "no_new_privileges": True,
            "platform": "linux/amd64",
            "read_only_root": True,
        },
        "distribution_build_manifest_sha256": _sha256(distribution_manifest_path.read_bytes()),
        "environment": calibration_entrypoint.EXPECTED_ENVIRONMENT,
        "image_build": image_build,
        "instrumentation_patch_sha256": _sha256(instrumentation_patch_path.read_bytes()),
        "python_coerce_c_locale": "0",
        "resource_ceiling": calibration_entrypoint.EXPECTED_RESOURCE_CEILING,
        "runner_source_manifest_sha256": _sha256(runner_manifest_content),
        "runtime_gid": 65532,
        "runtime_uid": 65532,
        "schema": "crux.oaf-calibration-bootstrap-request/v1",
        "upstream_source_manifest_sha256": _sha256(upstream_manifest_content),
    }
    bootstrap_request_content = _write_canonical(bootstrap_request_path, bootstrap_request)
    config_digest = "sha256:" + "a" * 64
    bootstrap_evidence_path = tmp_path / "calibration-bootstrap-evidence.json"
    bootstrap_evidence = {
        "base_image_config_digest": "sha256:" + "b" * 64,
        "base_image_layer_diff_ids": ["sha256:" + "c" * 64],
        "base_image_layer_digests": ["sha256:" + "d" * 64],
        "build_context_manifest_sha256": bootstrap_request["build_context_manifest_sha256"],
        "calibration_bootstrap_request_sha256": _sha256(bootstrap_request_content),
        "image_build": image_build,
        "native_host_attestation_bundle_sha256": "e" * 64,
        "native_host_evidence": host_evidence,
        "oci_layout_archive": {"name": "runtime.oci.tar", "sha256": "f" * 64, "size": 1},
        "oci_layout_manifest_sha256": "0" * 64,
        "runtime_image_config_digest": config_digest,
        "runtime_image_index_digest": "sha256:" + "1" * 64,
        "runtime_image_layer_diff_ids": [
            "sha256:" + "c" * 64,
            "sha256:" + "2" * 64,
        ],
        "runtime_image_layer_digests": [
            "sha256:" + "d" * 64,
            "sha256:" + "3" * 64,
        ],
        "runtime_image_manifest_digest": "sha256:" + "4" * 64,
        "schema": "crux.oaf-calibration-bootstrap-evidence/v1",
    }
    _write_canonical(bootstrap_evidence_path, bootstrap_evidence)
    runtime_config_path = tmp_path / "runtime-image-config-digest.txt"
    runtime_config_path.write_text(config_digest + "\n", encoding="ascii")
    return {
        "base_system_evidence_path": base_evidence_path,
        "base_system_request_path": base_request_path,
        "bootstrap_evidence_path": bootstrap_evidence_path,
        "bootstrap_request_path": bootstrap_request_path,
        "checkpoint_evidence_path": checkpoint_evidence_path,
        "checkpoint_request_path": checkpoint_request_path,
        "distribution_build_manifest_path": distribution_manifest_path,
        "instrumentation_patch_path": instrumentation_patch_path,
        "model_cache_root": model_cache,
        "runner_source_manifest_path": runner_manifest_path,
        "runner_source_root": runner_root,
        "runtime_image_config_digest_path": runtime_config_path,
        "upstream_source_manifest_path": upstream_manifest_path,
        "upstream_source_root": upstream_root,
    }


def test_calibration_protocol_accepts_only_exact_measurement_shape() -> None:
    request = validate_calibration_request(
        _canonical_bytes(_request()),
        authorized_max_input_audio_frames=8,
    )

    assert request.request_type == "measure"
    assert request.audio_frame_count == 4
    assert request.max_input_audio_frames == 8

    for mutation in (
        {"unknown": True},
        {"type": "transcribe"},
        {"audio_frame_count": True},
        {"audio_sha256": "A" * 64},
        {"audio_path": "../fixture.wav"},
    ):
        with pytest.raises(CalibrationProtocolFailure):
            validate_calibration_request(
                _canonical_bytes(_request(**mutation)),
                authorized_max_input_audio_frames=8,
            )


def test_calibration_protocol_accepts_over_bound_probe_for_typed_rejection() -> None:
    request = validate_calibration_request(
        _canonical_bytes(
            _request(
                audio_frame_count=9,
                request_id="probe-over-bound",
                type="calibration_probe",
            )
        ),
        authorized_max_input_audio_frames=8,
    )

    assert request.audio_frame_count == 9
    assert request.request_type == "calibration_probe"

    with pytest.raises(CalibrationProtocolFailure, match="bound"):
        validate_calibration_request(
            _canonical_bytes(_request(max_input_audio_frames=9)),
            authorized_max_input_audio_frames=8,
        )


def test_calibration_runner_rejects_over_bound_probe_without_audio_or_inference(
    tmp_path: Path,
) -> None:
    stdin = io.BytesIO(
        _canonical_bytes(
            _request(
                audio_frame_count=9,
                audio_path="missing.wav",
                request_id="probe-over-bound",
                type="calibration_probe",
            )
        )
    )
    stdout = io.BytesIO()
    calls = 0

    def transcribe(_verified: object) -> list[object]:
        nonlocal calls
        calls += 1
        return []

    calibration_entrypoint.serve_calibration_requests(
        stdin=stdin,
        stdout=stdout,
        transcribe=transcribe,
        input_root=tmp_path,
        authorized_max_input_audio_frames=8,
        stdout_max_line_bytes=4096,
    )

    assert calls == 0
    assert json.loads(stdout.getvalue()) == {
        "audio_sha256": "a" * 64,
        "inference_call_count_after": 0,
        "inference_call_count_before": 0,
        "native_events": [],
        "prediction_sha256": None,
        "rejected_before_inference": True,
        "request_id": "probe-over-bound",
        "type": "calibration_probe",
    }


def test_calibration_runner_hashes_native_events_and_counts_inference(tmp_path: Path) -> None:
    audio = _canonical_wav(4)
    (tmp_path / "fixture.wav").write_bytes(audio)
    stdin = io.BytesIO(
        _canonical_bytes(
            _request(
                audio_sha256=_sha256(audio),
                request_id="measurement-1",
            )
        )
    )
    stdout = io.BytesIO()
    native_events = [{"frame_index": 3, "native_midi_note": 36}]

    calibration_entrypoint.serve_calibration_requests(
        stdin=stdin,
        stdout=stdout,
        transcribe=lambda _verified: native_events,
        input_root=tmp_path,
        authorized_max_input_audio_frames=8,
        stdout_max_line_bytes=4096,
    )

    assert json.loads(stdout.getvalue()) == {
        "audio_sha256": _sha256(audio),
        "inference_call_count_after": 1,
        "inference_call_count_before": 0,
        "native_events": native_events,
        "prediction_sha256": _sha256(
            json.dumps(
                native_events,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ),
        "rejected_before_inference": False,
        "request_id": "measurement-1",
        "type": "measure",
    }


def test_calibration_protocol_verifies_hash_and_declared_frames(tmp_path: Path) -> None:
    content = _canonical_wav(4)
    (tmp_path / "fixture.wav").write_bytes(content)
    request = validate_calibration_request(
        _canonical_bytes(_request(audio_sha256=hashlib.sha256(content).hexdigest())),
        authorized_max_input_audio_frames=8,
    )

    verified = read_verified_calibration_wav(request, tmp_path)

    assert verified.audio_frame_count == 4
    assert verified.content == content

    bad_hash = validate_calibration_request(
        _canonical_bytes(_request()),
        authorized_max_input_audio_frames=8,
    )
    with pytest.raises(CalibrationProtocolFailure, match="hash"):
        read_verified_calibration_wav(bad_hash, tmp_path)


def test_calibration_preflight_failures_happen_before_tensorflow_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith(("numpy", "tensorflow")):
            raise AssertionError("numeric import happened before calibration preflight")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(CalibrationProtocolFailure):
        calibration_entrypoint.preflight_request(
            _canonical_bytes(_request(audio_frame_count=9)),
            authorized_max_input_audio_frames=8,
        )


def test_calibration_startup_authenticates_every_authority_before_numeric_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorities = _calibration_authorities(tmp_path)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith(("numpy", "tensorflow")):
            raise AssertionError("numeric import happened before calibration authority checks")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    startup = calibration_entrypoint.authenticate_calibration_startup(**authorities)

    assert startup.runtime_image_config_digest == "sha256:" + "a" * 64


def test_calibration_startup_rejects_authority_hash_drift_before_numeric_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorities = _calibration_authorities(tmp_path)
    authorities["runner_source_manifest_path"].write_bytes(b"{}\n")
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith(("numpy", "tensorflow")):
            raise AssertionError("numeric import happened before calibration authority checks")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(calibration_entrypoint.CalibrationAuthorityFailure):
        calibration_entrypoint.authenticate_calibration_startup(**authorities)


def test_calibration_environment_drift_fails_before_numeric_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNREVIEWED", "1")

    with pytest.raises(SystemExit) as error:
        calibration_entrypoint.validate_process_environment()

    assert error.value.code == 2


def test_calibration_tensor_evidence_is_readable_by_the_host(tmp_path: Path) -> None:
    model = SimpleNamespace(
        checkpoint_inventory=[],
        non_inference_inventory=[],
        required_inference_inventory=[],
    )

    calibration_entrypoint.publish_calibration_tensor_coverage(model, tmp_path)

    target = tmp_path / "tensor-coverage.json"
    assert stat.S_IMODE(target.stat().st_mode) == 0o444


def test_production_entrypoint_rejects_calibration_mounts_before_numeric_import(
    tmp_path: Path,
) -> None:
    mount = tmp_path / "calibration-bootstrap-request.json"
    mount.write_bytes(b"calibration")

    with pytest.raises(SystemExit) as error:
        entrypoint.reject_calibration_mode_mounts((mount,))

    assert error.value.code == 2
