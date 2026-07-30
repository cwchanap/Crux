from __future__ import annotations

# pylint: disable=duplicate-code,too-many-lines
import hashlib
import json
import os
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

import src.benchmark.backend_lock as backend_lock_module
import src.benchmark.backend_prepare as backend_prepare_module
from src.benchmark.backend_lock import (
    BackendLockError,
    LoadedBackendLock,
    LoadedConversionAudit,
    LoadedRuntimeLock,
    LoadedSealEvidence,
    load_backend_lock,
    load_conversion_audit,
    load_runtime_lock,
    load_seal_evidence,
    validate_oaf_lock_set,
)
from src.benchmark.backend_prepare import PrepareBackendRequest, prepare_oaf_backend
from src.benchmark.backend_registry import OFFICIAL_BACKEND_ID

ARCHIVE_SHA256 = "09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0"
COMPONENTS = [
    {
        "name": "model.ckpt-569400.data-00000-of-00001",
        "sha256": "6312cd6f8e29ee6f21f4a763822d554f46393e0e336812ba4a8cb324d2e8b0b5",
        "size": 222,
    },
    {
        "name": "model.ckpt-569400.index",
        "sha256": "475ca21993e102d52e2e6c7066b598b9b9e9a097bfb2fa6b770d40eedaa6139a",
        "size": 111,
    },
    {
        "name": "model.ckpt-569400.meta",
        "sha256": "e36105b27c17f2367292682df825b8ad16eb887ccad465bbb5ddf955ef75d422",
        "size": 333,
    },
]
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
}
GROUPS = [
    {"base_midi": 36, "group_id": "kick", "member_pitches": [36], "output_bin": 15},
    {
        "base_midi": 38,
        "group_id": "snare",
        "member_pitches": [38, 40, 37, 39],
        "output_bin": 17,
    },
    {
        "base_midi": 48,
        "group_id": "toms",
        "member_pitches": [48, 50, 45, 47, 43, 58, 64],
        "output_bin": 27,
    },
    {
        "base_midi": 46,
        "group_id": "hihat",
        "member_pitches": [46, 26, 42, 22, 44, 54, 70],
        "output_bin": 25,
    },
    {
        "base_midi": 51,
        "group_id": "ride",
        "member_pitches": [51, 59],
        "output_bin": 30,
    },
    {
        "base_midi": 53,
        "group_id": "ride_bell",
        "member_pitches": [53, 56],
        "output_bin": 32,
    },
    {
        "base_midi": 49,
        "group_id": "crash",
        "member_pitches": [49, 55, 57, 52],
        "output_bin": 28,
    },
    {"base_midi": 75, "group_id": "sticks", "member_pitches": [75], "output_bin": 54},
]
HPARAMS = {
    "acoustic_rnn_dropout_keep_prob": 0.5,
    "bidirectional": True,
    "combined_lstm_dropout_keep_prob": 1,
    "combined_lstm_units": 256,
    "conv_dropout_keep_amts": [1, 0.25, 0.25],
    "conv_filters": [16, 16, 32],
    "drum_data_map": "8-hit",
    "drum_note_duration": 0.05,
    "drum_prediction_map": "",
    "drums_only": True,
    "fc_dropout_keep_amt": 0.5,
    "fc_size": 256,
    "frame_lstm_units": 0,
    "frame_threshold": 0.5,
    "hop_length": 441,
    "log_amplitude": True,
    "min_gap": None,
    "num_pitches": 88,
    "offset_lstm_units": 256,
    "offset_network": True,
    "offset_threshold": 0,
    "onset_length": 0,
    "onset_lstm_units": 64,
    "onset_threshold": 0.5,
    "peak_picking": False,
    "sample_rate": 44100,
    "share_conv_features": False,
    "spec_fmin": 30,
    "spec_hop_length": 512,
    "spec_htk": True,
    "spec_mel_bins": 250,
    "spec_type": "mel",
    "transform_audio": False,
    "use_librosa": False,
    "velocity_bias": 0,
    "velocity_lstm_units": 0,
    "velocity_scale": 127,
    "viterbi_decoding": False,
}
METADATA_FIELDS = [
    {"name": "frame_index", "nullable": False, "type": "integer"},
    {"name": "upstream_group_id", "nullable": True, "type": "string"},
]
PYTHON_DISTRIBUTIONS = [
    {
        "filename": "tensorflow-1.15.5-cp37-cp37m-manylinux2010_x86_64.whl",
        "name": "tensorflow",
        "sha256": "29831dda98d668067de75403b2fca0d06a2f026ef6f217fa2ca873c20b4ee4d3",
        "version": "1.15.5",
    },
    {
        "filename": "numpy-1.18.5-cp37-cp37m-manylinux2010_x86_64.whl",
        "name": "numpy",
        "sha256": "1" * 64,
        "version": "1.18.5",
    },
]
SYSTEM_PACKAGES = [
    {
        "filename": "libgomp1_10.2.1-6_amd64.deb",
        "name": "libgomp1",
        "sha256": "2" * 64,
        "version": "10.2.1-6",
    }
]


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
    )


def write_json(path: Path, payload: Any) -> Path:
    path.write_bytes(canonical_bytes(payload))
    return path


def content_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def identity_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)[:-1]).hexdigest()


def inventories() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint = [
        {"dtype": "float32", "name": f"tensor_{index:03d}", "shape": [index + 1]}
        for index in range(130)
    ]
    required = deepcopy(checkpoint[:78])
    non_inference = [{**row, "reason": "optimizer_state"} for row in deepcopy(checkpoint[78:])]
    return checkpoint, required, non_inference


def backend_payload(
    *,
    runtime_sha256: str = "a" * 64,
    seal_sha256: str = "b" * 64,
    audit_sha256: str = "c" * 64,
) -> dict[str, Any]:
    checkpoint, required, non_inference = inventories()
    return {
        "architecture_id": "magenta-oaf-model-tpu-drums-v1",
        "backend_id": "magenta-egmd-tf1-94529798-8hit-v1",
        "checkpoint_archive": {
            "name": "e-gmd_checkpoint.zip",
            "sha256": ARCHIVE_SHA256,
            "size": 999,
        },
        "checkpoint_components": deepcopy(COMPONENTS),
        "checkpoint_inventory": checkpoint,
        "checkpoint_url": (
            "https://storage.googleapis.com/magentadata/models/"
            "onsets_frames_transcription/e-gmd_checkpoint.zip"
        ),
        "descriptor_schema": "crux.transcription-backend-descriptor/v1",
        "drum_prediction_map": "",
        "execution_report_schema": "crux.backend-execution-report/v1",
        "host_adapter_source_manifest_sha256": "3" * 64,
        "hparams": deepcopy(HPARAMS),
        "hparams_source": "magenta/models/onsets_frames_transcription/configs.py:drums",
        "legacy_conversion_coverage_sha256": audit_sha256,
        "legacy_score_report_schema": "crux.legacy-score-report/v1",
        "max_input_audio_frames": 441000,
        "model_id": "magenta-egmd-ckpt-569400-v1",
        "native_metadata_fields": deepcopy(METADATA_FIELDS),
        "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
        "native_output_bins": [
            {
                "model_output_bin": output_bin,
                "native_class_id": f"midi_{output_bin + 21}",
                "native_midi_note": output_bin + 21,
            }
            for output_bin in range(88)
        ],
        "native_output_space_id": "magenta-oaf-midi88-a0-v1",
        "non_inference_inventory": non_inference,
        "prediction_schema": "crux.drum-prediction-events/v1",
        "protocol_schema": "crux.transcription-runner/v1",
        "required_inference_inventory": required,
        "runtime_image_manifest_digest": f"sha256:{'4' * 64}",
        "runtime_lock_sha256": runtime_sha256,
        "schema": "crux.transcription-backend-lock/v1",
        "seal_evidence_sha256": seal_sha256,
        "serialization": {
            "encoding": "utf-8",
            "final_newline": True,
            "key_order": "lexicographic",
            "whitespace": "none",
        },
        "smoke_audio_sha256": "5" * 64,
        "smoke_oracle_sha256": "6" * 64,
        "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
        "training_groups": deepcopy(GROUPS),
        "upstream_repository": "https://github.com/magenta/magenta",
        "upstream_source_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
        "upstream_source_manifest_sha256": "7" * 64,
        "verification_report_schema": "crux.backend-verification-report/v1",
    }


def runtime_payload(*, seal_sha256: str = "b" * 64) -> dict[str, Any]:
    return {
        "base_image": "python:3.7.17-slim-bullseye",
        "base_image_manifest_digest": (
            "sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673"
        ),
        "debian_release_sha256": "8" * 64,
        "debian_snapshot_repository": "https://snapshot.debian.org/archive/debian/20240101T000000Z",
        "distribution_build_manifest_sha256": "d" * 64,
        "environment": deepcopy(ENVIRONMENT),
        "oci_layout_manifest_sha256": "3" * 64,
        "platform": "linux/amd64",
        "python_distributions": deepcopy(PYTHON_DISTRIBUTIONS),
        "python_version": "3.7.17",
        "runner_source_manifest_sha256": "9" * 64,
        "runtime_image_manifest_digest": f"sha256:{'4' * 64}",
        "schema": "crux.transcription-runtime-lock/v1",
        "seal_evidence_sha256": seal_sha256,
        "stdout_max_line_bytes": 262144,
        "stderr_max_line_bytes": 8192,
        "stderr_read_chunk_bytes": 4096,
        "stderr_ring_buffer_bytes": 65536,
        "system_packages": deepcopy(SYSTEM_PACKAGES),
        "tensorflow_abi": "cp37-cp37m-manylinux2010_x86_64",
        "tensorflow_build": "v1.15.5-0-g590d6eef7e",
        "upstream_source_manifest_sha256": "7" * 64,
    }


def seal_payload(*, audit_sha256: str = "c" * 64) -> dict[str, Any]:
    checkpoint, required, non_inference = inventories()
    return {
        "advisory_snapshot_sha256": "a" * 64,
        "base_image_manifest_digest": (
            "sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673"
        ),
        "checkpoint_archive": {
            "name": "e-gmd_checkpoint.zip",
            "sha256": ARCHIVE_SHA256,
            "size": 999,
        },
        "checkpoint_components": deepcopy(COMPONENTS),
        "checkpoint_inventory": checkpoint,
        "cpu_limit_millis": 2000,
        "debian_release_sha256": "8" * 64,
        "debian_snapshot_repository": "https://snapshot.debian.org/archive/debian/20240101T000000Z",
        "distribution_build_manifest_sha256": "d" * 64,
        "host_adapter_source_manifest_sha256": "3" * 64,
        "instrumentation_patch_sha256": "b" * 64,
        "legacy_conversion_coverage_sha256": audit_sha256,
        "max_input_audio_frames": 441000,
        "measurements": {
            "peak_cpu_millis": 1000,
            "peak_pid_count": 8,
            "peak_rss_bytes": 536870912,
            "peak_shm_bytes": 1048576,
            "peak_tmp_bytes": 1048576,
            "request_millis": 2000,
            "startup_millis": 10000,
        },
        "memory_limit_bytes": 1073741824,
        "native_host_evidence": {"form": "github_hosted_linux_x64", "sha256": "c" * 64},
        "non_inference_inventory": non_inference,
        "oci_layout_archive": {"name": "oaf-runtime.oci.tar", "sha256": "d" * 64, "size": 777},
        "oci_layout_manifest_sha256": "3" * 64,
        "pid_limit": 64,
        "python_distributions": deepcopy(PYTHON_DISTRIBUTIONS),
        "request_deadline_seconds": 60,
        "required_inference_inventory": required,
        "runner_source_manifest_sha256": "9" * 64,
        "runtime_gid": 10001,
        "runtime_image_config_digest": f"sha256:{'e' * 64}",
        "runtime_image_layer_digests": [f"sha256:{'f' * 64}"],
        "runtime_image_manifest_digest": f"sha256:{'4' * 64}",
        "runtime_uid": 10001,
        "schema": "crux.backend-seal-evidence/v1",
        "security_scan_sha256": "0" * 64,
        "shm_bytes": 67108864,
        "smoke_audio_sha256": "5" * 64,
        "smoke_oracle_sha256": "6" * 64,
        "smoke_prediction_sha256": "1" * 64,
        "startup_deadline_seconds": 120,
        "stdout_max_line_bytes": 262144,
        "stderr_max_line_bytes": 8192,
        "stderr_read_chunk_bytes": 4096,
        "stderr_ring_buffer_bytes": 65536,
        "system_packages": deepcopy(SYSTEM_PACKAGES),
        "tensor_coverage_sha256": "2" * 64,
        "tensorflow_abi": "cp37-cp37m-manylinux2010_x86_64",
        "tensorflow_build": "v1.15.5-0-g590d6eef7e",
        "tmp_bytes": 134217728,
        "upstream_source_manifest_sha256": "7" * 64,
    }


def audit_payload() -> dict[str, Any]:
    _, required, _ = inventories()
    return {
        "candidate_matches": [],
        "converter_source_manifest_sha256": "3" * 64,
        "matching_algorithm": "exact_assignment_trace",
        "matching_algorithm_version": "v1",
        "model_artifact_set_sha256": identity_sha256(COMPONENTS),
        "observed_hdf5_sha256": (
            "d36ced8b2ee241bc37ad6fbb918ba38e95d666350dd4888bca59a1243bf4d10e"
        ),
        "required_inference_inventory_sha256": identity_sha256(required),
        "restored_required": [],
        "restored_required_count": 0,
        "schema": "crux.legacy-tf2-conversion-coverage/v1",
        "tf2_model_source_manifest_sha256": "4" * 64,
        "unmatched_required": [row["name"] for row in required],
    }


# pylint: disable=too-many-arguments
def candidate_match(
    *,
    required_name: str = "tensor_000",
    candidate_name: str = "conv_0/kernel",
    match_kind: str = "loose_substring",
    dtype_compatible: bool = True,
    shape_compatible: bool = False,
    assigned: bool = False,
) -> dict[str, Any]:
    return {
        "assigned": assigned,
        "candidate_name": candidate_name,
        "dtype_compatible": dtype_compatible,
        "match_kind": match_kind,
        "required_name": required_name,
        "shape_compatible": shape_compatible,
    }


def write_lock_set(
    tmp_path: Path,
) -> tuple[LoadedBackendLock, LoadedRuntimeLock, LoadedSealEvidence, LoadedConversionAudit]:
    audit_data = audit_payload()
    audit = load_conversion_audit(write_json(tmp_path / "audit.json", audit_data))
    seal_data = seal_payload(audit_sha256=audit.sha256)
    seal = load_seal_evidence(write_json(tmp_path / "seal.json", seal_data))
    runtime_data = runtime_payload(seal_sha256=seal.sha256)
    runtime = load_runtime_lock(write_json(tmp_path / "runtime.json", runtime_data))
    backend_data = backend_payload(
        runtime_sha256=runtime.sha256,
        seal_sha256=seal.sha256,
        audit_sha256=audit.sha256,
    )
    backend = load_backend_lock(write_json(tmp_path / "backend.json", backend_data))
    return backend, runtime, seal, audit


def test_valid_lock_set_reproduces_descriptor_and_immutable_records(tmp_path: Path) -> None:
    backend, runtime, seal, audit = write_lock_set(tmp_path)

    validate_oaf_lock_set(backend, runtime, seal, audit)

    assert backend.max_input_audio_frames == 441000
    assert backend.descriptor.payload["backend_lock_sha256"] == backend.sha256
    assert backend.descriptor.payload["runtime_lock_sha256"] == runtime.sha256
    assert backend.descriptor.payload["model_artifact_set_sha256"] == identity_sha256(COMPONENTS)
    assert backend.descriptor.sha256 == identity_sha256(dict(backend.descriptor.payload))
    for record, field in (
        (backend, "sha256"),
        (runtime, "sha256"),
        (seal, "sha256"),
        (audit, "sha256"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, "0" * 64)
    with pytest.raises(TypeError):
        backend.payload["schema"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        backend.payload["hparams"]["hop_length"] = 512  # type: ignore[index]
    with pytest.raises(TypeError):
        backend.payload["checkpoint_components"][0]["size"] = 0  # type: ignore[index]


def test_backend_lock_rejects_missing_audio_frame_bound(tmp_path: Path) -> None:
    payload = backend_payload()
    del payload["max_input_audio_frames"]

    with pytest.raises(BackendLockError, match="backend lock fields"):
        load_backend_lock(write_json(tmp_path / "backend.json", payload))


def test_backend_lock_rejects_unknown_field_and_wrong_schema(tmp_path: Path) -> None:
    payload = backend_payload()
    payload["unknown"] = True
    with pytest.raises(BackendLockError, match="backend lock fields"):
        load_backend_lock(write_json(tmp_path / "unknown.json", payload))

    payload = backend_payload()
    payload["schema"] = "crux.transcription-backend-lock/v2"
    with pytest.raises(BackendLockError, match="backend lock schema"):
        load_backend_lock(write_json(tmp_path / "schema.json", payload))


def test_loaders_reject_duplicate_keys_noncanonical_bytes_and_symlinks(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"schema":"a","schema":"b"}\n')
    with pytest.raises(BackendLockError, match="duplicate key"):
        load_backend_lock(duplicate)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(b'{"schema": "value"}\n')
    with pytest.raises(BackendLockError, match="canonical"):
        load_backend_lock(noncanonical)

    missing_newline = tmp_path / "missing-newline.json"
    missing_newline.write_bytes(canonical_bytes(backend_payload())[:-1])
    with pytest.raises(BackendLockError, match="one final newline"):
        load_backend_lock(missing_newline)

    target = write_json(tmp_path / "target.json", backend_payload())
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(BackendLockError, match="regular file"):
        load_backend_lock(symlink)


def test_backend_lock_rejects_uppercase_hash_and_fixed_component_drift(tmp_path: Path) -> None:
    payload = backend_payload()
    payload["smoke_audio_sha256"] = "A" * 64
    with pytest.raises(BackendLockError, match="lowercase SHA-256"):
        load_backend_lock(write_json(tmp_path / "uppercase.json", payload))

    payload = backend_payload()
    payload["checkpoint_components"][0]["sha256"] = "a" * 64
    with pytest.raises(BackendLockError, match="checkpoint component identity"):
        load_backend_lock(write_json(tmp_path / "component.json", payload))


def test_backend_lock_rejects_duplicate_checkpoint_names_and_inventory_overlap(
    tmp_path: Path,
) -> None:
    payload = backend_payload()
    payload["checkpoint_inventory"][1]["name"] = payload["checkpoint_inventory"][0]["name"]
    with pytest.raises(BackendLockError, match="checkpoint inventory names"):
        load_backend_lock(write_json(tmp_path / "duplicate.json", payload))

    payload = backend_payload()
    payload["non_inference_inventory"][0]["name"] = payload["required_inference_inventory"][0][
        "name"
    ]
    with pytest.raises(BackendLockError, match="inventory overlap"):
        load_backend_lock(write_json(tmp_path / "overlap.json", payload))


def test_backend_lock_accepts_scalar_checkpoint_tensor_shape(tmp_path: Path) -> None:
    payload = backend_payload()
    payload["checkpoint_inventory"][-1]["shape"] = []
    payload["non_inference_inventory"][-1]["shape"] = []

    loaded = load_backend_lock(write_json(tmp_path / "scalar.json", payload))

    assert loaded.payload["non_inference_inventory"][-1]["shape"] == ()


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("checkpoint_inventory", "130"),
        ("required_inference_inventory", "78"),
        ("non_inference_inventory", "52"),
    ],
)
def test_backend_lock_requires_exact_inventory_counts(
    tmp_path: Path,
    field: str,
    expected: str,
) -> None:
    payload = backend_payload()
    payload[field].pop()

    with pytest.raises(BackendLockError, match=expected):
        load_backend_lock(write_json(tmp_path / f"{field}.json", payload))


def test_backend_lock_requires_complete_hparams_and_disabled_prediction_map(
    tmp_path: Path,
) -> None:
    payload = backend_payload()
    del payload["hparams"]["hop_length"]
    with pytest.raises(BackendLockError, match="resolved hparams"):
        load_backend_lock(write_json(tmp_path / "hparams.json", payload))

    payload = backend_payload()
    payload["hparams"]["spec_hop_length"] = 441
    with pytest.raises(BackendLockError, match="resolved hparams"):
        load_backend_lock(write_json(tmp_path / "spec-hop.json", payload))

    payload = backend_payload()
    payload["drum_prediction_map"] = "8-hit"
    payload["hparams"]["drum_prediction_map"] = "8-hit"
    with pytest.raises(BackendLockError, match="prediction map"):
        load_backend_lock(write_json(tmp_path / "prediction-map.json", payload))


def test_backend_lock_rejects_hparams_source_drift(tmp_path: Path) -> None:
    payload = backend_payload()
    payload["hparams_source"] = "some/other/config.py:drums"

    with pytest.raises(BackendLockError, match="hparams source"):
        load_backend_lock(write_json(tmp_path / "hparams-source.json", payload))


def test_backend_lock_requires_exact_bins_groups_and_metadata_schema(tmp_path: Path) -> None:
    payload = backend_payload()
    payload["native_output_bins"][40]["native_midi_note"] = 60
    with pytest.raises(BackendLockError, match="88-bin"):
        load_backend_lock(write_json(tmp_path / "bins.json", payload))

    payload = backend_payload()
    payload["training_groups"][1]["member_pitches"].pop()
    with pytest.raises(BackendLockError, match="8-hit groups"):
        load_backend_lock(write_json(tmp_path / "groups.json", payload))

    payload = backend_payload()
    payload["native_metadata_fields"][0]["type"] = "number"
    with pytest.raises(BackendLockError, match="metadata schema"):
        load_backend_lock(write_json(tmp_path / "metadata.json", payload))


@pytest.mark.parametrize(
    ("loader", "payload_factory", "field"),
    [
        (load_backend_lock, backend_payload, "max_input_audio_frames"),
        (load_runtime_lock, runtime_payload, "stderr_read_chunk_bytes"),
        (load_runtime_lock, runtime_payload, "stdout_max_line_bytes"),
        (load_seal_evidence, seal_payload, "runtime_uid"),
        (load_seal_evidence, seal_payload, "stdout_max_line_bytes"),
        (load_seal_evidence, seal_payload, "cpu_limit_millis"),
        (load_seal_evidence, seal_payload, "request_deadline_seconds"),
    ],
)
def test_lock_loaders_reject_zero_resource_values(
    tmp_path: Path,
    loader: Any,
    payload_factory: Any,
    field: str,
) -> None:
    payload = payload_factory()
    payload[field] = 0

    with pytest.raises(BackendLockError, match="positive integer"):
        loader(write_json(tmp_path / f"{field}.json", payload))


def test_seal_evidence_rejects_sentinels_and_final_lock_hashes(tmp_path: Path) -> None:
    payload = seal_payload()
    payload["measurements"]["request_millis"] = "unlimited"
    with pytest.raises(BackendLockError, match="positive integer"):
        load_seal_evidence(write_json(tmp_path / "sentinel.json", payload))

    payload = seal_payload()
    payload["native_host_evidence"]["backend_lock_sha256"] = "a" * 64
    with pytest.raises(BackendLockError, match="final lock hashes"):
        load_seal_evidence(write_json(tmp_path / "cycle.json", payload))


def test_lock_set_rejects_stdout_protocol_line_bound_drift(tmp_path: Path) -> None:
    audit = load_conversion_audit(write_json(tmp_path / "audit.json", audit_payload()))
    seal_data = seal_payload(audit_sha256=audit.sha256)
    seal_data["stdout_max_line_bytes"] += 1
    seal = load_seal_evidence(write_json(tmp_path / "seal.json", seal_data))
    runtime = load_runtime_lock(
        write_json(tmp_path / "runtime.json", runtime_payload(seal_sha256=seal.sha256))
    )
    backend = load_backend_lock(
        write_json(
            tmp_path / "backend.json",
            backend_payload(
                runtime_sha256=runtime.sha256,
                seal_sha256=seal.sha256,
                audit_sha256=audit.sha256,
            ),
        )
    )

    with pytest.raises(BackendLockError, match="runtime evidence mismatch"):
        validate_oaf_lock_set(backend, runtime, seal, audit)


@pytest.mark.parametrize(
    ("limit_field", "measurement_field", "scale"),
    [
        ("memory_limit_bytes", "peak_rss_bytes", 1),
        ("pid_limit", "peak_pid_count", 1),
        ("tmp_bytes", "peak_tmp_bytes", 1),
        ("shm_bytes", "peak_shm_bytes", 1),
        ("startup_deadline_seconds", "startup_millis", 1000),
        ("request_deadline_seconds", "request_millis", 1000),
    ],
)
def test_seal_evidence_rejects_underprovisioned_resource_measurement(
    tmp_path: Path,
    limit_field: str,
    measurement_field: str,
    scale: int,
) -> None:
    payload = seal_payload()
    measured = payload["measurements"][measurement_field]
    payload[limit_field] = (measured - 1) // scale

    with pytest.raises(BackendLockError, match=rf"{limit_field} must cover {measurement_field}"):
        load_seal_evidence(write_json(tmp_path / f"{limit_field}.json", payload))


@pytest.mark.parametrize(
    ("limit_field", "measurement_field", "scale"),
    [
        ("memory_limit_bytes", "peak_rss_bytes", 1),
        ("pid_limit", "peak_pid_count", 1),
        ("tmp_bytes", "peak_tmp_bytes", 1),
        ("shm_bytes", "peak_shm_bytes", 1),
        ("startup_deadline_seconds", "startup_millis", 1000),
        ("request_deadline_seconds", "request_millis", 1000),
    ],
)
def test_seal_evidence_accepts_equal_resource_measurement_boundary(
    tmp_path: Path,
    limit_field: str,
    measurement_field: str,
    scale: int,
) -> None:
    payload = seal_payload()
    measured = payload["measurements"][measurement_field]
    assert measured % scale == 0
    payload[limit_field] = measured // scale

    loaded = load_seal_evidence(write_json(tmp_path / f"{limit_field}.json", payload))

    assert loaded.payload[limit_field] * scale == measured


def test_runtime_lock_requires_exact_platform_image_environment_and_tensorflow(
    tmp_path: Path,
) -> None:
    payload = runtime_payload()
    payload["platform"] = "linux/arm64"
    with pytest.raises(BackendLockError, match="platform"):
        load_runtime_lock(write_json(tmp_path / "platform.json", payload))

    payload = runtime_payload()
    payload["environment"]["OMP_NUM_THREADS"] = "2"
    with pytest.raises(BackendLockError, match="environment"):
        load_runtime_lock(write_json(tmp_path / "environment.json", payload))

    payload = runtime_payload()
    payload["python_distributions"][0]["version"] = "2.0.0"
    with pytest.raises(BackendLockError, match="TensorFlow distribution"):
        load_runtime_lock(write_json(tmp_path / "tensorflow.json", payload))


@pytest.mark.parametrize(
    "repository",
    [
        "https://deb.debian.org/debian",
        "https://snapshot.debian.org/archive/debian/latest",
        "https://snapshot.debian.org/archive/debian/20240101",
        "http://snapshot.debian.org/archive/debian/20240101T000000Z",
        "https://snapshot.debian.org/archive/debian/٢٠٢٤٠١٠١T٠٠٠٠٠٠Z",
        "https://snapshot.debian.org/archive/debian/20241301T000000Z",
        "https://snapshot.debian.org/archive/debian/20240230T000000Z",
        "https://snapshot.debian.org/archive/debian/20240101T240000Z",
        "https://snapshot.debian.org/archive/debian/20240101T006000Z",
        "https://snapshot.debian.org/archive/debian/20240101T000060Z",
        "https://snapshot.debian.org/archive/debian/20240101T000000Z/extra",
        "https://snapshot.debian.org/archive/debian/20240101T000000Z?mirror=latest",
        "https://snapshot.debian.org/archive/debian/20240101T000000Z#latest",
    ],
)
def test_runtime_lock_rejects_nonimmutable_debian_repository(
    tmp_path: Path,
    repository: str,
) -> None:
    payload = runtime_payload()
    payload["debian_snapshot_repository"] = repository

    with pytest.raises(BackendLockError, match="snapshot-addressed"):
        load_runtime_lock(write_json(tmp_path / "runtime.json", payload))


@pytest.mark.parametrize(
    "repository",
    [
        "https://snapshot.debian.org/archive/debian/19700101T000000Z",
        "https://snapshot.debian.org/archive/debian/20000229T235959Z/",
        "https://snapshot.debian.org/archive/debian/99991231T235959Z",
    ],
)
def test_runtime_lock_accepts_calendar_valid_debian_snapshot_boundaries(
    tmp_path: Path,
    repository: str,
) -> None:
    payload = runtime_payload()
    payload["debian_snapshot_repository"] = repository

    loaded = load_runtime_lock(write_json(tmp_path / "runtime.json", payload))

    assert loaded.payload["debian_snapshot_repository"] == repository


def test_seal_evidence_requires_explicit_tensorflow_identity(tmp_path: Path) -> None:
    payload = seal_payload()
    del payload["tensorflow_abi"]
    del payload["tensorflow_build"]

    with pytest.raises(BackendLockError, match="seal evidence fields"):
        load_seal_evidence(write_json(tmp_path / "missing-tensorflow.json", payload))


def test_seal_evidence_accepts_explicit_tensorflow_identity(tmp_path: Path) -> None:
    payload = seal_payload()
    payload["tensorflow_abi"] = "cp37-cp37m-manylinux2010_x86_64"
    payload["tensorflow_build"] = "v1.15.5-0-g590d6eef7e"

    loaded = load_seal_evidence(write_json(tmp_path / "seal.json", payload))

    assert loaded.payload["tensorflow_abi"] == "cp37-cp37m-manylinux2010_x86_64"
    assert loaded.payload["tensorflow_build"] == "v1.15.5-0-g590d6eef7e"


def test_oci_layout_manifest_hash_is_required_by_runtime_and_seal(tmp_path: Path) -> None:
    runtime = runtime_payload()
    del runtime["oci_layout_manifest_sha256"]
    with pytest.raises(BackendLockError, match="runtime lock fields"):
        load_runtime_lock(write_json(tmp_path / "runtime.json", runtime))
    seal = seal_payload()
    del seal["oci_layout_manifest_sha256"]
    with pytest.raises(BackendLockError, match="seal evidence fields"):
        load_seal_evidence(write_json(tmp_path / "seal.json", seal))


def test_distribution_build_manifest_hash_is_required_by_runtime_and_seal(
    tmp_path: Path,
) -> None:
    runtime = runtime_payload()
    del runtime["distribution_build_manifest_sha256"]
    with pytest.raises(BackendLockError, match="runtime lock fields"):
        load_runtime_lock(write_json(tmp_path / "runtime.json", runtime))

    seal = seal_payload()
    del seal["distribution_build_manifest_sha256"]
    with pytest.raises(BackendLockError, match="seal evidence fields"):
        load_seal_evidence(write_json(tmp_path / "seal.json", seal))


@pytest.mark.parametrize("payload_factory", [runtime_payload, seal_payload])
def test_distribution_build_manifest_hash_must_be_lowercase_sha256(
    tmp_path: Path,
    payload_factory: Any,
) -> None:
    payload = payload_factory()
    payload["distribution_build_manifest_sha256"] = "D" * 64

    with pytest.raises(BackendLockError, match="lowercase SHA-256"):
        (
            load_runtime_lock
            if payload["schema"] == "crux.transcription-runtime-lock/v1"
            else load_seal_evidence
        )(write_json(tmp_path / "invalid-build-manifest.json", payload))


def test_lock_set_rejects_distribution_build_manifest_cross_reference_drift(
    tmp_path: Path,
) -> None:
    audit = load_conversion_audit(write_json(tmp_path / "audit.json", audit_payload()))
    seal_data = seal_payload(audit_sha256=audit.sha256)
    seal_data["distribution_build_manifest_sha256"] = "e" * 64
    seal = load_seal_evidence(write_json(tmp_path / "seal.json", seal_data))
    runtime = load_runtime_lock(
        write_json(tmp_path / "runtime.json", runtime_payload(seal_sha256=seal.sha256))
    )
    backend = load_backend_lock(
        write_json(
            tmp_path / "backend.json",
            backend_payload(
                runtime_sha256=runtime.sha256,
                seal_sha256=seal.sha256,
                audit_sha256=audit.sha256,
            ),
        )
    )

    with pytest.raises(
        BackendLockError,
        match="distribution build manifest evidence mismatch",
    ):
        validate_oaf_lock_set(backend, runtime, seal, audit)


def test_seal_evidence_accepts_reviewed_oci_layout_manifest_hash(tmp_path: Path) -> None:
    payload = seal_payload()
    payload["oci_layout_manifest_sha256"] = "3" * 64

    loaded = load_seal_evidence(write_json(tmp_path / "seal.json", payload))

    assert loaded.payload["oci_layout_manifest_sha256"] == "3" * 64


def test_seal_evidence_rejects_oci_layout_manifest_hash_drift(tmp_path: Path) -> None:
    payload = seal_payload()
    payload["oci_layout_manifest_sha256"] = "A" * 64

    with pytest.raises(BackendLockError, match="lowercase SHA-256"):
        load_seal_evidence(write_json(tmp_path / "hash-drift.json", payload))


def test_seal_evidence_with_oci_manifest_rejects_unknown_field(tmp_path: Path) -> None:
    payload = seal_payload()
    payload["oci_layout_manifest_sha256"] = "3" * 64
    payload["unreviewed_oci_claim"] = "4" * 64

    with pytest.raises(BackendLockError, match="seal evidence fields"):
        load_seal_evidence(write_json(tmp_path / "unknown.json", payload))


def test_lock_set_rejects_oci_layout_manifest_cross_reference_drift(tmp_path: Path) -> None:
    audit = load_conversion_audit(write_json(tmp_path / "audit.json", audit_payload()))
    seal_data = seal_payload(audit_sha256=audit.sha256)
    seal_data["oci_layout_manifest_sha256"] = "4" * 64
    seal = load_seal_evidence(write_json(tmp_path / "seal.json", seal_data))
    runtime = load_runtime_lock(
        write_json(tmp_path / "runtime.json", runtime_payload(seal_sha256=seal.sha256))
    )
    backend = load_backend_lock(
        write_json(
            tmp_path / "backend.json",
            backend_payload(
                runtime_sha256=runtime.sha256,
                seal_sha256=seal.sha256,
                audit_sha256=audit.sha256,
            ),
        )
    )

    with pytest.raises(BackendLockError, match="OCI layout manifest evidence mismatch"):
        validate_oaf_lock_set(backend, runtime, seal, audit)


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    [
        ("tensorflow_abi", "cp37-cp37m-drifted"),
        ("tensorflow_build", "v1.15.5-drifted"),
    ],
)
def test_lock_set_rejects_tensorflow_identity_evidence_mismatch(
    tmp_path: Path,
    field: str,
    drifted_value: str,
) -> None:
    audit = load_conversion_audit(write_json(tmp_path / "audit.json", audit_payload()))
    seal_data = seal_payload(audit_sha256=audit.sha256)
    seal_data[field] = drifted_value
    seal = load_seal_evidence(write_json(tmp_path / "seal.json", seal_data))
    runtime = load_runtime_lock(
        write_json(tmp_path / "runtime.json", runtime_payload(seal_sha256=seal.sha256))
    )
    backend = load_backend_lock(
        write_json(
            tmp_path / "backend.json",
            backend_payload(
                runtime_sha256=runtime.sha256,
                seal_sha256=seal.sha256,
                audit_sha256=audit.sha256,
            ),
        )
    )

    with pytest.raises(BackendLockError, match="TensorFlow runtime identity mismatch"):
        validate_oaf_lock_set(backend, runtime, seal, audit)


def test_lock_set_rejects_runtime_hash_mismatch(tmp_path: Path) -> None:
    backend, runtime, seal, audit = write_lock_set(tmp_path)
    payload = backend_payload(
        runtime_sha256="a" * 64,
        seal_sha256=seal.sha256,
        audit_sha256=audit.sha256,
    )
    backend = load_backend_lock(write_json(tmp_path / "mismatched-backend.json", payload))

    with pytest.raises(BackendLockError, match="runtime lock SHA-256 mismatch"):
        validate_oaf_lock_set(backend, runtime, seal, audit)


def test_lock_set_rejects_archive_component_and_inventory_mismatch(tmp_path: Path) -> None:
    backend, runtime, seal, audit = write_lock_set(tmp_path)
    payload = seal_payload(audit_sha256=audit.sha256)
    payload["checkpoint_archive"]["size"] += 1
    seal = load_seal_evidence(write_json(tmp_path / "archive-seal.json", payload))
    with pytest.raises(BackendLockError, match="checkpoint archive mismatch"):
        validate_oaf_lock_set(backend, runtime, seal, audit)

    payload = seal_payload(audit_sha256=audit.sha256)
    payload["checkpoint_components"][0]["size"] += 1
    seal = load_seal_evidence(write_json(tmp_path / "component-seal.json", payload))
    with pytest.raises(BackendLockError, match="checkpoint components mismatch"):
        validate_oaf_lock_set(backend, runtime, seal, audit)

    payload = seal_payload(audit_sha256=audit.sha256)
    payload["checkpoint_inventory"][0]["shape"] = [999]
    payload["required_inference_inventory"][0]["shape"] = [999]
    seal = load_seal_evidence(write_json(tmp_path / "inventory-seal.json", payload))
    with pytest.raises(BackendLockError, match="tensor inventories mismatch"):
        validate_oaf_lock_set(backend, runtime, seal, audit)


def test_lock_set_rejects_image_and_source_manifest_mismatch(tmp_path: Path) -> None:
    backend, _, seal, audit = write_lock_set(tmp_path)
    payload = runtime_payload(seal_sha256=seal.sha256)
    payload["runtime_image_manifest_digest"] = f"sha256:{'a' * 64}"
    runtime = load_runtime_lock(write_json(tmp_path / "image-runtime.json", payload))
    backend_data = backend_payload(
        runtime_sha256=runtime.sha256,
        seal_sha256=seal.sha256,
        audit_sha256=audit.sha256,
    )
    backend = load_backend_lock(write_json(tmp_path / "image-backend.json", backend_data))
    with pytest.raises(BackendLockError, match="runtime image manifest mismatch"):
        validate_oaf_lock_set(backend, runtime, seal, audit)

    backend, _, seal, audit = write_lock_set(tmp_path)
    payload = runtime_payload(seal_sha256=seal.sha256)
    payload["upstream_source_manifest_sha256"] = "a" * 64
    runtime = load_runtime_lock(write_json(tmp_path / "source-runtime.json", payload))
    backend_data = backend_payload(
        runtime_sha256=runtime.sha256,
        seal_sha256=seal.sha256,
        audit_sha256=audit.sha256,
    )
    backend = load_backend_lock(write_json(tmp_path / "source-backend.json", backend_data))
    with pytest.raises(BackendLockError, match="upstream source manifest mismatch"):
        validate_oaf_lock_set(backend, runtime, seal, audit)


def test_lock_set_rejects_smoke_and_audit_evidence_mismatch(tmp_path: Path) -> None:
    backend, runtime, seal, audit = write_lock_set(tmp_path)
    payload = seal_payload(audit_sha256=audit.sha256)
    payload["smoke_oracle_sha256"] = "a" * 64
    seal = load_seal_evidence(write_json(tmp_path / "smoke-seal.json", payload))
    with pytest.raises(BackendLockError, match="smoke evidence mismatch"):
        validate_oaf_lock_set(backend, runtime, seal, audit)

    payload = audit_payload()
    payload["required_inference_inventory_sha256"] = "a" * 64
    audit = load_conversion_audit(write_json(tmp_path / "inventory-audit.json", payload))
    seal_data = seal_payload(audit_sha256=audit.sha256)
    seal = load_seal_evidence(write_json(tmp_path / "audit-seal.json", seal_data))
    runtime_data = runtime_payload(seal_sha256=seal.sha256)
    runtime = load_runtime_lock(write_json(tmp_path / "audit-runtime.json", runtime_data))
    backend_data = backend_payload(
        runtime_sha256=runtime.sha256,
        seal_sha256=seal.sha256,
        audit_sha256=audit.sha256,
    )
    backend = load_backend_lock(write_json(tmp_path / "audit-backend.json", backend_data))
    with pytest.raises(BackendLockError, match="required inventory audit mismatch"):
        validate_oaf_lock_set(backend, runtime, seal, audit)


def test_conversion_audit_requires_exact_zero_of_78_evidence(tmp_path: Path) -> None:
    payload = audit_payload()
    payload["restored_required"] = ["tensor_000"]
    payload["restored_required_count"] = 1
    payload["unmatched_required"].pop(0)

    with pytest.raises(BackendLockError, match="zero restored"):
        load_conversion_audit(write_json(tmp_path / "restored.json", payload))


def test_conversion_audit_rejects_observed_hdf5_hash_drift(tmp_path: Path) -> None:
    payload = audit_payload()
    payload["observed_hdf5_sha256"] = "a" * 64

    with pytest.raises(BackendLockError, match="observed HDF5"):
        load_conversion_audit(write_json(tmp_path / "hdf5.json", payload))


def test_conversion_audit_requires_semantic_candidate_order(tmp_path: Path) -> None:
    payload = audit_payload()
    payload["candidate_matches"] = [
        candidate_match(required_name="tensor_002", candidate_name="candidate_a"),
        candidate_match(required_name="tensor_001", candidate_name="candidate_z"),
    ]

    with pytest.raises(BackendLockError, match="candidate matches must follow semantic key order"):
        load_conversion_audit(write_json(tmp_path / "candidate-order.json", payload))


def test_conversion_audit_accepts_semantic_order_independent_of_flags(tmp_path: Path) -> None:
    payload = audit_payload()
    matches = [
        candidate_match(
            required_name="tensor_001",
            candidate_name="candidate_z",
            dtype_compatible=True,
        ),
        candidate_match(
            required_name="tensor_002",
            candidate_name="candidate_a",
            dtype_compatible=False,
        ),
    ]
    payload["candidate_matches"] = matches

    loaded = load_conversion_audit(write_json(tmp_path / "semantic-order.json", payload))

    assert loaded.payload["candidate_matches"][0]["required_name"] == "tensor_001"
    assert [backend_lock_module.candidate_match_sort_key(match) for match in matches] == [
        (b"tensor_001", b"candidate_z", b"loose_substring"),
        (b"tensor_002", b"candidate_a", b"loose_substring"),
    ]


def test_conversion_audit_rejects_conflicting_duplicate_candidate_relation(
    tmp_path: Path,
) -> None:
    payload = audit_payload()
    payload["candidate_matches"] = [
        candidate_match(dtype_compatible=False),
        candidate_match(dtype_compatible=True),
    ]

    with pytest.raises(BackendLockError, match="candidate match relations must be unique"):
        load_conversion_audit(write_json(tmp_path / "conflicting-candidates.json", payload))


@pytest.mark.parametrize("match_kind", ["exact_name", "loose_substring", "dense_transpose"])
def test_conversion_audit_accepts_exact_candidate_match_kinds(
    tmp_path: Path,
    match_kind: str,
) -> None:
    payload = audit_payload()
    payload["candidate_matches"] = [candidate_match(match_kind=match_kind)]

    loaded = load_conversion_audit(write_json(tmp_path / f"{match_kind}.json", payload))

    assert loaded.payload["candidate_matches"][0]["match_kind"] == match_kind


@pytest.mark.parametrize(
    ("dtype_compatible", "shape_compatible", "message"),
    [
        (True, True, "zero restored"),
        (False, True, "assigned candidate must be dtype and shape compatible"),
        (True, False, "assigned candidate must be dtype and shape compatible"),
    ],
)
def test_conversion_audit_rejects_assigned_candidate_claims(
    tmp_path: Path,
    dtype_compatible: bool,
    shape_compatible: bool,
    message: str,
) -> None:
    payload = audit_payload()
    payload["candidate_matches"] = [
        candidate_match(
            assigned=True,
            dtype_compatible=dtype_compatible,
            shape_compatible=shape_compatible,
        )
    ]

    with pytest.raises(BackendLockError, match=message):
        load_conversion_audit(write_json(tmp_path / "assigned.json", payload))


def test_conversion_audit_rejects_arbitrary_candidate_match_kind(tmp_path: Path) -> None:
    payload = audit_payload()
    payload["candidate_matches"] = [candidate_match(match_kind="name_guess")]

    with pytest.raises(BackendLockError, match="candidate match kind"):
        load_conversion_audit(write_json(tmp_path / "match-kind.json", payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("required_name", "outside_locked_inventory", "locked required inventory"),
        ("candidate_name", "", "candidate match candidate_name"),
    ],
)
def test_conversion_audit_rejects_invalid_candidate_names(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    payload = audit_payload()
    match = candidate_match()
    match[field] = value
    payload["candidate_matches"] = [match]

    with pytest.raises(BackendLockError, match=message):
        load_conversion_audit(write_json(tmp_path / f"{field}.json", payload))


def test_conversion_audit_rejects_duplicate_candidate_rows(tmp_path: Path) -> None:
    payload = audit_payload()
    match = candidate_match()
    payload["candidate_matches"] = [match, deepcopy(match)]

    with pytest.raises(BackendLockError, match="candidate match relations must be unique"):
        load_conversion_audit(write_json(tmp_path / "duplicate-candidates.json", payload))


def test_backend_lock_rejects_directory_path(tmp_path: Path) -> None:
    with pytest.raises(BackendLockError, match="no-follow regular file"):
        load_backend_lock(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are unavailable")
def test_backend_lock_rejects_fifo_path_without_blocking(tmp_path: Path) -> None:
    fifo_path = tmp_path / "backend-lock.fifo"
    os.mkfifo(fifo_path)

    with pytest.raises(BackendLockError, match="no-follow regular file"):
        load_backend_lock(fifo_path)


def test_backend_lock_reads_normal_lock_with_one_open_and_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_json(tmp_path / "backend-lock.json", backend_payload())
    real_open = backend_lock_module.os.open
    real_read = backend_lock_module.os.read
    calls = {"open": 0, "read": 0}

    def counting_open(*args: Any, **kwargs: Any) -> int:
        calls["open"] += 1
        return real_open(*args, **kwargs)

    def counting_read(*args: Any, **kwargs: Any) -> bytes:
        calls["read"] += 1
        return real_read(*args, **kwargs)

    monkeypatch.setattr(backend_lock_module.os, "open", counting_open)
    monkeypatch.setattr(backend_lock_module.os, "read", counting_read)

    loaded = load_backend_lock(path)

    assert loaded.payload["schema"] == "crux.transcription-backend-lock/v1"
    assert calls == {"open": 1, "read": 1}


def test_loaded_backend_lock_revalidation_reproduces_every_identity(tmp_path: Path) -> None:
    loaded = load_backend_lock(write_json(tmp_path / "backend-lock.json", backend_payload()))

    reproduced = backend_lock_module.revalidate_loaded_backend_lock(loaded)

    assert reproduced == loaded
    assert reproduced is not loaded
    assert reproduced.payload is not loaded.payload
    assert reproduced.descriptor.payload["backend_lock_sha256"] == reproduced.sha256
    assert reproduced.descriptor.sha256 == loaded.descriptor.sha256


@pytest.mark.parametrize("forgery", ["lock_sha256", "descriptor_sha256", "payload"])
def test_loaded_backend_lock_revalidation_rejects_forged_dataclass(
    tmp_path: Path,
    forgery: str,
) -> None:
    loaded = load_backend_lock(write_json(tmp_path / "backend-lock.json", backend_payload()))
    if forgery == "lock_sha256":
        forged = LoadedBackendLock(
            path=loaded.path,
            payload=loaded.payload,
            sha256="f" * 64,
            descriptor=loaded.descriptor,
            max_input_audio_frames=loaded.max_input_audio_frames,
        )
    elif forgery == "descriptor_sha256":
        forged = LoadedBackendLock(
            path=loaded.path,
            payload=loaded.payload,
            sha256=loaded.sha256,
            descriptor=type(loaded.descriptor)(
                payload=loaded.descriptor.payload,
                sha256="f" * 64,
            ),
            max_input_audio_frames=loaded.max_input_audio_frames,
        )
    else:
        forged = LoadedBackendLock(
            path=loaded.path,
            payload={"backend_id": loaded.payload["backend_id"]},
            sha256=loaded.sha256,
            descriptor=loaded.descriptor,
            max_input_audio_frames=loaded.max_input_audio_frames,
        )

    with pytest.raises(BackendLockError):
        backend_lock_module.revalidate_loaded_backend_lock(forged)


@pytest.mark.parametrize(
    "forgery",
    [
        "path_type",
        "payload_type",
        "lock_sha256_type",
        "descriptor_type",
        "descriptor_payload_type",
        "descriptor_sha256_type",
        "frame_bound_type",
    ],
)
def test_loaded_backend_lock_revalidation_normalizes_malformed_runtime_types(
    tmp_path: Path,
    forgery: str,
) -> None:
    loaded = load_backend_lock(write_json(tmp_path / "backend-lock.json", backend_payload()))
    if forgery == "descriptor_payload_type":
        descriptor = replace(loaded.descriptor, payload=None)  # type: ignore[arg-type]
        malformed = replace(loaded, descriptor=descriptor)
    elif forgery == "descriptor_sha256_type":
        descriptor = replace(loaded.descriptor, sha256=None)  # type: ignore[arg-type]
        malformed = replace(loaded, descriptor=descriptor)
    else:
        replacements: dict[str, object] = {
            "path_type": {"path": None},
            "payload_type": {"payload": None},
            "lock_sha256_type": {"sha256": None},
            "descriptor_type": {"descriptor": None},
            "frame_bound_type": {"max_input_audio_frames": True},
        }
        malformed = replace(loaded, **replacements[forgery])  # type: ignore[arg-type]

    with pytest.raises(BackendLockError):
        backend_lock_module.revalidate_loaded_backend_lock(malformed)


@pytest.mark.parametrize(
    "forgery",
    [
        "path_type",
        "payload_type",
        "lock_sha256_type",
        "descriptor_type",
        "frame_bound_type",
    ],
)
def test_public_prepare_normalizes_malformed_loaded_lock_before_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forgery: str,
) -> None:
    loaded = load_backend_lock(write_json(tmp_path / "backend-lock.json", backend_payload()))
    replacements: dict[str, object] = {
        "path_type": {"path": None},
        "payload_type": {"payload": None},
        "lock_sha256_type": {"sha256": None},
        "descriptor_type": {"descriptor": None},
        "frame_bound_type": {"max_input_audio_frames": True},
    }
    malformed = replace(loaded, **replacements[forgery])  # type: ignore[arg-type]
    monkeypatch.setattr(
        backend_prepare_module,
        "_open_directory_chain",
        lambda *args, **kwargs: pytest.fail("malformed lock reached the filesystem"),
    )
    monkeypatch.setattr(
        backend_prepare_module,
        "_open_download_url",
        lambda *args, **kwargs: pytest.fail("malformed lock reached the network"),
    )

    outcome = prepare_oaf_backend(
        PrepareBackendRequest(
            backend_id=OFFICIAL_BACKEND_ID,
            cache_root=tmp_path / "cache",
            archive_path=tmp_path / "checkpoint.zip",
            download=False,
        ),
        backend_lock=malformed,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2
    assert not (tmp_path / "cache").exists()


def test_sealed_prepare_rejects_a_request_that_contradicts_the_final_lock_before_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = write_json(tmp_path / "backend-lock.json", backend_payload())
    backend_lock = load_backend_lock(lock_path)
    request_path = (
        Path(__file__).parents[2]
        / "config"
        / "benchmark"
        / "backends"
        / f"{OFFICIAL_BACKEND_ID}.checkpoint-acquisition-request.json"
    )
    monkeypatch.setattr(
        backend_prepare_module,
        "_open_directory_chain",
        lambda *args, **kwargs: pytest.fail("contradictory request reached the cache"),
    )

    outcome = prepare_oaf_backend(
        PrepareBackendRequest(
            backend_id=OFFICIAL_BACKEND_ID,
            cache_root=tmp_path / "cache",
            archive_path=None,
            download=False,
            acquisition_request_path=request_path,
            evidence_output_path=None,
            backend_lock_path=lock_path,
        ),
        backend_lock=backend_lock,
    )

    assert (outcome.status, outcome.exit_code, outcome.model_cache_path) == (
        "integrity_failed",
        2,
        None,
    )
