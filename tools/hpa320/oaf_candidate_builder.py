#!/usr/bin/env python3
"""Generate the complete pre-publication OaF seal candidate from native evidence."""

# Candidate generation deliberately keeps the acyclic payload order visible.
# pylint: disable=too-many-arguments,too-many-locals,too-many-statements,too-many-lines

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from src.benchmark.backend_identity import JsonValue, canonical_json_bytes, sha256_hex
from src.benchmark.backend_lock import (
    NATIVE_OUTPUT_BINS,
    REQUIRED_ENVIRONMENT,
    REQUIRED_HPARAMS,
    TRAINING_GROUPS,
)
from src.benchmark.backend_publication import read_regular_file_no_follow
from src.benchmark.checkpoint_acquisition import (
    CheckpointAcquisitionEvidence,
    CheckpointAcquisitionRequest,
)
from tools.hpa320.audit_legacy_tf2_conversion import (
    CANDIDATE_MANIFEST_NAME,
    CANDIDATE_MANIFEST_SCHEMA,
)
from tools.hpa320.oaf_host_attestation import NativeHostAttestationBundle
from tools.hpa320.oaf_system_packages import (
    BaseSystemPackageEvidence,
    BaseSystemPackageRequest,
)
from tools.hpa320.seal_oaf_backend import (
    _BACKEND_ID,
    _CANDIDATE_ARTIFACT_PATHS,
    _CANDIDATE_ARTIFACTS,
    _HASH_FIELDS,
    CalibrationBootstrapEvidence,
    CalibrationBootstrapRequest,
    CalibrationMeasurementRequest,
    MeasurementRow,
    SealIntegrityError,
    SealProfileRequest,
    build_host_adapter_source_manifest,
)

_REQUIREMENT_COMMENT = re.compile(
    r"# filename=(?P<filename>[^ ]+) byte_length=(?P<size>[1-9][0-9]*)\Z"
)
_REQUIREMENT_ROW = re.compile(
    r"(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^ ]+) " r"--hash=sha256:(?P<sha256>[0-9a-f]{64})\Z"
)


def build_native_candidate(
    *,
    staging: Path,
    repository_root: Path,
    tensor_coverage_path: Path,
    native_events: Sequence[Mapping[str, Any]],
    bootstrap_request: CalibrationBootstrapRequest,
    bootstrap: CalibrationBootstrapEvidence,
    bootstrap_evidence_path: Path,
    measurement_request: CalibrationMeasurementRequest,
    measurement_sha256: str,
    measurement_rows: Sequence[MeasurementRow],
    profile: SealProfileRequest,
    checkpoint_request: CheckpointAcquisitionRequest,
    checkpoint: CheckpointAcquisitionEvidence,
    model_cache: Path,
    base_request: BaseSystemPackageRequest,
    base: BaseSystemPackageEvidence,
    bundle: NativeHostAttestationBundle,
    host_payload: Mapping[str, JsonValue],
    boundary_probes: Sequence[JsonValue],
) -> None:
    """Generate every candidate byte before the seal validator sees the directory."""

    repository = Path(repository_root)
    tensor_content = _read(Path(tensor_coverage_path), "calibration tensor coverage")
    tensor_payload = _canonical_object(tensor_content, "calibration tensor coverage")
    checkpoint_inventory = _inventory(tensor_payload, "checkpoint_inventory")
    required_inventory = _inventory(tensor_payload, "required_inference_inventory")
    non_inference_inventory = _inventory(tensor_payload, "non_inference_inventory")
    _write(
        Path(staging) / _CANDIDATE_ARTIFACT_PATHS["tensor_coverage"],
        tensor_content,
    )

    audit_target = Path(staging) / _CANDIDATE_ARTIFACT_PATHS["conversion_audit"]
    _generate_conversion_audit(
        repository=repository,
        model_cache=Path(model_cache),
        required_inventory=required_inventory,
        output=audit_target,
    )
    audit_sha256 = sha256_hex(_read(audit_target, "conversion audit"))

    distributions = _runtime_distributions(repository)
    advisory_target = Path(staging) / _CANDIDATE_ARTIFACT_PATHS["advisory_snapshot"]
    _write_json(
        advisory_target,
        {
            "assessment_status": "review_required",
            "python_distributions": distributions,
            "requirements_lock_sha256": sha256_hex(
                _read(
                    repository / "runtime/oaf_tf1/requirements.lock",
                    "runtime requirements lock",
                )
            ),
            "schema": "crux.oaf-advisory-snapshot/v1",
        },
    )
    advisory_sha256 = sha256_hex(_read(advisory_target, "advisory snapshot"))
    security_target = Path(staging) / _CANDIDATE_ARTIFACT_PATHS["security_scan"]
    _write_json(
        security_target,
        {
            "advisory_snapshot_sha256": advisory_sha256,
            "findings": None,
            "review_status": "candidate_review_required",
            "scan_status": "not_performed_no_pinned_advisory_database",
            "schema": "crux.oaf-security-scan/v1",
        },
    )
    security_sha256 = sha256_hex(_read(security_target, "security scan"))

    archive_identity = _mapping(bootstrap.payload["oci_layout_archive"], "OCI archive identity")
    archive_source = _locate_archive(
        repository=repository,
        bootstrap_evidence_path=Path(bootstrap_evidence_path),
        identity=archive_identity,
    )
    archive_target = Path(staging) / _CANDIDATE_ARTIFACT_PATHS["oci_layout_archive"]
    _write(archive_target, _read(archive_source, "authenticated OCI archive"))
    oci_target = Path(staging) / _CANDIDATE_ARTIFACT_PATHS["oci_layout_manifest"]
    _write_json(
        oci_target,
        {
            "archive": _plain(archive_identity),
            "base_image_config_digest": bootstrap.payload["base_image_config_digest"],
            "base_image_layer_diff_ids": bootstrap.payload["base_image_layer_diff_ids"],
            "base_image_layer_digests": bootstrap.payload["base_image_layer_digests"],
            "config_digest": bootstrap.payload["runtime_image_config_digest"],
            "image_manifest_digest": bootstrap.payload["runtime_image_manifest_digest"],
            "index_digest": bootstrap.payload["runtime_image_index_digest"],
            "layer_diff_ids": bootstrap.payload["runtime_image_layer_diff_ids"],
            "layer_digests": bootstrap.payload["runtime_image_layer_digests"],
            "schema": "crux.oaf-oci-layout-manifest/v1",
        },
    )
    oci_sha256 = sha256_hex(_read(oci_target, "OCI layout manifest"))

    smoke_content = _read(
        repository / "tests/fixtures/oaf_tf1_smoke/canonical.wav",
        "canonical smoke audio",
    )
    smoke_target = Path(staging) / _CANDIDATE_ARTIFACT_PATHS["smoke_audio"]
    _write(smoke_target, smoke_content)
    if not native_events:
        raise SealIntegrityError("calibration smoke prediction is empty")
    smoke_events = _plain(list(native_events))
    smoke_prediction = canonical_json_bytes(smoke_events, trailing_newline=False)
    _write(
        Path(staging) / _CANDIDATE_ARTIFACT_PATHS["smoke_prediction"],
        smoke_prediction,
    )
    smoke_sha256 = sha256_hex(smoke_content)
    oracle_target = Path(staging) / _CANDIDATE_ARTIFACT_PATHS["smoke_oracle"]
    _write_json(
        oracle_target,
        {
            "input_audio_frame_count": (len(smoke_content) - 44) // 2,
            "input_audio_sha256": smoke_sha256,
            "input_view_id": "oaf-tf1-canonical-smoke-v1",
            "native_events": smoke_events,
            "schema": "crux.oaf-smoke-oracle/v1",
            "source_audio_id": "oaf-tf1-canonical-smoke-v1",
            "source_audio_sha256": smoke_sha256,
        },
    )
    oracle_sha256 = sha256_hex(_read(oracle_target, "smoke oracle"))

    host_manifest_target = Path(staging) / _CANDIDATE_ARTIFACT_PATHS["host_adapter_source_manifest"]
    _write_json(host_manifest_target, build_host_adapter_source_manifest(repository))
    host_manifest_sha256 = sha256_hex(_read(host_manifest_target, "host adapter source manifest"))
    source_hashes = {
        field: sha256_hex(_read(repository / relative, relative))
        for relative, field in _HASH_FIELDS
    }
    components = [
        {"name": row.name, "sha256": row.sha256, "size": row.size}
        for row in checkpoint_request.archive_members
        if row.role == "published_component"
    ]
    archive = {
        "name": checkpoint_request.archive.name,
        "sha256": checkpoint_request.archive.sha256,
        "size": checkpoint_request.archive.size,
    }
    packages = [
        {
            "architecture": row.architecture,
            "name": row.name,
            "version": row.version,
        }
        for row in base.package_inventory
    ]
    fingerprint = _host_fingerprint(host_payload)
    seal = _seal_payload(
        profile=profile,
        bootstrap_request=bootstrap_request,
        bootstrap=bootstrap,
        measurement_request=measurement_request,
        measurement_sha256=measurement_sha256,
        measurement_rows=measurement_rows,
        checkpoint=checkpoint,
        archive=archive,
        components=components,
        checkpoint_inventory=checkpoint_inventory,
        required_inventory=required_inventory,
        non_inference_inventory=non_inference_inventory,
        base_request=base_request,
        base=base,
        packages=packages,
        bundle=bundle,
        host_payload=host_payload,
        fingerprint=fingerprint,
        boundary_probes=boundary_probes,
        distributions=distributions,
        source_hashes=source_hashes,
        host_manifest_sha256=host_manifest_sha256,
        audit_sha256=audit_sha256,
        advisory_sha256=advisory_sha256,
        security_sha256=security_sha256,
        archive_identity=archive_identity,
        oci_sha256=oci_sha256,
        smoke_sha256=smoke_sha256,
        oracle_sha256=oracle_sha256,
        smoke_prediction_sha256=sha256_hex(smoke_prediction),
        tensor_sha256=sha256_hex(tensor_content),
    )
    seal_target = Path(staging) / _CANDIDATE_ARTIFACT_PATHS["seal_evidence"]
    _write_json(seal_target, seal)
    seal_sha256 = sha256_hex(_read(seal_target, "seal evidence"))

    runtime = _runtime_payload(
        profile=profile,
        bootstrap_request=bootstrap_request,
        bootstrap=bootstrap,
        base_request=base_request,
        base=base,
        packages=packages,
        distributions=distributions,
        source_hashes=source_hashes,
        oci_sha256=oci_sha256,
        seal_sha256=seal_sha256,
    )
    runtime_target = Path(staging) / _CANDIDATE_ARTIFACT_PATHS["runtime_lock"]
    _write_json(runtime_target, runtime)
    runtime_sha256 = sha256_hex(_read(runtime_target, "runtime lock"))

    backend = _backend_payload(
        profile=profile,
        bootstrap=bootstrap,
        checkpoint_request=checkpoint_request,
        checkpoint=checkpoint,
        archive=archive,
        components=components,
        checkpoint_inventory=checkpoint_inventory,
        required_inventory=required_inventory,
        non_inference_inventory=non_inference_inventory,
        source_hashes=source_hashes,
        host_manifest_sha256=host_manifest_sha256,
        audit_sha256=audit_sha256,
        smoke_sha256=smoke_sha256,
        oracle_sha256=oracle_sha256,
        runtime_sha256=runtime_sha256,
        seal_sha256=seal_sha256,
    )
    backend_target = Path(staging) / _CANDIDATE_ARTIFACT_PATHS["backend_lock"]
    _write_json(backend_target, backend)
    backend_sha256 = sha256_hex(_read(backend_target, "backend lock"))

    component_identity = sha256_hex(canonical_json_bytes(components, trailing_newline=False))
    artifacts = [
        {
            "path": relative,
            "role": role,
            "sha256": sha256_hex(_read(Path(staging) / relative, f"candidate {role}")),
        }
        for role, relative in _CANDIDATE_ARTIFACTS
    ]
    _write_json(
        Path(staging) / CANDIDATE_MANIFEST_NAME,
        {
            "artifacts": artifacts,
            "backend_lock_payload_sha256": backend_sha256,
            "calibration_bootstrap_evidence_sha256": bootstrap.sha256,
            "calibration_measurement_evidence_sha256": measurement_sha256,
            "checkpoint_components": components,
            "checkpoint_prefix": (f"sha256/{component_identity}/model.ckpt-569400"),
            "model_artifact_set_sha256": component_identity,
            "native_host_attestation_bundle_sha256": bundle.sha256,
            "required_inference_inventory_sha256": sha256_hex(
                canonical_json_bytes(required_inventory, trailing_newline=False)
            ),
            "runtime_lock_payload_sha256": runtime_sha256,
            "schema": CANDIDATE_MANIFEST_SCHEMA,
            "seal_evidence_payload_sha256": seal_sha256,
            "seal_profile_request_sha256": profile.sha256,
        },
    )


def _seal_payload(  # noqa: PLR0913
    *,
    profile: SealProfileRequest,
    bootstrap_request: CalibrationBootstrapRequest,
    bootstrap: CalibrationBootstrapEvidence,
    measurement_request: CalibrationMeasurementRequest,
    measurement_sha256: str,
    measurement_rows: Sequence[MeasurementRow],
    checkpoint: CheckpointAcquisitionEvidence,
    archive: JsonValue,
    components: JsonValue,
    checkpoint_inventory: JsonValue,
    required_inventory: JsonValue,
    non_inference_inventory: JsonValue,
    base_request: BaseSystemPackageRequest,
    base: BaseSystemPackageEvidence,
    packages: JsonValue,
    bundle: NativeHostAttestationBundle,
    host_payload: Mapping[str, JsonValue],
    fingerprint: JsonValue,
    boundary_probes: Sequence[JsonValue],
    distributions: JsonValue,
    source_hashes: Mapping[str, str],
    host_manifest_sha256: str,
    audit_sha256: str,
    advisory_sha256: str,
    security_sha256: str,
    archive_identity: Mapping[str, JsonValue],
    oci_sha256: str,
    smoke_sha256: str,
    oracle_sha256: str,
    smoke_prediction_sha256: str,
    tensor_sha256: str,
) -> dict[str, JsonValue]:
    return {
        "additional_system_packages": [],
        "advisory_snapshot_sha256": advisory_sha256,
        "base_image_archive_keyring_sha256": (base_request.base_image_archive_keyring_sha256),
        "base_image_config_digest": bootstrap.payload["base_image_config_digest"],
        "base_image_layer_diff_ids": bootstrap.payload["base_image_layer_diff_ids"],
        "base_image_layer_digests": bootstrap.payload["base_image_layer_digests"],
        "base_image_manifest_digest": base_request.base_image_manifest_digest,
        "base_system_package_evidence_sha256": base.sha256,
        "base_system_package_inventory": packages,
        "base_system_package_inventory_sha256": base.package_inventory_sha256,
        "base_system_package_request_sha256": base.request_sha256,
        "boundary_probes": list(boundary_probes),
        "build_context_manifest_sha256": (bootstrap_request.build_context_manifest_sha256),
        "calibration_bootstrap_evidence_sha256": bootstrap.sha256,
        "calibration_bootstrap_request_sha256": bootstrap_request.sha256,
        "calibration_measurement_evidence_sha256": measurement_sha256,
        "calibration_measurement_request_sha256": measurement_request.sha256,
        "checkpoint_acquisition_evidence_sha256": checkpoint.sha256,
        "checkpoint_acquisition_request_sha256": checkpoint.request_sha256,
        "checkpoint_archive": archive,
        "checkpoint_components": components,
        "checkpoint_inventory": checkpoint_inventory,
        "cpu_limit_millis": profile.payload["cpu_limit_millis"],
        "distribution_build_manifest_sha256": source_hashes["distribution_build_manifest_sha256"],
        "host_adapter_source_manifest_sha256": host_manifest_sha256,
        "instrumentation_patch_sha256": source_hashes["instrumentation_patch_sha256"],
        "legacy_conversion_coverage_sha256": audit_sha256,
        "max_input_audio_frames": profile.max_input_audio_frames,
        "measurements": [_row(row) for row in measurement_rows],
        "memory_limit_bytes": profile.payload["memory_limit_bytes"],
        "native_host_attestation_bundle_sha256": bundle.sha256,
        "native_host_evidence": _plain(host_payload),
        "non_inference_inventory": non_inference_inventory,
        "oci_layout_archive": _plain(archive_identity),
        "oci_layout_manifest_sha256": oci_sha256,
        "pid_limit": profile.payload["pid_limit"],
        "python_distributions": distributions,
        "reference_host_numeric_fingerprint": fingerprint,
        "request_deadline_seconds": profile.payload["request_deadline_seconds"],
        "required_inference_inventory": required_inventory,
        "runner_source_manifest_sha256": source_hashes["runner_source_manifest_sha256"],
        "runtime_gid": profile.payload["runtime_gid"],
        "runtime_image_config_digest": bootstrap.payload["runtime_image_config_digest"],
        "runtime_image_index_digest": bootstrap.payload["runtime_image_index_digest"],
        "runtime_image_layer_diff_ids": bootstrap.payload["runtime_image_layer_diff_ids"],
        "runtime_image_layer_digests": bootstrap.payload["runtime_image_layer_digests"],
        "runtime_image_manifest_digest": bootstrap.payload["runtime_image_manifest_digest"],
        "runtime_uid": profile.payload["runtime_uid"],
        "schema": "crux.backend-seal-evidence/v1",
        "seal_profile_request_sha256": profile.sha256,
        "security_scan_sha256": security_sha256,
        "shm_bytes": profile.payload["shm_bytes"],
        "smoke_audio_sha256": smoke_sha256,
        "smoke_oracle_sha256": oracle_sha256,
        "smoke_prediction_sha256": smoke_prediction_sha256,
        "startup_deadline_seconds": profile.payload["startup_deadline_seconds"],
        "stderr_max_line_bytes": profile.payload["stderr_max_line_bytes"],
        "stderr_read_chunk_bytes": profile.payload["stderr_read_chunk_bytes"],
        "stderr_ring_buffer_bytes": profile.payload["stderr_ring_buffer_bytes"],
        "stdout_max_line_bytes": profile.payload["stdout_max_line_bytes"],
        "tensor_coverage_sha256": tensor_sha256,
        "tensorflow_abi": "cp37-cp37m-manylinux2010_x86_64",
        "tensorflow_build": "v1.15.5-0-g590d6eef7e",
        "tmp_bytes": profile.payload["tmp_bytes"],
        "upstream_source_manifest_sha256": source_hashes["upstream_source_manifest_sha256"],
    }


def _runtime_payload(  # noqa: PLR0913
    *,
    profile: SealProfileRequest,
    bootstrap_request: CalibrationBootstrapRequest,
    bootstrap: CalibrationBootstrapEvidence,
    base_request: BaseSystemPackageRequest,
    base: BaseSystemPackageEvidence,
    packages: JsonValue,
    distributions: JsonValue,
    source_hashes: Mapping[str, str],
    oci_sha256: str,
    seal_sha256: str,
) -> dict[str, JsonValue]:
    return {
        "additional_system_packages": [],
        "base_image": base_request.base_image,
        "base_image_archive_keyring_sha256": (base_request.base_image_archive_keyring_sha256),
        "base_image_config_digest": bootstrap.payload["base_image_config_digest"],
        "base_image_layer_diff_ids": bootstrap.payload["base_image_layer_diff_ids"],
        "base_image_layer_digests": bootstrap.payload["base_image_layer_digests"],
        "base_image_manifest_digest": base_request.base_image_manifest_digest,
        "base_system_package_evidence_sha256": base.sha256,
        "base_system_package_inventory": packages,
        "base_system_package_inventory_sha256": base.package_inventory_sha256,
        "base_system_package_request_sha256": base.request_sha256,
        "build_context_manifest_sha256": (bootstrap_request.build_context_manifest_sha256),
        "calibration_bootstrap_evidence_sha256": bootstrap.sha256,
        "calibration_bootstrap_request_sha256": bootstrap_request.sha256,
        "distribution_build_manifest_sha256": source_hashes["distribution_build_manifest_sha256"],
        "environment": dict(REQUIRED_ENVIRONMENT),
        "image_build": _plain(bootstrap.payload["image_build"]),
        "oci_layout_manifest_sha256": oci_sha256,
        "platform": base_request.platform,
        "python_distributions": distributions,
        "python_version": "3.7.17",
        "runner_source_manifest_sha256": source_hashes["runner_source_manifest_sha256"],
        "runtime_image_config_digest": bootstrap.payload["runtime_image_config_digest"],
        "runtime_image_manifest_digest": bootstrap.payload["runtime_image_manifest_digest"],
        "schema": "crux.transcription-runtime-lock/v1",
        "seal_evidence_sha256": seal_sha256,
        "stderr_max_line_bytes": profile.payload["stderr_max_line_bytes"],
        "stderr_read_chunk_bytes": profile.payload["stderr_read_chunk_bytes"],
        "stderr_ring_buffer_bytes": profile.payload["stderr_ring_buffer_bytes"],
        "stdout_max_line_bytes": profile.payload["stdout_max_line_bytes"],
        "tensorflow_abi": "cp37-cp37m-manylinux2010_x86_64",
        "tensorflow_build": "v1.15.5-0-g590d6eef7e",
        "upstream_source_manifest_sha256": source_hashes["upstream_source_manifest_sha256"],
    }


def _backend_payload(  # noqa: PLR0913
    *,
    profile: SealProfileRequest,
    bootstrap: CalibrationBootstrapEvidence,
    checkpoint_request: CheckpointAcquisitionRequest,
    checkpoint: CheckpointAcquisitionEvidence,
    archive: JsonValue,
    components: JsonValue,
    checkpoint_inventory: JsonValue,
    required_inventory: JsonValue,
    non_inference_inventory: JsonValue,
    source_hashes: Mapping[str, str],
    host_manifest_sha256: str,
    audit_sha256: str,
    smoke_sha256: str,
    oracle_sha256: str,
    runtime_sha256: str,
    seal_sha256: str,
) -> dict[str, JsonValue]:
    return {
        "architecture_id": "magenta-oaf-model-tpu-drums-v1",
        "backend_id": _BACKEND_ID,
        "checkpoint_acquisition_evidence_sha256": checkpoint.sha256,
        "checkpoint_acquisition_request_sha256": checkpoint.request_sha256,
        "checkpoint_archive": archive,
        "checkpoint_components": components,
        "checkpoint_inventory": checkpoint_inventory,
        "checkpoint_url": checkpoint_request.checkpoint_url,
        "descriptor_schema": "crux.transcription-backend-descriptor/v1",
        "drum_prediction_map": "",
        "execution_report_schema": "crux.backend-execution-report/v1",
        "host_adapter_source_manifest_sha256": host_manifest_sha256,
        "hparams": _plain(REQUIRED_HPARAMS),
        "hparams_source": ("magenta/models/onsets_frames_transcription/configs.py:drums"),
        "legacy_conversion_coverage_sha256": audit_sha256,
        "legacy_score_report_schema": "crux.legacy-score-report/v1",
        "max_input_audio_frames": profile.max_input_audio_frames,
        "model_id": "magenta-egmd-ckpt-569400-v1",
        "native_metadata_fields": [
            {
                "name": "upstream_8hit_group_id",
                "nullable": True,
                "type": "string",
            }
        ],
        "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
        "native_output_bins": _plain(NATIVE_OUTPUT_BINS),
        "native_output_space_id": "magenta-oaf-midi88-a0-v1",
        "non_inference_inventory": non_inference_inventory,
        "prediction_schema": "crux.drum-prediction-events/v1",
        "protocol_schema": "crux.transcription-runner/v1",
        "required_inference_inventory": required_inventory,
        "runtime_image_manifest_digest": bootstrap.payload["runtime_image_manifest_digest"],
        "runtime_lock_sha256": runtime_sha256,
        "schema": "crux.transcription-backend-lock/v1",
        "seal_evidence_sha256": seal_sha256,
        "serialization": {
            "encoding": "utf-8",
            "final_newline": True,
            "key_order": "lexicographic",
            "whitespace": "none",
        },
        "smoke_audio_sha256": smoke_sha256,
        "smoke_oracle_sha256": oracle_sha256,
        "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
        "training_groups": _plain(TRAINING_GROUPS),
        "upstream_repository": "https://github.com/magenta/magenta",
        "upstream_source_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
        "upstream_source_manifest_sha256": source_hashes["upstream_source_manifest_sha256"],
        "verification_report_schema": "crux.backend-verification-report/v1",
    }


def _generate_conversion_audit(
    *,
    repository: Path,
    model_cache: Path,
    required_inventory: list[JsonValue],
    output: Path,
) -> None:
    try:
        import tensorflow as tf

        from src.cli.convert import convert_tf1_checkpoint_to_tf2
        from tools.hpa320.audit_legacy_tf2_conversion import main as audit_main
    except ImportError:
        raise SealIntegrityError("legacy conversion audit dependencies are unavailable") from None
    with tempfile.TemporaryDirectory(prefix=".hpa320-legacy-audit-") as directory:
        root = Path(directory)
        required = root / "required-inventory.json"
        required.write_bytes(canonical_json_bytes(required_inventory, trailing_newline=True))
        hdf5 = root / "legacy-conversion.weights.h5"
        tf.keras.utils.set_random_seed(0)
        checkpoint_prefix = Path(model_cache) / "model.ckpt-569400"
        convert_tf1_checkpoint_to_tf2(os.fspath(checkpoint_prefix), os.fspath(hdf5))
        output.parent.mkdir(parents=True, exist_ok=True)
        result = audit_main(
            [
                "--checkpoint-prefix",
                os.fspath(checkpoint_prefix),
                "--hdf5",
                os.fspath(hdf5),
                "--expected-hdf5-sha256",
                sha256_hex(_read(hdf5, "legacy conversion HDF5")),
                "--required-inventory",
                os.fspath(required),
                "--converter-source",
                os.fspath(repository / "src/cli/convert.py"),
                "--tf2-model-source",
                os.fspath(repository / "src/app/tf2_magenta_model.py"),
                "--output",
                os.fspath(output),
            ]
        )
        if result != 0:
            raise SealIntegrityError("legacy conversion audit generation failed")


def _runtime_distributions(repository: Path) -> list[JsonValue]:
    content = _read(
        repository / "runtime/oaf_tf1/requirements.lock",
        "runtime requirements lock",
    )
    try:
        lines = content.decode("utf-8", errors="strict").splitlines()
    except UnicodeError:
        raise SealIntegrityError("runtime requirements lock is not UTF-8") from None
    rows: list[JsonValue] = []
    pending: tuple[str, int] | None = None
    for line in lines:
        comment = _REQUIREMENT_COMMENT.fullmatch(line)
        if comment is not None:
            pending = (comment["filename"], int(comment["size"]))
            continue
        requirement = _REQUIREMENT_ROW.fullmatch(line)
        if requirement is None:
            continue
        if pending is None:
            raise SealIntegrityError("runtime requirement lacks wheel identity")
        filename, size = pending
        wheel_content = _read(
            repository / "runtime/oaf_tf1/wheelhouse/runtime" / filename,
            f"runtime wheel {filename}",
        )
        if len(wheel_content) != size or sha256_hex(wheel_content) != requirement["sha256"]:
            raise SealIntegrityError("runtime wheel differs from its requirements lock")
        rows.append(
            {
                "filename": filename,
                "name": requirement["name"].lower().replace("_", "-"),
                "sha256": requirement["sha256"],
                "version": requirement["version"],
            }
        )
        pending = None
    if not rows:
        raise SealIntegrityError("runtime distribution inventory is empty")
    return rows


def _locate_archive(
    *,
    repository: Path,
    bootstrap_evidence_path: Path,
    identity: Mapping[str, JsonValue],
) -> Path:
    name = cast(str, identity["name"])
    candidates = (
        Path(bootstrap_evidence_path).parent / name,
        repository / "artifacts/benchmark/backends/hpa320-bootstrap" / name,
    )
    matches: list[Path] = []
    for candidate in candidates:
        try:
            content = _read(candidate, "bootstrap OCI archive")
        except SealIntegrityError:
            continue
        if len(content) == identity["size"] and sha256_hex(content) == identity["sha256"]:
            matches.append(candidate)
    if not matches:
        raise SealIntegrityError("accepted bootstrap OCI archive bytes are unavailable")
    if len({candidate.resolve() for candidate in matches}) != 1:
        raise SealIntegrityError("multiple accepted bootstrap OCI archives are ambiguous")
    return matches[0]


def _host_fingerprint(host_payload: Mapping[str, JsonValue]) -> JsonValue:
    nested = _mapping(host_payload.get("payload"), "candidate host evidence payload")
    return _plain(
        _mapping(
            nested.get("host_numeric_fingerprint"),
            "candidate host numeric fingerprint",
        )
    )


def _row(row: MeasurementRow) -> dict[str, JsonValue]:
    return {
        "exit_code": row.exit_code,
        "inference_call_count_after": row.inference_call_count_after,
        "inference_call_count_before": row.inference_call_count_before,
        "input_audio_sha256": row.input_audio_sha256,
        "input_frame_count": row.input_frame_count,
        "oom_killed": row.oom_killed,
        "peak_cpu_millis": row.peak_cpu_millis,
        "peak_pid_count": row.peak_pid_count,
        "peak_rss_bytes": row.peak_rss_bytes,
        "peak_shm_bytes": row.peak_shm_bytes,
        "peak_tmp_bytes": row.peak_tmp_bytes,
        "prediction_sha256": row.prediction_sha256,
        "process_instance_id": row.process_instance_id,
        "repetition": row.repetition,
        "request_millis": row.request_millis,
        "signal": row.signal,
        "startup_millis": row.startup_millis,
        "stderr_max_line_bytes": row.stderr_max_line_bytes,
        "stdout_max_line_bytes": row.stdout_max_line_bytes,
    }


def _inventory(payload: Mapping[str, JsonValue], field: str) -> list[JsonValue]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise SealIntegrityError(f"calibration tensor coverage {field} is invalid")
    return value


def _canonical_object(content: bytes, label: str) -> Mapping[str, JsonValue]:
    from src.benchmark.backend_identity import strict_json_loads

    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise SealIntegrityError(f"{label} is not canonical JSON")
    try:
        value = strict_json_loads(content[:-1], require_canonical=True)
    except ValueError:
        raise SealIntegrityError(f"{label} is not canonical JSON") from None
    return _mapping(value, label)


def _mapping(value: object, label: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise SealIntegrityError(f"{label} is not an object")
    return cast(Mapping[str, JsonValue], value)


def _plain(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return cast(JsonValue, value)


def _read(path: Path, label: str) -> bytes:
    try:
        return read_regular_file_no_follow(Path(path))
    except OSError:
        raise SealIntegrityError(f"{label} is missing or unsafe") from None


def _write(path: Path, content: bytes) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _write_json(path: Path, payload: JsonValue) -> None:
    _write(path, canonical_json_bytes(payload, trailing_newline=True))
