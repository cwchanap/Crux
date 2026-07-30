"""Calibration-only CPython entrypoint with a stdlib pre-import boundary."""

from __future__ import annotations

import hashlib
import os
import re
import socket
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Sequence

try:
    from calibration_protocol import (
        CalibrationProtocolFailure,
        read_verified_calibration_wav,
        validate_calibration_request,
    )
    from protocol import (
        ProtocolFailure,
        canonical_json_bytes,
        load_authenticated_object,
    )
except ImportError:
    from runtime.oaf_tf1.calibration_protocol import (
        CalibrationProtocolFailure,
        read_verified_calibration_wav,
        validate_calibration_request,
    )
    from runtime.oaf_tf1.protocol import (
        ProtocolFailure,
        canonical_json_bytes,
        load_authenticated_object,
    )

EXPECTED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
}
EXPECTED_RESOURCE_CEILING = {
    "cpu_limit_millis": 2_000,
    "memory_limit_bytes": 4_294_967_296,
    "monitor_interval_millis": 10,
    "pid_limit": 256,
    "request_deadline_seconds": 1_800,
    "shm_bytes": 1_073_741_824,
    "startup_deadline_seconds": 300,
    "stderr_max_line_bytes": 65_536,
    "stderr_read_chunk_bytes": 65_536,
    "stderr_ring_buffer_bytes": 1_048_576,
    "stdout_max_line_bytes": 134_217_728,
    "tmp_bytes": 1_073_741_824,
}
EXPECTED_IMAGE_BUILD = {
    "annotations": [],
    "buildkit_image": (
        "moby/buildkit@sha256:63db51c9b30208a7c2b1c40392c7ebb9ce2f85ba238a18a85420f8f5ea2d4684"
    ),
    "buildkit_version": "v0.31.2",
    "buildx_binary_sha256": ("d41ece72044243b4f58b343441ae37446d9c29a7d6b5e11c61847bbcf8f7dfda"),
    "buildx_binary_size": 65_265_826,
    "buildx_binary_url": (
        "https://github.com/docker/buildx/releases/download/v0.35.0/buildx-v0.35.0.linux-amd64"
    ),
    "buildx_version": "v0.35.0",
    "compression": "gzip",
    "compression_level": 6,
    "dockerfile_frontend": (
        "docker/dockerfile-upstream@sha256:"
        "3d6d54b33351b396a910d33248754b86b1d7dd838b4eeb9575d8903a209f6516"
    ),
    "dockerfile_frontend_version": "1.25.0",
    "exporter": "oci",
    "exporter_tar": False,
    "force_compression": False,
    "inline_cache": False,
    "multi_platform_deterministic": True,
    "oci_archive": {
        "compression": "none",
        "final_zero_blocks": 2,
        "format": "posix-ustar",
        "gid": 0,
        "gname": "",
        "member_mode": 420,
        "member_types": "regular-files-only",
        "mtime": 0,
        "path_order": "utf8-byte",
        "uid": 0,
        "uname": "",
    },
    "oci_media_types": True,
    "platform": "linux/amd64",
    "provenance": False,
    "rewrite_timestamp": True,
    "sbom": False,
    "source_date_epoch": 0,
}
FINAL_MODE_MOUNTS = (
    Path("/run/crux/backend-lock.json"),
    Path("/run/crux/runtime-lock.json"),
    Path("/run/crux/seal-evidence.json"),
)
BOOTSTRAP_REQUEST_PATH = Path("/run/crux/calibration-bootstrap-request.json")
BOOTSTRAP_EVIDENCE_PATH = Path("/run/crux/calibration-bootstrap-evidence.json")
CHECKPOINT_REQUEST_PATH = Path("/run/crux/checkpoint-acquisition-request.json")
CHECKPOINT_EVIDENCE_PATH = Path("/run/crux/checkpoint-acquisition-evidence.json")
BASE_SYSTEM_REQUEST_PATH = Path("/run/crux/base-system-package-request.json")
BASE_SYSTEM_EVIDENCE_PATH = Path("/run/crux/base-system-package-evidence.json")
RUNTIME_IMAGE_CONFIG_DIGEST_PATH = Path("/run/crux/runtime-image-config-digest.txt")
RUNNER_SOURCE_MANIFEST_PATH = Path("/opt/crux/runtime/runner-source-manifest.json")
RUNNER_SOURCE_ROOT = Path("/opt/crux")
UPSTREAM_SOURCE_MANIFEST_PATH = Path("/opt/crux/vendor/source-manifest.json")
UPSTREAM_SOURCE_ROOT = Path("/opt/crux/upstream")
DISTRIBUTION_BUILD_MANIFEST_PATH = Path(
    "/opt/crux/runtime/oaf_tf1/distribution-build-manifest.json"
)
INSTRUMENTATION_PATCH_PATH = Path("/opt/crux/runtime/oaf_tf1/patches/capture-emitted-frame.patch")
MODEL_CACHE_ROOT = Path("/model")
CALIBRATION_INPUT_ROOT = Path("/input")
CALIBRATION_EVIDENCE_ROOT = Path("/output")
CALIBRATION_REQUEST_MAX_BYTES = 4_096

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_BOOTSTRAP_REQUEST_KEYS = frozenset(
    {
        "backend_id",
        "base_image_manifest_digest",
        "base_system_package_request_sha256",
        "build_context_manifest_sha256",
        "checkpoint_acquisition_request_sha256",
        "container_restrictions",
        "distribution_build_manifest_sha256",
        "environment",
        "image_build",
        "instrumentation_patch_sha256",
        "python_coerce_c_locale",
        "resource_ceiling",
        "runner_source_manifest_sha256",
        "runtime_gid",
        "runtime_uid",
        "schema",
        "upstream_source_manifest_sha256",
    }
)
_BOOTSTRAP_EVIDENCE_KEYS = frozenset(
    {
        "base_image_config_digest",
        "base_image_layer_diff_ids",
        "base_image_layer_digests",
        "build_context_manifest_sha256",
        "calibration_bootstrap_request_sha256",
        "image_build",
        "native_host_attestation_bundle_sha256",
        "native_host_evidence",
        "oci_layout_archive",
        "oci_layout_manifest_sha256",
        "runtime_image_config_digest",
        "runtime_image_index_digest",
        "runtime_image_layer_diff_ids",
        "runtime_image_layer_digests",
        "runtime_image_manifest_digest",
        "schema",
    }
)
_CHECKPOINT_REQUEST_KEYS = frozenset(
    {
        "archive",
        "archive_members",
        "backend_id",
        "checkpoint_url",
        "published_component_names",
        "schema",
    }
)
_CHECKPOINT_EVIDENCE_KEYS = frozenset(
    {
        "acquisition_mode",
        "archive",
        "archive_members",
        "cache_path",
        "model_artifact_set_sha256",
        "published_components",
        "request_sha256",
        "schema",
    }
)
_BASE_SYSTEM_REQUEST_KEYS = frozenset(
    {
        "additional_system_packages",
        "base_image",
        "base_image_archive_keyring_sha256",
        "base_image_manifest_digest",
        "platform",
        "required_probes",
        "schema",
    }
)
_BASE_SYSTEM_EVIDENCE_KEYS = frozenset(
    {
        "additional_system_packages",
        "base_image_archive_keyring_sha256",
        "base_image_manifest_digest",
        "native_host_evidence",
        "package_inventory",
        "package_inventory_sha256",
        "probes",
        "request_sha256",
        "schema",
    }
)
_RUNNER_MANIFEST_KEYS = frozenset({"covered_roots", "files", "schema"})
_UPSTREAM_MANIFEST_KEYS = frozenset(
    {
        "covered_roots",
        "files",
        "schema",
        "upstream_commit",
        "upstream_repository",
    }
)
_CONTAINER_RESTRICTIONS = {
    "drop_capabilities": ["ALL"],
    "network": "none",
    "no_new_privileges": True,
    "platform": "linux/amd64",
    "read_only_root": True,
}
_REQUIRED_PROBES = (
    "base_python_version",
    "runtime_python_version",
    "runtime_tensorflow_version",
    "runtime_smoke",
)


class CalibrationAuthorityFailure(Exception):
    """One mounted calibration authority failed closed before numeric imports."""


@dataclass(frozen=True)
class CalibrationStartup:
    """Authenticated identities needed by the measurement runtime."""

    bootstrap_request_sha256: str
    bootstrap_evidence_sha256: str
    base_system_package_evidence_sha256: str
    checkpoint_acquisition_evidence_sha256: str
    checkpoint_components: tuple[Mapping[str, Any], ...]
    runner_source_manifest_sha256: str
    runtime_image_config_digest: str
    upstream_source_manifest_sha256: str


def discard_interpreter_bootstrap_environment() -> None:
    if os.environ.pop("PYTHONCOERCECLOCALE", None) != "0":
        raise SystemExit(2)


def validate_process_environment() -> None:
    if dict(os.environ) != EXPECTED_ENVIRONMENT:
        raise SystemExit(2)


def reject_final_mode_mounts(paths: tuple[Path, ...] = FINAL_MODE_MOUNTS) -> None:
    if any(os.path.lexists(path) for path in paths):
        raise SystemExit(2)


def preflight_request(
    content: bytes,
    *,
    authorized_max_input_audio_frames: int,
):
    """Validate the calibration request before importing any numeric dependency."""

    return validate_calibration_request(
        content,
        authorized_max_input_audio_frames=authorized_max_input_audio_frames,
    )


def serve_calibration_requests(
    *,
    stdin: BinaryIO,
    stdout: BinaryIO,
    transcribe: Callable[[Any], Sequence[Mapping[str, Any]]],
    input_root: Path,
    authorized_max_input_audio_frames: int,
    stdout_max_line_bytes: int,
) -> None:
    """Serve the calibration-only protocol with an explicit inference counter."""

    inference_call_count = 0
    while True:
        content = stdin.readline(CALIBRATION_REQUEST_MAX_BYTES + 1)
        if not content:
            return
        if len(content) > CALIBRATION_REQUEST_MAX_BYTES:
            raise CalibrationProtocolFailure("calibration request exceeds its byte bound")
        request = preflight_request(
            content,
            authorized_max_input_audio_frames=authorized_max_input_audio_frames,
        )
        before = inference_call_count
        if request.audio_frame_count > authorized_max_input_audio_frames:
            response: Mapping[str, Any] = {
                "audio_sha256": request.audio_sha256,
                "inference_call_count_after": before,
                "inference_call_count_before": before,
                "native_events": [],
                "prediction_sha256": None,
                "rejected_before_inference": True,
                "request_id": request.request_id,
                "type": request.request_type,
            }
        else:
            verified = read_verified_calibration_wav(request, Path(input_root))
            native_events = list(transcribe(verified))
            prediction = canonical_json_bytes(native_events, trailing_newline=False)
            inference_call_count += 1
            response = {
                "audio_sha256": verified.sha256,
                "inference_call_count_after": inference_call_count,
                "inference_call_count_before": before,
                "native_events": native_events,
                "prediction_sha256": hashlib.sha256(prediction).hexdigest(),
                "rejected_before_inference": False,
                "request_id": request.request_id,
                "type": request.request_type,
            }
        encoded = canonical_json_bytes(response, trailing_newline=True)
        if len(encoded) > stdout_max_line_bytes:
            raise CalibrationProtocolFailure("calibration response exceeds its byte bound")
        stdout.write(encoded)
        stdout.flush()


def publish_calibration_tensor_coverage(model: Any, output_root: Path) -> None:
    """Publish diagnostic-only tensor evidence when the candidate mount is present."""

    root = Path(output_root)
    if not root.exists():
        return
    try:
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise OSError("calibration output root is unsafe")
        payload = {
            "active_predict_dropout": False,
            "checkpoint_inventory": list(model.checkpoint_inventory),
            "non_inference_inventory": list(model.non_inference_inventory),
            "note_sequence_byte_parity": True,
            "required_inference_inventory": list(model.required_inference_inventory),
            "schema": "crux.oaf-tensor-coverage/v1",
            "uninitialized_required": [],
        }
        content = canonical_json_bytes(payload, trailing_newline=True)
        target = root / "tensor-coverage.json"
        if target.exists():
            if _read_regular(target) != content:
                raise OSError("calibration tensor evidence already differs")
            return
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
    except OSError:
        raise CalibrationAuthorityFailure(
            "calibration tensor evidence publication failed"
        ) from None


def authenticate_calibration_startup(
    *,
    bootstrap_request_path: Path = BOOTSTRAP_REQUEST_PATH,
    bootstrap_evidence_path: Path = BOOTSTRAP_EVIDENCE_PATH,
    checkpoint_request_path: Path = CHECKPOINT_REQUEST_PATH,
    checkpoint_evidence_path: Path = CHECKPOINT_EVIDENCE_PATH,
    base_system_request_path: Path = BASE_SYSTEM_REQUEST_PATH,
    base_system_evidence_path: Path = BASE_SYSTEM_EVIDENCE_PATH,
    runner_source_manifest_path: Path = RUNNER_SOURCE_MANIFEST_PATH,
    runner_source_root: Path = RUNNER_SOURCE_ROOT,
    upstream_source_manifest_path: Path = UPSTREAM_SOURCE_MANIFEST_PATH,
    upstream_source_root: Path = UPSTREAM_SOURCE_ROOT,
    distribution_build_manifest_path: Path = DISTRIBUTION_BUILD_MANIFEST_PATH,
    instrumentation_patch_path: Path = INSTRUMENTATION_PATCH_PATH,
    model_cache_root: Path = MODEL_CACHE_ROOT,
    runtime_image_config_digest_path: Path = RUNTIME_IMAGE_CONFIG_DIGEST_PATH,
) -> CalibrationStartup:
    """Authenticate the complete calibration authority chain using only stdlib."""

    try:
        bootstrap_request = _load(
            bootstrap_request_path,
            "calibration_bootstrap_request",
            _BOOTSTRAP_REQUEST_KEYS,
            "crux.oaf-calibration-bootstrap-request/v1",
        )
        _validate_bootstrap_request(bootstrap_request.payload)
        bootstrap_evidence = _load(
            bootstrap_evidence_path,
            "calibration_bootstrap_evidence",
            _BOOTSTRAP_EVIDENCE_KEYS,
            "crux.oaf-calibration-bootstrap-evidence/v1",
        )
        _validate_bootstrap_evidence(
            bootstrap_request.payload,
            bootstrap_request.sha256,
            bootstrap_evidence.payload,
        )
        runtime_config_digest = _read_runtime_config_digest(runtime_image_config_digest_path)
        if runtime_config_digest != bootstrap_evidence.payload["runtime_image_config_digest"]:
            _fail("runtime image config digest does not match bootstrap evidence")

        checkpoint_request = _load(
            checkpoint_request_path,
            "checkpoint_acquisition_request",
            _CHECKPOINT_REQUEST_KEYS,
            "crux.oaf-checkpoint-acquisition-request/v1",
            expected_sha256=bootstrap_request.payload["checkpoint_acquisition_request_sha256"],
        )
        published_components = _validate_checkpoint_request(checkpoint_request.payload)
        checkpoint_evidence = _load(
            checkpoint_evidence_path,
            "checkpoint_acquisition_evidence",
            _CHECKPOINT_EVIDENCE_KEYS,
            "crux.oaf-checkpoint-acquisition-evidence/v1",
        )
        _validate_checkpoint_evidence(
            checkpoint_request.payload,
            checkpoint_request.sha256,
            checkpoint_evidence.payload,
            published_components,
        )
        _validate_checkpoint_cache(model_cache_root, published_components)

        base_request = _load(
            base_system_request_path,
            "base_system_package_request",
            _BASE_SYSTEM_REQUEST_KEYS,
            "crux.oaf-base-system-package-request/v1",
            expected_sha256=bootstrap_request.payload["base_system_package_request_sha256"],
        )
        _validate_base_system_request(
            base_request.payload,
            bootstrap_request.payload["base_image_manifest_digest"],
        )
        base_evidence = _load(
            base_system_evidence_path,
            "base_system_package_evidence",
            _BASE_SYSTEM_EVIDENCE_KEYS,
            "crux.oaf-base-system-package-evidence/v1",
        )
        _validate_base_system_evidence(
            base_request.payload,
            base_request.sha256,
            base_evidence.payload,
        )
        if (
            base_evidence.payload["native_host_evidence"]
            != bootstrap_evidence.payload["native_host_evidence"]
        ):
            _fail("base-system and bootstrap host evidence differ")

        runner_manifest = _load(
            runner_source_manifest_path,
            "runner_source_manifest",
            _RUNNER_MANIFEST_KEYS,
            "crux.oaf-runner-source-manifest/v1",
            expected_sha256=bootstrap_request.payload["runner_source_manifest_sha256"],
        )
        upstream_manifest = _load(
            upstream_source_manifest_path,
            "upstream_source_manifest",
            _UPSTREAM_MANIFEST_KEYS,
            "crux.oaf-upstream-source-manifest/v1",
            expected_sha256=bootstrap_request.payload["upstream_source_manifest_sha256"],
        )
        _validate_source_manifest(
            runner_manifest.payload,
            runner_source_root,
            frozenset({"byte_length", "path", "sha256"}),
        )
        _validate_source_manifest(
            upstream_manifest.payload,
            upstream_source_root,
            frozenset({"license", "path", "sha256"}),
        )
        _require_file_hash(
            distribution_build_manifest_path,
            bootstrap_request.payload["distribution_build_manifest_sha256"],
            "distribution build manifest",
        )
        _require_file_hash(
            instrumentation_patch_path,
            bootstrap_request.payload["instrumentation_patch_sha256"],
            "instrumentation patch",
        )
        return CalibrationStartup(
            bootstrap_request_sha256=bootstrap_request.sha256,
            bootstrap_evidence_sha256=bootstrap_evidence.sha256,
            base_system_package_evidence_sha256=base_evidence.sha256,
            checkpoint_acquisition_evidence_sha256=checkpoint_evidence.sha256,
            checkpoint_components=tuple(published_components),
            runner_source_manifest_sha256=runner_manifest.sha256,
            runtime_image_config_digest=runtime_config_digest,
            upstream_source_manifest_sha256=upstream_manifest.sha256,
        )
    except (CalibrationAuthorityFailure, ProtocolFailure, OSError, TypeError, ValueError):
        raise CalibrationAuthorityFailure("calibration startup authority is invalid") from None


def _load(path, label, keys, schema, expected_sha256=None):
    return load_authenticated_object(
        Path(path),
        label=label,
        exact_keys=keys,
        expected_schema=schema,
        expected_sha256=expected_sha256,
    )


def _validate_bootstrap_request(payload: Mapping[str, Any]) -> None:
    if (
        payload["backend_id"] != "magenta-egmd-tf1-94529798-8hit-v1"
        or payload["runtime_uid"] != 65532
        or type(payload["runtime_uid"]) is not int
        or payload["runtime_gid"] != 65532
        or type(payload["runtime_gid"]) is not int
        or payload["python_coerce_c_locale"] != "0"
        or payload["environment"] != EXPECTED_ENVIRONMENT
        or payload["container_restrictions"] != _CONTAINER_RESTRICTIONS
        or payload["image_build"] != EXPECTED_IMAGE_BUILD
    ):
        _fail("bootstrap request fixed authority differs")
    _require_digest(payload["base_image_manifest_digest"])
    for field in (
        "base_system_package_request_sha256",
        "build_context_manifest_sha256",
        "checkpoint_acquisition_request_sha256",
        "distribution_build_manifest_sha256",
        "instrumentation_patch_sha256",
        "runner_source_manifest_sha256",
        "upstream_source_manifest_sha256",
    ):
        _require_sha256(payload[field])
    ceiling = payload["resource_ceiling"]
    if (
        not isinstance(ceiling, Mapping)
        or ceiling != EXPECTED_RESOURCE_CEILING
        or any(type(value) is not int for value in ceiling.values())
    ):
        _fail("bootstrap resource ceiling is invalid")


def _validate_bootstrap_evidence(
    request: Mapping[str, Any],
    request_sha256: str,
    evidence: Mapping[str, Any],
) -> None:
    if (
        evidence["calibration_bootstrap_request_sha256"] != request_sha256
        or evidence["build_context_manifest_sha256"] != request["build_context_manifest_sha256"]
        or evidence["image_build"] != request["image_build"]
    ):
        _fail("bootstrap evidence does not reproduce its request")
    for field in (
        "native_host_attestation_bundle_sha256",
        "oci_layout_manifest_sha256",
    ):
        _require_sha256(evidence[field])
    for field in (
        "base_image_config_digest",
        "runtime_image_config_digest",
        "runtime_image_index_digest",
        "runtime_image_manifest_digest",
    ):
        _require_digest(evidence[field])
    base_layers = _digest_array(evidence["base_image_layer_digests"])
    base_diff_ids = _digest_array(evidence["base_image_layer_diff_ids"])
    layers = _digest_array(evidence["runtime_image_layer_digests"])
    diff_ids = _digest_array(evidence["runtime_image_layer_diff_ids"])
    if (
        not base_layers
        or len(base_layers) != len(base_diff_ids)
        or len(layers) != len(diff_ids)
        or layers[: len(base_layers)] != base_layers
        or diff_ids[: len(base_diff_ids)] != base_diff_ids
    ):
        _fail("bootstrap evidence layer identity is invalid")
    _validate_identity(evidence["oci_layout_archive"])
    _validate_native_host_evidence(evidence["native_host_evidence"])


def _validate_checkpoint_request(
    payload: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if (
        payload["backend_id"] != "magenta-egmd-tf1-94529798-8hit-v1"
        or not isinstance(payload["checkpoint_url"], str)
        or not payload["checkpoint_url"].startswith("https://")
    ):
        _fail("checkpoint request identity is invalid")
    _validate_identity(payload["archive"])
    members = payload["archive_members"]
    if not isinstance(members, list) or not members:
        _fail("checkpoint request members are invalid")
    published = []
    pointer_count = 0
    for member in members:
        if not isinstance(member, Mapping) or set(member) != {
            "name",
            "role",
            "sha256",
            "size",
        }:
            _fail("checkpoint request member is invalid")
        _validate_identity({key: member[key] for key in ("name", "sha256", "size")})
        if member["role"] == "pointer":
            pointer_count += 1
        elif member["role"] == "published_component":
            published.append({key: member[key] for key in ("name", "sha256", "size")})
        else:
            _fail("checkpoint request member role is invalid")
    names = payload["published_component_names"]
    if (
        pointer_count != 1
        or not published
        or not isinstance(names, list)
        or names != [row["name"] for row in published]
        or len(set(names)) != len(names)
    ):
        _fail("checkpoint request published components are invalid")
    return published


def _validate_checkpoint_evidence(
    request: Mapping[str, Any],
    request_sha256: str,
    evidence: Mapping[str, Any],
    published_components: list[Mapping[str, Any]],
) -> None:
    if (
        evidence["request_sha256"] != request_sha256
        or evidence["acquisition_mode"] not in {"download", "archive", "cache_verify"}
        or evidence["archive"] != request["archive"]
        or evidence["archive_members"] != request["archive_members"]
        or evidence["published_components"] != published_components
        or not _safe_relative_path(evidence["cache_path"])
    ):
        _fail("checkpoint evidence does not reproduce its request")
    expected_set_sha256 = hashlib.sha256(
        canonical_json_bytes(published_components, trailing_newline=False)
    ).hexdigest()
    if evidence["model_artifact_set_sha256"] != expected_set_sha256:
        _fail("checkpoint artifact-set hash differs")


def _validate_checkpoint_cache(
    root: Path,
    published_components: list[Mapping[str, Any]],
) -> None:
    expected_names = [row["name"] for row in published_components]
    try:
        entries = sorted(os.scandir(str(root)), key=lambda entry: entry.name.encode("utf-8"))
    except OSError:
        _fail("checkpoint cache is unavailable")
    if [entry.name for entry in entries] != sorted(
        expected_names, key=lambda name: name.encode("utf-8")
    ):
        _fail("checkpoint cache entry set differs")
    by_name = {row["name"]: row for row in published_components}
    for entry in entries:
        row = by_name[entry.name]
        content = _read_regular(Path(entry.path))
        if len(content) != row["size"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
            _fail("checkpoint cache component differs")


def _validate_base_system_request(
    payload: Mapping[str, Any],
    expected_manifest_digest: Any,
) -> None:
    if (
        payload["base_image"] != "python:3.7.17-slim-bullseye"
        or payload["base_image_manifest_digest"] != expected_manifest_digest
        or payload["platform"] != "linux/amd64"
        or payload["additional_system_packages"] != []
        or tuple(payload["required_probes"]) != _REQUIRED_PROBES
    ):
        _fail("base-system request is invalid")
    _require_sha256(payload["base_image_archive_keyring_sha256"])
    _require_digest(payload["base_image_manifest_digest"])


def _validate_base_system_evidence(
    request: Mapping[str, Any],
    request_sha256: str,
    evidence: Mapping[str, Any],
) -> None:
    if (
        evidence["request_sha256"] != request_sha256
        or evidence["base_image_manifest_digest"] != request["base_image_manifest_digest"]
        or evidence["base_image_archive_keyring_sha256"]
        != request["base_image_archive_keyring_sha256"]
        or evidence["additional_system_packages"] != []
    ):
        _fail("base-system evidence does not reproduce its request")
    inventory = evidence["package_inventory"]
    if not isinstance(inventory, list) or not inventory:
        _fail("base-system inventory is invalid")
    ordering = []
    for row in inventory:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"architecture", "name", "version"}
            or any(not isinstance(row[field], str) or not row[field] for field in row)
        ):
            _fail("base-system package row is invalid")
        ordering.append(
            (
                row["name"].encode("utf-8"),
                row["version"].encode("utf-8"),
                row["architecture"].encode("utf-8"),
            )
        )
    if ordering != sorted(ordering) or len(set(ordering)) != len(ordering):
        _fail("base-system package ordering is invalid")
    inventory_sha256 = hashlib.sha256(
        canonical_json_bytes(inventory, trailing_newline=True)
    ).hexdigest()
    if evidence["package_inventory_sha256"] != inventory_sha256:
        _fail("base-system inventory hash differs")
    probes = evidence["probes"]
    if (
        not isinstance(probes, list)
        or len(probes) != len(_REQUIRED_PROBES)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"name", "value"}
            or row["name"] != _REQUIRED_PROBES[index]
            or not isinstance(row["value"], str)
            or not row["value"]
            for index, row in enumerate(probes)
        )
    ):
        _fail("base-system probes are invalid")
    _validate_native_host_evidence(evidence["native_host_evidence"])


def _validate_native_host_evidence(value: Any) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"kind", "official_execution_allowed", "payload", "sha256"}
        or value["kind"] not in {"github_hosted", "orchestrator_signed", "approved_local"}
        or value["official_execution_allowed"] is not True
        or not isinstance(value["payload"], Mapping)
    ):
        _fail("native host evidence is invalid")
    _require_sha256(value["sha256"])
    expected = hashlib.sha256(
        canonical_json_bytes(value["payload"], trailing_newline=False)
    ).hexdigest()
    if value["sha256"] != expected:
        _fail("native host evidence payload hash differs")


def _validate_source_manifest(
    payload: Mapping[str, Any],
    source_root: Path,
    file_keys: frozenset[str],
) -> None:
    roots = payload["covered_roots"]
    files = payload["files"]
    if (
        not isinstance(roots, list)
        or not roots
        or roots != sorted(roots, key=lambda value: value.encode("utf-8"))
        or len(set(roots)) != len(roots)
        or any(not _safe_relative_path(root) for root in roots)
        or not isinstance(files, list)
        or not files
    ):
        _fail("source manifest roots or files are invalid")
    for root in roots:
        try:
            metadata = os.lstat(str(Path(source_root) / root))
        except OSError:
            _fail("source manifest root is unavailable")
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            _fail("source manifest root is unsafe")
    paths = []
    for row in files:
        if not isinstance(row, Mapping) or set(row) != file_keys:
            _fail("source manifest file row is invalid")
        relative = row["path"]
        if not _safe_relative_path(relative) or not _path_is_covered(relative, roots):
            _fail("source manifest path is invalid")
        content = _read_regular(Path(source_root) / relative)
        _require_sha256(row["sha256"])
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            _fail("source manifest file hash differs")
        if "byte_length" in row and (
            type(row["byte_length"]) is not int or row["byte_length"] != len(content)
        ):
            _fail("source manifest byte length differs")
        if "license" in row and (not isinstance(row["license"], str) or not row["license"]):
            _fail("source manifest license is invalid")
        paths.append(relative)
    if paths != sorted(paths, key=lambda value: value.encode("utf-8")) or len(set(paths)) != len(
        paths
    ):
        _fail("source manifest file ordering is invalid")


def _path_is_covered(path: str, roots: list[str]) -> bool:
    if "/" not in path:
        return True
    if any(path == root or path.startswith(root + "/") for root in roots):
        return True
    parent = path.rsplit("/", 1)[0]
    return any(root.startswith(parent + "/") for root in roots)


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    return not value.startswith("/") and all(
        part not in {"", ".", ".."} for part in value.split("/")
    )


def _validate_identity(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"name", "sha256", "size"}:
        _fail("file identity shape is invalid")
    if (
        not isinstance(value["name"], str)
        or not value["name"]
        or any(character in value["name"] for character in ("/", "\\", "\x00"))
        or value["name"] in {".", ".."}
        or type(value["size"]) is not int
        or value["size"] <= 0
    ):
        _fail("file identity is invalid")
    _require_sha256(value["sha256"])


def _digest_array(value: Any) -> list[str]:
    if not isinstance(value, list) or any(_DIGEST.fullmatch(item) is None for item in value):
        _fail("digest array is invalid")
    return value


def _require_sha256(value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail("SHA-256 identity is invalid")
    return value


def _require_digest(value: Any) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        _fail("digest identity is invalid")
    return value


def _read_regular(path: Path, maximum_bytes: int = 16 * 1024 * 1024) -> bytes:
    descriptor = os.open(
        str(path),
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            _fail("mounted file is unsafe or exceeds its bound")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(content) != before.st_size or (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            _fail("mounted file changed while being read")
        return content
    finally:
        os.close(descriptor)


def _require_file_hash(path: Path, expected_sha256: Any, label: str) -> None:
    expected = _require_sha256(expected_sha256)
    if hashlib.sha256(_read_regular(Path(path))).hexdigest() != expected:
        _fail(label + " hash differs")


def _read_runtime_config_digest(path: Path) -> str:
    content = _read_regular(Path(path), maximum_bytes=72)
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        _fail("runtime image config digest file is not canonical")
    try:
        value = content[:-1].decode("ascii")
    except UnicodeDecodeError:
        _fail("runtime image config digest is not ASCII")
    return _require_digest(value)


def _fail(message: str):
    raise CalibrationAuthorityFailure(message)


def main() -> int:
    discard_interpreter_bootstrap_environment()
    validate_process_environment()
    reject_final_mode_mounts()
    try:
        startup = authenticate_calibration_startup()
    except CalibrationAuthorityFailure:
        os.write(2, b"code=calibration_authority_invalid count=1\n")
        return 2
    vendor_root = "/opt/crux/vendor"
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)
    try:
        import random

        import numpy as np
        import tensorflow.compat.v1 as tf

        try:
            from oaf_backend import build_calibration_model, transcribe_canonical_wav
        except ImportError:
            from runtime.oaf_tf1.oaf_backend import (
                build_calibration_model,
                transcribe_canonical_wav,
            )

        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)
        if tf.__version__ != "1.15.5" or tf.version.GIT_VERSION != "v1.15.5-0-g590d6eef7e":
            raise CalibrationAuthorityFailure("TensorFlow build identity differs")
        model = build_calibration_model(startup.checkpoint_components)
        publish_calibration_tensor_coverage(model, CALIBRATION_EVIDENCE_ROOT)
        coverage = model.handle.coverage
        ready = {
            "base_system_package_evidence_sha256": (startup.base_system_package_evidence_sha256),
            "calibration_bootstrap_request_sha256": (startup.bootstrap_request_sha256),
            "checkpoint_acquisition_evidence_sha256": (
                startup.checkpoint_acquisition_evidence_sha256
            ),
            "checkpoint_inventory_sha256": coverage.checkpoint_inventory_sha256,
            "non_inference_count": coverage.non_inference_count,
            "non_inference_inventory_sha256": (coverage.non_inference_inventory_sha256),
            "process_instance_id": socket.gethostname(),
            "protocol_schema": "crux.oaf-calibration-runner/v1",
            "required_inference_count": coverage.required_count,
            "required_inference_inventory_sha256": (coverage.required_inventory_sha256),
            "restored_inference_count": coverage.restored_count,
            "runner_source_manifest_sha256": startup.runner_source_manifest_sha256,
            "runtime_image_config_digest": startup.runtime_image_config_digest,
            "tensorflow_abi": "cp37-cp37m-manylinux2010_x86_64",
            "tensorflow_build": tf.version.GIT_VERSION,
            "type": "ready",
            "upstream_source_manifest_sha256": startup.upstream_source_manifest_sha256,
        }
        encoded = canonical_json_bytes(ready, trailing_newline=True)
        if len(encoded) > EXPECTED_RESOURCE_CEILING["stdout_max_line_bytes"]:
            raise CalibrationProtocolFailure("calibration ready response is oversized")
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.flush()
        serve_calibration_requests(
            stdin=sys.stdin.buffer,
            stdout=sys.stdout.buffer,
            transcribe=lambda verified: transcribe_canonical_wav(model.handle, verified),
            input_root=CALIBRATION_INPUT_ROOT,
            authorized_max_input_audio_frames=26_214_378,
            stdout_max_line_bytes=EXPECTED_RESOURCE_CEILING["stdout_max_line_bytes"],
        )
        return 0
    except (CalibrationAuthorityFailure, CalibrationProtocolFailure):
        os.write(2, b"code=calibration_execution_invalid count=1\n")
        return 2
    except BaseException:  # pylint: disable=broad-exception-caught
        os.write(2, b"code=calibration_internal_failure count=1\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
