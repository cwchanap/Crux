from __future__ import annotations

# The sibling lock suite supplies hand-written, independently validated frozen payloads.
# pylint: disable=duplicate-code,too-many-arguments,too-many-lines,too-many-locals
# pylint: disable=too-many-positional-arguments,too-many-statements
import hashlib
import importlib.util
import json
import os
import platform
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pytest

import tools.hpa320.seal_oaf_backend as seal_module
from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex, strict_json_loads
from tools.hpa320.seal_oaf_backend import SealError

HOST_PATHS = (
    "src/benchmark/backend_attestation.py",
    "src/benchmark/backend_identity.py",
    "src/benchmark/backend_lock.py",
    "src/benchmark/backend_process.py",
    "src/benchmark/backend_publication.py",
    "src/benchmark/backends/base.py",
    "src/benchmark/backends/oaf_tf1.py",
    "src/benchmark/input_view.py",
    "src/benchmark/prediction_artifact.py",
)


def _lock_fixtures() -> ModuleType:
    path = Path(__file__).with_name("test_backend_lock.py")
    spec = importlib.util.spec_from_file_location("_oaf_lock_fixture_payloads", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOCKS = _lock_fixtures()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    return path


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _native_host_payload() -> dict[str, object]:
    return {
        "api_record_sha256": "a" * 64,
        "approved_labels": ["Linux", "X64"],
        "job_id": 123,
        "run_url": "https://github.com/acme/crux/actions/runs/456/job/123",
        "runner_arch": "X64",
        "runner_os": "Linux",
        "workflow_commit": "b" * 40,
    }


def _native_host_record(
    *,
    kind: str = "github_hosted",
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    evidence_payload = _native_host_payload() if payload is None else payload
    return {
        "kind": kind,
        "official_execution_allowed": True,
        "payload": evidence_payload,
        "sha256": sha256_hex(canonical_json_bytes(evidence_payload)),
    }


def _set_native_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("RUNNER_OS", "Linux")
    monkeypatch.setenv("RUNNER_ARCH", "X64")


def _host_manifest_payload(repository: Path) -> dict[str, object]:
    return {
        "covered_roots": list(HOST_PATHS),
        "files": [
            {
                "path": relative,
                "sha256": hashlib.sha256((repository / relative).read_bytes()).hexdigest(),
            }
            for relative in HOST_PATHS
        ],
        "schema": "crux.oaf-host-adapter-source-manifest/v1",
    }


def _make_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    for index, relative in enumerate(HOST_PATHS):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"host source {index}\n".encode())
    _write_json(
        repository / "runtime/oaf_tf1/source-manifest.json",
        {
            "covered_roots": ["runtime/oaf_tf1/vendor/magenta"],
            "files": [],
            "schema": "crux.oaf-upstream-source-manifest/v1",
            "upstream_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
            "upstream_repository": "https://github.com/magenta/magenta.git",
        },
    )
    _write_json(
        repository / "runtime/oaf_tf1/runner-source-manifest.json",
        {
            "covered_roots": ["runtime/oaf_tf1"],
            "files": [],
            "schema": "crux.oaf-runner-source-manifest/v1",
        },
    )
    _write_json(
        repository / "runtime/oaf_tf1/distribution-build-manifest.json",
        {"schema": "crux.distribution-build-manifest/v2"},
    )
    patch = repository / "runtime/oaf_tf1/patches/capture-emitted-frame.patch"
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_bytes(b"reviewed instrumentation patch\n")
    return repository


def _tensor_payload(seal: dict[str, Any]) -> dict[str, object]:
    return {
        "active_predict_dropout": False,
        "checkpoint_inventory": deepcopy(seal["checkpoint_inventory"]),
        "non_inference_inventory": deepcopy(seal["non_inference_inventory"]),
        "note_sequence_byte_parity": True,
        "required_inference_inventory": deepcopy(seal["required_inference_inventory"]),
        "schema": "crux.oaf-tensor-coverage/v1",
        "uninitialized_required": [],
    }


def _output_paths(repository: Path) -> dict[str, Path]:
    backend_id = "magenta-egmd-tf1-94529798-8hit-v1"
    config = repository / "config/benchmark/backends"
    evidence = repository / "docs/superpowers/evidence/hpa-320"
    return {
        "backend_lock": config / f"{backend_id}.backend-lock.json",
        "host_adapter_source_manifest": (
            repository / "runtime/oaf_tf1/host-adapter-source-manifest.json"
        ),
        "oci_layout_manifest": evidence / "oaf-oci-layout-manifest.json",
        "runtime_lock": config / f"{backend_id}.runtime-lock.json",
        "seal_evidence": config / f"{backend_id}.seal-evidence.json",
        "security_scan": evidence / "oaf-security-scan.json",
        "smoke_oracle": repository / "tests/fixtures/oaf_tf1_smoke/smoke-oracle.json",
        "tensor_coverage": evidence / "oaf-tensor-coverage.json",
    }


def _build_candidate(
    repository: Path,
    *,
    audit_mutation: Callable[[dict[str, Any]], None] | None = None,
    seal_mutation: Callable[[dict[str, Any]], None] | None = None,
    tensor_payload: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    candidate = repository / "candidate"
    candidate.mkdir(parents=True)
    host_record = _native_host_record()
    _write_json(candidate / "native-host-evidence.json", host_record)

    audit = LOCKS.audit_payload()
    if audit_mutation is not None:
        audit_mutation(audit)
    audit_path = _write_json(repository / "conversion-audit.json", audit)

    seal = LOCKS.seal_payload(audit_sha256=_content_hash(audit_path))
    seal["native_host_evidence"] = {
        "form": "github_hosted_linux_x64",
        "sha256": host_record["sha256"],
    }

    upstream = repository / "runtime/oaf_tf1/source-manifest.json"
    runner = repository / "runtime/oaf_tf1/runner-source-manifest.json"
    distribution = repository / "runtime/oaf_tf1/distribution-build-manifest.json"
    instrumentation = repository / "runtime/oaf_tf1/patches/capture-emitted-frame.patch"
    host_manifest_content = (
        json.dumps(
            _host_manifest_payload(repository),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    seal["upstream_source_manifest_sha256"] = _content_hash(upstream)
    seal["runner_source_manifest_sha256"] = _content_hash(runner)
    seal["distribution_build_manifest_sha256"] = _content_hash(distribution)
    seal["instrumentation_patch_sha256"] = _content_hash(instrumentation)
    seal["host_adapter_source_manifest_sha256"] = hashlib.sha256(host_manifest_content).hexdigest()

    advisory_path = _write_json(
        candidate / "advisory-snapshot.json",
        {"advisories": [], "schema": "crux.test-advisory-snapshot/v1"},
    )
    security_path = _write_json(
        candidate / "security-scan.json",
        {
            "advisory_snapshot_sha256": _content_hash(advisory_path),
            "findings": [],
            "schema": "crux.oaf-security-scan/v1",
        },
    )
    seal["advisory_snapshot_sha256"] = _content_hash(advisory_path)
    seal["security_scan_sha256"] = _content_hash(security_path)

    audio = candidate / "canonical.wav"
    audio.write_bytes(b"RIFF deterministic smoke bytes")
    smoke_prediction = candidate / "smoke-prediction.jsonl"
    smoke_prediction.write_bytes(b'{"record_type":"header"}\n{"record_type":"terminal"}\n')
    oracle_path = _write_json(
        candidate / "smoke-oracle.json",
        {
            "input_audio_frame_count": 64,
            "input_audio_sha256": _content_hash(audio),
            "input_view_id": "oaf-smoke-canonical-v1",
            "native_events": [
                {
                    "confidence_raw": "0.75",
                    "frame_index": 12,
                    "model_output_bin": 15,
                    "native_class_id": "midi_36",
                    "native_midi_note": 36,
                    "time_sec_raw": "0.139319",
                    "upstream_group_id": "kick",
                    "velocity": 100,
                }
            ],
            "schema": "crux.oaf-smoke-oracle/v1",
            "source_audio_id": "oaf-smoke-source-v1",
            "source_audio_sha256": _content_hash(audio),
        },
    )
    seal["smoke_audio_sha256"] = _content_hash(audio)
    seal["smoke_oracle_sha256"] = _content_hash(oracle_path)
    seal["smoke_prediction_sha256"] = _content_hash(smoke_prediction)

    archive = candidate / "oaf-runtime.oci.tar"
    archive.write_bytes(b"complete deterministic OCI archive bytes")
    seal["oci_layout_archive"] = {
        "name": archive.name,
        "sha256": _content_hash(archive),
        "size": archive.stat().st_size,
    }
    oci_path = _write_json(
        candidate / "oci-layout-manifest.json",
        {
            "archive": deepcopy(seal["oci_layout_archive"]),
            "config_digest": seal["runtime_image_config_digest"],
            "image_manifest_digest": seal["runtime_image_manifest_digest"],
            "layer_digests": deepcopy(seal["runtime_image_layer_digests"]),
            "schema": "crux.oaf-oci-layout-manifest/v1",
        },
    )
    seal["oci_layout_manifest_sha256"] = _content_hash(oci_path)

    tensor = _tensor_payload(seal) if tensor_payload is None else tensor_payload
    tensor_path = _write_json(candidate / "tensor-coverage.json", tensor)
    seal["tensor_coverage_sha256"] = _content_hash(tensor_path)
    if seal_mutation is not None:
        seal_mutation(seal)
    seal_path = _write_json(candidate / "proposed-seal-evidence.json", seal)

    runtime = LOCKS.runtime_payload(seal_sha256=_content_hash(seal_path))
    for field in (
        "base_image_manifest_digest",
        "debian_release_sha256",
        "debian_snapshot_repository",
        "distribution_build_manifest_sha256",
        "oci_layout_manifest_sha256",
        "python_distributions",
        "runner_source_manifest_sha256",
        "runtime_image_manifest_digest",
        "system_packages",
        "tensorflow_abi",
        "tensorflow_build",
        "upstream_source_manifest_sha256",
    ):
        runtime[field] = deepcopy(seal[field])
    runtime_path = _write_json(candidate / "proposed-runtime-lock.json", runtime)

    backend = LOCKS.backend_payload(
        runtime_sha256=_content_hash(runtime_path),
        seal_sha256=_content_hash(seal_path),
        audit_sha256=_content_hash(audit_path),
    )
    for field in (
        "checkpoint_archive",
        "checkpoint_components",
        "checkpoint_inventory",
        "host_adapter_source_manifest_sha256",
        "max_input_audio_frames",
        "non_inference_inventory",
        "required_inference_inventory",
        "runtime_image_manifest_digest",
        "smoke_audio_sha256",
        "smoke_oracle_sha256",
        "upstream_source_manifest_sha256",
    ):
        backend[field] = deepcopy(seal[field])
    _write_json(candidate / "proposed-backend-lock.json", backend)

    model_identity = LOCKS.identity_sha256(backend["checkpoint_components"])
    _write_json(
        candidate / "candidate-manifest.json",
        {
            "checkpoint_components": deepcopy(backend["checkpoint_components"]),
            "checkpoint_prefix": f"sha256/{model_identity}/model.ckpt-569400",
            "model_artifact_set_sha256": model_identity,
            "required_inference_inventory_sha256": LOCKS.identity_sha256(
                backend["required_inference_inventory"]
            ),
            "schema": "crux.oaf-seal-candidate/v1",
        },
    )
    return candidate, audit_path


def _seal(
    repository: Path,
    candidate: Path,
    audit: Path,
) -> tuple[seal_module.PublishedSeal, dict[str, Path]]:
    paths = _output_paths(repository)
    published = seal_module.seal_candidate(
        candidate=candidate,
        conversion_audit=audit,
        repository_root=repository,
        **paths,
    )
    return published, paths


def test_validate_host_strict_reads_existing_native_evidence_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_native_host(monkeypatch)
    path = _write_json(tmp_path / "native-host-evidence.json", _native_host_record())
    original = path.read_bytes()

    assert seal_module.main(["validate-host", "--evidence", os.fspath(path)]) == 0

    assert path.read_bytes() == original
    summary = strict_json_loads(capsys.readouterr().out.encode().rstrip(b"\n"))
    assert summary["official_execution_allowed"] is True  # type: ignore[index]
    assert summary["kind"] == "github_hosted"  # type: ignore[index]


@pytest.mark.parametrize("mutation", ["duplicate", "extra", "noncanonical"])
def test_validate_host_rejects_non_strict_wrapper_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _set_native_host(monkeypatch)
    record = _native_host_record()
    path = tmp_path / "native-host-evidence.json"
    if mutation == "duplicate":
        path.write_bytes(
            canonical_json_bytes(record)[:-1]
            + b',"sha256":"'
            + str(record["sha256"]).encode()
            + b'"}\n'
        )
    elif mutation == "extra":
        record["schema"] = "invented-wrapper-schema"
        _write_json(path, record)
    else:
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(SealError, match="host evidence"):
        seal_module.load_native_host_evidence(path)


@pytest.mark.parametrize(
    ("system", "machine", "runner_arch", "kind"),
    [
        ("Darwin", "arm64", "X64", "github_hosted"),
        ("Linux", "aarch64", "X64", "github_hosted"),
        ("Linux", "x86_64", "ARM64", "github_hosted"),
        ("Linux", "x86_64", "X64", "emulated"),
    ],
)
def test_validate_host_rejects_non_native_or_emulated_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    runner_arch: str,
    kind: str,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setenv("RUNNER_OS", "Linux")
    monkeypatch.setenv("RUNNER_ARCH", runner_arch)
    path = _write_json(
        tmp_path / "native-host-evidence.json",
        _native_host_record(kind=kind),
    )

    with pytest.raises(SealError):
        seal_module.load_native_host_evidence(path)


def test_unspecified_native_producers_fail_closed_without_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_native_host(monkeypatch)
    evidence = _write_json(tmp_path / "native-host-evidence.json", _native_host_record())
    model_cache = tmp_path / "model-cache"
    model_cache.mkdir()
    bundle = tmp_path / "system-packages"
    build_args = tmp_path / "build-args.env"
    candidate = tmp_path / "candidate"

    assert (
        seal_module.main(
            [
                "materialize-system-packages",
                "--host-evidence",
                os.fspath(evidence),
                "--bundle",
                os.fspath(bundle),
                "--build-args-output",
                os.fspath(build_args),
            ]
        )
        == 1
    )
    assert not bundle.exists()
    assert not build_args.exists()

    assert (
        seal_module.main(
            [
                "calibrate",
                "--image",
                "crux-oaf-tf1:hpa320-seal",
                "--host-evidence",
                os.fspath(evidence),
                "--model-cache",
                os.fspath(model_cache),
                "--output",
                os.fspath(candidate),
            ]
        )
        == 1
    )
    assert not candidate.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("memory_limit_bytes"),
        lambda payload: payload.__setitem__("memory_limit_bytes", "auto"),
        lambda payload: payload.__setitem__("max_input_audio_frames", 0),
    ],
)
def test_seal_rejects_missing_sentinel_or_zero_calibration_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    _set_native_host(monkeypatch)
    repository = _make_repository(tmp_path)
    candidate, audit = _build_candidate(repository, seal_mutation=mutation)
    paths = _output_paths(repository)

    with pytest.raises(SealError, match="seal evidence|backend lock"):
        _seal(repository, candidate, audit)

    assert not any(path.exists() for path in paths.values())


def test_seal_rejects_count_only_tensor_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_native_host(monkeypatch)
    repository = _make_repository(tmp_path)
    candidate, audit = _build_candidate(
        repository,
        tensor_payload={
            "checkpoint_count": 130,
            "non_inference_count": 52,
            "required_count": 78,
            "restored_count": 78,
            "schema": "crux.oaf-tensor-coverage/v1",
        },
    )

    with pytest.raises(SealError, match="exact tensor inventories"):
        _seal(repository, candidate, audit)


def test_seal_rejects_oci_archive_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_native_host(monkeypatch)
    repository = _make_repository(tmp_path)
    candidate, audit = _build_candidate(repository)
    archive = candidate / "oaf-runtime.oci.tar"
    archive.write_bytes(archive.read_bytes() + b"drift")

    with pytest.raises(SealError, match="OCI archive"):
        _seal(repository, candidate, audit)


def test_seal_rejects_observed_hdf5_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_native_host(monkeypatch)
    repository = _make_repository(tmp_path)
    candidate, audit = _build_candidate(
        repository,
        audit_mutation=lambda payload: payload.__setitem__("observed_hdf5_sha256", "0" * 64),
    )

    with pytest.raises(SealError, match="HDF5"):
        _seal(repository, candidate, audit)


def test_seal_rejects_final_lock_hash_cycle_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_native_host(monkeypatch)
    repository = _make_repository(tmp_path)
    candidate, audit = _build_candidate(
        repository,
        seal_mutation=lambda payload: payload.__setitem__("runtime_lock_sha256", "f" * 64),
    )

    with pytest.raises(SealError, match="final lock hashes"):
        _seal(repository, candidate, audit)

    assert not any(path.exists() for path in _output_paths(repository).values())


def test_runtime_publication_failure_never_publishes_backend_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_native_host(monkeypatch)
    repository = _make_repository(tmp_path)
    candidate, audit = _build_candidate(repository)
    paths = _output_paths(repository)
    paths["runtime_lock"].parent.mkdir(parents=True, exist_ok=True)
    paths["runtime_lock"].write_bytes(b"prior immutable runtime\n")

    with pytest.raises(SealError, match="publication"):
        _seal(repository, candidate, audit)

    assert paths["runtime_lock"].read_bytes() == b"prior immutable runtime\n"
    assert not paths["backend_lock"].exists()


def test_seal_is_deterministic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_native_host(monkeypatch)
    repository = _make_repository(tmp_path)
    candidate, audit = _build_candidate(repository)

    first, paths = _seal(repository, candidate, audit)
    first_bytes = {name: path.read_bytes() for name, path in paths.items()}
    second, _ = _seal(repository, candidate, audit)

    assert first == second
    assert {name: path.read_bytes() for name, path in paths.items()} == first_bytes
    assert list(first.publication_order)[-2:] == ["runtime_lock", "backend_lock"]
    host_manifest = strict_json_loads(
        paths["host_adapter_source_manifest"].read_bytes()[:-1],
        require_canonical=True,
    )
    assert [row["path"] for row in host_manifest["files"]] == list(  # type: ignore[index]
        HOST_PATHS
    )
    assert host_manifest["covered_roots"] == list(HOST_PATHS)  # type: ignore[index]
