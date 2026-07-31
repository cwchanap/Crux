"""Immutable candidate-artifact roles and paths shared by native seal boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

CANDIDATE_ARTIFACTS: tuple[tuple[str, str], ...] = (
    (
        "conversion_audit",
        "docs/superpowers/evidence/hpa-320/legacy-conversion-audit.json",
    ),
    (
        "native_host_attestation_bundle",
        "docs/superpowers/evidence/hpa-320/native/"
        "candidate-host-attestation/attestation-bundle.json",
    ),
    (
        "native_host_evidence",
        "docs/superpowers/evidence/hpa-320/native/"
        "candidate-host-attestation/native-host-evidence.json",
    ),
    (
        "native_host_observation",
        "docs/superpowers/evidence/hpa-320/native/"
        "candidate-host-attestation/native-host-observation.json",
    ),
    (
        "host_adapter_source_manifest",
        "runtime/oaf_tf1/host-adapter-source-manifest.json",
    ),
    (
        "tensor_coverage",
        "docs/superpowers/evidence/hpa-320/oaf-tensor-coverage.json",
    ),
    (
        "advisory_snapshot",
        "docs/superpowers/evidence/hpa-320/oaf-advisory-snapshot.json",
    ),
    (
        "security_scan",
        "docs/superpowers/evidence/hpa-320/oaf-security-scan.json",
    ),
    (
        "oci_layout_archive",
        "artifacts/benchmark/backends/oaf-tf1/runtime.oci.tar",
    ),
    (
        "oci_layout_manifest",
        "docs/superpowers/evidence/hpa-320/oaf-oci-layout-manifest.json",
    ),
    ("smoke_audio", "tests/fixtures/oaf_tf1_smoke/canonical.wav"),
    (
        "smoke_prediction",
        "docs/superpowers/evidence/hpa-320/oaf-smoke-prediction.jsonl",
    ),
    ("smoke_oracle", "tests/fixtures/oaf_tf1_smoke/smoke-oracle.json"),
    (
        "seal_evidence",
        "config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-evidence.json",
    ),
    (
        "runtime_lock",
        "config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.runtime-lock.json",
    ),
    (
        "backend_lock",
        "config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.backend-lock.json",
    ),
)
CANDIDATE_ARTIFACT_PATHS: Mapping[str, str] = MappingProxyType(dict(CANDIDATE_ARTIFACTS))
