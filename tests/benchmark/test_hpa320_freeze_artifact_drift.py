from __future__ import annotations

import json
from pathlib import Path

from src.benchmark.backend_identity import sha256_hex
from src.benchmark.backend_publication import read_regular_file_no_follow
from tools.hpa320.generate_runner_source_manifest import (
    build_runner_source_manifest,
    canonical_manifest_bytes,
)
from tools.hpa320.oaf_build_context import load_build_context_manifest
from tools.hpa320.seal_oaf_backend import load_calibration_bootstrap_request

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _manifest_drift_paths(generated: bytes, checked: bytes) -> tuple[str, ...]:
    generated_rows = {row["path"]: row for row in json.loads(generated)["files"]}
    checked_rows = {row["path"]: row for row in json.loads(checked)["files"]}
    return tuple(
        sorted(
            {
                path
                for path in generated_rows.keys() | checked_rows.keys()
                if generated_rows.get(path) != checked_rows.get(path)
            },
            key=lambda path: path.encode("utf-8"),
        )
    )


def test_checked_runner_source_manifest_matches_final_sources() -> None:
    generated = canonical_manifest_bytes(build_runner_source_manifest(REPOSITORY_ROOT))
    checked = read_regular_file_no_follow(
        REPOSITORY_ROOT / "runtime/oaf_tf1/runner-source-manifest.json"
    )
    drift = _manifest_drift_paths(generated, checked)
    assert generated == checked, f"runner source manifest drift: {drift!r}"


def test_build_context_repository_rows_match_final_checkout() -> None:
    manifest = load_build_context_manifest(
        REPOSITORY_ROOT / "runtime/oaf_tf1/build-context-manifest.json"
    )
    repository_rows = tuple(
        row for row in manifest.files if not row.path.startswith("runtime/oaf_tf1/wheelhouse/")
    )
    drift: list[str] = []
    for row in repository_rows:
        path = REPOSITORY_ROOT / row.path
        if not path.is_file() or path.is_symlink():
            drift.append(row.path)
            continue
        content = read_regular_file_no_follow(path)
        if len(content) != row.byte_length or sha256_hex(content) != row.sha256:
            drift.append(row.path)
    assert not drift, f"build-context repository drift: {tuple(drift)!r}"


def test_bootstrap_request_cross_hashes_every_current_input() -> None:
    request = load_calibration_bootstrap_request(
        REPOSITORY_ROOT / "config/benchmark/backends/"
        "magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json"
    )
    expected_paths = {
        "runner_source_manifest_sha256": "runtime/oaf_tf1/runner-source-manifest.json",
        "build_context_manifest_sha256": "runtime/oaf_tf1/build-context-manifest.json",
        "upstream_source_manifest_sha256": "runtime/oaf_tf1/source-manifest.json",
        "checkpoint_acquisition_request_sha256": (
            "config/benchmark/backends/"
            "magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json"
        ),
        "base_system_package_request_sha256": ("runtime/oaf_tf1/base-system-package-request.json"),
        "distribution_build_manifest_sha256": ("runtime/oaf_tf1/distribution-build-manifest.json"),
        "instrumentation_patch_sha256": ("runtime/oaf_tf1/patches/capture-emitted-frame.patch"),
    }
    for field, relative_path in expected_paths.items():
        content = read_regular_file_no_follow(REPOSITORY_ROOT / relative_path)
        assert request.payload[field] == sha256_hex(content)
