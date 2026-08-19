#!/usr/bin/env python3
"""Freeze one already-working external separator runtime into a lock."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.benchmark.artifact_io import ArtifactPublicationError, publish_immutable_file
from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex
from src.benchmark.separators import (
    _SEPARATOR_POLICIES,
    HTDEMUCS_SEPARATOR_ID,
    SEPARATOR_LOCK_SCHEMA,
    SPLEETER_SEPARATOR_ID,
    SeparatorExecutionError,
    SeparatorLock,
    _require_absolute_model_root,
    _resolve_separator_interpreter,
    _run_separator_environment_probe,
    attest_separator_runtime,
    inventory_separator_model_root,
    separator_environment_manifest_payload,
)


class FreezeError(RuntimeError):
    """The mechanical separator lock freeze could not complete."""


def freeze_separator_runtime(
    *,
    separator_id: str,
    interpreter: Path,
    model_root: Path,
    repository_revision: str,
    output: Path,
) -> SeparatorLock:
    """Attest and publish one policy-owned separator runtime as a fixed pair."""
    if not isinstance(interpreter, Path):
        raise TypeError("interpreter must be a Path")
    if not isinstance(model_root, Path):
        raise TypeError("model_root must be a Path")
    if not isinstance(output, Path):
        raise TypeError("output must be a Path")
    _require_absolute_model_root(model_root)
    policy = _SEPARATOR_POLICIES.get(separator_id)
    if policy is None:
        raise FreezeError("separator_id is unsupported")

    resolved_interpreter, interpreter_sha256 = _resolve_separator_interpreter(interpreter)
    environment = _run_separator_environment_probe(resolved_interpreter)
    if (
        environment.separator_id != separator_id
        or environment.package_name != policy["package_name"]
        or environment.package_version != policy["package_version"]
        or environment.interpreter_sha256 != interpreter_sha256
    ):
        raise SeparatorExecutionError(
            "separator_environment_mismatch",
            "separator environment does not match the selected policy",
        )
    model_files = inventory_separator_model_root(separator_id, model_root)

    environment_bytes = canonical_json_bytes(
        separator_environment_manifest_payload(environment),
        trailing_newline=True,
    )
    environment_manifest_sha256 = sha256_hex(environment_bytes)
    lock_payload = {
        "schema": SEPARATOR_LOCK_SCHEMA,
        "separator_id": separator_id,
        "repository_url": policy["repository_url"],
        "repository_revision": repository_revision,
        "package_name": environment.package_name,
        "package_version": environment.package_version,
        "model_id": policy["model_id"],
        "model_files": [
            {"name": model_file.name, "sha256": model_file.sha256} for model_file in model_files
        ],
        "code_license": policy["code_license"],
        "model_license": policy["model_license"],
        "argv": list(policy["argv"]),
        "expected_drum_stem_relative_path": policy["expected_drum_stem_relative_path"],
        "output_container": policy["output_container"],
        "interpreter_sha256": interpreter_sha256,
        "environment_manifest_sha256": environment_manifest_sha256,
        "model_root_kind": policy["model_root_kind"],
    }
    lock_bytes = canonical_json_bytes(lock_payload, trailing_newline=True)
    try:
        publish_immutable_file(output.parent / "environment.json", environment_bytes)
        publish_immutable_file(output, lock_bytes)
    except (ArtifactPublicationError, OSError, TypeError) as error:
        raise FreezeError("separator lock publication failed") from error

    attested = attest_separator_runtime(output, interpreter, model_root)
    try:
        if attested.lock.separator_id != separator_id:
            raise FreezeError("published separator lock did not round-trip")
        return attested.lock
    finally:
        attested.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--separator-id",
        choices=(SPLEETER_SEPARATOR_ID, HTDEMUCS_SEPARATOR_ID),
        required=True,
    )
    parser.add_argument("--interpreter", "--python", dest="interpreter", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--repository-revision",
        "--revision",
        dest="repository_revision",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        lock = freeze_separator_runtime(
            separator_id=args.separator_id,
            interpreter=args.interpreter,
            model_root=args.model_root,
            repository_revision=args.repository_revision,
            output=args.output,
        )
    except (FreezeError, OSError, ValueError) as error:
        print(f"freeze failed: {error}", file=sys.stderr)
        return 1
    print(lock.sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
