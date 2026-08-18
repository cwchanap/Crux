#!/usr/bin/env python3
"""Freeze one already-working external separator runtime into a lock."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from src.benchmark.artifact_io import publish_immutable_file, read_regular_file_no_follow
from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex
from src.benchmark.separators import (
    _SEPARATOR_POLICIES,
    HTDEMUCS_SEPARATOR_ID,
    SPLEETER_SEPARATOR_ID,
    SeparatorLock,
    SeparatorModelFile,
    load_separator_lock,
    separator_lock_payload,
)


class FreezeError(RuntimeError):
    """The mechanical separator lock freeze could not complete."""


def freeze_separator_runtime(
    *,
    separator_id: str,
    interpreter: Path,
    model_files: Mapping[str, Path],
    repository_revision: str,
    output: Path,
) -> SeparatorLock:
    """Hash explicit model files and publish one lock without running inference."""
    if separator_id not in _SEPARATOR_POLICIES:
        raise FreezeError("separator_id is unsupported")
    if not isinstance(interpreter, Path) or not isinstance(output, Path):
        raise TypeError("interpreter and output must be Paths")
    if not isinstance(model_files, Mapping) or not model_files:
        raise FreezeError("model_files must be a nonempty name-to-Path mapping")
    if not isinstance(repository_revision, str):
        raise TypeError("repository_revision must be a string")

    package_name = _SEPARATOR_POLICIES[separator_id]["package_name"]
    package_version = _package_version(interpreter, package_name)
    frozen_files: list[SeparatorModelFile] = []
    for name, path in sorted(model_files.items()):
        if not isinstance(path, Path):
            raise TypeError("model file paths must be Paths")
        try:
            content = read_regular_file_no_follow(path)
        except (OSError, TypeError) as error:
            raise FreezeError(f"model file is unavailable: {name}") from error
        frozen_files.append(SeparatorModelFile(name=name, sha256=sha256_hex(content)))

    policy = _SEPARATOR_POLICIES[separator_id]
    provisional = SeparatorLock(
        separator_id=separator_id,
        repository_url=policy["repository_url"],
        repository_revision=repository_revision,
        package_name=package_name,
        package_version=package_version,
        model_id=policy["model_id"],
        model_files=tuple(frozen_files),
        code_license="MIT",
        model_license="MIT",
        argv=policy["argv"],
        expected_drum_stem_relative_path=policy["expected_drum_stem_relative_path"],
        output_container=policy["output_container"],
        sha256="0" * 64,
    )
    content = canonical_json_bytes(separator_lock_payload(provisional), trailing_newline=True)
    lock = SeparatorLock(
        separator_id=provisional.separator_id,
        repository_url=provisional.repository_url,
        repository_revision=provisional.repository_revision,
        package_name=provisional.package_name,
        package_version=provisional.package_version,
        model_id=provisional.model_id,
        model_files=provisional.model_files,
        code_license=provisional.code_license,
        model_license=provisional.model_license,
        argv=provisional.argv,
        expected_drum_stem_relative_path=provisional.expected_drum_stem_relative_path,
        output_container=provisional.output_container,
        sha256=sha256_hex(content),
    )
    publish_immutable_file(output, content)
    loaded = load_separator_lock(output)
    if loaded != lock:
        raise FreezeError("published separator lock did not round-trip")
    return loaded


def _package_version(interpreter: Path, package_name: str) -> str:
    version_code = "import importlib.metadata, sys; print(importlib.metadata.version(sys.argv[1]))"
    try:
        result = subprocess.run(
            [str(interpreter), "-c", version_code, package_name],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FreezeError(
            f"could not read {package_name} version from isolated interpreter"
        ) from error
    version = result.stdout.strip()
    if not version or len(version.splitlines()) != 1:
        raise FreezeError(f"isolated interpreter returned an invalid {package_name} version")
    return version


def _parse_model_file(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("model files must use NAME=PATH")
    return name, Path(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--separator-id",
        choices=(SPLEETER_SEPARATOR_ID, HTDEMUCS_SEPARATOR_ID),
        required=True,
    )
    parser.add_argument("--interpreter", "--python", dest="interpreter", type=Path, required=True)
    parser.add_argument(
        "--model-file",
        action="append",
        type=_parse_model_file,
        required=True,
        metavar="NAME=PATH",
    )
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
    model_files: dict[str, Path] = {}
    for name, path in args.model_file:
        if name in model_files:
            print(f"freeze failed: duplicate model file name {name}", file=sys.stderr)
            return 1
        model_files[name] = path
    try:
        lock = freeze_separator_runtime(
            separator_id=args.separator_id,
            interpreter=args.interpreter,
            model_files=model_files,
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
