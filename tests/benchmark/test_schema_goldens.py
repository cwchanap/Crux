from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath

import pytest

from src.benchmark.backend_identity import canonical_json_bytes, strict_json_loads

_MANIFEST_PATH = PurePosixPath("tests/benchmark/schema_goldens/manifest.json")
_MANIFEST_ROW_KEYS = frozenset({"schema", "validator_modules", "golden_path"})


@dataclass(frozen=True)
class SchemaGoldenEntry:
    schema: str
    validator_modules: tuple[str, ...]
    golden_path: PurePosixPath


def load_schema_golden_manifest(repository_root: Path) -> tuple[SchemaGoldenEntry, ...]:
    content = (repository_root / _MANIFEST_PATH).read_bytes()
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError("schema-golden manifest must have one final newline")
    value = strict_json_loads(content[:-1], require_canonical=True)
    if not isinstance(value, list) or not value:
        raise ValueError("schema-golden manifest must be a nonempty array")

    entries: list[SchemaGoldenEntry] = []
    schemas: set[str] = set()
    golden_paths: set[PurePosixPath] = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != _MANIFEST_ROW_KEYS:
            raise ValueError("schema-golden manifest rows must contain the exact key set")
        schema = row["schema"]
        modules = row["validator_modules"]
        raw_path = row["golden_path"]
        if not isinstance(schema, str) or not schema:
            raise ValueError("schema-golden schema must be nonempty")
        if (
            not isinstance(modules, list)
            or not modules
            or any(not isinstance(module, str) or not module for module in modules)
        ):
            raise ValueError("schema-golden validator_modules must be a nonempty string array")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("schema-golden golden_path must be nonempty")
        golden_path = PurePosixPath(raw_path)
        if golden_path.is_absolute() or any(part in {"", ".", ".."} for part in golden_path.parts):
            raise ValueError("schema-golden golden_path must be relative POSIX")
        if schema in schemas:
            raise ValueError("schema-golden manifest contains duplicate schema")
        if golden_path in golden_paths:
            raise ValueError("schema-golden manifest contains duplicate golden path")
        schemas.add(schema)
        golden_paths.add(golden_path)
        entries.append(
            SchemaGoldenEntry(
                schema=schema,
                validator_modules=tuple(modules),
                golden_path=golden_path,
            )
        )
    return tuple(entries)


def validate_schema_golden_entry(entry: SchemaGoldenEntry, repository_root: Path) -> None:
    content = (repository_root / entry.golden_path).read_bytes()
    for module_name in entry.validator_modules:
        validator = getattr(importlib.import_module(module_name), "validate_schema_golden")
        validator(entry.schema, content)


@pytest.fixture
def repository_root() -> Path:
    return Path(__file__).parents[2]


def test_phase_a_and_b_schema_goldens_are_complete_and_strict(repository_root: Path) -> None:
    entries = load_schema_golden_manifest(repository_root)
    schemas = {entry.schema for entry in entries}

    assert {
        "crux.transcription-backend-descriptor/v1",
        "crux.heuristic-backend-descriptor/v1",
        "crux.input-view-manifest/v1",
        "crux.backend-execution-item-id/v1",
        "crux.backend-execution-attestation/v1",
        "crux.backend-verification-report/v1",
        "crux.backend-execution-report/v1",
        "crux.legacy-score-report/v1",
        "crux.drum-prediction-events/v1",
        "crux.transcription-backend-lock/v1",
        "crux.transcription-runtime-lock/v1",
        "crux.backend-seal-evidence/v1",
        "crux.legacy-tf2-conversion-coverage/v1",
        "crux.oaf-checkpoint-acquisition-request/v1",
        "crux.oaf-base-system-package-request/v1",
        "crux.oaf-calibration-measurement-request/v1",
        "crux.oaf-seal-profile-request/v1",
        "crux.transcription-runner/v1",
    }.issubset(schemas)
    for entry in entries:
        validate_schema_golden_entry(entry, repository_root)


@pytest.mark.parametrize("duplicate", ["schema", "golden_path"])
def test_schema_golden_manifest_rejects_duplicate_schema_ids_and_paths(
    tmp_path: Path,
    duplicate: str,
) -> None:
    manifest_path = tmp_path / _MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True)
    rows: list[dict[str, object]] = [
        {
            "schema": "example.schema/v1",
            "validator_modules": ["example.validator"],
            "golden_path": "goldens/first.json",
        },
        {
            "schema": "example.other/v1",
            "validator_modules": ["example.validator"],
            "golden_path": "goldens/second.json",
        },
    ]
    rows[1][duplicate] = rows[0][duplicate]
    manifest_path.write_bytes(canonical_json_bytes(rows, trailing_newline=True))

    with pytest.raises(ValueError, match=f"duplicate {duplicate.replace('_', ' ')}"):
        load_schema_golden_manifest(tmp_path)


Mutation = Callable[[bytes], bytes]


def _json_root(content: bytes) -> tuple[dict[str, object], list[dict[str, object]] | None]:
    lines = content.splitlines(keepends=True)
    records = [json.loads(line, parse_float=Decimal) for line in lines]
    if len(records) == 1:
        return records[0], None
    return records[0], records


def _render_mutated(
    content: bytes, root: dict[str, object], records: list[dict[str, object]] | None
) -> bytes:
    if records is None:
        return canonical_json_bytes(root, trailing_newline=True)
    records[0] = root
    return b"".join(canonical_json_bytes(record, trailing_newline=True) for record in records)


def _remove_one_key(content: bytes) -> bytes:
    root, records = _json_root(content)
    root.pop(sorted(root)[0])
    return _render_mutated(content, root, records)


def _add_unexpected_key(content: bytes) -> bytes:
    root, records = _json_root(content)
    root["unexpected"] = "unexpected"
    return _render_mutated(content, root, records)


def _duplicate_one_json_key(content: bytes) -> bytes:
    lines = content.splitlines(keepends=True)
    root = json.loads(lines[0])
    key = sorted(root)[0]
    body = lines[0][:-1]
    lines[0] = body[:-1] + f',"{key}":null}}'.encode("ascii") + b"\n"
    return b"".join(lines)


def _replace_one_typed_value(content: bytes) -> bytes:
    root, records = _json_root(content)
    root[sorted(root)[0]] = 0
    return _render_mutated(content, root, records)


@pytest.mark.parametrize(
    "mutation",
    [_remove_one_key, _add_unexpected_key, _duplicate_one_json_key, _replace_one_typed_value],
    ids=["remove-key", "unexpected-key", "duplicate-key", "wrong-type"],
)
def test_phase_b_schema_golden_validators_reject_structural_mutations(
    repository_root: Path,
    mutation: Mutation,
) -> None:
    for entry in load_schema_golden_manifest(repository_root):
        mutated = mutation((repository_root / entry.golden_path).read_bytes())
        for module_name in entry.validator_modules:
            validator = getattr(importlib.import_module(module_name), "validate_schema_golden")
            with pytest.raises(ValueError):
                validator(entry.schema, mutated)


@pytest.mark.parametrize(
    ("schema", "field"),
    [
        ("crux.transcription-backend-lock/v1", "checkpoint_acquisition_request_sha256"),
        ("crux.transcription-runtime-lock/v1", "base_system_package_evidence_sha256"),
    ],
)
def test_phase_a_schema_golden_validators_reject_important_hash_type_drift(
    repository_root: Path,
    schema: str,
    field: str,
) -> None:
    entry = next(
        candidate
        for candidate in load_schema_golden_manifest(repository_root)
        if candidate.schema == schema
    )
    payload = json.loads(
        (repository_root / entry.golden_path).read_bytes(),
        parse_float=Decimal,
    )
    payload[field] = 0
    mutated = canonical_json_bytes(payload, trailing_newline=True)

    for module_name in entry.validator_modules:
        validator = getattr(importlib.import_module(module_name), "validate_schema_golden")
        with pytest.raises(ValueError):
            validator(entry.schema, mutated)


@pytest.mark.parametrize(
    ("schema", "field"),
    [
        ("crux.oaf-calibration-measurement-evidence/v1", "base_system_package_evidence_sha256"),
        ("crux.oaf-seal-candidate/v1", "seal_profile_request_sha256"),
        ("crux.oaf-oci-layout-manifest/v1", "config_digest"),
        ("crux.oaf-smoke-oracle/v1", "source_audio_sha256"),
    ],
)
def test_phase_a_schema_goldens_reject_nonfirst_critical_field_drift(
    repository_root: Path, schema: str, field: str
) -> None:
    entry = next(
        item for item in load_schema_golden_manifest(repository_root) if item.schema == schema
    )
    payload = json.loads((repository_root / entry.golden_path).read_bytes(), parse_float=Decimal)
    payload[field] = 0
    mutated = canonical_json_bytes(payload, trailing_newline=True)
    for module_name in entry.validator_modules:
        validator = getattr(importlib.import_module(module_name), "validate_schema_golden")
        with pytest.raises(ValueError):
            validator(entry.schema, mutated)


@pytest.mark.parametrize(
    "audio_path",
    [
        "/absolute.wav",
        "../escape.wav",
        "./dot.wav",
        "dir\\escape.wav",
        "audio.mp3",
        "bad\x00.wav",
        "e\u0301.wav",
    ],
)
def test_host_and_runner_reject_the_same_invalid_audio_paths(
    repository_root: Path, audio_path: str
) -> None:
    entry = next(
        item
        for item in load_schema_golden_manifest(repository_root)
        if item.schema == "crux.transcription-runner/v1"
    )
    payload = json.loads((repository_root / entry.golden_path).read_bytes())
    payload["audio_path"] = audio_path
    content = canonical_json_bytes(payload, trailing_newline=True)
    for module_name in entry.validator_modules:
        validator = getattr(importlib.import_module(module_name), "validate_schema_golden")
        with pytest.raises(ValueError):
            validator(entry.schema, content)
