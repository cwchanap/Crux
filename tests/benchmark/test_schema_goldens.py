from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from src.benchmark.backend_identity import strict_json_loads
from src.benchmark.prediction_artifact import validate_schema_golden


@dataclass(frozen=True)
class SchemaGoldenEntry:
    schema: str
    validator_modules: tuple[str, ...]
    golden_path: PurePosixPath


def load_schema_golden_manifest(repository_root: Path) -> tuple[SchemaGoldenEntry, ...]:
    path = repository_root / "tests/benchmark/schema_goldens/manifest.json"
    content = path.read_bytes()
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError("schema-golden manifest must have one final newline")
    value = strict_json_loads(content[:-1], require_canonical=True)
    if not isinstance(value, list):
        raise ValueError("schema-golden manifest must be an array")
    entries: list[SchemaGoldenEntry] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {"schema", "validator_modules", "golden_path"}:
            raise ValueError("schema-golden manifest row is invalid")
        raw_path = row["golden_path"]
        modules = row["validator_modules"]
        if not isinstance(row["schema"], str) or not isinstance(raw_path, str):
            raise ValueError("schema-golden manifest value is invalid")
        if not isinstance(modules, list) or any(not isinstance(item, str) for item in modules):
            raise ValueError("schema-golden validator_modules is invalid")
        entries.append(SchemaGoldenEntry(row["schema"], tuple(modules), PurePosixPath(raw_path)))
    return tuple(entries)


def validate_schema_golden_entry(entry: SchemaGoldenEntry, repository_root: Path) -> None:
    content = (repository_root / entry.golden_path).read_bytes()
    for module_name in entry.validator_modules:
        validator = getattr(importlib.import_module(module_name), "validate_schema_golden")
        validator(entry.schema, content)


def test_schema_golden_manifest_contains_only_task_d_rows() -> None:
    manifest = json.loads(
        (Path(__file__).parent / "schema_goldens" / "manifest.json").read_text(encoding="utf-8")
    )
    assert [row["schema"] for row in manifest] == [
        "crux.input-view-manifest/v1",
        "crux.drum-prediction-events/v2",
        "crux.dtx-reference-event/v1",
        "crux.reference-chart-manifest/v1",
        "crux.reference-timing-manifest/v1",
        "crux.benchmark-reference-manifest/v1",
    ]


def test_prediction_v2_schema_golden_is_valid() -> None:
    path = Path(__file__).parent / "schema_goldens" / "crux.drum-prediction-events-v2.jsonl"
    validate_schema_golden("crux.drum-prediction-events/v2", path.read_bytes())
