from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.separators import (
    HTDEMUCS_SEPARATOR_ID,
    SEPARATOR_LOCK_SCHEMA,
    SPLEETER_SEPARATOR_ID,
    SeparatorLock,
    load_separator_lock,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "separators"


def _fixture_path(separator_id: str) -> Path:
    filename = (
        "spleeter-model.json" if separator_id == SPLEETER_SEPARATOR_ID else "htdemucs-model.json"
    )
    return FIXTURE_ROOT / filename


def _fixture_payload(separator_id: str) -> dict[str, object]:
    return json.loads(_fixture_path(separator_id).read_text(encoding="utf-8"))


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(canonical_json_bytes(payload, trailing_newline=True))


@pytest.mark.parametrize("separator_id", [SPLEETER_SEPARATOR_ID, HTDEMUCS_SEPARATOR_ID])
def test_loads_fixture_lock_and_hashes_exact_canonical_bytes(
    separator_id: str,
) -> None:
    path = _fixture_path(separator_id)

    lock = load_separator_lock(path)

    assert isinstance(lock, SeparatorLock)
    assert lock.separator_id == separator_id
    assert lock.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert lock.model_files
    assert lock.argv[0:2] == (
        "-m",
        "spleeter" if separator_id == SPLEETER_SEPARATOR_ID else "demucs",
    )
    assert set(_fixture_payload(separator_id)) == {
        "schema",
        "separator_id",
        "repository_url",
        "repository_revision",
        "package_name",
        "package_version",
        "model_id",
        "model_files",
        "code_license",
        "model_license",
        "argv",
        "expected_drum_stem_relative_path",
        "output_container",
    }
    assert _fixture_payload(separator_id)["schema"] == SEPARATOR_LOCK_SCHEMA


@pytest.mark.parametrize("separator_id", [SPLEETER_SEPARATOR_ID, HTDEMUCS_SEPARATOR_ID])
@pytest.mark.parametrize("mutation", ["unknown_key", "missing_key"])
def test_loader_rejects_unknown_or_missing_keys(
    tmp_path: Path,
    separator_id: str,
    mutation: str,
) -> None:
    payload = _fixture_payload(separator_id)
    if mutation == "unknown_key":
        payload["unexpected"] = True
    else:
        payload.pop("package_version")
    path = tmp_path / "separator.json"
    _write_payload(path, payload)

    with pytest.raises(ValueError):
        load_separator_lock(path)


def test_loader_rejects_noncanonical_json(tmp_path: Path) -> None:
    path = tmp_path / "separator.json"
    content = _fixture_path(SPLEETER_SEPARATOR_ID).read_bytes()
    path.write_bytes(b" " + content)

    with pytest.raises(ValueError, match="canonical"):
        load_separator_lock(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository_revision", "A" * 40),
        ("model_files", [{"name": "weights.bin", "sha256": "not-a-hash"}]),
    ],
)
def test_loader_rejects_malformed_hashes(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    payload[field] = value
    path = tmp_path / "separator.json"
    _write_payload(path, payload)

    with pytest.raises(ValueError, match="hash|revision"):
        load_separator_lock(path)


@pytest.mark.parametrize(
    "model_files",
    [
        [{"name": "weights.bin", "sha256": "a" * 64}, {"name": "weights.bin", "sha256": "b" * 64}],
        [{"name": "/weights.bin", "sha256": "a" * 64}],
    ],
)
def test_loader_rejects_duplicate_or_absolute_model_names(
    tmp_path: Path,
    model_files: list[dict[str, str]],
) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    payload["model_files"] = model_files
    path = tmp_path / "separator.json"
    _write_payload(path, payload)

    with pytest.raises(ValueError, match="model file"):
        load_separator_lock(path)


def test_loader_rejects_unsupported_separator_id(tmp_path: Path) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    payload["separator_id"] = "other-separator-v1"
    path = tmp_path / "separator.json"
    _write_payload(path, payload)

    with pytest.raises(ValueError, match="separator_id"):
        load_separator_lock(path)


@pytest.mark.parametrize(
    ("separator_id", "wrong_model_token"),
    [
        (SPLEETER_SEPARATOR_ID, "spleeter:2stems"),
        (HTDEMUCS_SEPARATOR_ID, "htdemucs_ft"),
    ],
)
def test_loader_rejects_command_model_mismatch(
    tmp_path: Path,
    separator_id: str,
    wrong_model_token: str,
) -> None:
    payload = _fixture_payload(separator_id)
    argv = list(payload["argv"])
    model_index = argv.index(
        "spleeter:4stems" if separator_id == SPLEETER_SEPARATOR_ID else "htdemucs"
    )
    argv[model_index] = wrong_model_token
    payload["argv"] = argv
    path = tmp_path / "separator.json"
    _write_payload(path, payload)

    with pytest.raises(ValueError, match="argv|model"):
        load_separator_lock(path)


def test_freeze_script_hashes_explicit_files_and_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import freeze_separator_runtime as freezer

    model_file = tmp_path / "weights.bin"
    model_bytes = b"synthetic separator model bytes"
    model_file.write_bytes(model_bytes)
    output = tmp_path / "separator.json"
    monkeypatch.setattr(freezer, "_package_version", lambda interpreter, package: "2.4.2")

    lock = freezer.freeze_separator_runtime(
        separator_id=SPLEETER_SEPARATOR_ID,
        interpreter=Path("/isolated/python"),
        model_files={"weights.bin": model_file},
        repository_revision="a" * 40,
        output=output,
    )

    assert lock == load_separator_lock(output)
    assert lock.package_version == "2.4.2"
    assert lock.model_files[0].sha256 == hashlib.sha256(model_bytes).hexdigest()
