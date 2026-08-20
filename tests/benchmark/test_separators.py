from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import signal
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import src.benchmark.separators as separators
from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.separators import (
    HTDEMUCS_SEPARATOR_ID,
    SEPARATOR_LOCK_SCHEMA,
    SPLEETER_SEPARATOR_ID,
    SeparatorExecutionError,
    SeparatorLock,
    SeparatorLockError,
    SeparatorModelFile,
    load_separator_environment_manifest,
    load_separator_lock,
)
from tests.benchmark.test_separator_environment_probe import _synthetic_environment

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "separators"


def _fixture_path(separator_id: str) -> Path:
    directory = "spleeter" if separator_id == SPLEETER_SEPARATOR_ID else "htdemucs"
    return FIXTURE_ROOT / directory / "model.json"


def _fixture_payload(separator_id: str) -> dict[str, object]:
    return json.loads(_fixture_path(separator_id).read_text(encoding="utf-8"))


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(canonical_json_bytes(payload, trailing_newline=True))


@pytest.mark.parametrize(
    ("separator_id", "package_version"),
    [
        (SPLEETER_SEPARATOR_ID, "2.4.2"),
        (HTDEMUCS_SEPARATOR_ID, "4.1.0"),
    ],
)
def test_separator_policy_pins_package_version(
    separator_id: str,
    package_version: str,
) -> None:
    policy = separators._SEPARATOR_POLICIES[separator_id]

    assert policy.get("package_version") == package_version
    assert load_separator_lock(_fixture_path(separator_id)).package_version == package_version


def test_loader_rejects_package_version_outside_policy(tmp_path: Path) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    payload["package_version"] = "9.9.9"
    path = tmp_path / "separator.json"
    _write_payload(path, payload)

    with pytest.raises(SeparatorLockError, match="package_version"):
        load_separator_lock(path)


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
        "interpreter_sha256",
        "environment_manifest_sha256",
        "model_root_kind",
    }
    assert _fixture_payload(separator_id)["schema"] == SEPARATOR_LOCK_SCHEMA


def test_v2_lock_requires_its_fixed_canonical_environment_sibling(
    tmp_path: Path,
) -> None:
    fixture_directory = _fixture_path(SPLEETER_SEPARATOR_ID).parent
    copied_directory = tmp_path / "fixture-pair"
    shutil.copytree(fixture_directory, copied_directory)
    lock_path = copied_directory / "model.json"
    lock = load_separator_lock(lock_path)
    manifest = load_separator_environment_manifest(lock_path, lock)

    assert manifest.separator_id == lock.separator_id
    assert manifest.package_name == lock.package_name
    assert manifest.package_version == lock.package_version
    assert manifest.interpreter_sha256 == lock.interpreter_sha256
    assert manifest.sha256 == lock.environment_manifest_sha256

    sibling = lock_path.parent / "environment.json"
    sibling.write_bytes(b"{}\n")
    with pytest.raises(SeparatorLockError, match="companion|environment"):
        load_separator_environment_manifest(lock_path, lock)


@pytest.mark.parametrize("separator_id", [SPLEETER_SEPARATOR_ID, HTDEMUCS_SEPARATOR_ID])
def test_loader_rejects_v1_lock_schema(
    tmp_path: Path,
    separator_id: str,
) -> None:
    payload = _fixture_payload(separator_id)
    payload["schema"] = "crux.separator-lock/v1"
    payload.pop("interpreter_sha256")
    payload.pop("environment_manifest_sha256")
    payload.pop("model_root_kind")
    path = tmp_path / "separator.json"
    _write_payload(path, payload)

    with pytest.raises(SeparatorLockError, match="schema"):
        load_separator_lock(path)


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


_ATTESTATION_FAILURE_CODES = frozenset(
    {
        "separator_lock_companion_mismatch",
        "separator_interpreter_mismatch",
        "separator_environment_mismatch",
        "separator_model_root_invalid",
        "separator_environment_probe_failed",
    }
)


def _freeze_separator_runtime(
    *,
    separator_id: str,
    interpreter: Path,
    model_root: Path,
    repository_revision: str,
    output: Path,
) -> object:
    implementation = getattr(separators, "freeze_separator_runtime", None)
    assert callable(implementation), "freezer implementation is missing"
    has_model_root = "model_root" in inspect.signature(implementation).parameters
    assert has_model_root, "freezer must accept the policy-owned model root"
    return implementation(
        separator_id=separator_id,
        interpreter=interpreter,
        model_root=model_root,
        repository_revision=repository_revision,
        output=output,
    )


def _attest_separator_runtime(
    lock_path: Path,
    interpreter: Path,
    model_root: Path,
) -> object:
    implementation = getattr(separators, "attest_separator_runtime", None)
    assert callable(implementation), "live separator attester is missing"
    return implementation(lock_path, interpreter, model_root)


def _assert_no_attestation_artifacts(tmp_path: Path) -> None:
    for name in ("cache", "stems", "predictions", "reports"):
        assert not (tmp_path / name).exists()


def test_freeze_and_attest_round_trip_with_synthetic_runtime(tmp_path: Path) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"

    lock = _freeze_separator_runtime(
        separator_id=SPLEETER_SEPARATOR_ID,
        interpreter=interpreter,
        model_root=model_root,
        repository_revision="a" * 40,
        output=lock_path,
    )
    runtime = _attest_separator_runtime(lock_path, interpreter, model_root)

    assert runtime.lock == lock
    assert runtime.interpreter == interpreter.resolve(strict=True)
    assert runtime.model_files == lock.model_files
    assert (lock_path.parent / "environment.json").is_file()
    assert getattr(separators, "ATTESTATION_FAILURE_CODES", frozenset()) == (
        _ATTESTATION_FAILURE_CODES
    )


def test_attested_runtime_close_releases_model_root_descriptor(
    tmp_path: Path,
) -> None:
    model_root, expected = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    runtime = _attested_runtime(
        SPLEETER_SEPARATOR_ID,
        model_root=model_root,
        model_files=expected,
    )
    descriptor = runtime.model_root_fd  # type: ignore[attr-defined]
    assert isinstance(descriptor, int)

    runtime.close()  # type: ignore[attr-defined]
    with pytest.raises(OSError):
        os.fstat(descriptor)
    assert runtime.model_root_fd is None  # type: ignore[attr-defined]
    runtime.close()  # type: ignore[attr-defined]


def test_freezer_rejects_live_package_version_before_publishing(tmp_path: Path) -> None:
    interpreter, _ = _synthetic_environment(tmp_path, package_version="9.9.9")
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"

    with pytest.raises(SeparatorExecutionError) as raised:
        _freeze_separator_runtime(
            separator_id=SPLEETER_SEPARATOR_ID,
            interpreter=interpreter,
            model_root=model_root,
            repository_revision="a" * 40,
            output=lock_path,
        )

    assert raised.value.code == "separator_environment_mismatch"
    assert not lock_path.exists()
    assert not (lock_path.parent / "environment.json").exists()


def test_freezer_rejects_relative_model_root_before_probe_or_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    relative_model_root = Path(os.path.relpath(model_root, Path.cwd()))
    lock_path = tmp_path / "frozen" / "model.json"

    def unexpected_probe(_interpreter: Path) -> object:
        raise AssertionError("relative model roots must be rejected before probing")

    monkeypatch.setattr(separators, "_run_separator_environment_probe", unexpected_probe)

    with pytest.raises(SeparatorExecutionError) as raised:
        _freeze_separator_runtime(
            separator_id=SPLEETER_SEPARATOR_ID,
            interpreter=interpreter,
            model_root=relative_model_root,
            repository_revision="a" * 40,
            output=lock_path,
        )

    assert raised.value.code == "separator_model_root_invalid"
    assert not lock_path.exists()
    assert not (lock_path.parent / "environment.json").exists()


def test_freezer_rejects_invalid_repository_revision_before_probe_or_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"

    def unexpected_probe(_interpreter: Path) -> object:
        raise AssertionError("invalid revision must be rejected before probing")

    monkeypatch.setattr(separators, "_run_separator_environment_probe", unexpected_probe)

    with pytest.raises(SeparatorLockError, match="repository_revision"):
        _freeze_separator_runtime(
            separator_id=SPLEETER_SEPARATOR_ID,
            interpreter=interpreter,
            model_root=model_root,
            repository_revision="not-a-sha",
            output=lock_path,
        )

    assert not lock_path.exists()
    assert not (lock_path.parent / "environment.json").exists()


def test_freezer_preserves_preexisting_environment_on_lock_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"
    environment_path = lock_path.parent / "environment.json"

    # First freeze succeeds and publishes both files.
    _freeze_separator_runtime(
        separator_id=SPLEETER_SEPARATOR_ID,
        interpreter=interpreter,
        model_root=model_root,
        repository_revision="a" * 40,
        output=lock_path,
    )
    environment_bytes_before = environment_path.read_bytes()
    lock_bytes_before = lock_path.read_bytes()

    # Second freeze with a different revision reuses environment.json (identical
    # bytes) but must conflict on model.json (different lock payload).  The
    # pre-existing environment.json must survive the conflict.
    real_publish = separators.publish_immutable_file

    def conflict_on_lock(path: Path, content: bytes) -> object:
        if path == lock_path and content != lock_bytes_before:
            from src.benchmark.artifact_io import ArtifactPublicationError

            raise ArtifactPublicationError("artifact already exists with different bytes")
        return real_publish(path, content)

    monkeypatch.setattr(separators, "publish_immutable_file", conflict_on_lock)

    with pytest.raises(SeparatorExecutionError, match="separator_lock_publication_failed"):
        _freeze_separator_runtime(
            separator_id=SPLEETER_SEPARATOR_ID,
            interpreter=interpreter,
            model_root=model_root,
            repository_revision="b" * 40,
            output=lock_path,
        )

    assert environment_path.exists()
    assert environment_path.read_bytes() == environment_bytes_before
    assert lock_path.read_bytes() == lock_bytes_before


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("companion", "separator_lock_companion_mismatch"),
        ("wrong_interpreter_hash", "separator_interpreter_mismatch"),
        ("changed_recorded_package_file", "separator_environment_mismatch"),
        ("changed_model_file", "separator_model_root_invalid"),
        ("bad_probe_stdout", "separator_environment_probe_failed"),
    ],
)
def test_attest_translates_synthetic_mismatches_to_closed_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"
    _freeze_separator_runtime(
        separator_id=SPLEETER_SEPARATOR_ID,
        interpreter=interpreter,
        model_root=model_root,
        repository_revision="a" * 40,
        output=lock_path,
    )

    environment_path = lock_path.parent / "environment.json"
    if mutation == "companion":
        environment_path.write_bytes(b"{}\n")
    elif mutation == "wrong_interpreter_hash":
        wrong_hash = "0" * 64
        environment_payload = json.loads(environment_path.read_text(encoding="utf-8"))
        environment_payload["interpreter_sha256"] = wrong_hash
        environment_bytes = canonical_json_bytes(environment_payload, trailing_newline=True)
        environment_path.write_bytes(environment_bytes)
        lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_payload["interpreter_sha256"] = wrong_hash
        lock_payload["environment_manifest_sha256"] = hashlib.sha256(environment_bytes).hexdigest()
        lock_path.write_bytes(canonical_json_bytes(lock_payload, trailing_newline=True))
    elif mutation == "changed_recorded_package_file":
        (purelib / "spleeter" / "__init__.py").write_text("changed\n", encoding="utf-8")
    elif mutation == "changed_model_file":
        (model_root / "4stems" / "model.meta").write_bytes(b"changed model bytes")
    elif mutation == "bad_probe_stdout":

        def fail_probe(_interpreter: Path) -> object:
            raise SeparatorExecutionError("separator_environment_probe_failed")

        monkeypatch.setattr(separators, "_run_separator_environment_probe", fail_probe)
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(SeparatorExecutionError) as raised:
        _attest_separator_runtime(lock_path, interpreter, model_root)

    assert raised.value.code == expected_code
    assert raised.value.code in _ATTESTATION_FAILURE_CODES
    _assert_no_attestation_artifacts(tmp_path)


def test_freezer_cli_requires_model_root_and_rejects_model_file(
    tmp_path: Path,
) -> None:
    from src.cli import freeze_separator_runtime as cli

    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"
    arguments = [
        "--separator-id",
        SPLEETER_SEPARATOR_ID,
        "--interpreter",
        str(interpreter),
        "--model-root",
        str(model_root),
        "--repository-revision",
        "a" * 40,
        "--output",
        str(lock_path),
    ]

    result = cli.main(arguments)
    assert result == 0
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(arguments + ["--model-file", "weights.bin=x"])
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(arguments + ["--environment", "environment.json"])


class _FakePopen:
    def __init__(
        self,
        *,
        returncode: int = 0,
        write_output: object | None = None,
        timeout: bool = False,
        terminate_wait_timeout: bool = False,
        leader_exits_after_term: bool = False,
    ) -> None:
        self.returncode = returncode
        self.write_output = write_output
        self.timeout = timeout
        self.terminate_wait_timeout = terminate_wait_timeout
        self.leader_exits_after_term = leader_exits_after_term
        self.pid = 4321
        self.argv: list[str] = []
        self.kwargs: dict[str, object] = {}
        self.wait_calls: list[float | None] = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        if self.timeout:
            raise subprocess.TimeoutExpired(self.argv or "separator", timeout)
        if self.write_output is not None:
            assert callable(self.write_output)
            self.write_output(Path(self.kwargs["cwd"]))
        return b"", b""

    def wait(self, timeout: float | None = None) -> None:
        self.wait_calls.append(timeout)
        if self.leader_exits_after_term and len(self.wait_calls) == 1:
            return
        if self.terminate_wait_timeout and len(self.wait_calls) == 1:
            raise subprocess.TimeoutExpired(self.argv or "separator", timeout)

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1


def _synthetic_model_root(
    tmp_path: Path,
    separator_id: str,
) -> tuple[Path, tuple[SeparatorModelFile, ...]]:
    expected_files = (
        {
            "4stems/checkpoint": b"checkpoint bytes",
            "4stems/.probe": b"probe bytes",
            "4stems/model.index": b"index bytes",
            "4stems/model.data-00000-of-00001": b"data bytes",
            "4stems/model.meta": b"meta bytes",
        }
        if separator_id == SPLEETER_SEPARATOR_ID
        else {
            "htdemucs.yaml": b"yaml bytes",
            "955717e8-8726e21a.th": b"weights bytes",
        }
    )
    root = tmp_path / separator_id
    for relative_path, content in expected_files.items():
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    model_files = tuple(
        SeparatorModelFile(
            name=relative_path,
            sha256=hashlib.sha256(content).hexdigest(),
        )
        for relative_path, content in sorted(expected_files.items())
    )
    return root, model_files


def _attested_runtime(
    separator_id: str,
    *,
    interpreter: Path = Path("/isolated/python"),
    model_root: Path = Path("/isolated/model-root"),
    model_files: tuple[SeparatorModelFile, ...] | None = None,
) -> object:
    lock_path = _lock_path(separator_id)
    lock = load_separator_lock(lock_path)
    if model_files is not None:
        lock = replace(lock, model_files=model_files)
    environment = load_separator_environment_manifest(lock_path, load_separator_lock(lock_path))
    factory = getattr(separators, "_build_attested_separator_runtime", None)
    assert callable(factory), "Task 3 internal runtime construction gate is missing"
    return factory(
        interpreter=interpreter,
        lock=lock,
        model_root=model_root,
        model_files=lock.model_files,
        environment=environment,
    )


@pytest.mark.parametrize("separator_id", [SPLEETER_SEPARATOR_ID, HTDEMUCS_SEPARATOR_ID])
def test_inventory_separator_model_root_returns_exact_policy_files(
    tmp_path: Path,
    separator_id: str,
) -> None:
    model_root, expected = _synthetic_model_root(tmp_path, separator_id)

    assert separators.inventory_separator_model_root(separator_id, model_root) == expected


@pytest.mark.parametrize("separator_id", [SPLEETER_SEPARATOR_ID, HTDEMUCS_SEPARATOR_ID])
def test_inventory_separator_model_root_rejects_extra_regular_file(
    tmp_path: Path,
    separator_id: str,
) -> None:
    model_root, _ = _synthetic_model_root(tmp_path, separator_id)
    (model_root / "extra.bin").write_bytes(b"unexpected")

    with pytest.raises(SeparatorExecutionError, match="separator_model_root_invalid") as raised:
        separators.inventory_separator_model_root(separator_id, model_root)
    assert raised.value.detail_code == "separator_model_root_invalid"


@pytest.mark.parametrize("separator_id", [SPLEETER_SEPARATOR_ID, HTDEMUCS_SEPARATOR_ID])
def test_inventory_separator_model_root_rejects_symlink(
    tmp_path: Path,
    separator_id: str,
) -> None:
    model_root, _ = _synthetic_model_root(tmp_path, separator_id)
    relative_path = (
        "4stems/model.index" if separator_id == SPLEETER_SEPARATOR_ID else "htdemucs.yaml"
    )
    destination = model_root / relative_path
    destination.unlink()
    destination.symlink_to(
        model_root
        / ("4stems/checkpoint" if separator_id == SPLEETER_SEPARATOR_ID else "955717e8-8726e21a.th")
    )

    with pytest.raises(SeparatorExecutionError, match="separator_model_root_invalid") as raised:
        separators.inventory_separator_model_root(separator_id, model_root)
    assert raised.value.detail_code == "separator_model_root_invalid"


def test_inventory_separator_model_root_rejects_parent_symlink(tmp_path: Path) -> None:
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    nested_root = model_root / "4stems"
    replacement = tmp_path / "real-4stems"
    nested_root.rename(replacement)
    nested_root.symlink_to(replacement, target_is_directory=True)

    with pytest.raises(SeparatorExecutionError, match="separator_model_root_invalid") as raised:
        separators.inventory_separator_model_root(SPLEETER_SEPARATOR_ID, model_root)
    assert raised.value.detail_code == "separator_model_root_invalid"


def test_inventory_separator_model_root_rejects_child_directory_identity_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    original_directory = model_root / "4stems"
    replacement_directory = tmp_path / "replacement-4stems"
    shutil.copytree(original_directory, replacement_directory)
    moved_original = tmp_path / "original-4stems"
    original_open = separators.os.open
    swapped = False

    def swap_directory_before_open(
        path: object,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal swapped
        if path == "4stems" and kwargs.get("dir_fd") is not None and not swapped:
            original_directory.rename(moved_original)
            replacement_directory.rename(original_directory)
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(separators.os, "open", swap_directory_before_open)
    monkeypatch.setattr(
        separators.os,
        "supports_dir_fd",
        set(separators.os.supports_dir_fd) | {swap_directory_before_open},
    )

    with pytest.raises(SeparatorExecutionError, match="separator_model_root_invalid") as raised:
        separators.inventory_separator_model_root(SPLEETER_SEPARATOR_ID, model_root)
    assert raised.value.detail_code == "separator_model_root_invalid"
    assert swapped is True


def test_runner_rejects_direct_unattested_runtime_instance(
    tmp_path: Path,
) -> None:
    source_path = _source_wav(tmp_path)
    model_root, expected = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = _lock_path(SPLEETER_SEPARATOR_ID)
    lock = replace(load_separator_lock(lock_path), model_files=expected)
    environment = load_separator_environment_manifest(lock_path, load_separator_lock(lock_path))
    runtime = separators.AttestedSeparatorRuntime(
        interpreter=Path("/isolated/python"),
        lock=lock,
        model_root=model_root,
        model_files=expected,
        environment=environment,
        launch_environment={"PATH": "/isolated/bin", "MODEL_PATH": str(model_root)},
    )

    with pytest.raises(SeparatorExecutionError, match="separator_runtime_unattested") as raised:
        _run_separator(
            SPLEETER_SEPARATOR_ID,
            source_path,
            cache_root=tmp_path / "cache",
            runtime=runtime,
        )
    assert raised.value.detail_code == "separator_runtime_unattested"


def test_revalidate_separator_model_root_compares_the_attested_inventory(
    tmp_path: Path,
) -> None:
    model_root, expected = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    runtime = _attested_runtime(
        SPLEETER_SEPARATOR_ID,
        model_root=model_root,
        model_files=expected,
    )

    separators.revalidate_separator_model_root(runtime)

    (model_root / "4stems" / "model.meta").write_bytes(b"changed")
    with pytest.raises(SeparatorExecutionError, match="separator_model_root_invalid") as raised:
        separators.revalidate_separator_model_root(runtime)
    assert raised.value.detail_code == "separator_model_root_invalid"


def test_htdemucs_renderer_propagates_the_attested_local_repository(
    tmp_path: Path,
) -> None:
    model_root, expected = _synthetic_model_root(tmp_path, HTDEMUCS_SEPARATOR_ID)
    runtime = _attested_runtime(
        HTDEMUCS_SEPARATOR_ID,
        interpreter=Path("/isolated/demucs/python"),
        model_root=model_root,
        model_files=expected,
    )

    argv = separators._render_separator_argv(
        runtime,
        input_path=tmp_path / "input.wav",
        output_dir=tmp_path / "output",
    )

    assert argv[0] == "/isolated/demucs/python"
    assert argv[argv.index("--repo") + 1] == str(runtime.model_root_launch_path)


def test_spleeter_launch_environment_replaces_inherited_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    monkeypatch.setenv("MODEL_PATH", "/ambient/model-cache")
    monkeypatch.setenv("PYTHONPATH", "/ambient/python-path")
    monkeypatch.setenv("PYTHONHOME", "/ambient/python-home")
    monkeypatch.setenv("PYTHONUSERBASE", "/ambient/python-userbase")

    environment = separators._build_separator_launch_environment(
        SPLEETER_SEPARATOR_ID,
        model_root,
    )

    assert environment["MODEL_PATH"] == str(model_root)
    assert all(
        key not in environment
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONNOUSERSITE")
    )


def test_environment_probe_uses_isolated_interpreter_and_sanitized_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONNOUSERSITE"):
        monkeypatch.setenv(key, "/attacker/" + key.lower())

    observed: dict[str, object] = {}
    original_run = separators.subprocess.run

    def recording_run(argv: object, **kwargs: object) -> object:
        observed["argv"] = argv
        observed["env"] = kwargs.get("env")
        return original_run(argv, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(separators.subprocess, "run", recording_run)
    manifest = separators._run_separator_environment_probe(interpreter)

    argv = observed["argv"]
    assert isinstance(argv, list)
    assert argv[0] == str(interpreter.resolve(strict=True))
    assert argv[1] == "-I"
    environment = observed["env"]
    assert isinstance(environment, dict)
    assert all(
        key not in environment
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONNOUSERSITE")
    )
    assert manifest.package_name == "spleeter"


def test_demucs_launch_environment_removes_model_cache_and_endpoint_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root, _ = _synthetic_model_root(tmp_path, HTDEMUCS_SEPARATOR_ID)
    policy = separators._SEPARATOR_POLICIES[HTDEMUCS_SEPARATOR_ID]
    discovery_keys = tuple(policy["environment_discovery_keys"])
    for key in discovery_keys:
        monkeypatch.setenv(key, "/ambient/" + key.lower())

    environment = separators._build_separator_launch_environment(
        HTDEMUCS_SEPARATOR_ID,
        model_root,
    )

    assert all(key not in environment for key in discovery_keys)


def test_separator_passes_runtime_launch_environment_to_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _source_wav(tmp_path)
    model_root, expected = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONNOUSERSITE"):
        monkeypatch.setenv(key, "/attacker/" + key.lower())
    runtime = _attested_runtime(
        SPLEETER_SEPARATOR_ID,
        model_root=model_root,
        model_files=expected,
    )
    fake = _FakePopen(
        write_output=lambda workdir: _write_stem(
            workdir,
            separator_id=SPLEETER_SEPARATOR_ID,
            samples=np.full((44100, 1), 0.25, dtype=np.float32),
        )
    )
    _install_fake_popen(monkeypatch, fake)

    result = _run_separator(
        SPLEETER_SEPARATOR_ID,
        source_path,
        cache_root=tmp_path / "cache",
        runtime=runtime,
    )

    assert fake.kwargs["env"] == runtime.launch_environment
    assert fake.argv[1] == "-I"
    assert fake.kwargs["pass_fds"] == (runtime.model_root_fd,)  # type: ignore[attr-defined]
    assert all(
        key not in fake.kwargs["env"]
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONNOUSERSITE")
    )
    assert str(model_root) not in str(result.path)


def test_separator_launch_stays_bound_to_attested_model_root_after_alias_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _source_wav(tmp_path)
    model_root, expected = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    runtime = _attested_runtime(
        SPLEETER_SEPARATOR_ID,
        model_root=model_root,
        model_files=expected,
    )
    model_root_fd = runtime.model_root_fd  # type: ignore[attr-defined]
    assert isinstance(model_root_fd, int)

    replacement = tmp_path / "replacement-model-root"
    shutil.copytree(model_root, replacement)
    (replacement / "4stems" / "model.meta").write_bytes(b"unverified replacement")
    verified = tmp_path / "verified-model-root"
    model_root.rename(verified)
    replacement.rename(model_root)

    fake = _FakePopen(
        write_output=lambda workdir: _write_stem(
            workdir,
            separator_id=SPLEETER_SEPARATOR_ID,
            samples=np.full((44100, 1), 0.25, dtype=np.float32),
        )
    )
    _install_fake_popen(monkeypatch, fake)
    try:
        _run_separator(
            SPLEETER_SEPARATOR_ID,
            source_path,
            cache_root=tmp_path / "cache",
            runtime=runtime,
        )
        bound_root = Path(fake.kwargs["env"]["MODEL_PATH"])
        pass_fds = fake.kwargs["pass_fds"]
        if bound_root == Path("."):
            original_cwd_fd = os.open(".", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                preexec = fake.kwargs["preexec_fn"]
                assert callable(preexec)
                preexec()
                bound_model_bytes = Path("4stems/model.meta").read_bytes()
            finally:
                os.fchdir(original_cwd_fd)
                os.close(original_cwd_fd)
        else:
            bound_model_bytes = (bound_root / "4stems" / "model.meta").read_bytes()
    finally:
        runtime.close()  # type: ignore[attr-defined]

    assert bound_root != model_root
    assert bound_model_bytes == b"meta bytes"
    assert pass_fds == (model_root_fd,)


def _source_wav(tmp_path: Path, *, duration_sec: float = 1.0) -> Path:
    path = tmp_path / "source.wav"
    frame_count = int(duration_sec * 44100)
    sf.write(path, np.full(frame_count, 0.25, dtype=np.float32), 44100, format="WAV")
    return path


def _lock_path(separator_id: str) -> Path:
    return _fixture_path(separator_id)


def _run_separator(
    separator_id: str,
    source_path: Path,
    *,
    cache_root: Path,
    runtime: object | None = None,
    **kwargs: object,
) -> object:
    implementation = (
        getattr(separators, "run_spleeter_drums", None)
        if separator_id == SPLEETER_SEPARATOR_ID
        else getattr(separators, "run_htdemucs_drums", None)
    )
    assert callable(implementation), "Task 4 separator execution API is missing"
    if runtime is None:
        model_root, model_files = _synthetic_model_root(cache_root, separator_id)
        runtime = _attested_runtime(
            separator_id,
            model_root=model_root,
            model_files=model_files,
        )
    return implementation(
        source_path,
        source_audio_sha256="a" * 64,
        source_duration_sec=1.0,
        runtime=runtime,
        cache_root=cache_root,
        **kwargs,
    )


def _install_fake_popen(
    monkeypatch: pytest.MonkeyPatch,
    fake: _FakePopen,
) -> None:
    def factory(argv: list[str], **kwargs: object) -> _FakePopen:
        fake.argv = argv
        fake.kwargs = kwargs
        return fake

    monkeypatch.setattr(subprocess, "Popen", factory)


def _write_stem(
    workdir: Path,
    *,
    separator_id: str,
    samples: np.ndarray,
    sample_rate: int = 44100,
) -> None:
    lock = load_separator_lock(_lock_path(separator_id))
    output_path = workdir / "output" / lock.expected_drum_stem_relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, samples, sample_rate, format="WAV")


@pytest.mark.parametrize("separator_id", [SPLEETER_SEPARATOR_ID, HTDEMUCS_SEPARATOR_ID])
def test_separator_stages_input_wav_and_publishes_exact_native_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    separator_id: str,
) -> None:
    source_path = _source_wav(tmp_path)
    native_bytes: bytes | None = None

    def write_output(workdir: Path) -> None:
        nonlocal native_bytes
        assert (workdir / "input.wav").read_bytes() == source_path.read_bytes()
        samples = np.full((44100, 1), 0.25, dtype=np.float32)
        _write_stem(workdir, separator_id=separator_id, samples=samples)
        native_bytes = (
            workdir
            / "output"
            / load_separator_lock(_lock_path(separator_id)).expected_drum_stem_relative_path
        ).read_bytes()

    fake = _FakePopen(write_output=write_output)
    _install_fake_popen(monkeypatch, fake)

    result = _run_separator(separator_id, source_path, cache_root=tmp_path / "cache")

    assert fake.argv[-1] == str(Path(fake.kwargs["cwd"]) / "input.wav")
    assert fake.argv[-2] == str(Path(fake.kwargs["cwd"]) / "output")
    assert native_bytes is not None
    assert result.path == (
        tmp_path
        / "cache"
        / "derived"
        / "stems"
        / separator_id
        / ("a" * 64)
        / load_separator_lock(_lock_path(separator_id)).sha256
        / "drums.wav"
    )
    assert result.path.read_bytes() == native_bytes
    assert result.cache_hit is False


@pytest.mark.parametrize(
    ("returncode", "detail"),
    [(9, "separator_nonzero_exit"), (0, "separator_output_missing")],
)
def test_separator_reports_nonzero_exit_and_missing_output_with_stable_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    detail: str,
) -> None:
    source_path = _source_wav(tmp_path)
    fake = _FakePopen(returncode=returncode)
    _install_fake_popen(monkeypatch, fake)

    implementation = (
        separators.run_spleeter_drums if hasattr(separators, "run_spleeter_drums") else None
    )
    assert callable(implementation), "Task 4 separator execution API is missing"
    with pytest.raises(ValueError, match=detail) as raised:
        _run_separator(SPLEETER_SEPARATOR_ID, source_path, cache_root=tmp_path / "cache")
    assert raised.value.detail_code == detail


@pytest.mark.parametrize(
    ("samples", "detail"),
    [
        (b"not a wav", "stem_decode_failed"),
        (np.zeros((44100, 1), dtype=np.float32), "stem_near_silent"),
        (np.zeros((44100 + 44100, 1), dtype=np.float32), "stem_duration_mismatch"),
        (np.zeros((44100, 3), dtype=np.float32), "stem_channel_count"),
        (np.full((44100, 1), np.nan, dtype=np.float32), "stem_nonfinite"),
    ],
)
def test_separator_rejects_invalid_stems_with_stable_qc_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    samples: object,
    detail: str,
) -> None:
    source_path = _source_wav(tmp_path)

    def write_output(workdir: Path) -> None:
        lock = load_separator_lock(_lock_path(SPLEETER_SEPARATOR_ID))
        output_path = workdir / "output" / lock.expected_drum_stem_relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(samples, bytes):
            output_path.write_bytes(samples)
        else:
            sf.write(output_path, samples, 44100, format="WAV", subtype="FLOAT")

    fake = _FakePopen(write_output=write_output)
    _install_fake_popen(monkeypatch, fake)
    with pytest.raises(ValueError, match=detail) as raised:
        _run_separator(SPLEETER_SEPARATOR_ID, source_path, cache_root=tmp_path / "cache")
    assert raised.value.detail_code == detail


def test_separator_records_clipping_as_qc_warning_without_rejecting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _source_wav(tmp_path)
    fake = _FakePopen(
        write_output=lambda workdir: _write_stem(
            workdir,
            separator_id=SPLEETER_SEPARATOR_ID,
            samples=np.full((44100, 1), 1.0, dtype=np.float32),
        )
    )
    _install_fake_popen(monkeypatch, fake)

    result = _run_separator(SPLEETER_SEPARATOR_ID, source_path, cache_root=tmp_path / "cache")

    assert result.qc.clipping_detected is True
    assert result.qc.peak_abs >= 0.9999
    assert "stem_clipping" in result.qc.warnings
    assert "stem_clipping" in result.warnings


def test_separator_cache_hit_bypasses_popen_but_reruns_qc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _source_wav(tmp_path)
    first_fake = _FakePopen(
        write_output=lambda workdir: _write_stem(
            workdir,
            separator_id=SPLEETER_SEPARATOR_ID,
            samples=np.full((44100, 1), 0.25, dtype=np.float32),
        )
    )
    _install_fake_popen(monkeypatch, first_fake)
    cache_root = tmp_path / "cache"
    _run_separator(SPLEETER_SEPARATOR_ID, source_path, cache_root=cache_root)

    def unexpected_popen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("cache hit must bypass separator subprocess")

    monkeypatch.setattr(subprocess, "Popen", unexpected_popen)
    read_calls = 0
    original_read = separators.soundfile.read

    def recording_read(*args: object, **kwargs: object) -> object:
        nonlocal read_calls
        read_calls += 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(separators.soundfile, "read", recording_read)
    result = _run_separator(SPLEETER_SEPARATOR_ID, source_path, cache_root=cache_root)

    assert result.cache_hit is True
    assert read_calls == 1
    assert result.qc.rms_dbfs > -80.0


def test_separator_cache_hit_still_rejects_near_silent_cached_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _source_wav(tmp_path)
    cache_root = tmp_path / "cache"
    cache_path = (
        cache_root
        / "derived"
        / "stems"
        / SPLEETER_SEPARATOR_ID
        / ("a" * 64)
        / load_separator_lock(_lock_path(SPLEETER_SEPARATOR_ID)).sha256
        / "drums.wav"
    )
    cache_path.parent.mkdir(parents=True)
    sf.write(cache_path, np.zeros((44100, 1), dtype=np.float32), 44100, format="WAV")

    def unexpected_popen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("cached output must not invoke separator")

    monkeypatch.setattr(subprocess, "Popen", unexpected_popen)
    with pytest.raises(ValueError, match="stem_near_silent") as raised:
        _run_separator(SPLEETER_SEPARATOR_ID, source_path, cache_root=cache_root)
    assert raised.value.detail_code == "stem_near_silent"


def test_separator_qc_uses_captured_fresh_stem_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _source_wav(tmp_path)
    fake = _FakePopen(
        write_output=lambda workdir: _write_stem(
            workdir,
            separator_id=SPLEETER_SEPARATOR_ID,
            samples=np.full((44100, 1), 0.25, dtype=np.float32),
        )
    )
    _install_fake_popen(monkeypatch, fake)
    original_read = separators.read_regular_file_no_follow

    def mutate_after_capture(path: Path) -> bytes:
        content = original_read(path)
        if path.name == "drums.wav":
            sf.write(path, np.zeros((44100, 1), dtype=np.float32), 44100, format="WAV")
        return content

    monkeypatch.setattr(separators, "read_regular_file_no_follow", mutate_after_capture)

    result = _run_separator(SPLEETER_SEPARATOR_ID, source_path, cache_root=tmp_path / "cache")

    assert result.qc.rms_dbfs > -80.0
    assert result.path.read_bytes() != b""


def test_separator_qc_uses_captured_cached_stem_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _source_wav(tmp_path)
    first_fake = _FakePopen(
        write_output=lambda workdir: _write_stem(
            workdir,
            separator_id=SPLEETER_SEPARATOR_ID,
            samples=np.full((44100, 1), 0.25, dtype=np.float32),
        )
    )
    _install_fake_popen(monkeypatch, first_fake)
    cache_root = tmp_path / "cache"
    first = _run_separator(SPLEETER_SEPARATOR_ID, source_path, cache_root=cache_root)
    captured_bytes = first.path.read_bytes()
    original_read = separators.read_regular_file_no_follow

    def mutate_after_capture(path: Path) -> bytes:
        content = original_read(path)
        if path == first.path:
            sf.write(path, np.zeros((44100, 1), dtype=np.float32), 44100, format="WAV")
        return content

    monkeypatch.setattr(separators, "read_regular_file_no_follow", mutate_after_capture)
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("cache hit must bypass separator subprocess"),
    )

    result = _run_separator(SPLEETER_SEPARATOR_ID, source_path, cache_root=cache_root)

    assert result.qc.rms_dbfs > -80.0
    assert result.sha256 == hashlib.sha256(captured_bytes).hexdigest()


def test_separator_maps_immutable_publication_conflict_to_stable_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _source_wav(tmp_path)
    fake = _FakePopen(
        write_output=lambda workdir: _write_stem(
            workdir,
            separator_id=SPLEETER_SEPARATOR_ID,
            samples=np.full((44100, 1), 0.25, dtype=np.float32),
        )
    )
    _install_fake_popen(monkeypatch, fake)

    def conflict(*_args: object, **_kwargs: object) -> object:
        from src.benchmark.artifact_io import ArtifactPublicationError

        raise ArtifactPublicationError("artifact already exists with different bytes")

    monkeypatch.setattr(separators, "publish_immutable_file", conflict, raising=False)
    with pytest.raises(ValueError, match="stem_publish_conflict") as raised:
        _run_separator(SPLEETER_SEPARATOR_ID, source_path, cache_root=tmp_path / "cache")
    assert raised.value.detail_code == "stem_publish_conflict"


def test_separator_timeout_terminates_process_group_then_kills_after_grace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _source_wav(tmp_path)
    fake = _FakePopen(timeout=True, terminate_wait_timeout=True)
    _install_fake_popen(monkeypatch, fake)
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        os,
        "killpg",
        lambda process_group, signal_number: killpg_calls.append((process_group, signal_number)),
    )

    with pytest.raises(ValueError, match="separator_timeout") as raised:
        _run_separator(SPLEETER_SEPARATOR_ID, source_path, cache_root=tmp_path / "cache")

    assert raised.value.detail_code == "separator_timeout"
    assert killpg_calls == [
        (fake.pid, signal.SIGTERM),
        (fake.pid, signal.SIGKILL),
    ]
    assert fake.wait_calls == [5.0, None]
    assert fake.terminate_calls == 0
    assert fake.kill_calls == 0


def test_separator_timeout_kills_group_even_when_leader_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _source_wav(tmp_path)
    fake = _FakePopen(timeout=True, leader_exits_after_term=True)
    _install_fake_popen(monkeypatch, fake)
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        os,
        "killpg",
        lambda process_group, signal_number: killpg_calls.append((process_group, signal_number)),
    )

    with pytest.raises(ValueError, match="separator_timeout"):
        _run_separator(SPLEETER_SEPARATOR_ID, source_path, cache_root=tmp_path / "cache")

    assert killpg_calls == [
        (fake.pid, signal.SIGTERM),
        (fake.pid, signal.SIGKILL),
    ]
    assert fake.wait_calls == [5.0]
