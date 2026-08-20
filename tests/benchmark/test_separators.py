from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import signal
import subprocess
import threading
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


def test_concurrent_freezes_same_env_different_revision_retain_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent freezes with the same env but different revisions must not
    orphan the winning lock's companion manifest.

    Without per-runtime publication serialization, both freezes observe
    ``environment_preexisting == False``; the loser reuses the winner's
    environment.json, conflicts on model.json, and then deletes the manifest it
    does not own.  The coordinated publisher below forces that interleaving so
    the regression is deterministic.
    """
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"
    environment_path = lock_path.parent / "environment.json"

    real_publish = separators.publish_immutable_file
    env_first_done = threading.Event()
    lock_first_done = threading.Event()
    counter_lock = threading.Lock()
    counters = {"environment": 0, "lock": 0}

    def coordinated_publish(path: Path, content: bytes) -> object:
        if path == environment_path:
            with counter_lock:
                counters["environment"] += 1
                rank = counters["environment"]
            if rank == 1:
                result = real_publish(path, content)
                env_first_done.set()
                return result
            env_first_done.wait(timeout=10)
            return real_publish(path, content)
        if path == lock_path:
            with counter_lock:
                counters["lock"] += 1
                rank = counters["lock"]
            if rank == 1:
                result = real_publish(path, content)
                lock_first_done.set()
                return result
            lock_first_done.wait(timeout=10)
            return real_publish(path, content)
        return real_publish(path, content)

    monkeypatch.setattr(separators, "publish_immutable_file", coordinated_publish)

    outcomes: dict[str, str] = {}

    def freeze(revision: str) -> None:
        try:
            separators.freeze_separator_runtime(
                separator_id=SPLEETER_SEPARATOR_ID,
                interpreter=interpreter,
                model_root=model_root,
                repository_revision=revision,
                output=lock_path,
            )
            outcomes[revision] = "success"
        except Exception as error:  # noqa: BLE001
            outcomes[revision] = type(error).__name__

    revision_a = "a" * 40
    revision_b = "b" * 40
    thread_a = threading.Thread(target=freeze, args=(revision_a,), name="freeze-a")
    thread_b = threading.Thread(target=freeze, args=(revision_b,), name="freeze-b")
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=30)
    thread_b.join(timeout=30)

    # Exactly one freeze wins; the other conflicts on the lock payload.
    winners = [rev for rev, outcome in outcomes.items() if outcome == "success"]
    losers = [rev for rev, outcome in outcomes.items() if outcome != "success"]
    assert len(winners) == 1, outcomes
    assert len(losers) == 1, outcomes
    assert outcomes[losers[0]] == "SeparatorExecutionError", outcomes

    # The winning lock must retain its companion manifest.
    assert lock_path.is_file(), "winning lock was not published"
    assert environment_path.is_file(), "winning lock's companion manifest was deleted"
    lock = separators.load_separator_lock(lock_path)
    manifest = separators.load_separator_environment_manifest(lock_path, lock)
    assert lock.repository_revision == winners[0]
    assert manifest.sha256 == lock.environment_manifest_sha256


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


# ---------------------------------------------------------------------------
# SeparatorExecutionError construction validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("detail_code", ["", 123, None])
def test_execution_error_rejects_invalid_detail_code(detail_code: object) -> None:
    with pytest.raises(ValueError, match="detail_code"):
        SeparatorExecutionError(detail_code)  # type: ignore[arg-type]


def test_execution_error_code_property_returns_detail_code() -> None:
    error = SeparatorExecutionError("my_code", "description")
    assert error.code == "my_code"
    assert error.detail_code == "my_code"
    assert "my_code" in str(error)
    assert "description" in str(error)


def test_execution_error_defaults_message_to_detail_code() -> None:
    error = SeparatorExecutionError("my_code")
    assert str(error) == "my_code: my_code"


# ---------------------------------------------------------------------------
# load_separator_lock error paths
# ---------------------------------------------------------------------------


def test_load_separator_lock_rejects_non_path(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="path must be a Path"):
        load_separator_lock("not-a-path")  # type: ignore[arg-type]


def test_load_separator_lock_rejects_missing_trailing_newline(tmp_path: Path) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    path = tmp_path / "separator.json"
    path.write_bytes(canonical_json_bytes(payload, trailing_newline=False))

    with pytest.raises(SeparatorLockError, match="final newline"):
        load_separator_lock(path)


def test_load_separator_lock_rejects_double_trailing_newline(tmp_path: Path) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    path = tmp_path / "separator.json"
    path.write_bytes(canonical_json_bytes(payload, trailing_newline=True) + b"\n")

    with pytest.raises(SeparatorLockError, match="final newline"):
        load_separator_lock(path)


def test_load_separator_lock_rejects_non_dict_json(tmp_path: Path) -> None:
    path = tmp_path / "separator.json"
    path.write_bytes(b"[]\n")

    with pytest.raises(SeparatorLockError, match="exact key set"):
        load_separator_lock(path)


def test_load_separator_lock_rejects_model_files_not_list(tmp_path: Path) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    payload["model_files"] = "not-a-list"
    path = tmp_path / "separator.json"
    _write_payload(path, payload)

    with pytest.raises(SeparatorLockError, match="model_files must be a list"):
        load_separator_lock(path)


def test_load_separator_lock_rejects_argv_not_list_of_strings(tmp_path: Path) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    payload["argv"] = ["-m", 123]
    path = tmp_path / "separator.json"
    _write_payload(path, payload)

    with pytest.raises(SeparatorLockError, match="argv must be a list of strings"):
        load_separator_lock(path)


def test_load_separator_lock_rejects_fields_invalid_via_type_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    path = tmp_path / "separator.json"
    _write_payload(path, payload)

    original_model_file_from_value = separators._model_file_from_value

    def raise_type_error(value: object) -> object:
        if isinstance(value, dict) and set(value) == separators._MODEL_FILE_KEYS:
            raise TypeError("synthetic type error")
        return original_model_file_from_value(value)

    monkeypatch.setattr(separators, "_model_file_from_value", raise_type_error)
    with pytest.raises(SeparatorLockError, match="fields are invalid"):
        load_separator_lock(path)


def test_load_separator_lock_rejects_unavailable_file(tmp_path: Path) -> None:
    with pytest.raises(SeparatorLockError, match="unavailable"):
        load_separator_lock(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# separator_lock_payload / separator_environment_manifest_payload type guards
# ---------------------------------------------------------------------------


def test_separator_lock_payload_rejects_non_lock() -> None:
    with pytest.raises(TypeError, match="lock must be a SeparatorLock"):
        separators.separator_lock_payload("not-a-lock")  # type: ignore[arg-type]


def test_separator_environment_manifest_payload_rejects_non_manifest() -> None:
    with pytest.raises(TypeError, match="manifest must be a SeparatorEnvironmentManifest"):
        separators.separator_environment_manifest_payload("not-a-manifest")  # type: ignore[arg-type]


def test_separator_lock_payload_round_trips_fixture() -> None:
    lock = load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID))
    payload = separators.separator_lock_payload(lock)
    assert payload["schema"] == SEPARATOR_LOCK_SCHEMA
    assert payload["separator_id"] == SPLEETER_SEPARATOR_ID
    assert payload["model_files"] == [
        {"name": mf.name, "sha256": mf.sha256} for mf in lock.model_files
    ]


def test_separator_environment_manifest_payload_round_trips_fixture() -> None:
    lock = load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID))
    manifest = load_separator_environment_manifest(_fixture_path(SPLEETER_SEPARATOR_ID), lock)
    payload = separators.separator_environment_manifest_payload(manifest)
    assert payload["schema"] == separators.SEPARATOR_ENVIRONMENT_SCHEMA
    assert payload["separator_id"] == SPLEETER_SEPARATOR_ID
    assert isinstance(payload["distributions"], list)


# ---------------------------------------------------------------------------
# _parse_separator_environment_manifest error paths
# ---------------------------------------------------------------------------


def test_parse_environment_manifest_rejects_non_bytes() -> None:
    with pytest.raises(TypeError, match="content must be bytes"):
        separators._parse_separator_environment_manifest("not-bytes")  # type: ignore[arg-type]


def test_parse_environment_manifest_rejects_missing_trailing_newline() -> None:
    content = _fixture_path(SPLEETER_SEPARATOR_ID).parent.joinpath("environment.json").read_bytes()
    with pytest.raises(SeparatorLockError, match="final newline"):
        separators._parse_separator_environment_manifest(content[:-1])


def test_parse_environment_manifest_rejects_noncanonical_json(tmp_path: Path) -> None:
    path = _fixture_path(SPLEETER_SEPARATOR_ID).parent / "environment.json"
    content = path.read_bytes()
    with pytest.raises(SeparatorLockError, match="invalid"):
        separators._parse_separator_environment_manifest(b" " + content)


def test_parse_environment_manifest_rejects_non_dict(tmp_path: Path) -> None:
    with pytest.raises(SeparatorLockError, match="exact key set"):
        separators._parse_separator_environment_manifest(b"[]\n")


def test_parse_environment_manifest_rejects_wrong_schema() -> None:
    content = _fixture_path(SPLEETER_SEPARATOR_ID).parent / "environment.json"
    payload = json.loads(content.read_text(encoding="utf-8"))
    payload["schema"] = "wrong-schema"
    bad_bytes = canonical_json_bytes(payload, trailing_newline=True)
    with pytest.raises(SeparatorLockError, match="schema is invalid"):
        separators._parse_separator_environment_manifest(bad_bytes)


def test_parse_environment_manifest_rejects_distributions_not_list() -> None:
    content = _fixture_path(SPLEETER_SEPARATOR_ID).parent / "environment.json"
    payload = json.loads(content.read_text(encoding="utf-8"))
    payload["distributions"] = "not-a-list"
    bad_bytes = canonical_json_bytes(payload, trailing_newline=True)
    with pytest.raises(SeparatorLockError, match="distributions must be a list"):
        separators._parse_separator_environment_manifest(bad_bytes)


def test_parse_environment_manifest_rejects_invalid_distribution_field() -> None:
    content = _fixture_path(SPLEETER_SEPARATOR_ID).parent / "environment.json"
    payload = json.loads(content.read_text(encoding="utf-8"))
    payload["distributions"][0]["name"] = 123
    bad_bytes = canonical_json_bytes(payload, trailing_newline=True)
    with pytest.raises(SeparatorLockError, match="name must be a nonempty string"):
        separators._parse_separator_environment_manifest(bad_bytes)


def test_parse_environment_manifest_rejects_fields_invalid_via_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _fixture_path(SPLEETER_SEPARATOR_ID).parent / "environment.json"
    payload = json.loads(content.read_text(encoding="utf-8"))
    bad_bytes = canonical_json_bytes(payload, trailing_newline=True)

    original_dist_from_value = separators._environment_distribution_from_value

    def raise_type_error(value: object) -> object:
        if isinstance(value, dict) and set(value) == separators._ENVIRONMENT_DISTRIBUTION_KEYS:
            raise TypeError("synthetic type error")
        return original_dist_from_value(value)

    monkeypatch.setattr(separators, "_environment_distribution_from_value", raise_type_error)
    with pytest.raises(SeparatorLockError, match="fields are invalid"):
        separators._parse_separator_environment_manifest(bad_bytes)


# ---------------------------------------------------------------------------
# load_separator_environment_manifest error paths
# ---------------------------------------------------------------------------


def test_load_environment_manifest_rejects_non_path_lock_path() -> None:
    lock = load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID))
    with pytest.raises(TypeError, match="lock_path must be a Path"):
        load_separator_environment_manifest("not-a-path", lock)  # type: ignore[arg-type]


def test_load_environment_manifest_rejects_non_lock(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="lock must be a SeparatorLock"):
        load_separator_environment_manifest(_fixture_path(SPLEETER_SEPARATOR_ID), "not-a-lock")  # type: ignore[arg-type]


def test_load_environment_manifest_rejects_unavailable_companion(tmp_path: Path) -> None:
    fixture_dir = _fixture_path(SPLEETER_SEPARATOR_ID).parent
    copied = tmp_path / "fixture"
    shutil.copytree(fixture_dir, copied)
    (copied / "environment.json").unlink()
    lock = load_separator_lock(copied / "model.json")
    with pytest.raises(SeparatorLockError, match="unavailable"):
        load_separator_environment_manifest(copied / "model.json", lock)


def test_load_environment_manifest_rejects_hash_mismatch(tmp_path: Path) -> None:
    fixture_dir = _fixture_path(SPLEETER_SEPARATOR_ID).parent
    copied = tmp_path / "fixture"
    shutil.copytree(fixture_dir, copied)
    lock = load_separator_lock(copied / "model.json")
    env_path = copied / "environment.json"
    env_payload = json.loads(env_path.read_text(encoding="utf-8"))
    env_payload["platform"] = "different-platform"
    env_path.write_bytes(canonical_json_bytes(env_payload, trailing_newline=True))
    with pytest.raises(SeparatorLockError, match="hash does not match"):
        load_separator_environment_manifest(copied / "model.json", lock)


def test_load_environment_manifest_rejects_separator_id_mismatch(tmp_path: Path) -> None:
    fixture_dir = _fixture_path(SPLEETER_SEPARATOR_ID).parent
    copied = tmp_path / "fixture"
    shutil.copytree(fixture_dir, copied)
    lock = load_separator_lock(copied / "model.json")
    env_path = copied / "environment.json"
    env_payload = json.loads(env_path.read_text(encoding="utf-8"))
    env_payload["separator_id"] = "other-separator"
    env_bytes = canonical_json_bytes(env_payload, trailing_newline=True)
    env_path.write_bytes(env_bytes)
    lock_payload = json.loads(copied.joinpath("model.json").read_text(encoding="utf-8"))
    lock_payload["environment_manifest_sha256"] = hashlib.sha256(env_bytes).hexdigest()
    model_path = copied / "model.json"
    model_path.write_bytes(canonical_json_bytes(lock_payload, trailing_newline=True))
    lock = load_separator_lock(model_path)
    with pytest.raises(SeparatorLockError, match="separator_id does not match"):
        load_separator_environment_manifest(model_path, lock)


def test_load_environment_manifest_rejects_package_name_mismatch(tmp_path: Path) -> None:
    fixture_dir = _fixture_path(SPLEETER_SEPARATOR_ID).parent
    copied = tmp_path / "fixture"
    shutil.copytree(fixture_dir, copied)
    lock = load_separator_lock(copied / "model.json")
    env_path = copied / "environment.json"
    env_payload = json.loads(env_path.read_text(encoding="utf-8"))
    env_payload["package_name"] = "other-package"
    env_bytes = canonical_json_bytes(env_payload, trailing_newline=True)
    env_path.write_bytes(env_bytes)
    lock_payload = json.loads(copied.joinpath("model.json").read_text(encoding="utf-8"))
    lock_payload["environment_manifest_sha256"] = hashlib.sha256(env_bytes).hexdigest()
    model_path = copied / "model.json"
    model_path.write_bytes(canonical_json_bytes(lock_payload, trailing_newline=True))
    lock = load_separator_lock(model_path)
    with pytest.raises(SeparatorLockError, match="package_name does not match"):
        load_separator_environment_manifest(model_path, lock)


def test_load_environment_manifest_rejects_package_version_mismatch(tmp_path: Path) -> None:
    fixture_dir = _fixture_path(SPLEETER_SEPARATOR_ID).parent
    copied = tmp_path / "fixture"
    shutil.copytree(fixture_dir, copied)
    lock = load_separator_lock(copied / "model.json")
    env_path = copied / "environment.json"
    env_payload = json.loads(env_path.read_text(encoding="utf-8"))
    env_payload["package_version"] = "9.9.9"
    env_bytes = canonical_json_bytes(env_payload, trailing_newline=True)
    env_path.write_bytes(env_bytes)
    lock_payload = json.loads(copied.joinpath("model.json").read_text(encoding="utf-8"))
    lock_payload["environment_manifest_sha256"] = hashlib.sha256(env_bytes).hexdigest()
    model_path = copied / "model.json"
    model_path.write_bytes(canonical_json_bytes(lock_payload, trailing_newline=True))
    lock = load_separator_lock(model_path)
    with pytest.raises(SeparatorLockError, match="package_version does not match"):
        load_separator_environment_manifest(model_path, lock)


def test_load_environment_manifest_rejects_interpreter_sha256_mismatch(tmp_path: Path) -> None:
    fixture_dir = _fixture_path(SPLEETER_SEPARATOR_ID).parent
    copied = tmp_path / "fixture"
    shutil.copytree(fixture_dir, copied)
    lock = load_separator_lock(copied / "model.json")
    env_path = copied / "environment.json"
    env_payload = json.loads(env_path.read_text(encoding="utf-8"))
    env_payload["interpreter_sha256"] = "f" * 64
    env_bytes = canonical_json_bytes(env_payload, trailing_newline=True)
    env_path.write_bytes(env_bytes)
    lock_payload = json.loads(copied.joinpath("model.json").read_text(encoding="utf-8"))
    lock_payload["environment_manifest_sha256"] = hashlib.sha256(env_bytes).hexdigest()
    model_path = copied / "model.json"
    model_path.write_bytes(canonical_json_bytes(lock_payload, trailing_newline=True))
    lock = load_separator_lock(model_path)
    with pytest.raises(SeparatorLockError, match="interpreter_sha256 does not match"):
        load_separator_environment_manifest(model_path, lock)


# ---------------------------------------------------------------------------
# _resolve_separator_interpreter error paths
# ---------------------------------------------------------------------------


def test_resolve_separator_interpreter_rejects_non_path() -> None:
    with pytest.raises(TypeError, match="interpreter must be a Path"):
        separators._resolve_separator_interpreter("not-a-path")  # type: ignore[arg-type]


def test_resolve_separator_interpreter_rejects_missing_executable(tmp_path: Path) -> None:
    with pytest.raises(SeparatorExecutionError, match="separator_interpreter_mismatch") as raised:
        separators._resolve_separator_interpreter(tmp_path / "nonexistent-python")
    assert raised.value.detail_code == "separator_interpreter_mismatch"


# ---------------------------------------------------------------------------
# _run_separator_environment_probe error paths
# ---------------------------------------------------------------------------


def test_environment_probe_rejects_non_path() -> None:
    with pytest.raises(TypeError, match="interpreter must be a Path"):
        separators._run_separator_environment_probe("not-a-path")  # type: ignore[arg-type]


def test_environment_probe_rejects_unstartable_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_run(*_args: object, **_kwargs: object) -> object:
        raise OSError("cannot start")

    monkeypatch.setattr(separators.subprocess, "run", fail_run)
    with pytest.raises(SeparatorExecutionError, match="probe could not be started") as raised:
        separators._run_separator_environment_probe(tmp_path / "python")
    assert raised.value.detail_code == "separator_environment_probe_failed"


def test_environment_probe_rejects_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"err")

    monkeypatch.setattr(separators.subprocess, "run", failing_run)
    with pytest.raises(SeparatorExecutionError, match="did not complete cleanly") as raised:
        separators._run_separator_environment_probe(tmp_path / "python")
    assert raised.value.detail_code == "separator_environment_probe_failed"


def test_environment_probe_rejects_noncanonical_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_stdout_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=b"not-json\n", stderr=b"")

    monkeypatch.setattr(separators.subprocess, "run", bad_stdout_run)
    with pytest.raises(SeparatorExecutionError, match="not canonical") as raised:
        separators._run_separator_environment_probe(tmp_path / "python")
    assert raised.value.detail_code == "separator_environment_probe_failed"


# ---------------------------------------------------------------------------
# _require_absolute_model_root
# ---------------------------------------------------------------------------


def test_require_absolute_model_root_rejects_non_path() -> None:
    with pytest.raises(TypeError, match="model_root must be a Path"):
        separators._require_absolute_model_root("not-a-path")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# attest_separator_runtime type guards
# ---------------------------------------------------------------------------


def test_attest_rejects_non_path_lock_path(tmp_path: Path) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    with pytest.raises(TypeError, match="lock_path must be a Path"):
        separators.attest_separator_runtime("not-a-path", interpreter, model_root)  # type: ignore[arg-type]


def test_attest_rejects_non_path_interpreter(tmp_path: Path) -> None:
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = _fixture_path(SPLEETER_SEPARATOR_ID)
    with pytest.raises(TypeError, match="interpreter must be a Path"):
        separators.attest_separator_runtime(lock_path, "not-a-path", model_root)  # type: ignore[arg-type]


def test_attest_translates_probe_non_probe_error_to_probe_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"
    _freeze_separator_runtime(
        separator_id=SPLEETER_SEPARATOR_ID,
        interpreter=interpreter,
        model_root=model_root,
        repository_revision="a" * 40,
        output=lock_path,
    )

    def probe_raises_value_error(_interpreter: Path) -> object:
        raise ValueError("unexpected error")

    monkeypatch.setattr(separators, "_run_separator_environment_probe", probe_raises_value_error)
    with pytest.raises(SeparatorExecutionError, match="probe failed") as raised:
        separators.attest_separator_runtime(lock_path, interpreter, model_root)
    assert raised.value.detail_code == "separator_environment_probe_failed"


def test_attest_rejects_non_manifest_probe_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"
    _freeze_separator_runtime(
        separator_id=SPLEETER_SEPARATOR_ID,
        interpreter=interpreter,
        model_root=model_root,
        repository_revision="a" * 40,
        output=lock_path,
    )

    monkeypatch.setattr(
        separators, "_run_separator_environment_probe", lambda _interp: "not-a-manifest"
    )
    with pytest.raises(SeparatorExecutionError, match="probe output is invalid") as raised:
        separators.attest_separator_runtime(lock_path, interpreter, model_root)
    assert raised.value.detail_code == "separator_environment_probe_failed"


def test_attest_translates_model_root_non_model_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"
    _freeze_separator_runtime(
        separator_id=SPLEETER_SEPARATOR_ID,
        interpreter=interpreter,
        model_root=model_root,
        repository_revision="a" * 40,
        output=lock_path,
    )

    def inventory_raises_value_error(*_args: object, **_kwargs: object) -> object:
        raise ValueError("unexpected inventory error")

    monkeypatch.setattr(separators, "_inventory_model_root_bound", inventory_raises_value_error)
    with pytest.raises(SeparatorExecutionError, match="model root does not match") as raised:
        separators.attest_separator_runtime(lock_path, interpreter, model_root)
    assert raised.value.detail_code == "separator_model_root_invalid"


def test_attest_translates_build_runtime_non_attestation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"
    _freeze_separator_runtime(
        separator_id=SPLEETER_SEPARATOR_ID,
        interpreter=interpreter,
        model_root=model_root,
        repository_revision="a" * 40,
        output=lock_path,
    )

    def build_raises_os_error(*_args: object, **_kwargs: object) -> object:
        raise OSError("build failure")

    monkeypatch.setattr(separators, "_build_attested_separator_runtime", build_raises_os_error)
    with pytest.raises(SeparatorExecutionError, match="could not be constructed") as raised:
        separators.attest_separator_runtime(lock_path, interpreter, model_root)
    assert raised.value.detail_code == "separator_model_root_invalid"


# ---------------------------------------------------------------------------
# _model_file_from_value / _validate_lock error paths
# ---------------------------------------------------------------------------


def test_model_file_from_value_rejects_non_dict() -> None:
    with pytest.raises(SeparatorLockError, match="model file must contain the exact key set"):
        separators._model_file_from_value("not-a-dict")


def test_model_file_from_value_rejects_wrong_keys() -> None:
    with pytest.raises(SeparatorLockError, match="model file must contain the exact key set"):
        separators._model_file_from_value({"name": "f.bin", "sha256": "a" * 64, "extra": True})


def test_validate_lock_rejects_empty_string_field(tmp_path: Path) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    payload["repository_url"] = ""
    path = tmp_path / "separator.json"
    _write_payload(path, payload)
    with pytest.raises(SeparatorLockError, match="repository_url must be a nonempty string"):
        load_separator_lock(path)


def test_validate_lock_rejects_repository_url_mismatch(tmp_path: Path) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    payload["repository_url"] = "https://wrong.example/repo"
    path = tmp_path / "separator.json"
    _write_payload(path, payload)
    with pytest.raises(SeparatorLockError, match="repository_url does not match"):
        load_separator_lock(path)


def test_validate_lock_rejects_package_name_mismatch(tmp_path: Path) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    payload["package_name"] = "wrong-package"
    path = tmp_path / "separator.json"
    _write_payload(path, payload)
    with pytest.raises(SeparatorLockError, match="package_name does not match"):
        load_separator_lock(path)


def test_validate_lock_rejects_model_id_mismatch(tmp_path: Path) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    payload["model_id"] = "wrong-model"
    path = tmp_path / "separator.json"
    _write_payload(path, payload)
    with pytest.raises(SeparatorLockError, match="model_id does not match"):
        load_separator_lock(path)


def test_validate_lock_rejects_model_root_kind_mismatch(tmp_path: Path) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    payload["model_root_kind"] = "wrong-kind"
    path = tmp_path / "separator.json"
    _write_payload(path, payload)
    with pytest.raises(SeparatorLockError, match="model_root_kind does not match"):
        load_separator_lock(path)


def test_validate_lock_rejects_expected_stem_path_mismatch(tmp_path: Path) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    payload["expected_drum_stem_relative_path"] = "wrong/path.wav"
    path = tmp_path / "separator.json"
    _write_payload(path, payload)
    with pytest.raises(SeparatorLockError, match="expected drum stem path"):
        load_separator_lock(path)


def test_validate_lock_rejects_output_container_mismatch(tmp_path: Path) -> None:
    payload = _fixture_payload(SPLEETER_SEPARATOR_ID)
    payload["output_container"] = "flac"
    path = tmp_path / "separator.json"
    _write_payload(path, payload)
    with pytest.raises(SeparatorLockError, match="output_container does not match"):
        load_separator_lock(path)


# ---------------------------------------------------------------------------
# Environment validation error paths
# ---------------------------------------------------------------------------


def test_environment_file_from_value_rejects_wrong_keys() -> None:
    with pytest.raises(SeparatorLockError, match="environment file must contain the exact key set"):
        separators._environment_file_from_value({"root": "purelib", "path": "f.py"})


def test_environment_distribution_from_value_rejects_wrong_keys() -> None:
    with pytest.raises(
        SeparatorLockError, match="environment distribution must contain the exact key set"
    ):
        separators._environment_distribution_from_value({"name": "pkg"})


def test_environment_distribution_from_value_rejects_files_not_list() -> None:
    with pytest.raises(SeparatorLockError, match="distribution files must be a list"):
        separators._environment_distribution_from_value(
            {"name": "pkg", "version": "1.0", "files": "not-a-list"}
        )


def test_validate_environment_file_rejects_empty_root() -> None:
    with pytest.raises(SeparatorLockError, match="environment file root must be a nonempty string"):
        separators.SeparatorEnvironmentFile(
            root="",
            path="f.py",
            byte_length=10,
            sha256="a" * 64,
        )


def test_validate_environment_file_rejects_unsupported_root() -> None:
    with pytest.raises(SeparatorLockError, match="environment file root is unsupported"):
        separators.SeparatorEnvironmentFile(
            root="badroot",
            path="f.py",
            byte_length=10,
            sha256="a" * 64,
        )


def test_validate_environment_file_rejects_negative_byte_length() -> None:
    with pytest.raises(SeparatorLockError, match="byte_length must be a nonnegative integer"):
        separators.SeparatorEnvironmentFile(
            root="purelib",
            path="f.py",
            byte_length=-1,
            sha256="a" * 64,
        )


def test_validate_environment_distribution_rejects_empty_name() -> None:
    with pytest.raises(
        SeparatorLockError, match="environment distribution name must be a nonempty string"
    ):
        separators.SeparatorEnvironmentDistribution(name="", version="1.0", files=())


def test_validate_environment_distribution_rejects_unnormalized_name() -> None:
    file = separators.SeparatorEnvironmentFile(
        root="purelib", path="f.py", byte_length=1, sha256="a" * 64
    )
    with pytest.raises(SeparatorLockError, match="distribution name must be normalized"):
        separators.SeparatorEnvironmentDistribution(name="Pkg_Name", version="1.0", files=(file,))


def test_validate_environment_distribution_rejects_empty_files() -> None:
    with pytest.raises(
        SeparatorLockError, match="environment distribution files must be a nonempty tuple"
    ):
        separators.SeparatorEnvironmentDistribution(name="pkg", version="1.0", files=())


def test_validate_environment_distribution_rejects_unsorted_files() -> None:
    file_a = separators.SeparatorEnvironmentFile(
        root="purelib", path="b.py", byte_length=1, sha256="a" * 64
    )
    file_b = separators.SeparatorEnvironmentFile(
        root="purelib", path="a.py", byte_length=1, sha256="b" * 64
    )
    with pytest.raises(SeparatorLockError, match="environment file tuples must be sorted"):
        separators.SeparatorEnvironmentDistribution(
            name="pkg", version="1.0", files=(file_a, file_b)
        )


def test_validate_environment_manifest_rejects_empty_field() -> None:
    file = separators.SeparatorEnvironmentFile(
        root="purelib", path="f.py", byte_length=1, sha256="a" * 64
    )
    dist = separators.SeparatorEnvironmentDistribution(name="pkg", version="1.0", files=(file,))
    with pytest.raises(
        SeparatorLockError, match="environment manifest separator_id must be a nonempty string"
    ):
        separators.SeparatorEnvironmentManifest(
            separator_id="",
            package_name="pkg",
            package_version="1.0",
            python_implementation="CPython",
            python_version="3.12.0",
            python_abi="cp312",
            platform="linux",
            interpreter_sha256="a" * 64,
            distributions=(dist,),
            sha256="b" * 64,
        )


def test_validate_environment_manifest_rejects_empty_distributions() -> None:
    with pytest.raises(
        SeparatorLockError, match="environment distributions must be a nonempty tuple"
    ):
        separators.SeparatorEnvironmentManifest(
            separator_id="spleeter4-drums-v1",
            package_name="pkg",
            package_version="1.0",
            python_implementation="CPython",
            python_version="3.12.0",
            python_abi="cp312",
            platform="linux",
            interpreter_sha256="a" * 64,
            distributions=(),
            sha256="b" * 64,
        )


def test_validate_environment_manifest_rejects_unsorted_distributions() -> None:
    file = separators.SeparatorEnvironmentFile(
        root="purelib", path="f.py", byte_length=1, sha256="a" * 64
    )
    dist_b = separators.SeparatorEnvironmentDistribution(name="zpkg", version="1.0", files=(file,))
    dist_a = separators.SeparatorEnvironmentDistribution(name="apkg", version="1.0", files=(file,))
    with pytest.raises(SeparatorLockError, match="distributions must be sorted and unique"):
        separators.SeparatorEnvironmentManifest(
            separator_id="spleeter4-drums-v1",
            package_name="pkg",
            package_version="1.0",
            python_implementation="CPython",
            python_version="3.12.0",
            python_abi="cp312",
            platform="linux",
            interpreter_sha256="a" * 64,
            distributions=(dist_b, dist_a),
            sha256="b" * 64,
        )


def test_normalize_distribution_name_rejects_invalid_name() -> None:
    with pytest.raises(SeparatorLockError, match="distribution name is invalid"):
        separators._normalize_distribution_name("!!!")


def test_validate_environment_relative_path_rejects_null_byte() -> None:
    with pytest.raises(SeparatorLockError, match="environment file path is invalid"):
        separators._validate_environment_relative_path("f\x00.py")


def test_validate_environment_relative_path_rejects_absolute() -> None:
    with pytest.raises(SeparatorLockError, match="environment file path must be relative"):
        separators._validate_environment_relative_path("/absolute/path.py")


def test_validate_environment_relative_path_rejects_dot_component() -> None:
    with pytest.raises(SeparatorLockError, match="environment file path must be normalized"):
        separators._validate_environment_relative_path("./f.py")


def test_validate_model_file_name_rejects_null_byte() -> None:
    with pytest.raises(SeparatorLockError, match="model file name is invalid"):
        separators._validate_model_file_name("f\x00.bin")


def test_validate_model_file_name_rejects_dot_component() -> None:
    with pytest.raises(SeparatorLockError, match="model file name must be normalized"):
        separators._validate_model_file_name("./f.bin")


def test_require_hash_rejects_non_string() -> None:
    with pytest.raises(SeparatorLockError, match="must be a lowercase SHA-256 hash"):
        separators._require_hash(123, "test_field")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# inventory_separator_model_root / freeze_separator_runtime error paths
# ---------------------------------------------------------------------------


def test_inventory_separator_model_root_rejects_unsupported_separator(tmp_path: Path) -> None:
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    with pytest.raises(SeparatorExecutionError, match="separator_id_unsupported") as raised:
        separators.inventory_separator_model_root("unknown-separator", model_root)
    assert raised.value.detail_code == "separator_id_unsupported"


def test_inventory_separator_model_root_rejects_non_absolute_model_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(SeparatorExecutionError, match="separator_model_root_invalid") as raised:
        separators.inventory_separator_model_root(SPLEETER_SEPARATOR_ID, Path("relative/path"))
    assert raised.value.detail_code == "separator_model_root_invalid"


def test_freeze_rejects_non_path_interpreter(tmp_path: Path) -> None:
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    with pytest.raises(TypeError, match="interpreter must be a Path"):
        separators.freeze_separator_runtime(
            separator_id=SPLEETER_SEPARATOR_ID,
            interpreter="not-a-path",  # type: ignore[arg-type]
            model_root=model_root,
            repository_revision="a" * 40,
            output=tmp_path / "model.json",
        )


def test_freeze_rejects_non_path_model_root(tmp_path: Path) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    with pytest.raises(TypeError, match="model_root must be a Path"):
        separators.freeze_separator_runtime(
            separator_id=SPLEETER_SEPARATOR_ID,
            interpreter=interpreter,
            model_root="not-a-path",  # type: ignore[arg-type]
            repository_revision="a" * 40,
            output=tmp_path / "model.json",
        )


def test_freeze_rejects_non_path_output(tmp_path: Path) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    with pytest.raises(TypeError, match="output must be a Path"):
        separators.freeze_separator_runtime(
            separator_id=SPLEETER_SEPARATOR_ID,
            interpreter=interpreter,
            model_root=model_root,
            repository_revision="a" * 40,
            output="not-a-path",  # type: ignore[arg-type]
        )


def test_freeze_rejects_unsupported_separator_id(tmp_path: Path) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    with pytest.raises(ValueError, match="separator_id is unsupported"):
        separators.freeze_separator_runtime(
            separator_id="unknown-separator",
            interpreter=interpreter,
            model_root=model_root,
            repository_revision="a" * 40,
            output=tmp_path / "model.json",
        )


def test_freeze_cleans_up_environment_on_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"
    environment_path = lock_path.parent / "environment.json"

    def fail_publish(path: Path, content: bytes) -> object:
        from src.benchmark.artifact_io import ArtifactPublicationError

        if path == lock_path:
            raise ArtifactPublicationError("conflict")
        from src.benchmark.artifact_io import publish_immutable_file

        return publish_immutable_file(path, content)

    monkeypatch.setattr(separators, "publish_immutable_file", fail_publish)
    with pytest.raises(SeparatorExecutionError, match="separator_lock_publication_failed"):
        separators.freeze_separator_runtime(
            separator_id=SPLEETER_SEPARATOR_ID,
            interpreter=interpreter,
            model_root=model_root,
            repository_revision="a" * 40,
            output=lock_path,
        )
    assert not environment_path.exists()
    assert not lock_path.exists()


# ---------------------------------------------------------------------------
# revalidate_separator_model_root error paths
# ---------------------------------------------------------------------------


def test_revalidate_rejects_closed_descriptor(tmp_path: Path) -> None:
    model_root, expected = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    runtime = _attested_runtime(
        SPLEETER_SEPARATOR_ID,
        model_root=model_root,
        model_files=expected,
    )
    runtime.close()  # type: ignore[attr-defined]
    with pytest.raises(SeparatorExecutionError, match="separator_runtime_unattested") as raised:
        separators.revalidate_separator_model_root(runtime)
    assert raised.value.detail_code == "separator_runtime_unattested"


def test_revalidate_translates_inventory_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root, expected = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    runtime = _attested_runtime(
        SPLEETER_SEPARATOR_ID,
        model_root=model_root,
        model_files=expected,
    )

    def inventory_raises_os_error(*_args: object, **_kwargs: object) -> object:
        raise OSError("inventory failure")

    monkeypatch.setattr(separators, "_inventory_model_root_bound", inventory_raises_os_error)
    with pytest.raises(SeparatorExecutionError, match="separator_model_root_invalid") as raised:
        separators.revalidate_separator_model_root(runtime)
    assert raised.value.detail_code == "separator_model_root_invalid"
    runtime.close()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _build_separator_launch_environment error paths
# ---------------------------------------------------------------------------


def test_build_launch_environment_rejects_unsupported_separator() -> None:
    with pytest.raises(SeparatorExecutionError, match="separator_id_unsupported") as raised:
        separators._build_separator_launch_environment("unknown", Path("/model"))
    assert raised.value.detail_code == "separator_id_unsupported"


def test_build_launch_environment_rejects_non_path_model_root() -> None:
    with pytest.raises(TypeError, match="model_root must be a Path"):
        separators._build_separator_launch_environment(SPLEETER_SEPARATOR_ID, "not-a-path")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _build_attested_separator_runtime error paths
# ---------------------------------------------------------------------------


def test_build_runtime_rejects_non_path_interpreter(tmp_path: Path) -> None:
    lock = load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID))
    env = load_separator_environment_manifest(_fixture_path(SPLEETER_SEPARATOR_ID), lock)
    model_root, model_files = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    with pytest.raises(TypeError, match="interpreter must be a Path"):
        separators._build_attested_separator_runtime(
            interpreter="not-a-path",  # type: ignore[arg-type]
            lock=lock,
            model_root=model_root,
            model_files=model_files,
            environment=env,
        )


def test_build_runtime_rejects_non_lock(tmp_path: Path) -> None:
    env = load_separator_environment_manifest(
        _fixture_path(SPLEETER_SEPARATOR_ID),
        load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID)),
    )
    model_root, model_files = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    with pytest.raises(TypeError, match="lock must be a SeparatorLock"):
        separators._build_attested_separator_runtime(
            interpreter=Path("/python"),
            lock="not-a-lock",  # type: ignore[arg-type]
            model_root=model_root,
            model_files=model_files,
            environment=env,
        )


def test_build_runtime_rejects_non_path_model_root(tmp_path: Path) -> None:
    lock = load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID))
    env = load_separator_environment_manifest(_fixture_path(SPLEETER_SEPARATOR_ID), lock)
    _, model_files = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    with pytest.raises(TypeError, match="model_root must be a Path"):
        separators._build_attested_separator_runtime(
            interpreter=Path("/python"),
            lock=lock,
            model_root="not-a-path",  # type: ignore[arg-type]
            model_files=model_files,
            environment=env,
        )


def test_build_runtime_rejects_non_tuple_model_files(tmp_path: Path) -> None:
    lock = load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID))
    env = load_separator_environment_manifest(_fixture_path(SPLEETER_SEPARATOR_ID), lock)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    with pytest.raises(TypeError, match="model_files must be a tuple of SeparatorModelFile"):
        separators._build_attested_separator_runtime(
            interpreter=Path("/python"),
            lock=lock,
            model_root=model_root,
            model_files="not-a-tuple",  # type: ignore[arg-type]
            environment=env,
        )


def test_build_runtime_rejects_non_manifest_environment(tmp_path: Path) -> None:
    lock = load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID))
    model_root, model_files = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    with pytest.raises(TypeError, match="environment must be a SeparatorEnvironmentManifest"):
        separators._build_attested_separator_runtime(
            interpreter=Path("/python"),
            lock=lock,
            model_root=model_root,
            model_files=model_files,
            environment="not-a-manifest",  # type: ignore[arg-type]
        )


def test_build_runtime_rejects_mismatched_model_files(tmp_path: Path) -> None:
    lock = load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID))
    env = load_separator_environment_manifest(_fixture_path(SPLEETER_SEPARATOR_ID), lock)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    wrong_files = (SeparatorModelFile(name="4stems/.probe", sha256="0" * 64),)
    with pytest.raises(SeparatorExecutionError, match="separator_runtime_unattested") as raised:
        separators._build_attested_separator_runtime(
            interpreter=Path("/python"),
            lock=lock,
            model_root=model_root,
            model_files=wrong_files,
            environment=env,
        )
    assert raised.value.detail_code == "separator_runtime_unattested"


# ---------------------------------------------------------------------------
# _require_attested_runtime error paths
# ---------------------------------------------------------------------------


def test_require_attested_runtime_rejects_non_runtime() -> None:
    with pytest.raises(TypeError, match="runtime must be an AttestedSeparatorRuntime"):
        separators._require_attested_runtime("not-a-runtime")  # type: ignore[arg-type]


def test_require_attested_runtime_rejects_unregistered_runtime(tmp_path: Path) -> None:
    lock = load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID))
    env = load_separator_environment_manifest(_fixture_path(SPLEETER_SEPARATOR_ID), lock)
    model_root, model_files = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    runtime = separators.AttestedSeparatorRuntime(
        interpreter=Path("/python"),
        lock=lock,
        model_root=model_root,
        model_files=model_files,
        environment=env,
        launch_environment={"PATH": "/bin"},
    )
    with pytest.raises(SeparatorExecutionError, match="separator_runtime_unattested") as raised:
        separators._require_attested_runtime(runtime)
    assert raised.value.detail_code == "separator_runtime_unattested"


# ---------------------------------------------------------------------------
# _run_separator_drums execution error paths
# ---------------------------------------------------------------------------


def test_run_separator_rejects_lock_separator_id_mismatch(tmp_path: Path) -> None:
    source_path = _source_wav(tmp_path)
    model_root, expected = _synthetic_model_root(tmp_path, HTDEMUCS_SEPARATOR_ID)
    runtime = _attested_runtime(
        HTDEMUCS_SEPARATOR_ID,
        model_root=model_root,
        model_files=expected,
    )
    with pytest.raises(SeparatorExecutionError, match="separator_lock_mismatch") as raised:
        separators.run_spleeter_drums(
            source_path,
            source_audio_sha256="a" * 64,
            source_duration_sec=1.0,
            runtime=runtime,
            cache_root=tmp_path / "cache",
        )
    assert raised.value.detail_code == "separator_lock_mismatch"
    runtime.close()  # type: ignore[attr-defined]


def test_run_separator_rejects_unavailable_source_audio(tmp_path: Path) -> None:
    model_root, expected = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    runtime = _attested_runtime(
        SPLEETER_SEPARATOR_ID,
        model_root=model_root,
        model_files=expected,
    )
    with pytest.raises(SeparatorExecutionError, match="source_audio_unavailable") as raised:
        separators.run_spleeter_drums(
            tmp_path / "nonexistent.wav",
            source_audio_sha256="a" * 64,
            source_duration_sec=1.0,
            runtime=runtime,
            cache_root=tmp_path / "cache",
        )
    assert raised.value.detail_code == "source_audio_unavailable"
    runtime.close()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _validate_execution_inputs error paths
# ---------------------------------------------------------------------------


def test_validate_execution_inputs_rejects_unsupported_separator(tmp_path: Path) -> None:
    source_path = _source_wav(tmp_path)
    model_root, expected = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    runtime = _attested_runtime(
        SPLEETER_SEPARATOR_ID,
        model_root=model_root,
        model_files=expected,
    )
    with pytest.raises(SeparatorExecutionError, match="separator_id_unsupported") as raised:
        separators._validate_execution_inputs(
            "unknown-separator",
            source_path,
            "a" * 64,
            1.0,
            runtime,
            tmp_path / "cache",
        )
    assert raised.value.detail_code == "separator_id_unsupported"
    runtime.close()  # type: ignore[attr-defined]


def test_validate_execution_inputs_rejects_non_path_source(tmp_path: Path) -> None:
    model_root, expected = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    runtime = _attested_runtime(
        SPLEETER_SEPARATOR_ID,
        model_root=model_root,
        model_files=expected,
    )
    with pytest.raises(TypeError, match="source_audio_path must be a Path"):
        separators._validate_execution_inputs(
            SPLEETER_SEPARATOR_ID,
            "not-a-path",  # type: ignore[arg-type]
            "a" * 64,
            1.0,
            runtime,
            tmp_path / "cache",
        )
    runtime.close()  # type: ignore[attr-defined]


def test_validate_execution_inputs_rejects_non_path_cache_root(tmp_path: Path) -> None:
    source_path = _source_wav(tmp_path)
    model_root, expected = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    runtime = _attested_runtime(
        SPLEETER_SEPARATOR_ID,
        model_root=model_root,
        model_files=expected,
    )
    with pytest.raises(TypeError, match="cache_root must be a Path"):
        separators._validate_execution_inputs(
            SPLEETER_SEPARATOR_ID,
            source_path,
            "a" * 64,
            1.0,
            runtime,
            "not-a-path",  # type: ignore[arg-type]
        )
    runtime.close()  # type: ignore[attr-defined]


def test_validate_execution_inputs_rejects_invalid_sha256(tmp_path: Path) -> None:
    source_path = _source_wav(tmp_path)
    model_root, expected = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    runtime = _attested_runtime(
        SPLEETER_SEPARATOR_ID,
        model_root=model_root,
        model_files=expected,
    )
    with pytest.raises(SeparatorExecutionError, match="source_audio_identity_invalid") as raised:
        separators._validate_execution_inputs(
            SPLEETER_SEPARATOR_ID,
            source_path,
            "not-a-hash",
            1.0,
            runtime,
            tmp_path / "cache",
        )
    assert raised.value.detail_code == "source_audio_identity_invalid"
    runtime.close()  # type: ignore[attr-defined]


def test_validate_execution_inputs_rejects_invalid_duration(tmp_path: Path) -> None:
    source_path = _source_wav(tmp_path)
    model_root, expected = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    runtime = _attested_runtime(
        SPLEETER_SEPARATOR_ID,
        model_root=model_root,
        model_files=expected,
    )
    with pytest.raises(SeparatorExecutionError, match="source_duration_invalid") as raised:
        separators._validate_execution_inputs(
            SPLEETER_SEPARATOR_ID,
            source_path,
            "a" * 64,
            -1.0,
            runtime,
            tmp_path / "cache",
        )
    assert raised.value.detail_code == "source_duration_invalid"
    runtime.close()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _read_cached_stem error path
# ---------------------------------------------------------------------------


def test_read_cached_stem_rejects_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "cache.wav"
    path.symlink_to(tmp_path / "nonexistent-target")
    with pytest.raises(SeparatorExecutionError, match="stem_cache_unavailable") as raised:
        separators._read_cached_stem(path)
    assert raised.value.detail_code == "stem_cache_unavailable"


# ---------------------------------------------------------------------------
# _run_separator_process start failure
# ---------------------------------------------------------------------------


def test_run_separator_process_rejects_start_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def popen_os_error(*_args: object, **_kwargs: object) -> object:
        raise OSError("cannot start")

    monkeypatch.setattr(subprocess, "Popen", popen_os_error)
    with pytest.raises(SeparatorExecutionError, match="separator_start_failed") as raised:
        separators._run_separator_process(
            ["python"],
            cwd=tmp_path,
            env={"PATH": "/bin"},
        )
    assert raised.value.detail_code == "separator_start_failed"


# ---------------------------------------------------------------------------
# _stop_separator_process_group error handling
# ---------------------------------------------------------------------------


def test_stop_separator_process_group_swallows_killpg_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakePopen()
    monkeypatch.setattr(
        os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(OSError("no such process")),
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: fake,
    )
    separators._stop_separator_process_group(fake)


def test_stop_separator_process_group_waits_after_grace_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakePopen(terminate_wait_timeout=True)
    monkeypatch.setattr(os, "killpg", lambda *_args: None)
    separators._stop_separator_process_group(fake)
    assert fake.wait_calls == [5.0, None]


# ---------------------------------------------------------------------------
# _qc_stem_bytes direct error paths
# ---------------------------------------------------------------------------


def test_qc_stem_bytes_rejects_non_wav_bytes() -> None:
    with pytest.raises(SeparatorExecutionError, match="stem_decode_failed") as raised:
        separators._qc_stem_bytes(b"not a wav file", 1.0)
    assert raised.value.detail_code == "stem_decode_failed"


def test_qc_stem_bytes_rejects_zero_frame_count(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros((0, 1), dtype=np.float32), 44100, format="WAV")
    with pytest.raises(SeparatorExecutionError, match="stem_duration_invalid") as raised:
        separators._qc_stem_bytes(path.read_bytes(), 1.0)
    assert raised.value.detail_code == "stem_duration_invalid"


def test_qc_stem_bytes_rejects_too_many_channels(tmp_path: Path) -> None:
    path = tmp_path / "multi.wav"
    sf.write(path, np.zeros((44100, 3), dtype=np.float32), 44100, format="WAV")
    with pytest.raises(SeparatorExecutionError, match="stem_channel_count") as raised:
        separators._qc_stem_bytes(path.read_bytes(), 1.0)
    assert raised.value.detail_code == "stem_channel_count"


def test_qc_stem_bytes_rejects_nonfinite_samples(tmp_path: Path) -> None:
    path = tmp_path / "nan.wav"
    sf.write(
        path,
        np.full((44100, 1), np.nan, dtype=np.float32),
        44100,
        format="WAV",
        subtype="FLOAT",
    )
    with pytest.raises(SeparatorExecutionError, match="stem_nonfinite") as raised:
        separators._qc_stem_bytes(path.read_bytes(), 1.0)
    assert raised.value.detail_code == "stem_nonfinite"


def test_qc_stem_bytes_rejects_near_silent(tmp_path: Path) -> None:
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros((44100, 1), dtype=np.float32), 44100, format="WAV")
    with pytest.raises(SeparatorExecutionError, match="stem_near_silent") as raised:
        separators._qc_stem_bytes(path.read_bytes(), 1.0)
    assert raised.value.detail_code == "stem_near_silent"


def test_qc_stem_bytes_rejects_duration_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "long.wav"
    sf.write(path, np.full((88200, 1), 0.25, dtype=np.float32), 44100, format="WAV")
    with pytest.raises(SeparatorExecutionError, match="stem_duration_mismatch") as raised:
        separators._qc_stem_bytes(path.read_bytes(), 1.0)
    assert raised.value.detail_code == "stem_duration_mismatch"


def test_qc_stem_bytes_records_clipping_warning(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    sf.write(path, np.full((44100, 1), 1.0, dtype=np.float32), 44100, format="WAV")
    qc = separators._qc_stem_bytes(path.read_bytes(), 1.0)
    assert qc.clipping_detected is True
    assert "stem_clipping" in qc.warnings


def test_qc_stem_bytes_accepts_stereo_stem(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    sf.write(path, np.full((44100, 2), 0.25, dtype=np.float32), 44100, format="WAV")
    qc = separators._qc_stem_bytes(path.read_bytes(), 1.0)
    assert qc.channel_count == 2
    assert qc.rms_dbfs > separators.STEM_NEAR_SILENT_DBFS


# ---------------------------------------------------------------------------
# _model_root_launch_path platform paths
# ---------------------------------------------------------------------------


def test_model_root_launch_path_returns_dot_on_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(separators.os, "name", "posix")
    monkeypatch.setattr(separators.sys, "platform", "darwin")
    result = separators._model_root_launch_path(42)
    assert result == Path(".")


def test_model_root_launch_path_rejects_non_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(separators.os, "name", "nt")
    with pytest.raises(OSError, match="descriptor-backed model-root paths are unavailable"):
        separators._model_root_launch_path(42)


# ---------------------------------------------------------------------------
# _open_model_root_path error paths
# ---------------------------------------------------------------------------


def test_open_model_root_path_rejects_non_absolute(tmp_path: Path) -> None:
    no_follow, directory_flag, close_on_exec = separators._model_root_descriptor_flags()
    with pytest.raises(OSError, match="model root must be absolute"):
        separators._open_model_root_path(
            Path("relative/path"),
            directory_flag=directory_flag,
            no_follow=no_follow,
            close_on_exec=close_on_exec,
        )


def test_open_model_root_path_rejects_unnormalized_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(separators.os, "name", "posix")
    no_follow, directory_flag, close_on_exec = separators._model_root_descriptor_flags()
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    bad_path = subdir / ".." / "subdir"
    with pytest.raises(OSError, match="model root path is not normalized"):
        separators._open_model_root_path(
            bad_path,
            directory_flag=directory_flag,
            no_follow=no_follow,
            close_on_exec=close_on_exec,
        )


# ---------------------------------------------------------------------------
# _inventory_model_root_bound special-file rejection
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX FIFO support")
def test_inventory_model_root_rejects_special_file(tmp_path: Path) -> None:
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    (model_root / "4stems" / "model.meta").unlink()
    os.mkfifo(model_root / "4stems" / "model.meta")
    with pytest.raises(SeparatorExecutionError, match="separator_model_root_invalid") as raised:
        separators.inventory_separator_model_root(SPLEETER_SEPARATOR_ID, model_root)
    assert raised.value.detail_code == "separator_model_root_invalid"


# ---------------------------------------------------------------------------
# _hash_model_file_descriptor error paths
# ---------------------------------------------------------------------------


def test_hash_model_file_descriptor_rejects_non_regular_file(tmp_path: Path) -> None:
    dir_path = tmp_path / "subdir"
    dir_path.mkdir()
    fd = os.open(dir_path, os.O_RDONLY)
    try:
        metadata = os.fstat(fd)
        with pytest.raises(OSError, match="model file is not an ordinary file"):
            separators._hash_model_file_descriptor(fd, metadata)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Additional coverage for remaining uncovered lines
# ---------------------------------------------------------------------------


def test_load_environment_manifest_wraps_invalid_companion_error(
    tmp_path: Path,
) -> None:
    fixture_dir = _fixture_path(SPLEETER_SEPARATOR_ID).parent
    copied = tmp_path / "fixture"
    shutil.copytree(fixture_dir, copied)
    lock = load_separator_lock(copied / "model.json")
    env_path = copied / "environment.json"
    env_payload = json.loads(env_path.read_text(encoding="utf-8"))
    env_payload["schema"] = "wrong-schema"
    env_bytes = canonical_json_bytes(env_payload, trailing_newline=True)
    env_path.write_bytes(env_bytes)
    lock_payload = json.loads(copied.joinpath("model.json").read_text(encoding="utf-8"))
    lock_payload["environment_manifest_sha256"] = hashlib.sha256(env_bytes).hexdigest()
    model_path = copied / "model.json"
    model_path.write_bytes(canonical_json_bytes(lock_payload, trailing_newline=True))
    lock = load_separator_lock(model_path)
    with pytest.raises(SeparatorLockError, match="companion environment is invalid"):
        load_separator_environment_manifest(model_path, lock)


def test_attest_translates_probe_non_probe_code_to_probe_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"
    _freeze_separator_runtime(
        separator_id=SPLEETER_SEPARATOR_ID,
        interpreter=interpreter,
        model_root=model_root,
        repository_revision="a" * 40,
        output=lock_path,
    )

    def probe_wrong_code(_interpreter: Path) -> object:
        raise SeparatorExecutionError("some_other_code", "detail")

    monkeypatch.setattr(separators, "_run_separator_environment_probe", probe_wrong_code)
    with pytest.raises(SeparatorExecutionError, match="probe failed") as raised:
        separators.attest_separator_runtime(lock_path, interpreter, model_root)
    assert raised.value.detail_code == "separator_environment_probe_failed"


def test_attest_translates_model_root_non_model_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"
    _freeze_separator_runtime(
        separator_id=SPLEETER_SEPARATOR_ID,
        interpreter=interpreter,
        model_root=model_root,
        repository_revision="a" * 40,
        output=lock_path,
    )

    def inventory_wrong_code(*_args: object, **_kwargs: object) -> object:
        raise SeparatorExecutionError("some_other_code", "detail")

    monkeypatch.setattr(separators, "_inventory_model_root_bound", inventory_wrong_code)
    with pytest.raises(SeparatorExecutionError, match="model root does not match") as raised:
        separators.attest_separator_runtime(lock_path, interpreter, model_root)
    assert raised.value.detail_code == "separator_model_root_invalid"


def test_attest_translates_build_runtime_non_attestation_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"
    _freeze_separator_runtime(
        separator_id=SPLEETER_SEPARATOR_ID,
        interpreter=interpreter,
        model_root=model_root,
        repository_revision="a" * 40,
        output=lock_path,
    )

    def build_wrong_code(*_args: object, **_kwargs: object) -> object:
        raise SeparatorExecutionError("some_other_code", "detail")

    monkeypatch.setattr(separators, "_build_attested_separator_runtime", build_wrong_code)
    with pytest.raises(SeparatorExecutionError, match="could not be constructed") as raised:
        separators.attest_separator_runtime(lock_path, interpreter, model_root)
    assert raised.value.detail_code == "separator_model_root_invalid"


def test_validate_lock_rejects_empty_model_files() -> None:
    lock = load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID))
    with pytest.raises(SeparatorLockError, match="model_files must be a nonempty tuple"):
        replace(lock, model_files=())


def test_validate_lock_rejects_invalid_model_file_type() -> None:
    lock = load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID))
    with pytest.raises(SeparatorLockError, match="model_files contains an invalid model file"):
        replace(lock, model_files=("not-a-model-file",))  # type: ignore[arg-type]


def test_validate_environment_distribution_rejects_invalid_file_type() -> None:
    with pytest.raises(
        SeparatorLockError, match="environment distribution contains an invalid file"
    ):
        separators.SeparatorEnvironmentDistribution(
            name="pkg",
            version="1.0",
            files=("not-a-file",),  # type: ignore[arg-type]
        )


def test_validate_environment_manifest_rejects_invalid_distribution_type() -> None:
    with pytest.raises(
        SeparatorLockError, match="environment manifest contains an invalid distribution"
    ):
        separators.SeparatorEnvironmentManifest(
            separator_id="spleeter4-drums-v1",
            package_name="pkg",
            package_version="1.0",
            python_implementation="CPython",
            python_version="3.12.0",
            python_abi="cp312",
            platform="linux",
            interpreter_sha256="a" * 64,
            distributions=("not-a-distribution",),  # type: ignore[arg-type]
            sha256="b" * 64,
        )


def test_freeze_cleans_up_environment_on_os_error_during_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"

    def fail_publish(path: Path, content: bytes) -> object:
        from src.benchmark.artifact_io import ArtifactPublicationError

        if path == lock_path:
            raise ArtifactPublicationError("conflict")
        from src.benchmark.artifact_io import publish_immutable_file

        return publish_immutable_file(path, content)

    monkeypatch.setattr(separators, "publish_immutable_file", fail_publish)
    original_unlink = Path.unlink

    def fail_unlink(self: Path, *args: object, **kwargs: object) -> object:
        if self.name == "environment.json":
            raise OSError("cannot unlink")
        return original_unlink(self, *args, **kwargs)  # type: ignore[call-arg]

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(SeparatorExecutionError, match="separator_lock_publication_failed"):
        separators.freeze_separator_runtime(
            separator_id=SPLEETER_SEPARATOR_ID,
            interpreter=interpreter,
            model_root=model_root,
            repository_revision="a" * 40,
            output=lock_path,
        )


def test_freeze_detects_round_trip_separator_id_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"

    original_attest = separators.attest_separator_runtime

    def attest_wrong_id(
        lock_path: Path,
        interpreter: Path,
        model_root: Path,
    ) -> object:
        runtime = original_attest(lock_path, interpreter, model_root)
        object.__setattr__(runtime.lock, "separator_id", "wrong-id")
        return runtime

    monkeypatch.setattr(separators, "attest_separator_runtime", attest_wrong_id)
    with pytest.raises(SeparatorExecutionError, match="separator_lock_companion_mismatch"):
        separators.freeze_separator_runtime(
            separator_id=SPLEETER_SEPARATOR_ID,
            interpreter=interpreter,
            model_root=model_root,
            repository_revision="a" * 40,
            output=lock_path,
        )


def test_revalidate_reraises_separator_execution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root, expected = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    runtime = _attested_runtime(
        SPLEETER_SEPARATOR_ID,
        model_root=model_root,
        model_files=expected,
    )

    def inventory_raises_exec_error(*_args: object, **_kwargs: object) -> object:
        raise SeparatorExecutionError("separator_model_root_invalid", "detail")

    monkeypatch.setattr(separators, "_inventory_model_root_bound", inventory_raises_exec_error)
    with pytest.raises(SeparatorExecutionError, match="separator_model_root_invalid") as raised:
        separators.revalidate_separator_model_root(runtime)
    assert raised.value.detail_code == "separator_model_root_invalid"
    runtime.close()  # type: ignore[attr-defined]


def test_inventory_model_root_bound_rejects_non_int_root_fd(
    tmp_path: Path,
) -> None:
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    with pytest.raises(OSError, match="model root descriptor is unavailable"):
        separators._inventory_model_root_bound(
            model_root,
            separators._SPLEETER_MODEL_ROOT_FILES,
            root_fd="not-an-int",  # type: ignore[arg-type]
        )


def test_inventory_model_root_bound_rejects_non_directory_fd(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "regular.txt"
    file_path.write_text("content")
    fd = os.open(file_path, os.O_RDONLY)
    try:
        with pytest.raises(OSError, match="model root is not an ordinary directory"):
            separators._inventory_model_root_bound(
                tmp_path,
                separators._SPLEETER_MODEL_ROOT_FILES,
                root_fd=fd,
            )
    finally:
        os.close(fd)


def test_open_model_root_path_rejects_non_posix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_follow, directory_flag, close_on_exec = separators._model_root_descriptor_flags()
    monkeypatch.setattr(separators.os.path, "sep", "\\")
    with pytest.raises(OSError, match="descriptor-relative model paths are unavailable"):
        separators._open_model_root_path(
            Path("/some/path"),
            directory_flag=directory_flag,
            no_follow=no_follow,
            close_on_exec=close_on_exec,
        )


def test_close_model_root_fd_swallows_os_error() -> None:
    separators._close_model_root_fd(999999)


def test_model_root_launch_path_falls_back_when_no_proc_or_dev_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(separators.os, "name", "posix")
    monkeypatch.setattr(separators.sys, "platform", "linux")

    original_is_dir = Path.is_dir

    def not_dir(self: Path) -> bool:
        if str(self) in ("/proc/self/fd", "/dev/fd"):
            return False
        return original_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", not_dir)
    with pytest.raises(OSError, match="descriptor-backed model-root paths are unavailable"):
        separators._model_root_launch_path(42)


def test_build_runtime_rejects_non_int_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID))
    env = load_separator_environment_manifest(_fixture_path(SPLEETER_SEPARATOR_ID), lock)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)

    def inventory_returns_none_fd(*_args: object, **_kwargs: object) -> object:
        return lock.model_files, None

    monkeypatch.setattr(separators, "_inventory_model_root_bound", inventory_returns_none_fd)
    with pytest.raises(OSError, match="separator model root descriptor is unavailable"):
        separators._build_attested_separator_runtime(
            interpreter=Path("/python"),
            lock=lock,
            model_root=model_root,
            model_files=lock.model_files,
            environment=env,
        )


def test_build_runtime_rejects_inventory_mismatch_after_reinventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID))
    env = load_separator_environment_manifest(_fixture_path(SPLEETER_SEPARATOR_ID), lock)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    wrong_files = (SeparatorModelFile(name="4stems/.probe", sha256="0" * 64),)

    def inventory_returns_wrong(*_args: object, **_kwargs: object) -> object:
        return wrong_files, 42

    monkeypatch.setattr(separators, "_inventory_model_root_bound", inventory_returns_wrong)
    with pytest.raises(SeparatorExecutionError, match="separator_runtime_unattested") as raised:
        separators._build_attested_separator_runtime(
            interpreter=Path("/python"),
            lock=lock,
            model_root=model_root,
            model_files=wrong_files,
            environment=env,
        )
    assert raised.value.detail_code == "separator_runtime_unattested"


def test_build_runtime_closes_descriptor_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID))
    env = load_separator_environment_manifest(_fixture_path(SPLEETER_SEPARATOR_ID), lock)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    closed_fds: list[int] = []
    original_close = separators._close_model_root_fd

    def tracking_close(fd: int) -> None:
        closed_fds.append(fd)
        original_close(fd)

    monkeypatch.setattr(separators, "_close_model_root_fd", tracking_close)

    def inventory_returns_fd(*_args: object, **_kwargs: object) -> object:
        real_fd = os.open(tmp_path, os.O_RDONLY)
        return lock.model_files, real_fd

    monkeypatch.setattr(separators, "_inventory_model_root_bound", inventory_returns_fd)

    def launch_path_raises(_fd: int) -> Path:
        raise OSError("launch path failure")

    monkeypatch.setattr(separators, "_model_root_launch_path", launch_path_raises)
    with pytest.raises(OSError, match="launch path failure"):
        separators._build_attested_separator_runtime(
            interpreter=Path("/python"),
            lock=lock,
            model_root=model_root,
            model_files=lock.model_files,
            environment=env,
        )
    assert len(closed_fds) == 1


def test_run_separator_rejects_staging_os_error(
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

    def fail_tempdir(*_args: object, **_kwargs: object) -> object:
        raise OSError("staging failure")

    monkeypatch.setattr(separators.tempfile, "TemporaryDirectory", fail_tempdir)
    with pytest.raises(SeparatorExecutionError, match="separator_staging_failed") as raised:
        separators.run_spleeter_drums(
            source_path,
            source_audio_sha256="a" * 64,
            source_duration_sec=1.0,
            runtime=runtime,
            cache_root=tmp_path / "cache",
        )
    assert raised.value.detail_code == "separator_staging_failed"
    runtime.close()  # type: ignore[attr-defined]


def test_run_separator_rejects_publish_os_error(
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
    fake = _FakePopen(
        write_output=lambda workdir: _write_stem(
            workdir,
            separator_id=SPLEETER_SEPARATOR_ID,
            samples=np.full((44100, 1), 0.25, dtype=np.float32),
        )
    )
    _install_fake_popen(monkeypatch, fake)

    def publish_os_error(_path: Path, _content: bytes) -> object:
        raise OSError("disk full")

    monkeypatch.setattr(separators, "publish_immutable_file", publish_os_error)
    with pytest.raises(SeparatorExecutionError, match="stem_publish_failed") as raised:
        separators.run_spleeter_drums(
            source_path,
            source_audio_sha256="a" * 64,
            source_duration_sec=1.0,
            runtime=runtime,
            cache_root=tmp_path / "cache",
        )
    assert raised.value.detail_code == "stem_publish_failed"
    runtime.close()  # type: ignore[attr-defined]


def test_stop_separator_process_group_swallows_final_wait_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakePopen(terminate_wait_timeout=True)
    monkeypatch.setattr(os, "killpg", lambda *_args: None)
    original_wait = fake.wait

    def wait_then_raise(timeout: float | None = None) -> None:
        original_wait(timeout)
        if timeout is None:
            raise OSError("wait error")

    monkeypatch.setattr(fake, "wait", wait_then_raise)
    separators._stop_separator_process_group(fake)


def test_require_same_model_state_detects_state_change() -> None:
    import os as os_module

    path = os_module.devnull
    fd = os_module.open(path, os_module.O_RDONLY)
    try:
        meta1 = os_module.fstat(fd)
        meta2 = os_module.stat_result((meta1.st_mode + 1,) + meta1[1:])
        with pytest.raises(OSError, match="model root entry changed"):
            separators._require_same_model_state(meta1, meta2)
    finally:
        os_module.close(fd)


# ---------------------------------------------------------------------------
# Final targeted tests for remaining uncovered lines
# ---------------------------------------------------------------------------


def test_attest_reraises_model_root_invalid_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"
    _freeze_separator_runtime(
        separator_id=SPLEETER_SEPARATOR_ID,
        interpreter=interpreter,
        model_root=model_root,
        repository_revision="a" * 40,
        output=lock_path,
    )

    def inventory_model_root_invalid(*_args: object, **_kwargs: object) -> object:
        raise SeparatorExecutionError("separator_model_root_invalid", "detail")

    monkeypatch.setattr(separators, "_inventory_model_root_bound", inventory_model_root_invalid)
    with pytest.raises(SeparatorExecutionError, match="separator_model_root_invalid") as raised:
        separators.attest_separator_runtime(lock_path, interpreter, model_root)
    assert raised.value.detail_code == "separator_model_root_invalid"


def test_attest_reraises_attestation_failure_code_from_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    lock_path = tmp_path / "frozen" / "model.json"
    _freeze_separator_runtime(
        separator_id=SPLEETER_SEPARATOR_ID,
        interpreter=interpreter,
        model_root=model_root,
        repository_revision="a" * 40,
        output=lock_path,
    )

    def build_attestation_code(*_args: object, **_kwargs: object) -> object:
        raise SeparatorExecutionError("separator_lock_companion_mismatch", "detail")

    monkeypatch.setattr(separators, "_build_attested_separator_runtime", build_attestation_code)
    with pytest.raises(
        SeparatorExecutionError, match="separator_lock_companion_mismatch"
    ) as raised:
        separators.attest_separator_runtime(lock_path, interpreter, model_root)
    assert raised.value.detail_code == "separator_lock_companion_mismatch"


def test_model_root_launch_path_uses_dev_fd_on_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(separators.os, "name", "posix")
    monkeypatch.setattr(separators.sys, "platform", "linux")
    result = separators._model_root_launch_path(42)
    assert result == Path("/dev/fd") / "42" or result == Path("/proc/self/fd") / "42"


def test_build_runtime_rejects_inventory_mismatch_with_matching_model_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = load_separator_lock(_fixture_path(SPLEETER_SEPARATOR_ID))
    env = load_separator_environment_manifest(_fixture_path(SPLEETER_SEPARATOR_ID), lock)
    model_root, _ = _synthetic_model_root(tmp_path, SPLEETER_SEPARATOR_ID)
    wrong_files = (SeparatorModelFile(name="4stems/.probe", sha256="0" * 64),)

    def inventory_returns_wrong(*_args: object, **_kwargs: object) -> object:
        return wrong_files, 42

    monkeypatch.setattr(separators, "_inventory_model_root_bound", inventory_returns_wrong)
    with pytest.raises(SeparatorExecutionError, match="separator_runtime_unattested") as raised:
        separators._build_attested_separator_runtime(
            interpreter=Path("/python"),
            lock=lock,
            model_root=model_root,
            model_files=lock.model_files,
            environment=env,
        )
    assert raised.value.detail_code == "separator_runtime_unattested"
