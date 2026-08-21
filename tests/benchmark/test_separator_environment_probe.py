from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import venv
from collections.abc import Iterator
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = REPO_ROOT / "src" / "benchmark" / "separator_environment_probe.py"


def canonical_json_bytes(value: object, *, trailing_newline: bool = False) -> bytes:
    content = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return content + (b"\n" if trailing_newline else b"")


def strict_json_loads(content: bytes, *, require_canonical: bool = False) -> object:
    def object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        content.decode("utf-8"),
        object_pairs_hook=object_from_pairs,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant: {constant}")
        ),
    )
    if require_canonical and canonical_json_bytes(value) != content:
        raise ValueError("JSON bytes are not canonical")
    return value


def _venv_interpreter(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _distribution_file_digest(content: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")


def _write_distribution(
    purelib: Path,
    package_name: str,
    package_version: str,
) -> None:
    package_dir = purelib / package_name
    package_dir.mkdir()
    init_path = package_dir / "__init__.py"
    init_path.write_text("__version__ = '2.4.2'\n", encoding="utf-8")

    distribution_dir = purelib / f"{package_name}-{package_version}.dist-info"
    distribution_dir.mkdir()
    metadata_path = distribution_dir / "METADATA"
    metadata_path.write_text(
        f"Metadata-Version: 2.1\nName: {package_name}\nVersion: {package_version}\n",
        encoding="utf-8",
    )
    record_path = distribution_dir / "RECORD"
    files = (init_path, metadata_path)
    rows = []
    for file_path in files:
        relative_path = file_path.relative_to(purelib).as_posix()
        content = file_path.read_bytes()
        rows.append(
            [
                relative_path,
                f"sha256={_distribution_file_digest(content)}",
                str(len(content)),
            ]
        )
    rows.append([record_path.relative_to(purelib).as_posix(), "", ""])
    with record_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


_SYNTHETIC_VENV_BASE: tuple[Path, Path, Path] | None = None


def _initialize_synthetic_venv_base() -> tuple[Path, Path, Path]:
    """Build the shared bare venv once per session (eagerly via the fixture)."""
    global _SYNTHETIC_VENV_BASE
    if _SYNTHETIC_VENV_BASE is None:
        environment = (Path(tempfile.mkdtemp(prefix="crux-synthetic-venv-")) / "venv").resolve()
        venv.EnvBuilder(with_pip=False, symlinks=False).create(environment)
        interpreter = _venv_interpreter(environment)
        purelib = Path(
            subprocess.check_output(
                [
                    str(interpreter),
                    "-c",
                    "import sysconfig; print(sysconfig.get_paths()['purelib'])",
                ],
                text=True,
            ).strip()
        )
        _SYNTHETIC_VENV_BASE = (environment, purelib, _venv_scripts(interpreter))
    return _SYNTHETIC_VENV_BASE


@pytest.fixture(scope="session", autouse=True)
def _initialize_synthetic_venv_base_fixture() -> Iterator[None]:
    global _SYNTHETIC_VENV_BASE
    _initialize_synthetic_venv_base()
    yield
    if _SYNTHETIC_VENV_BASE is not None:
        environment, _, _ = _SYNTHETIC_VENV_BASE  # pylint: disable=unpacking-non-sequence
        shutil.rmtree(environment.parent, ignore_errors=True)
        _SYNTHETIC_VENV_BASE = None


def _copy_synthetic_venv(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Copy the session venv into tmp_path and remap its scheme locations."""
    base_environment, base_purelib, base_scripts = _initialize_synthetic_venv_base()
    environment = tmp_path / "venv"
    shutil.copytree(base_environment, environment)

    def _remap(base_path: Path) -> Path:
        return environment / base_path.relative_to(base_environment)

    return _venv_interpreter(environment), _remap(base_purelib), _remap(base_scripts)


def _synthetic_environment(
    tmp_path: Path,
    *,
    package_name: str = "spleeter",
    package_version: str = "2.4.2",
) -> tuple[Path, Path]:
    interpreter, purelib, _scripts = _copy_synthetic_venv(tmp_path)
    _write_distribution(purelib, package_name, package_version)
    return interpreter, purelib


def _venv_scripts(interpreter: Path) -> Path:
    return Path(
        subprocess.check_output(
            [
                str(interpreter),
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['scripts'])",
            ],
            text=True,
        ).strip()
    )


def _write_console_script(
    purelib: Path,
    scripts: Path,
    package_name: str,
    package_version: str,
) -> Path:
    """Write a distribution whose RECORD references a console script in scripts.

    Mirrors the layout that ``pip install`` produces: the console-entry-point
    wrapper is installed into the scheme's scripts directory, and its RECORD
    entry uses a relative path with ``..`` components to escape the
    distribution root (``purelib``) before landing in ``scripts``.
    """
    package_dir = purelib / package_name
    package_dir.mkdir()
    init_path = package_dir / "__init__.py"
    init_path.write_text("__version__ = '2.4.2'\n", encoding="utf-8")

    distribution_dir = purelib / f"{package_name}-{package_version}.dist-info"
    distribution_dir.mkdir()
    metadata_path = distribution_dir / "METADATA"
    metadata_path.write_text(
        f"Metadata-Version: 2.1\nName: {package_name}\nVersion: {package_version}\n",
        encoding="utf-8",
    )

    scripts.mkdir(parents=True, exist_ok=True)
    console_script_path = scripts / package_name
    console_script_path.write_text(
        f"#!/usr/bin/env python\nimport {package_name}\n{package_name}.main()\n",
        encoding="utf-8",
    )

    record_path = distribution_dir / "RECORD"
    rows: list[list[str]] = []
    for file_path in (init_path, metadata_path):
        relative_path = file_path.relative_to(purelib).as_posix()
        content = file_path.read_bytes()
        rows.append(
            [
                relative_path,
                f"sha256={_distribution_file_digest(content)}",
                str(len(content)),
            ]
        )
    # Console script: path relative to purelib (the directory containing
    # .dist-info) uses ".." to reach the scripts directory.
    script_relative = os.path.relpath(console_script_path, purelib).replace(os.sep, "/")
    script_content = console_script_path.read_bytes()
    rows.append(
        [
            script_relative,
            f"sha256={_distribution_file_digest(script_content)}",
            str(len(script_content)),
        ]
    )
    rows.append([record_path.relative_to(purelib).as_posix(), "", ""])
    with record_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    return console_script_path


def _synthetic_environment_with_console_script(
    tmp_path: Path,
    *,
    package_name: str = "spleeter",
    package_version: str = "2.4.2",
) -> tuple[Path, Path, Path]:
    interpreter, purelib, scripts = _copy_synthetic_venv(tmp_path)
    console_script = _write_console_script(purelib, scripts, package_name, package_version)
    return interpreter, purelib, console_script


def _run_probe(interpreter: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(interpreter), str(PROBE_PATH)],
        check=False,
        capture_output=True,
    )


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("separator_environment_probe_test", PROBE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in _iter_strings(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _iter_strings(child)]
    return []


def _assert_no_absolute_paths(value: object) -> None:
    assert all(
        not (
            Path(item).is_absolute()
            or PurePosixPath(item).is_absolute()
            or PureWindowsPath(item).is_absolute()
            or bool(PureWindowsPath(item).drive)
        )
        for item in _iter_strings(value)
    )


def test_probe_emits_canonical_manifest_with_synthetic_distribution(tmp_path: Path) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)

    result = _run_probe(interpreter)

    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.endswith(b"\n")
    payload = strict_json_loads(result.stdout[:-1], require_canonical=True)
    assert isinstance(payload, dict)
    assert canonical_json_bytes(payload, trailing_newline=True) == result.stdout
    _assert_no_absolute_paths(payload)

    assert payload["schema"] == "crux.separator-environment/v1"
    assert payload["separator_id"] == "spleeter4-drums-v1"
    assert payload["package_name"] == "spleeter"
    assert payload["package_version"] == "2.4.2"
    assert (
        payload["interpreter_sha256"]
        == hashlib.sha256(interpreter.resolve(strict=True).read_bytes()).hexdigest()
    )

    distributions = payload["distributions"]
    assert isinstance(distributions, list)
    synthetic = next(item for item in distributions if item["name"] == "spleeter")
    assert synthetic["version"] == "2.4.2"
    files = {item["path"]: item for item in synthetic["files"]}
    for relative_path in (
        "spleeter/__init__.py",
        "spleeter-2.4.2.dist-info/METADATA",
        "spleeter-2.4.2.dist-info/RECORD",
    ):
        content = (purelib / relative_path).read_bytes()
        assert files[relative_path] == {
            "root": "purelib",
            "path": relative_path,
            "byte_length": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }


def test_probe_inventories_console_script_in_scripts_root(tmp_path: Path) -> None:
    """A console script referenced via a cross-root RECORD path is inventoried.

    Installer-generated RECORD files reference console-entry-point wrappers
    in the scripts directory using relative paths with ``..`` components that
    escape the distribution root.  The probe must normalize these paths,
    classify them under the ``scripts`` root, and verify them without
    rejecting the venv-owned files (activate, python, ...) that share that
    root.
    """
    interpreter, purelib, console_script = _synthetic_environment_with_console_script(tmp_path)

    result = _run_probe(interpreter)

    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.endswith(b"\n")
    payload = strict_json_loads(result.stdout[:-1], require_canonical=True)
    assert isinstance(payload, dict)
    assert canonical_json_bytes(payload, trailing_newline=True) == result.stdout
    _assert_no_absolute_paths(payload)

    distributions = payload["distributions"]
    assert isinstance(distributions, list)
    synthetic = next(item for item in distributions if item["name"] == "spleeter")
    files = {item["path"]: item for item in synthetic["files"]}

    # The console script is classified under the scripts root with a portable
    # path relative to that root (just the script name).
    script_name = console_script.name
    assert script_name in files
    script_entry = files[script_name]
    assert script_entry["root"] == "scripts"
    assert script_entry["path"] == script_name
    assert script_entry["sha256"] == hashlib.sha256(console_script.read_bytes()).hexdigest()
    assert script_entry["byte_length"] == console_script.stat().st_size

    # The purelib members are still present and correct.
    for relative_path in (
        "spleeter/__init__.py",
        "spleeter-2.4.2.dist-info/METADATA",
        "spleeter-2.4.2.dist-info/RECORD",
    ):
        content = (purelib / relative_path).read_bytes()
        assert files[relative_path] == {
            "root": "purelib",
            "path": relative_path,
            "byte_length": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }


def test_probe_ignores_stale_record_digest_and_size(tmp_path: Path) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    record_path = purelib / "spleeter-2.4.2.dist-info" / "RECORD"
    rows = list(csv.reader(record_path.read_text(encoding="utf-8").splitlines()))
    assert rows[0][0] == "spleeter/__init__.py"
    rows[0][1] = "sha256=stale"
    rows[0][2] = "0"
    with record_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)

    result = _run_probe(interpreter)

    assert result.returncode == 0, result.stderr.decode()
    payload = strict_json_loads(result.stdout[:-1], require_canonical=True)
    assert isinstance(payload, dict)
    distribution = next(item for item in payload["distributions"] if item["name"] == "spleeter")
    file = next(item for item in distribution["files"] if item["path"] == "spleeter/__init__.py")
    actual_bytes = (purelib / "spleeter" / "__init__.py").read_bytes()
    assert file["byte_length"] == len(actual_bytes)
    assert file["sha256"] == hashlib.sha256(actual_bytes).hexdigest()
    assert canonical_json_bytes(payload, trailing_newline=True) == result.stdout


def test_probe_reports_changed_declared_member_bytes(tmp_path: Path) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    baseline_result = _run_probe(interpreter)
    assert baseline_result.returncode == 0, baseline_result.stderr.decode()
    baseline = strict_json_loads(baseline_result.stdout[:-1], require_canonical=True)
    assert isinstance(baseline, dict)

    _mutate_distribution_tree(purelib, "record_content_changed")
    result = _run_probe(interpreter)

    assert result.returncode == 0, result.stderr.decode()
    payload = strict_json_loads(result.stdout[:-1], require_canonical=True)
    assert isinstance(payload, dict)
    assert canonical_json_bytes(payload, trailing_newline=True) == result.stdout
    baseline_distribution = next(
        item for item in baseline["distributions"] if item["name"] == "spleeter"
    )
    distribution = next(item for item in payload["distributions"] if item["name"] == "spleeter")
    baseline_file = next(
        item for item in baseline_distribution["files"] if item["path"] == "spleeter/__init__.py"
    )
    file = next(item for item in distribution["files"] if item["path"] == "spleeter/__init__.py")
    actual_bytes = (purelib / "spleeter" / "__init__.py").read_bytes()
    assert file["byte_length"] == len(actual_bytes)
    assert file["sha256"] == hashlib.sha256(actual_bytes).hexdigest()
    assert (file["byte_length"], file["sha256"]) != (
        baseline_file["byte_length"],
        baseline_file["sha256"],
    )


def _mutate_distribution_tree(purelib: Path, mutation: str) -> None:
    mutations = {
        "missing_record_member": lambda: (purelib / "spleeter" / "__init__.py").unlink(),
        "record_content_changed": lambda: (purelib / "spleeter" / "__init__.py").write_text(
            "changed\n", encoding="utf-8"
        ),
        "record_self_changed": lambda: (purelib / "spleeter-2.4.2.dist-info" / "RECORD").write_text(
            "changed\n", encoding="utf-8"
        ),
        "extra_python": lambda: (purelib / "injected.py").write_text("x = 1\n", encoding="utf-8"),
        "extra_pth": lambda: (purelib / "injected.pth").write_text(
            "/tmp/escape\n", encoding="utf-8"
        ),
        "sitecustomize": lambda: (purelib / "sitecustomize.py").write_text(
            "pass\n", encoding="utf-8"
        ),
    }
    if mutation in mutations:
        mutations[mutation]()
        return
    if mutation == "leaf_symlink":
        target = purelib / "missing-target"
        (purelib / "spleeter" / "__init__.py").unlink()
        (purelib / "spleeter" / "__init__.py").symlink_to(target)
        return
    if mutation == "parent_symlink":
        package_dir = purelib / "spleeter"
        shutil.rmtree(package_dir)
        package_dir.symlink_to(purelib / "missing-package", target_is_directory=True)
        return
    raise AssertionError(f"unknown mutation: {mutation}")


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_record_member",
        "record_self_changed",
        "extra_python",
        "extra_pth",
        "sitecustomize",
        "leaf_symlink",
        "parent_symlink",
    ),
)
def test_probe_rejects_distribution_tree_drift(tmp_path: Path, mutation: str) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    _mutate_distribution_tree(purelib, mutation)

    result = _run_probe(interpreter)

    assert result.returncode != 0
    assert result.stdout == b""


def test_probe_ignores_generated_bytecode(tmp_path: Path) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    baseline = _run_probe(interpreter)
    assert baseline.returncode == 0, baseline.stderr.decode()

    pycache = purelib / "spleeter" / "__pycache__"
    pycache.mkdir()
    (pycache / "module.cpython-313.pyc").write_bytes(b"synthetic bytecode")

    result = _run_probe(interpreter)

    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == baseline.stdout


def test_probe_hashes_resolved_interpreter_target(tmp_path: Path) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    linked_interpreter = tmp_path / "linked-python"
    linked_interpreter.symlink_to(interpreter)

    result = _run_probe(linked_interpreter)

    assert result.returncode == 0, result.stderr.decode()
    payload = strict_json_loads(result.stdout[:-1], require_canonical=True)
    assert isinstance(payload, dict)
    final_target = interpreter.resolve(strict=True)
    assert payload["interpreter_sha256"] == hashlib.sha256(final_target.read_bytes()).hexdigest()


def _synthetic_root_configuration(tmp_path: Path) -> dict[str, str]:
    tags = (
        "stdlib",
        "platstdlib",
        "purelib",
        "platlib",
        "include",
        "platinclude",
        "scripts",
        "data",
    )
    paths = {tag: tmp_path / tag for tag in tags}
    for path in paths.values():
        path.mkdir(parents=True)
    configuration = {tag: str(path) for tag, path in paths.items()}
    configuration["platstdlib"] = configuration["stdlib"]
    configuration["platlib"] = configuration["purelib"]
    configuration["platinclude"] = configuration["include"]
    return configuration


def test_root_aliases_are_primary_by_role_and_cross_role_collisions_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    configuration = _synthetic_root_configuration(tmp_path)
    monkeypatch.setattr(probe, "_configured_paths", lambda _: configuration)

    roots = probe._canonical_root_paths(Path("/synthetic/python"))

    assert tuple(roots) == ("stdlib", "purelib", "include", "scripts", "data")

    ambiguous = dict(configuration)
    ambiguous["platlib"] = ambiguous["stdlib"]
    monkeypatch.setattr(probe, "_configured_paths", lambda _: ambiguous)
    with pytest.raises(probe._ProbeError):
        probe._canonical_root_paths(Path("/synthetic/python"))


def test_configured_symlink_root_and_missing_no_follow_support_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    configuration = _synthetic_root_configuration(tmp_path / "configured")
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    configuration["purelib"] = str(link)
    monkeypatch.setattr(probe, "_configured_paths", lambda _: configuration)

    with pytest.raises(probe._ProbeError):
        probe._canonical_root_paths(Path("/synthetic/python"))

    monkeypatch.setattr(probe, "_NOFOLLOW", 0)
    with pytest.raises(probe._ProbeError):
        probe._canonical_root_paths(Path("/synthetic/python"))


def test_inventory_carries_first_hash_identity_to_tree_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    probe = _load_probe_module()
    roots = probe._canonical_root_paths(interpreter)
    descriptors = {}
    identities = {}
    try:
        for tag, path in roots.items():
            descriptor, identity = probe._open_root(path)
            descriptors[tag] = descriptor
            identities[tag] = identity
        distributions = list(probe.importlib.metadata.distributions(path=[str(purelib)]))
        distribution = next(item for item in distributions if item.metadata["Name"] == "spleeter")
        original_hash = probe._hash_relative_file
        mutated = False

        def racing_hash(root_descriptor, parts, *, capture=False):
            nonlocal mutated
            result = original_hash(root_descriptor, parts, capture=capture)
            if not capture and parts == ("spleeter", "__init__.py") and not mutated:
                (purelib / "spleeter" / "__init__.py").write_text("raced\n", encoding="utf-8")
                mutated = True
            return result

        monkeypatch.setattr(probe, "_hash_relative_file", racing_hash)
        item, expected, expected_identities = probe._inventory_distribution(
            distribution, roots, descriptors
        )
        assert item["name"] == "spleeter"
        assert expected_identities
        with pytest.raises(probe._ProbeError):
            probe._walk_tree(
                descriptors["purelib"],
                identities["purelib"],
                "purelib",
                expected,
                expected_identities,
            )
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


class _FakeImplementation:
    """Stand-in for ``sys.implementation`` with controllable attributes."""

    def __init__(self, name: str, cache_tag: str | None = None) -> None:
        self.name = name
        self.cache_tag = cache_tag


class _FakeDistribution:
    """Minimal stand-in for ``importlib.metadata.Distribution``."""

    def __init__(
        self,
        *,
        dist_path: Path | None,
        files: object,
        locate_file: object,
        metadata: dict[str, str] | None = None,
        version: str = "2.4.2",
        name: str = "spleeter",
    ) -> None:
        self._path = dist_path
        self.files = files
        self._locate_file = locate_file
        self._metadata = metadata if metadata is not None else {"Name": name}
        self._version = version
        self._name = name

    @property
    def metadata(self) -> dict[str, str]:
        return self._metadata

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def locate_file(self, record: object) -> object:
        return self._locate_file(record)


def _open_roots(probe, roots: dict[str, Path]) -> tuple[dict[str, int], dict[str, tuple[int, ...]]]:
    descriptors: dict[str, int] = {}
    identities: dict[str, tuple[int, ...]] = {}
    for tag, path in roots.items():
        descriptor, identity = probe._open_root(path)
        descriptors[tag] = descriptor
        identities[tag] = identity
    return descriptors, identities


def test_canonical_json_bytes_appends_trailing_newline() -> None:
    probe = _load_probe_module()
    assert probe.canonical_json_bytes({"a": 1}, trailing_newline=True) == b'{"a":1}\n'


def test_canonical_json_bytes_rejects_non_serializable() -> None:
    probe = _load_probe_module()
    with pytest.raises(probe._ProbeError):
        probe.canonical_json_bytes({object()})


def test_require_regular_rejects_directory(tmp_path: Path) -> None:
    probe = _load_probe_module()
    with pytest.raises(probe._ProbeError):
        probe._require_regular(os.stat(tmp_path))


def test_require_directory_rejects_regular_file(tmp_path: Path) -> None:
    probe = _load_probe_module()
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(probe._ProbeError):
        probe._require_directory(os.stat(file_path))


def test_require_dir_fd_support_rejects_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(probe.os, "supports_dir_fd", set())
    with pytest.raises(probe._ProbeError):
        probe._require_dir_fd_support()


def test_read_hash_fd_rejects_changed_file_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    file_path = tmp_path / "data.bin"
    file_path.write_bytes(b"payload")
    file_descriptor = os.open(file_path, os.O_RDONLY)
    real_identity = probe._identity
    calls = 0

    def shifting_identity(metadata: os.stat_result) -> tuple[int, ...]:
        nonlocal calls
        calls += 1
        base = real_identity(metadata)
        return (calls, *base[1:])

    monkeypatch.setattr(probe, "_identity", shifting_identity)
    try:
        with pytest.raises(probe._ProbeError):
            probe._read_hash_fd(file_descriptor)
    finally:
        os.close(file_descriptor)


def test_open_root_closes_descriptor_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    target = tmp_path / "realdir"
    target.mkdir()
    real_close = os.close

    def failing_fstat(fd: int) -> os.stat_result:
        raise OSError("boom")

    def failing_close(fd: int) -> None:
        monkeypatch.setattr(probe.os, "close", real_close)
        raise OSError("close boom")

    monkeypatch.setattr(probe.os, "fstat", failing_fstat)
    monkeypatch.setattr(probe.os, "close", failing_close)
    with pytest.raises(probe._ProbeError):
        probe._open_root(target)


def test_open_relative_rejects_empty_parts(tmp_path: Path) -> None:
    probe = _load_probe_module()
    root = tmp_path / "root"
    root.mkdir()
    file_descriptor = os.open(root, os.O_RDONLY)
    try:
        with pytest.raises(probe._ProbeError):
            probe._open_relative(file_descriptor, ())
    finally:
        os.close(file_descriptor)


def test_open_relative_wraps_missing_file_error(tmp_path: Path) -> None:
    probe = _load_probe_module()
    root = tmp_path / "root"
    root.mkdir()
    file_descriptor = os.open(root, os.O_RDONLY)
    try:
        with pytest.raises(probe._ProbeError):
            probe._open_relative(file_descriptor, ("missing",))
    finally:
        os.close(file_descriptor)


def test_open_relative_closes_descriptor_when_intermediate_not_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    root = tmp_path / "root"
    (root / "subdir").mkdir(parents=True)
    file_descriptor = os.open(root, os.O_RDONLY)
    monkeypatch.setattr(
        probe,
        "_require_directory",
        lambda metadata: (_ for _ in ()).throw(probe._ProbeError("not a dir")),
    )
    try:
        with pytest.raises(probe._ProbeError):
            probe._open_relative(file_descriptor, ("subdir", "file"))
    finally:
        os.close(file_descriptor)


@pytest.mark.parametrize(
    "value",
    ("", 123, "a\x00b", "/abs", "\\win", "a/../b", "a/./b", "a//b"),
)
def test_validate_relative_path_rejects_invalid(value: object) -> None:
    probe = _load_probe_module()
    with pytest.raises(probe._ProbeError):
        probe._validate_relative_path(value)


@pytest.mark.parametrize(
    "value",
    ("", 123, "a\x00b", "/abs", "\\win", "a/./b", "a//b"),
)
def test_validate_record_path_rejects_invalid(value: object) -> None:
    probe = _load_probe_module()
    with pytest.raises(probe._ProbeError):
        probe._validate_record_path(value)


def test_validate_record_path_allows_parent_references() -> None:
    probe = _load_probe_module()
    assert probe._validate_record_path("a/../b") == "a/../b"


@pytest.mark.parametrize(
    "value",
    ("", 123, "!bad", "-abc", "abc-", "a!b", "A-B.C_D", "ABC"),
)
def test_normalize_distribution_name_branches(value: object) -> None:
    probe = _load_probe_module()
    if value in ("", 123, "!bad", "-abc", "abc-", "a!b"):
        with pytest.raises(probe._ProbeError):
            probe._normalize_distribution_name(value)
    else:
        result = probe._normalize_distribution_name(value)
        assert isinstance(result, str) and result


def test_normalize_distribution_name_lowercases_and_normalizes_separators() -> None:
    probe = _load_probe_module()
    assert probe._normalize_distribution_name("A-B.C_D") == "a-b-c-d"
    assert probe._normalize_distribution_name("ABC") == "abc"


def test_resolved_virtual_environment_returns_none_without_pyvenv_cfg(tmp_path: Path) -> None:
    probe = _load_probe_module()
    interpreter = tmp_path / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    assert probe._resolved_virtual_environment(interpreter) is None


def test_resolved_virtual_environment_returns_none_for_non_regular_cfg(tmp_path: Path) -> None:
    probe = _load_probe_module()
    environment = tmp_path / "venv"
    (environment / "bin").mkdir(parents=True)
    (environment / "pyvenv.cfg").mkdir()
    interpreter = environment / "bin" / "python"
    assert probe._resolved_virtual_environment(interpreter) is None


def test_configured_paths_wraps_sysconfig_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _load_probe_module()

    def boom() -> dict[str, str]:
        raise KeyError("nope")

    monkeypatch.setattr(probe.sysconfig, "get_paths", boom)
    with pytest.raises(probe._ProbeError):
        probe._configured_paths(Path("/x/python"))


def test_configured_paths_overrides_windows_venv_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    probe = _load_probe_module()
    monkeypatch.setattr(probe.os, "name", "nt")
    configured = probe._configured_paths(interpreter)
    environment = interpreter.parent.parent
    assert configured["platstdlib"] == os.fspath(environment / "Lib")
    assert configured["scripts"] == os.fspath(environment / "Scripts")


def test_canonical_root_paths_rejects_missing_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    configuration = _synthetic_root_configuration(tmp_path)
    del configuration["scripts"]
    monkeypatch.setattr(probe, "_configured_paths", lambda _: configuration)
    with pytest.raises(probe._ProbeError):
        probe._canonical_root_paths(Path("/synthetic/python"))


def test_canonical_root_paths_rejects_empty_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    configuration = _synthetic_root_configuration(tmp_path)
    configuration["scripts"] = ""
    monkeypatch.setattr(probe, "_configured_paths", lambda _: configuration)
    with pytest.raises(probe._ProbeError):
        probe._canonical_root_paths(Path("/synthetic/python"))


def test_canonical_root_paths_rejects_relative_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    configuration = _synthetic_root_configuration(tmp_path)
    configuration["scripts"] = "relative/path"
    monkeypatch.setattr(probe, "_configured_paths", lambda _: configuration)
    with pytest.raises(probe._ProbeError):
        probe._canonical_root_paths(Path("/synthetic/python"))


def test_root_for_path_rejects_outside_roots(tmp_path: Path) -> None:
    probe = _load_probe_module()
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(probe._ProbeError):
        probe._root_for_path(outside, {"purelib": root})


def test_root_for_path_rejects_ambiguous_root(tmp_path: Path) -> None:
    probe = _load_probe_module()
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    with pytest.raises(probe._ProbeError):
        probe._root_for_path(root / "sub", {"a": root, "b": root})


def test_hash_relative_file_wraps_open_error(tmp_path: Path) -> None:
    probe = _load_probe_module()
    root = tmp_path / "root"
    root.mkdir()
    file_descriptor = os.open(root, os.O_RDONLY)
    try:
        with pytest.raises(probe._ProbeError):
            probe._hash_relative_file(file_descriptor, ("missing",))
    finally:
        os.close(file_descriptor)


def test_hash_absolute_file_returns_digest(tmp_path: Path) -> None:
    probe = _load_probe_module()
    file_path = tmp_path / "interp"
    file_path.write_bytes(b"binary")
    assert probe._hash_absolute_file(file_path) == hashlib.sha256(b"binary").hexdigest()


def test_hash_absolute_file_wraps_open_error(tmp_path: Path) -> None:
    probe = _load_probe_module()
    with pytest.raises(probe._ProbeError):
        probe._hash_absolute_file(tmp_path / "missing")


def test_hash_absolute_file_wraps_read_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    file_path = tmp_path / "interp"
    file_path.write_bytes(b"binary")

    def boom(fd: int, *, capture: bool = False) -> tuple[bytes, int, str, tuple[int, ...]]:
        raise probe._ProbeError("read boom")

    monkeypatch.setattr(probe, "_read_hash_fd", boom)
    with pytest.raises(probe._ProbeError):
        probe._hash_absolute_file(file_path)


def test_parse_record_rejects_malformed_encoding() -> None:
    probe = _load_probe_module()
    with pytest.raises(probe._ProbeError):
        probe._parse_record(b"\xff\xfe invalid utf8")


def test_parse_record_rejects_empty() -> None:
    probe = _load_probe_module()
    with pytest.raises(probe._ProbeError):
        probe._parse_record(b"")


def test_parse_record_rejects_malformed_row() -> None:
    probe = _load_probe_module()
    with pytest.raises(probe._ProbeError):
        probe._parse_record(b"a,b\n")


def test_parse_record_rejects_duplicate_paths() -> None:
    probe = _load_probe_module()
    with pytest.raises(probe._ProbeError):
        probe._parse_record(b"a,b,c\na,b,c\n")


def test_record_path_for_distribution_uses_files_candidates(tmp_path: Path) -> None:
    probe = _load_probe_module()
    dist_info = tmp_path / "pkg-1.0.dist-info"
    dist_info.mkdir()
    record = dist_info / "RECORD"
    record.write_text("", encoding="utf-8")
    distribution = _FakeDistribution(
        dist_path=None,
        files=[PurePosixPath("pkg-1.0.dist-info/RECORD")],
        locate_file=lambda r: record,
    )
    record_path, root = probe._record_path_for_distribution(distribution)
    assert record_path == record
    assert root == dist_info.parent


def test_record_path_for_distribution_rejects_ambiguous_candidates(tmp_path: Path) -> None:
    probe = _load_probe_module()
    distribution = _FakeDistribution(
        dist_path=None,
        files=[
            PurePosixPath("a.dist-info/RECORD"),
            PurePosixPath("b.dist-info/RECORD"),
        ],
        locate_file=lambda r: tmp_path / "RECORD",
    )
    with pytest.raises(probe._ProbeError):
        probe._record_path_for_distribution(distribution)


def test_record_path_for_distribution_wraps_locate_file_error(tmp_path: Path) -> None:
    probe = _load_probe_module()
    distribution = _FakeDistribution(
        dist_path=None,
        files=[PurePosixPath("pkg-1.0.dist-info/RECORD")],
        locate_file=lambda r: (_ for _ in ()).throw(TypeError("boom")),
    )
    with pytest.raises(probe._ProbeError):
        probe._record_path_for_distribution(distribution)


def test_record_path_for_distribution_rejects_non_dist_info_parent(tmp_path: Path) -> None:
    probe = _load_probe_module()
    other = tmp_path / "not-dist"
    other.mkdir()
    distribution = _FakeDistribution(dist_path=other, files=None, locate_file=None)
    with pytest.raises(probe._ProbeError):
        probe._record_path_for_distribution(distribution)


def test_inventory_distribution_rejects_malformed_name(tmp_path: Path) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    probe = _load_probe_module()
    roots = probe._canonical_root_paths(interpreter)
    descriptors, _ = _open_roots(probe, roots)
    try:
        dist_info = purelib / "spleeter-2.4.2.dist-info"
        distribution = _FakeDistribution(
            dist_path=dist_info,
            files=None,
            locate_file=None,
            metadata={"Name": "!invalid!"},
        )
        with pytest.raises(probe._ProbeError):
            probe._inventory_distribution(distribution, roots, descriptors)
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def test_inventory_distribution_rejects_malformed_version(tmp_path: Path) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    probe = _load_probe_module()
    roots = probe._canonical_root_paths(interpreter)
    descriptors, _ = _open_roots(probe, roots)
    try:
        dist_info = purelib / "spleeter-2.4.2.dist-info"
        distribution = _FakeDistribution(
            dist_path=dist_info,
            files=None,
            locate_file=None,
            version="",
        )
        with pytest.raises(probe._ProbeError):
            probe._inventory_distribution(distribution, roots, descriptors)
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def test_inventory_distribution_skips_bytecode_records(tmp_path: Path) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    probe = _load_probe_module()
    roots = probe._canonical_root_paths(interpreter)
    descriptors, _ = _open_roots(probe, roots)
    try:
        (purelib / "spleeter" / "module.pyc").write_bytes(b"bytecode")
        record_path = purelib / "spleeter-2.4.2.dist-info" / "RECORD"
        rows = list(csv.reader(record_path.read_text(encoding="utf-8").splitlines()))
        rows.insert(1, ["spleeter/module.pyc", "", ""])
        with record_path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)
        distributions = list(probe.importlib.metadata.distributions(path=[str(purelib)]))
        distribution = next(d for d in distributions if d.metadata["Name"] == "spleeter")
        item, _, _ = probe._inventory_distribution(distribution, roots, descriptors)
        paths = {file["path"] for file in item["files"]}
        assert "spleeter/module.pyc" not in paths
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def test_inventory_distribution_rejects_duplicate_resolved_files(tmp_path: Path) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    probe = _load_probe_module()
    roots = probe._canonical_root_paths(interpreter)
    descriptors, _ = _open_roots(probe, roots)
    try:
        record_path = purelib / "spleeter-2.4.2.dist-info" / "RECORD"
        rows = list(csv.reader(record_path.read_text(encoding="utf-8").splitlines()))
        rows.insert(1, ["spleeter/sub/../__init__.py", "", ""])
        with record_path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)
        distributions = list(probe.importlib.metadata.distributions(path=[str(purelib)]))
        distribution = next(d for d in distributions if d.metadata["Name"] == "spleeter")
        with pytest.raises(probe._ProbeError):
            probe._inventory_distribution(distribution, roots, descriptors)
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def test_inventory_distribution_adds_record_when_missing_from_rows(tmp_path: Path) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    probe = _load_probe_module()
    roots = probe._canonical_root_paths(interpreter)
    descriptors, _ = _open_roots(probe, roots)
    try:
        record_path = purelib / "spleeter-2.4.2.dist-info" / "RECORD"
        rows = [
            row
            for row in csv.reader(record_path.read_text(encoding="utf-8").splitlines())
            if row[0] != "spleeter-2.4.2.dist-info/RECORD"
        ]
        with record_path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)
        distributions = list(probe.importlib.metadata.distributions(path=[str(purelib)]))
        distribution = next(d for d in distributions if d.metadata["Name"] == "spleeter")
        item, _, _ = probe._inventory_distribution(distribution, roots, descriptors)
        paths = {file["path"] for file in item["files"]}
        assert "spleeter-2.4.2.dist-info/RECORD" in paths
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def test_walk_tree_rejects_changed_root_identity(tmp_path: Path) -> None:
    probe = _load_probe_module()
    root = tmp_path / "root"
    root.mkdir()
    file_descriptor = os.open(root, os.O_RDONLY)
    try:
        with pytest.raises(probe._ProbeError):
            probe._walk_tree(file_descriptor, (999, 999, 0, 0, 0, 0), "purelib", set(), {})
    finally:
        os.close(file_descriptor)


def test_walk_tree_rejects_symlink(tmp_path: Path) -> None:
    probe = _load_probe_module()
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (root / "link").symlink_to(target)
    file_descriptor = os.open(root, os.O_RDONLY)
    try:
        identity = probe._identity(os.fstat(file_descriptor))
        with pytest.raises(probe._ProbeError):
            probe._walk_tree(file_descriptor, identity, "purelib", set(), {})
    finally:
        os.close(file_descriptor)


def test_walk_tree_skips_pycache_directory(tmp_path: Path) -> None:
    probe = _load_probe_module()
    root = tmp_path / "root"
    root.mkdir()
    (root / "__pycache__").mkdir()
    (root / "real.py").write_text("x = 1\n", encoding="utf-8")
    file_descriptor = os.open(root, os.O_RDONLY)
    try:
        identity = probe._identity(os.fstat(file_descriptor))
        key = ("purelib", "real.py")
        observed = probe._walk_tree(
            file_descriptor,
            identity,
            "purelib",
            {key},
            {key: probe._identity(os.stat(root / "real.py"))},
        )
        assert observed == {key}
    finally:
        os.close(file_descriptor)


def test_walk_tree_skips_pyc_file(tmp_path: Path) -> None:
    probe = _load_probe_module()
    root = tmp_path / "root"
    root.mkdir()
    (root / "module.pyc").write_bytes(b"bytecode")
    file_descriptor = os.open(root, os.O_RDONLY)
    try:
        identity = probe._identity(os.fstat(file_descriptor))
        assert probe._walk_tree(file_descriptor, identity, "purelib", set(), {}) == set()
    finally:
        os.close(file_descriptor)


def test_walk_tree_rejects_unexpected_file(tmp_path: Path) -> None:
    probe = _load_probe_module()
    root = tmp_path / "root"
    root.mkdir()
    (root / "extra.py").write_text("x = 1\n", encoding="utf-8")
    file_descriptor = os.open(root, os.O_RDONLY)
    try:
        identity = probe._identity(os.fstat(file_descriptor))
        with pytest.raises(probe._ProbeError):
            probe._walk_tree(file_descriptor, identity, "purelib", set(), {})
    finally:
        os.close(file_descriptor)


def test_walk_tree_rejects_changed_file_identity(tmp_path: Path) -> None:
    probe = _load_probe_module()
    root = tmp_path / "root"
    root.mkdir()
    (root / "real.py").write_text("x = 1\n", encoding="utf-8")
    file_descriptor = os.open(root, os.O_RDONLY)
    try:
        identity = probe._identity(os.fstat(file_descriptor))
        key = ("purelib", "real.py")
        with pytest.raises(probe._ProbeError):
            probe._walk_tree(file_descriptor, identity, "purelib", {key}, {key: (0, 0, 0, 0, 0, 0)})
    finally:
        os.close(file_descriptor)


def test_walk_tree_rejects_special_file(tmp_path: Path) -> None:
    probe = _load_probe_module()
    root = tmp_path / "root"
    root.mkdir()
    os.mkfifo(root / "fifo")
    file_descriptor = os.open(root, os.O_RDONLY)
    try:
        identity = probe._identity(os.fstat(file_descriptor))
        with pytest.raises(probe._ProbeError):
            probe._walk_tree(file_descriptor, identity, "purelib", set(), {})
    finally:
        os.close(file_descriptor)


def test_walk_tree_rejects_root_changed_while_walking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    root = tmp_path / "root"
    root.mkdir()
    file_descriptor = os.open(root, os.O_RDONLY)
    real_identity = probe._identity
    calls = 0

    def shifting_identity(metadata: os.stat_result) -> tuple[int, ...]:
        nonlocal calls
        calls += 1
        base = real_identity(metadata)
        return (calls, *base[1:])

    monkeypatch.setattr(probe, "_identity", shifting_identity)
    try:
        real = real_identity(os.fstat(file_descriptor))
        root_identity = (1, *real[1:])
        with pytest.raises(probe._ProbeError):
            probe._walk_tree(file_descriptor, root_identity, "purelib", set(), {})
    finally:
        os.close(file_descriptor)


def test_walk_tree_wraps_scandir_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _load_probe_module()
    root = tmp_path / "root"
    root.mkdir()
    file_descriptor = os.open(root, os.O_RDONLY)

    def boom(*args: object, **kwargs: object) -> object:
        raise OSError("boom")

    monkeypatch.setattr(probe.os, "scandir", boom)
    try:
        identity = probe._identity(os.fstat(file_descriptor))
        with pytest.raises(probe._ProbeError):
            probe._walk_tree(file_descriptor, identity, "purelib", set(), {})
    finally:
        os.close(file_descriptor)


def test_verify_expected_files_in_shared_root_confirms_members(tmp_path: Path) -> None:
    probe = _load_probe_module()
    root = tmp_path / "scripts"
    root.mkdir()
    (root / "spleeter").write_text("script\n", encoding="utf-8")
    file_descriptor = os.open(root, os.O_RDONLY)
    try:
        identity = probe._identity(os.fstat(file_descriptor))
        key = ("scripts", "spleeter")
        observed = probe._verify_expected_files_in_shared_root(
            file_descriptor,
            identity,
            "scripts",
            {key},
            {key: probe._identity(os.stat(root / "spleeter"))},
        )
        assert observed == {key}
    finally:
        os.close(file_descriptor)


def test_verify_expected_files_in_shared_root_rejects_non_regular(tmp_path: Path) -> None:
    probe = _load_probe_module()
    root = tmp_path / "scripts"
    root.mkdir()
    (root / "subdir").mkdir()
    file_descriptor = os.open(root, os.O_RDONLY)
    try:
        identity = probe._identity(os.fstat(file_descriptor))
        with pytest.raises(probe._ProbeError):
            probe._verify_expected_files_in_shared_root(
                file_descriptor, identity, "scripts", {("scripts", "subdir")}, {}
            )
    finally:
        os.close(file_descriptor)


def test_verify_expected_files_in_shared_root_rejects_changed_identity(tmp_path: Path) -> None:
    probe = _load_probe_module()
    root = tmp_path / "scripts"
    root.mkdir()
    (root / "spleeter").write_text("script\n", encoding="utf-8")
    file_descriptor = os.open(root, os.O_RDONLY)
    try:
        identity = probe._identity(os.fstat(file_descriptor))
        key = ("scripts", "spleeter")
        with pytest.raises(probe._ProbeError):
            probe._verify_expected_files_in_shared_root(
                file_descriptor, identity, "scripts", {key}, {key: (0, 0, 0, 0, 0, 0)}
            )
    finally:
        os.close(file_descriptor)


def test_verify_expected_files_in_shared_root_rejects_root_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    root = tmp_path / "scripts"
    root.mkdir()
    file_descriptor = os.open(root, os.O_RDONLY)
    real_identity = probe._identity
    calls = 0

    def shifting_identity(metadata: os.stat_result) -> tuple[int, ...]:
        nonlocal calls
        calls += 1
        base = real_identity(metadata)
        return (calls, *base[1:])

    monkeypatch.setattr(probe, "_identity", shifting_identity)
    try:
        real = real_identity(os.fstat(file_descriptor))
        root_identity = (1, *real[1:])
        with pytest.raises(probe._ProbeError):
            probe._verify_expected_files_in_shared_root(
                file_descriptor, root_identity, "scripts", set(), {}
            )
    finally:
        os.close(file_descriptor)


def test_verify_expected_files_in_shared_root_wraps_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    root = tmp_path / "scripts"
    root.mkdir()
    file_descriptor = os.open(root, os.O_RDONLY)

    def boom(fd: int) -> os.stat_result:
        raise OSError("boom")

    monkeypatch.setattr(probe.os, "fstat", boom)
    try:
        with pytest.raises(probe._ProbeError):
            probe._verify_expected_files_in_shared_root(
                file_descriptor, (0, 0, 0, 0, 0, 0), "scripts", set(), {}
            )
    finally:
        os.close(file_descriptor)


def test_verify_expected_files_in_shared_root_rejects_changed_root_identity(
    tmp_path: Path,
) -> None:
    probe = _load_probe_module()
    root = tmp_path / "scripts"
    root.mkdir()
    (root / "spleeter").write_text("script\n", encoding="utf-8")
    file_descriptor = os.open(root, os.O_RDONLY)
    try:
        with pytest.raises(probe._ProbeError):
            probe._verify_expected_files_in_shared_root(
                file_descriptor, (999, 999, 0, 0, 0, 0), "scripts", set(), {}
            )
    finally:
        os.close(file_descriptor)


def test_python_version_returns_dotted_version() -> None:
    probe = _load_probe_module()
    version = probe._python_version()
    assert version.count(".") == 2


def test_python_implementation_returns_cpython() -> None:
    probe = _load_probe_module()
    if probe.sys.implementation.name == "cpython":
        assert probe._python_implementation() == "CPython"


def test_python_implementation_maps_pypy(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(probe.sys, "implementation", _FakeImplementation("pypy"))
    assert probe._python_implementation() == "PyPy"


def test_python_implementation_returns_other_name(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(probe.sys, "implementation", _FakeImplementation("graalpy"))
    assert probe._python_implementation() == "graalpy"


def test_python_implementation_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(probe.sys, "implementation", _FakeImplementation(""))
    with pytest.raises(probe._ProbeError):
        probe._python_implementation()


def test_python_abi_returns_value() -> None:
    probe = _load_probe_module()
    assert probe._python_abi()


def test_python_abi_falls_back_to_cache_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(probe.sysconfig, "get_config_var", lambda name: None)
    monkeypatch.setattr(probe.sys, "implementation", _FakeImplementation("cpython", "cpython-312"))
    assert probe._python_abi() == "cpython-312"


def test_python_abi_rejects_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(probe.sysconfig, "get_config_var", lambda name: None)
    monkeypatch.setattr(probe.sys, "implementation", _FakeImplementation("cpython", None))
    with pytest.raises(probe._ProbeError):
        probe._python_abi()


def test_distribution_search_paths_returns_none_outside_venv(tmp_path: Path) -> None:
    probe = _load_probe_module()
    interpreter = tmp_path / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    assert probe._distribution_search_paths(interpreter, {}) is None


def test_distribution_search_paths_returns_purelib_paths(tmp_path: Path) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    probe = _load_probe_module()
    roots = probe._canonical_root_paths(interpreter)
    paths = probe._distribution_search_paths(interpreter, roots)
    assert paths is not None
    assert os.fspath(roots["purelib"]) in paths


def test_build_environment_manifest_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    probe = _load_probe_module()
    monkeypatch.setattr(probe.sys, "executable", str(interpreter))
    manifest = probe.build_environment_manifest()
    assert manifest["schema"] == "crux.separator-environment/v1"
    assert manifest["separator_id"] == "spleeter4-drums-v1"
    assert manifest["package_name"] == "spleeter"
    assert manifest["package_version"] == "2.4.2"
    assert isinstance(manifest["interpreter_sha256"], str)
    assert any(distribution["name"] == "spleeter" for distribution in manifest["distributions"])


def test_build_environment_manifest_uses_global_distributions_when_not_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    probe = _load_probe_module()
    monkeypatch.setattr(probe.sys, "executable", str(interpreter))
    monkeypatch.setattr(probe, "_distribution_search_paths", lambda *_: None)
    distributions = list(probe.importlib.metadata.distributions(path=[str(purelib)]))
    monkeypatch.setattr(
        probe.importlib.metadata, "distributions", lambda *args, **kwargs: distributions
    )
    manifest = probe.build_environment_manifest()
    assert manifest["separator_id"] == "spleeter4-drums-v1"


def test_build_environment_manifest_rejects_no_separator_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    shutil.rmtree(purelib / "spleeter-2.4.2.dist-info")
    shutil.rmtree(purelib / "spleeter")
    probe = _load_probe_module()
    monkeypatch.setattr(probe.sys, "executable", str(interpreter))
    with pytest.raises(probe._ProbeError):
        probe.build_environment_manifest()


def test_build_environment_manifest_rejects_ambiguous_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    _write_distribution(purelib, "demucs", "4.0.0")
    probe = _load_probe_module()
    monkeypatch.setattr(probe.sys, "executable", str(interpreter))
    with pytest.raises(probe._ProbeError):
        probe.build_environment_manifest()


def test_build_environment_manifest_rejects_duplicate_distribution_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    package_dir = purelib / "spleeter_dup"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("__version__ = '9.9.9'\n", encoding="utf-8")
    distribution_dir = purelib / "spleeter_dup-9.9.9.dist-info"
    distribution_dir.mkdir()
    (distribution_dir / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: spleeter\nVersion: 9.9.9\n", encoding="utf-8"
    )
    record_path = distribution_dir / "RECORD"
    init_path = package_dir / "__init__.py"
    rows = [
        [
            init_path.relative_to(purelib).as_posix(),
            f"sha256={_distribution_file_digest(init_path.read_bytes())}",
            str(len(init_path.read_bytes())),
        ],
        [record_path.relative_to(purelib).as_posix(), "", ""],
    ]
    with record_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    probe = _load_probe_module()
    monkeypatch.setattr(probe.sys, "executable", str(interpreter))
    with pytest.raises(probe._ProbeError):
        probe.build_environment_manifest()


def test_main_writes_canonical_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    probe = _load_probe_module()
    monkeypatch.setattr(probe.sys, "executable", str(interpreter))
    return_code = probe.main()
    assert return_code == 0
    captured = capsysbinary.readouterr()
    assert captured.out.endswith(b"\n")
    payload = strict_json_loads(captured.out[:-1], require_canonical=True)
    assert payload["separator_id"] == "spleeter4-drums-v1"


def test_main_reports_probe_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(
        probe,
        "build_environment_manifest",
        lambda: (_ for _ in ()).throw(probe._ProbeError("boom")),
    )
    return_code = probe.main()
    assert return_code == 1
    assert capsys.readouterr().out == ""


def test_main_reports_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(
        probe,
        "build_environment_manifest",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    return_code = probe.main()
    assert return_code == 1
    assert "separator_environment_probe_failed" in capsys.readouterr().err


def test_build_environment_manifest_verifies_shared_root_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter, purelib, console_script = _synthetic_environment_with_console_script(tmp_path)
    probe = _load_probe_module()
    monkeypatch.setattr(probe.sys, "executable", str(interpreter))
    manifest = probe.build_environment_manifest()
    distribution = next(d for d in manifest["distributions"] if d["name"] == "spleeter")
    files = {file["path"]: file for file in distribution["files"]}
    assert console_script.name in files
    assert files[console_script.name]["root"] == "scripts"


def test_module_entry_point_raises_systemexit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    monkeypatch.setattr(sys, "executable", str(interpreter))
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(PROBE_PATH), run_name="__main__")
    assert excinfo.value.code == 0
