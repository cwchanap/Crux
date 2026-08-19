# HPA-328 Portable Separator Runtime Attestation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Make each Spleeter and HTDemucs derived view attributable only when its resolved interpreter, installed package tree, and policy-owned model root match a canonical v2 separator lock, without charging the attestation work to per-item separator RTF.

**Architecture:** Promote the current v1 separator lock to a v2 lock plus a fixed sibling environment manifest. A standard-library probe, launched only by the isolated interpreter, produces the portable installed-distribution inventory; separators.py owns the one verifier that compares it and the model root to the lock. The pilot preflights that verifier once per separator before its first mutable snapshot, passes the resulting typed runtime to cache/resume/fresh paths, and rechecks model roots before publishing derived evidence.

**Tech Stack:** Python 3.13, stdlib importlib.metadata/sysconfig/venv/subprocess/os, dataclasses, existing canonical JSON and no-follow artifact helpers, Click, pytest, Ruff, Black, Pylint.

**Spec:** docs/superpowers/specs/2026-08-18-hpa-328-portable-runtime-attestation-design.md

## Global Constraints

- Replace crux.separator-lock/v1 with crux.separator-lock/v2; reject every v1 lock.
- A v2 lock contains interpreter_sha256, environment_manifest_sha256, model_root_kind, and the complete policy-owned model_files inventory. Its exact canonical bytes remain the cache and persisted provenance identity.
- The only environment manifest path is lock_path.parent / environment.json. It uses schema crux.separator-environment/v1 and contains hashes plus normalized relative paths, never an absolute host path.
- The environment manifest records separator/package identity, Python implementation/version/ABI/platform, the resolved interpreter hash, and every installed distribution with root-tagged relative files, byte lengths, and SHA-256 values, including each RECORD file.
- The standalone probe lives at src/benchmark/separator_environment_probe.py, imports only the Python standard library, emits canonical JSON to stdout, and is invoked through the resolved isolated interpreter. Neither freezer nor pilot reimplements discovery.
- Probe membership comes from RECORD only; hash actual opened bytes, hash RECORD itself, reject malformed/missing/unexpected non-bytecode files, injected .py/.pth/sitecustomize.py files, symlinks, traversal, unstable file identity, and noncanonical output. Ignore __pycache__ entries and *.pyc.
- Resolve the supplied interpreter to its final existing regular-file target once before hashing. It is the sole permitted symlink exception; package and model trees contain no parent or leaf symlink.
- _SEPARATOR_POLICIES stays closed and data-only. Named functions, not callables stored in policy data, own inventory and launch-environment logic.
- Spleeter 2.4.2 uses a dedicated MODEL_PATH root containing exactly 4stems/checkpoint, 4stems/.probe, 4stems/model.index, 4stems/model.data-00000-of-00001, and 4stems/model.meta.
- HTDemucs 4.1.0 uses a dedicated demucs-local-repo-v1 root containing exactly htdemucs.yaml and 955717e8-8726e21a.th. Its fixed argv contains --repo {model_root}; no Hugging Face cache layout is accepted.
- New attestation codes are exactly separator_lock_companion_mismatch, separator_interpreter_mismatch, separator_environment_mismatch, separator_model_root_invalid, and separator_environment_probe_failed.
- A successful pilot attests each separator exactly once after authoritative parent/source validation and before constructing or writing the first mutable run.json; this work is outside separator_wall_time_sec and separator_rtf.
- Fresh, cache-hit, and retained-stem paths receive an AttestedSeparatorRuntime, never raw interpreter/lock/model-root inputs. Model/interpreter locations must not appear in run identity, snapshots, comparison reports, or handoff rows.
- A preflight attestation failure exits 2 with one closed failure_code in OafSeparationPilotOutcome and CLI JSON, runs neither control nor derived loops, and publishes no mutable pilot snapshot.
- Re-inventory preflighted model roots after all derived execution and before derived scoring, view reports, comparison reports, or a successful final snapshot. On drift, restore per-view preimages, clear newly attributed derived evidence, suppress derived/report publication, and return fatal separator_model_root_invalid.
- The cleanup path rechecks model roots if a separator invocation was attempted and the pilot otherwise raises. Do not claim generic host-level network isolation for native Popen.
- Do not run a production separator, download a model, generate Task 11 locks, change persisted HPA-328 run/comparison/handoff schemas, add a database/plugin framework, or weaken provenance to requirements hashes or package name/version lists.
- All automated coverage uses synthetic virtual environments, synthetic model roots, fake Popen, and fake pilot seams. Task 11 remains blocked until its immutable upstream inputs exist.

## File Structure

| File | Responsibility |
| --- | --- |
| src/benchmark/separators.py | v2 lock and sibling-manifest types/loaders, policy-owned model inventory, environment-probe bridge, typed runtime, attestation, launch environment, and separator process invocation. |
| src/benchmark/separator_environment_probe.py | Standalone stdlib-only child program that inventories the interpreter environment and prints canonical manifest JSON. |
| scripts/freeze_separator_runtime.py | Freezes one supplied model root/interpreter into environment.json plus model.json, then round-trips through the shared attester. |
| src/benchmark/separation_pilot.py | Required model-root request fields, one-per-separator preflight, typed runtime handoff, fatal diagnostic carrier, and postflight restoration. |
| src/cli/benchmark.py | Required model-root options and canonical outcome failure_code output. |
| tests/benchmark/test_separators.py | Lock/companion/model-root/launch/freezer/attestation unit coverage using synthetic files and fake Popen. |
| tests/benchmark/test_separator_environment_probe.py | Real subprocess coverage for the stdlib probe in a synthetic venv. |
| tests/benchmark/test_separation_pilot.py | Pilot preflight ordering, typed-runtime handoff, resume, fatal, and postflight-recovery coverage. |
| tests/benchmark/test_separation_pilot_acceptance.py | Offline end-to-end invariant coverage for native failure semantics and nonpublication. |
| tests/test_cli_benchmark.py | Click option, request construction, exit-code, and failure_code payload coverage. |
| tests/fixtures/separators/spleeter/{model.json,environment.json} | Static canonical v2 Spleeter lock pair for loader-only tests. |
| tests/fixtures/separators/htdemucs/{model.json,environment.json} | Static canonical v2 HTDemucs lock pair for loader-only tests. |

---

### Task 1: Define canonical v2 lock and sibling-environment parsing

**Files:**

- Modify: src/benchmark/separators.py:30-310
- Modify: tests/benchmark/test_separators.py:1-205
- Create: tests/fixtures/separators/spleeter/model.json
- Create: tests/fixtures/separators/spleeter/environment.json
- Create: tests/fixtures/separators/htdemucs/model.json
- Create: tests/fixtures/separators/htdemucs/environment.json
- Delete after migration: tests/fixtures/separators/spleeter-model.json
- Delete after migration: tests/fixtures/separators/htdemucs-model.json
- Verify unchanged: src/benchmark/separation_pilot.py:160-163

**Interfaces:**

    SEPARATOR_LOCK_SCHEMA = "crux.separator-lock/v2"
    SEPARATOR_ENVIRONMENT_SCHEMA = "crux.separator-environment/v1"

    @dataclass(frozen=True)
    class SeparatorEnvironmentFile:
        root: str
        path: str
        byte_length: int
        sha256: str

    @dataclass(frozen=True)
    class SeparatorEnvironmentDistribution:
        name: str
        version: str
        files: tuple[SeparatorEnvironmentFile, ...]

    @dataclass(frozen=True)
    class SeparatorEnvironmentManifest:
        separator_id: str
        package_name: str
        package_version: str
        python_implementation: str
        python_version: str
        python_abi: str
        platform: str
        interpreter_sha256: str
        distributions: tuple[SeparatorEnvironmentDistribution, ...]
        sha256: str

    @dataclass(frozen=True)
    class SeparatorLock:
        separator_id: str
        repository_url: str
        repository_revision: str
        package_name: str
        package_version: str
        model_id: str
        model_files: tuple[SeparatorModelFile, ...]
        code_license: str
        model_license: str
        argv: tuple[str, ...]
        expected_drum_stem_relative_path: str
        output_container: str
        interpreter_sha256: str
        environment_manifest_sha256: str
        model_root_kind: str
        sha256: str

    load_separator_environment_manifest(
        lock_path: Path,
        lock: SeparatorLock,
    ) -> SeparatorEnvironmentManifest

    separator_environment_manifest_payload(
        manifest: SeparatorEnvironmentManifest,
    ) -> dict[str, object]

- Consumes: existing canonical_json_bytes(), strict_json_loads(), read_regular_file_no_follow(), require_sha256(), and the current fixed separator policy identity values.
- Produces: strict v2 loader/serializer primitives consumed by Tasks 3 and 4. Loader-only tests must not require a real interpreter, model root, or production runtime.

- [ ] **Step 1: Write red v2/sibling parsing tests and relocate fixture lookup**

    Update _fixture_path() so each separator fixture resolves to:

        FIXTURE_ROOT / "spleeter" / "model.json"
        FIXTURE_ROOT / "htdemucs" / "model.json"

    Add import shutil and an exact-key/sibling-binding test:

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
            sibling.write_bytes(b"{}\\n")
            with pytest.raises(SeparatorLockError, match="companion|environment"):
                load_separator_environment_manifest(lock_path, lock)

    Add a parameterized v1 rejection test that writes a formerly valid v1 payload and asserts load_separator_lock() raises SeparatorLockError with schema in the message. Extend the existing key-set assertion with interpreter_sha256, environment_manifest_sha256, and model_root_kind.

- [ ] **Step 2: Verify the tests fail**

    Run:

        uv run pytest tests/benchmark/test_separators.py -k "v2_lock_requires or v1" -q

    Expected: FAIL because the v1 schema and old lock shape are still active and no sibling loader exists.

- [ ] **Step 3: Implement v2 data validation and canonical sibling parsing**

    Replace the v1 schema/key constants. Add model_root_kind values to the existing data-only policy for both separator IDs, while leaving model-root inventory and HTDemucs --repo launch wiring to Task 3. Extend SeparatorLock and separator_lock_payload() with the three v2 fields. Validate all three as lowercase SHA-256 or policy-exact values:

        if lock.model_root_kind != policy["model_root_kind"]:
            raise SeparatorLockError("model_root_kind does not match separator model")
        _require_hash(lock.interpreter_sha256, "interpreter_sha256")
        _require_hash(lock.environment_manifest_sha256, "environment_manifest_sha256")

    Add strict dataclass validation for:

    - exact manifest and nested object key sets;
    - sorted unique distributions by normalized package name;
    - sorted unique file tuples by (root, path);
    - nonempty string metadata;
    - nonnegative integer byte_length that is not bool;
    - normalized slash-only relative paths and a closed root-tag vocabulary;
    - no absolute path, drive, backslash, empty, dot, or dot-dot component;
    - canonical single-final-newline bytes.

    load_separator_environment_manifest() must derive only:

        sibling_path = lock_path.parent / "environment.json"

    It reads no-follow bytes, verifies their SHA-256 against lock.environment_manifest_sha256 before parsing, requires the environment schema and matching separator/package/version/interpreter values, and returns a manifest carrying the exact sibling SHA. It never accepts an alternate sibling path argument.

- [ ] **Step 4: Install minimal canonical static fixture pairs**

    Create one directory per fixture lock so each model.json owns exactly one environment.json sibling. Use canonical JSON with deterministic synthetic hashes, policy-exact argv, model_root_kind, and model file names:

        Spleeter:
          4stems/checkpoint
          4stems/.probe
          4stems/model.index
          4stems/model.data-00000-of-00001
          4stems/model.meta

        HTDemucs:
          htdemucs.yaml
          955717e8-8726e21a.th

    Keep the HTDemucs argv at its currently accepted fixed value in this schema-only task. Task 3 atomically changes the policy and its static fixture lock to the --repo form, then recomputes environment_manifest_sha256 from the unchanged exact sibling bytes. Do not put a host root, interpreter location, cache path, or real model byte in a fixture.

- [ ] **Step 5: Run focused lock regressions**

    Run:

        uv run pytest tests/benchmark/test_separators.py -q

    Expected: PASS. Existing cache-key and process tests retain their current behavior while all loader tests use only v2 fixture pairs.

- [ ] **Step 6: Commit**

    Run:

        git add src/benchmark/separators.py tests/benchmark/test_separators.py tests/fixtures/separators
        git commit -m "feat: add v2 separator lock companion manifest"

---

### Task 2: Build the isolated standard-library environment probe

**Files:**

- Create: src/benchmark/separator_environment_probe.py
- Create: tests/benchmark/test_separator_environment_probe.py
- Verify unchanged: src/benchmark/separators.py

**Interfaces:**

    build_environment_manifest() -> dict[str, object]
    main() -> int

    # stdout on success:
    canonical_json_bytes(build_environment_manifest(), trailing_newline=True)

- Consumes: only Python standard library modules: csv, hashlib, importlib.metadata, json, os, pathlib, stat, sys, sysconfig, and typing/collections utilities.
- Produces: canonical crux.separator-environment/v1 JSON compatible with Task 1's loader. It imports no src.* module and writes diagnostics only to stderr.

- [ ] **Step 1: Write a synthetic isolated-environment fixture and red probe tests**

    In the new test file, create a real temporary venv without pip and add one synthetic installed distribution:

        def _synthetic_environment(
            tmp_path: Path,
            *,
            package_name: str = "spleeter",
            package_version: str = "2.4.2",
        ) -> tuple[Path, Path]:
            environment = tmp_path / "venv"
            venv.EnvBuilder(with_pip=False, symlinks=False).create(environment)
            interpreter = _venv_interpreter(environment)
            purelib = Path(
                subprocess.check_output(
                    [str(interpreter), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                    text=True,
                ).strip()
            )
            _write_distribution(purelib, package_name, package_version)
            return interpreter, purelib

    _write_distribution() creates package_name/__init__.py, package_name-<version>.dist-info/METADATA, and a CSV RECORD containing every regular file plus its own RECORD row. Run the probe as a child:

        result = subprocess.run(
            [str(interpreter), str(PROBE_PATH)],
            check=False,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr.decode()
        payload = strict_json_loads(result.stdout[:-1], require_canonical=True)
        assert canonical_json_bytes(payload, trailing_newline=True) == result.stdout

    Assert the result has no absolute path strings, contains the synthetic distribution, and records the METADATA and RECORD SHA-256 values from their actual bytes.

- [ ] **Step 2: Verify red**

    Run:

        uv run pytest tests/benchmark/test_separator_environment_probe.py -q

    Expected: FAIL because the probe executable does not exist.

- [ ] **Step 3: Implement descriptor-safe inventory in the standalone child**

    Implement these exact phases in build_environment_manifest():

    1. Resolve Path(sys.executable) with strict=True, open the final target no-follow, and SHA-256 its stable regular bytes.
    2. Build the allowed absolute installation-root map from sysconfig.get_paths(). Reject missing/non-directory roots and duplicate ambiguous root tags. Persist only a stable tag and the relative path; never persist the absolute root.
    3. Enumerate importlib.metadata.distributions(), read each dist-info RECORD as metadata, and reject duplicate normalized distribution names.
    4. For every non-bytecode RECORD member, reject absolute/backslash/dot/dot-dot paths; use descriptor-relative no-follow open plus fstat-before/fstat-after hashing. Serialize the bytes actually opened as root, path, byte_length, sha256.
    5. Add the distribution's RECORD itself even when its RECORD row has no digest. Verify every declared file is a regular file under exactly one allowed root.
    6. For every allowed root with declared files, descriptor-walk its tree. Ignore any __pycache__ subtree and *.pyc leaf. Reject every other regular file, any symlink, special file, path traversal, or observed identity change.
    7. Sort roots/files/distributions, compose the exact manifest object, and emit canonical JSON plus one newline.

    Do not inspect RECORD's quoted digest or size as truth. Use its paths solely to establish expected membership. A probe failure must exit nonzero and print one stable, non-host-path diagnostic token to stderr.

- [ ] **Step 4: Add the required mutation tests**

    Add these parameterized cases to the subprocess suite:

        @pytest.mark.parametrize(
            "mutation",
            (
                "missing_record_member",
                "record_content_changed",
                "record_self_changed",
                "extra_python",
                "extra_pth",
                "sitecustomize",
                "leaf_symlink",
                "parent_symlink",
            ),
        )
        def test_probe_rejects_distribution_tree_drift(
            tmp_path: Path,
            mutation: str,
        ) -> None:
            interpreter, purelib = _synthetic_environment(tmp_path)
            _mutate_distribution_tree(purelib, mutation)
            result = _run_probe(interpreter)
            assert result.returncode != 0
            assert result.stdout == b""

    Implement _mutate_distribution_tree() with this exact case table:

        mutations = {
            "missing_record_member": lambda purelib: (purelib / "spleeter" / "__init__.py").unlink(),
            "record_content_changed": lambda purelib: (purelib / "spleeter" / "__init__.py").write_text("changed\\n"),
            "record_self_changed": lambda purelib: (
                purelib / "spleeter-2.4.2.dist-info" / "RECORD"
            ).write_text("changed\\n"),
            "extra_python": lambda purelib: (purelib / "injected.py").write_text("x = 1\\n"),
            "extra_pth": lambda purelib: (purelib / "injected.pth").write_text("/tmp/escape\\n"),
            "sitecustomize": lambda purelib: (purelib / "sitecustomize.py").write_text("pass\\n"),
        }

    Implement the two symlink cases separately with Path.symlink_to() so leaf_symlink replaces package/__init__.py and parent_symlink replaces the package directory. For every case, mutate only the synthetic venv after _write_distribution(), rerun the child, and assert nonzero return code with empty stdout. Add a separate bytecode case that creates package/__pycache__/module.cpython-313.pyc and asserts the canonical manifest bytes are unchanged. Add a resolved-interpreter test using a symlink to the venv interpreter and assert the child manifest hash equals the final target's hash.

- [ ] **Step 5: Run the probe suite**

    Run:

        uv run pytest tests/benchmark/test_separator_environment_probe.py -q

    Expected: PASS using only a fresh synthetic venv and no third-party separator package execution.

- [ ] **Step 6: Commit**

    Run:

        git add src/benchmark/separator_environment_probe.py tests/benchmark/test_separator_environment_probe.py
        git commit -m "feat: probe isolated separator environments"

---

### Task 3: Make model roots and process launches policy-owned

**Files:**

- Modify: src/benchmark/separators.py:45-598
- Modify: tests/benchmark/test_separators.py:185-640
- Modify: tests/fixtures/separators/htdemucs/model.json
- Verify unchanged: tests/benchmark/test_separator_environment_probe.py

**Interfaces:**

    @dataclass(frozen=True)
    class AttestedSeparatorRuntime:
        interpreter: Path
        lock: SeparatorLock
        model_root: Path
        model_files: tuple[SeparatorModelFile, ...]
        environment: SeparatorEnvironmentManifest
        launch_environment: Mapping[str, str]

    inventory_separator_model_root(
        separator_id: str,
        model_root: Path,
    ) -> tuple[SeparatorModelFile, ...]

    revalidate_separator_model_root(runtime: AttestedSeparatorRuntime) -> None

    run_spleeter_drums(
        source_audio_path: Path,
        *,
        source_audio_sha256: str,
        source_duration_sec: float,
        runtime: AttestedSeparatorRuntime,
        cache_root: Path,
    ) -> SeparatedStem

    run_htdemucs_drums(
        source_audio_path: Path,
        *,
        source_audio_sha256: str,
        source_duration_sec: float,
        runtime: AttestedSeparatorRuntime,
        cache_root: Path,
    ) -> SeparatedStem

- Consumes: v2 policy fields and types from Task 1. Task 4 supplies verified runtimes; these functions must not invoke the probe themselves.
- Produces: fixed model inventory, argv, Popen environment, and runtime-only raw inputs that Task 5 passes through the pilot.

- [ ] **Step 1: Add red policy-root and Popen-contract tests**

    Add a fixture helper that creates exact synthetic roots with regular bytes and returns their expected SeparatorModelFile hashes:

        root = tmp_path / separator_id
        (root / "4stems").mkdir(parents=True)
        for relative_path, content in expected_files.items():
            destination = root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)

    Test both inventories equal the exact ordered expected names and reject an extra regular file and a symlink. Test HTDemucs argv and process environment:

        runtime = _attested_runtime(
            HTDEMUCS_SEPARATOR_ID,
            interpreter=Path("/isolated/demucs/python"),
            model_root=demucs_root,
        )
        argv = separators._render_separator_argv(
            runtime,
            input_path=tmp_path / "input.wav",
            output_dir=tmp_path / "output",
        )
        assert argv[0] == "/isolated/demucs/python"
        assert argv[argv.index("--repo") + 1] == str(demucs_root)

    Use _FakePopen to assert Popen receives env=runtime.launch_environment. For Spleeter assert an inherited MODEL_PATH is removed and replaced with str(spleeter_root). For Demucs assert inherited HF_HOME, HF_HUB_CACHE, TORCH_HOME, and any policy-listed endpoint/repository variables are absent or replaced only by policy data; no model root is stored in SeparatedStem or cache path.

- [ ] **Step 2: Verify red**

    Run:

        uv run pytest tests/benchmark/test_separators.py -k "model_root or launch_environment or repo" -q

    Expected: FAIL because callers still pass raw interpreter and lock_path and no policy-owned root inventory exists.

- [ ] **Step 3: Implement data-only policy metadata and exact inventories**

    Keep _SEPARATOR_POLICIES declarative. Retain the Task 1 model_root_kind values, replace the HTDemucs argv with the exact --repo {model_root} sequence from Global Constraints, and add only tuple/list environment discovery key names. Do not put Python callables in it.

    Add named dispatch functions. The Spleeter branch must accept exactly the five nested files from Global Constraints. The HTDemucs branch must accept exactly:

        ("htdemucs.yaml", "955717e8-8726e21a.th")

    Each branch uses no-follow opens and validates ordinary directories/regular files only. It returns files sorted by their portable relative name. Any missing, extra, special, parent/leaf symlink, unreadable, or changed file raises:

        SeparatorExecutionError(
            "separator_model_root_invalid",
            "separator model root does not match the fixed policy",
        )

    Update tests/fixtures/separators/htdemucs/model.json to the same fixed --repo argv and recompute only its environment_manifest_sha256 from the existing sibling bytes. This keeps the static loader pair internally consistent when Task 3 changes the policy command.

    revalidate_separator_model_root() reruns the same inventory and compares it exactly to runtime.lock.model_files; it never reads a cache or writes evidence.

- [ ] **Step 4: Convert runner and process boundaries to the typed runtime**

    Replace raw interpreter/lock_path parameters with runtime in the two public runner functions and _run_separator_drums(). Validate runtime.lock.separator_id against the requested runner. Derive cache identity only from runtime.lock.sha256 exactly as today.

    Change the renderer to substitute all three private placeholders:

        replacements = {
            "{input_wav}": os.fspath(input_path),
            "{output_dir}": os.fspath(output_dir),
            "{model_root}": os.fspath(runtime.model_root),
        }

    Change _run_separator_process() to accept env: Mapping[str, str] and call:

        subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )

    Build launch_environment from a copy of os.environ, remove only the policy-declared discovery keys, then set exact Spleeter MODEL_PATH. The Demucs fixed --repo path is the resolver boundary; do not add a claim or implementation of a generic network sandbox.

- [ ] **Step 5: Run focused separator regressions**

    Run:

        uv run pytest tests/benchmark/test_separators.py -q

    Expected: PASS, including existing timeout process-group, cache, QC, and immutable-publication regressions with the new runtime argument.

- [ ] **Step 6: Commit**

    Run:

        git add src/benchmark/separators.py tests/benchmark/test_separators.py tests/fixtures/separators/htdemucs/model.json
        git commit -m "feat: bind separator launches to model root policy"

---

### Task 4: Add the single live attester and freeze through it

**Files:**

- Modify: src/benchmark/separators.py
- Modify: scripts/freeze_separator_runtime.py
- Modify: tests/benchmark/test_separators.py
- Verify unchanged: src/benchmark/separator_environment_probe.py

**Interfaces:**

    ATTESTATION_FAILURE_CODES = frozenset(
        {
            "separator_lock_companion_mismatch",
            "separator_interpreter_mismatch",
            "separator_environment_mismatch",
            "separator_model_root_invalid",
            "separator_environment_probe_failed",
        }
    )

    _run_separator_environment_probe(
        interpreter: Path,
    ) -> SeparatorEnvironmentManifest

    attest_separator_runtime(
        lock_path: Path,
        interpreter: Path,
        model_root: Path,
    ) -> AttestedSeparatorRuntime

    freeze_separator_runtime(
        *,
        separator_id: str,
        interpreter: Path,
        model_root: Path,
        repository_revision: str,
        output: Path,
    ) -> SeparatorLock

- Consumes: Task 1 parsers, Task 2 child probe, and Task 3 model/root launch APIs.
- Produces: one nonpublishing live verifier used identically by freezer and pilot. Freezer writes only output.parent / environment.json and output.

- [ ] **Step 1: Write red attestation and freezer CLI tests**

    Use the synthetic venv from Task 2 and exact Spleeter root from Task 3. Freeze one pair, then assert:

        lock = freezer.freeze_separator_runtime(
            separator_id=SPLEETER_SEPARATOR_ID,
            interpreter=interpreter,
            model_root=model_root,
            repository_revision="a" * 40,
            output=lock_path,
        )
        runtime = attest_separator_runtime(lock_path, interpreter, model_root)
        assert runtime.lock == lock
        assert runtime.interpreter == interpreter.resolve(strict=True)
        assert runtime.model_files == lock.model_files
        assert (lock_path.parent / "environment.json").is_file()

    Add one parameterized mismatch test per required code:

        ("companion", "separator_lock_companion_mismatch")
        ("wrong_interpreter_hash", "separator_interpreter_mismatch")
        ("changed_recorded_package_file", "separator_environment_mismatch")
        ("changed_model_file", "separator_model_root_invalid")
        ("bad_probe_stdout", "separator_environment_probe_failed")

    Mutate one synthetic input after freeze. Assert attest_separator_runtime() raises SeparatorExecutionError whose code is exactly the expected code and does not create cache/stem/prediction/report files. For bad_probe_stdout, monkeypatch separators._run_separator_environment_probe to raise SeparatorExecutionError("separator_environment_probe_failed") rather than implementing a second probe in the test.

    Add parser coverage:

        result = freezer.main(
            [
                "--separator-id", SPLEETER_SEPARATOR_ID,
                "--interpreter", str(interpreter),
                "--model-root", str(model_root),
                "--repository-revision", "a" * 40,
                "--output", str(lock_path),
            ]
        )
        assert result == 0

    Assert --model-file is rejected by the parser and no independent environment output option exists.

- [ ] **Step 2: Verify red**

    Run:

        uv run pytest tests/benchmark/test_separators.py -k "attest or freeze" -q

    Expected: FAIL because no live attester exists and freezer still accepts model_files.

- [ ] **Step 3: Implement one verifier with closed error translation**

    attest_separator_runtime() performs exactly this sequence:

    1. load the v2 lock;
    2. load its derived fixed sibling and verify the sibling hash;
    3. resolve/hash the supplied interpreter and compare lock.interpreter_sha256;
    4. execute the standalone probe with that resolved interpreter, parse only canonical stdout, and compare the complete manifest to the sibling;
    5. inventory the supplied policy-owned root and compare it exactly to lock.model_files;
    6. construct AttestedSeparatorRuntime with the resolved interpreter, parsed lock/manifest, inventory, root, and closed launch environment.

    Map malformed/missing sibling data only to separator_lock_companion_mismatch; a changed resolved executable only to separator_interpreter_mismatch; a successful but unequal probe inventory only to separator_environment_mismatch; root-layout/content changes only to separator_model_root_invalid; probe start/nonzero/stderr/malformed/canonical-output failures only to separator_environment_probe_failed. Do not expose host paths in any public code or persisted object.

- [ ] **Step 4: Refactor freezer to publish the fixed pair and round-trip**

    Remove Mapping import, _parse_model_file(), --model-file, and _package_version(). Add required:

        parser.add_argument("--model-root", type=Path, required=True)

    In freeze_separator_runtime(), resolve the interpreter, execute the same probe, inventory the exact model root, compose canonical environment bytes, and publish:

        environment_path = output.parent / "environment.json"
        publish_immutable_file(environment_path, environment_bytes)
        publish_immutable_file(output, lock_bytes)

    Read the return value only from:

        attested = attest_separator_runtime(output, interpreter, model_root)
        if attested.lock.separator_id != separator_id:
            raise FreezeError("published separator lock did not round-trip")
        return attested.lock

    The freezer does not install packages, resolve remote model files, or run a separator.

- [ ] **Step 5: Run attester/freezer and probe coverage**

    Run:

        uv run pytest tests/benchmark/test_separator_environment_probe.py tests/benchmark/test_separators.py -q

    Expected: PASS with all five closed error codes covered from actual synthetic inputs.

- [ ] **Step 6: Commit**

    Run:

        git add src/benchmark/separators.py scripts/freeze_separator_runtime.py tests/benchmark/test_separators.py
        git commit -m "feat: attest frozen separator runtimes"

---

### Task 5: Preflight both separator runtimes before a pilot snapshot

**Files:**

- Modify: src/benchmark/separation_pilot.py:160-269, 1215-1720, 1749-2033
- Modify: src/cli/benchmark.py:1315-1412
- Modify: tests/benchmark/test_separation_pilot.py
- Modify: tests/benchmark/test_separation_pilot_acceptance.py
- Modify: tests/benchmark/test_separation_comparison.py
- Modify: tests/benchmark/test_separation_handoff.py
- Modify: tests/test_cli_benchmark.py:840-1010

**Interfaces:**

    @dataclass(frozen=True)
    class OafSeparationPilotRequest:
        reference_manifest_path: Path
        timing_manifest_path: Path
        subset_manifest_path: Path
        oaf_run_path: Path
        cache_dir: Path
        output_dir: Path
        spleeter_python: Path
        demucs_python: Path
        spleeter_model_root: Path
        demucs_model_root: Path
        resume: bool = False
        crux_commit: str | None = None

    @dataclass(frozen=True)
    class OafSeparationPilotOutcome:
        overall_status: PilotStatus
        exit_code: PilotExitCode
        run_id: str | None
        run_path: Path | None
        reports_path: Path | None
        full_mix_reports_path: Path | None
        success_count: int
        failed_count: int
        skipped_count: int
        quarantined_count: int
        failure_code: str | None

    _fatal_outcome(
        failure_code: str | None = None,
    ) -> OafSeparationPilotOutcome

- Consumes: AttestedSeparatorRuntime and attest_separator_runtime() from Task 4.
- Produces: one runtime per separator, no per-item environment scan, typed runner handoff, and a native fatal diagnostic that only appears in outcome/CLI JSON.

- [ ] **Step 1: Update shared fake seams and add red preflight ordering tests**

    Extend _request() with separate temporary roots:

        spleeter_model_root=tmp_path / "spleeter-model-root"
        demucs_model_root=tmp_path / "demucs-model-root"

    Extend _task6_seams() to monkeypatch pilot.attest_separator_runtime with a fake that returns an actual AttestedSeparatorRuntime built from the static v2 fixture lock/manifest, records separator IDs, and does not run the probe. Update fake runners to receive runtime=runtime rather than interpreter/lock_path.

    Add:

        def test_pilot_attests_each_separator_once_before_snapshot_or_rtf(
            tmp_path: Path,
            monkeypatch: pytest.MonkeyPatch,
        ) -> None:
            import src.benchmark.separation_pilot as pilot
            from src.benchmark.separation_pilot import run_oaf_separation_pilot

            fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
            subset = _subset_path(tmp_path, fixture)
            request = _request(tmp_path, fixture, subset)
            calls = _task6_seams(tmp_path, fixture, monkeypatch)
            monkeypatch.setattr(
                pilot,
                "write_oaf_separation_run",
                _recording_snapshot_writer(calls["events"]),
            )
            run_oaf_separation_pilot(
                request,
                backend_factory=calls["factory"][0],
                perf_counter=_recording_perf_counter(calls["events"]),
            )
            assert calls["attest"] == [SPLEETER_SEPARATOR_ID, HTDEMUCS_SEPARATOR_ID]
            assert calls["events"].index("attest:htdemucs-drums-v1") < calls["events"].index("write")
            assert calls["events"].index("attest:htdemucs-drums-v1") < calls["events"].index("perf")

    The test must record write_oaf_separation_run and perf_counter events, prove neither is called before both preflights, and prove no extra attester call occurs across 20 fresh views or a resume cache-hit run.

    Add a fatal case that makes the Spleeter attester raise:

        SeparatorExecutionError("separator_environment_mismatch")

    Assert exit_code is 2, outcome.failure_code is separator_environment_mismatch, no run.json exists below output_dir, no full-mix scorer or separator runner is called, and the HTDemucs attester is not called after the first failure.

- [ ] **Step 2: Verify red**

    Run:

        uv run pytest tests/benchmark/test_separation_pilot.py -k "attest or request_exposes" -q

    Expected: FAIL because the request lacks model roots, no attester is called, and fatal outcomes have no failure_code.

- [ ] **Step 3: Move preflight before snapshot construction and pass typed runtimes**

    Add required Path validation for spleeter_model_root and demucs_model_root. After current authoritative parent, source, lock, descriptor, frozen OaF binding, and view-config validation but before _build_snapshot(), call:

        runtimes = {
            SPLEETER_SEPARATOR_ID: attest_separator_runtime(
                SEPARATOR_LOCK_PATHS[SPLEETER_SEPARATOR_ID],
                request.spleeter_python,
                request.spleeter_model_root,
            ),
            HTDEMUCS_SEPARATOR_ID: attest_separator_runtime(
                SEPARATOR_LOCK_PATHS[HTDEMUCS_SEPARATOR_ID],
                request.demucs_python,
                request.demucs_model_root,
            ),
        }

    Do not construct a snapshot or call write_oaf_separation_run before that block succeeds. On an attestation-code SeparatorExecutionError, return _fatal_outcome(error.code); on unrelated fatal preflight, return _fatal_outcome() as before.

    Change _execute_derived_view() to receive:

        runtime: AttestedSeparatorRuntime

    Use runtime.lock for retained-stem checks and pass runtime to the fresh runner. It must not call the attester, read a cached stem before runtime exists, or accept raw interpreter/lock inputs.

- [ ] **Step 4: Add the outcome/CLI diagnostic carrier**

    Add failure_code to OafSeparationPilotOutcome. Validate None or one ATTESTATION_FAILURE_CODES member; all normal and unrelated fatal outcomes use None. Add:

        @click.option(
            "--spleeter-model-root",
            type=click.Path(path_type=Path, file_okay=False),
            required=True,
        )
        @click.option(
            "--demucs-model-root",
            type=click.Path(path_type=Path, file_okay=False),
            required=True,
        )

    Include both roots in OafSeparationPilotRequest construction. Include only:

        "failure_code": outcome.failure_code

    in the canonical CLI payload. Do not add it to run.json, comparison, or handoff data. Update both fake outcomes in tests/test_cli_benchmark.py and assert normal output contains null while a synthetic fatal code is preserved exactly.

- [ ] **Step 5: Run pilot and CLI consumer regressions**

    Run:

        uv run pytest tests/benchmark/test_separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py tests/benchmark/test_separation_comparison.py tests/benchmark/test_separation_handoff.py tests/test_cli_benchmark.py -q

    Expected: PASS. Existing full-mix, derived partial, comparison, and handoff semantics remain unchanged after a successful preflight.

- [ ] **Step 6: Commit**

    Run:

        git add src/benchmark/separation_pilot.py src/cli/benchmark.py tests/benchmark/test_separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py tests/benchmark/test_separation_comparison.py tests/benchmark/test_separation_handoff.py tests/test_cli_benchmark.py
        git commit -m "feat: preflight separator runtime attestation"

---

### Task 6: Hold derived evidence provisional until postflight root validation

**Files:**

- Modify: src/benchmark/separation_pilot.py:1385-2033
- Modify: tests/benchmark/test_separation_pilot.py
- Modify: tests/benchmark/test_separation_pilot_acceptance.py
- Modify: tests/benchmark/test_separation_comparison.py

**Interfaces:**

    _capture_derived_view_preimages(
        snapshot: Mapping[str, object],
    ) -> dict[tuple[int, str], dict[str, object]]

    _restore_derived_view_preimages(
        snapshot: dict[str, object],
        preimages: Mapping[tuple[int, str], Mapping[str, object]],
    ) -> None

- Consumes: revalidate_separator_model_root(runtime) from Task 3 and the successful-preflight runtimes from Task 5.
- Produces: a restored, non-attributed mutable ledger on model-root drift and no newly published derived reports/comparison after that drift.

- [ ] **Step 1: Add red postflight/nonpublication tests**

    Add a fake runner that creates a normal synthetic stem and then writes an extra regular file into its supplied model root. Use the real or a targeted fake revalidator so it raises separator_model_root_invalid after the derived loops.

    Record calls to _score_derived_cohort and compare_oaf_separation:

        outcome = run_oaf_separation_pilot(
            request,
            backend_factory=calls["factory"][0],
        )

        assert outcome.exit_code == 2
        assert outcome.failure_code == "separator_model_root_invalid"
        assert not derived_score_calls
        assert not comparison_calls

    Inspect the mutable snapshot. For a fresh run, both derived views for all rows must be their original pending preimages, with no stem/input/prediction/runtime field newly retained. The synthetic immutable cache stem may still exist but has no snapshot reference.

    Add a resume case: start from a valid completed synthetic run, deliberately remove one retained prediction so resume would construct a new derived view, force model-root postflight failure, and assert every preexisting valid view dictionary is restored byte-for-byte from the prior snapshot rather than deleted or converted to prediction_output_conflict.

    Add a cleanup case where a fake fresh runner raises KeyboardInterrupt after its invocation begins. Assert revalidate_separator_model_root() is called from cleanup before the exception leaves run_oaf_separation_pilot().

- [ ] **Step 2: Verify red**

    Run:

        uv run pytest tests/benchmark/test_separation_pilot.py -k "postflight or provisional or cleanup" -q

    Expected: FAIL because derived reports are currently scored before any model-root recheck and mutated rows persist.

- [ ] **Step 3: Capture preimages, track attempts, and postflight before report work**

    Immediately after successful preflight and snapshot construction/recovery, deep-copy only each derived view into a map keyed by:

        (simfile_id, view_name)

    Before every fresh separator_runner() call, set a shared separator_invocation_attempted flag. Retained-stem-only resume reads do not set it.

    After all loops and any existing stop disposition handling, but before either _score_derived_cohort(), _comparison_reports_ready(), compare_oaf_separation(), or the normal complete/partial final write, run:

        for runtime in runtimes.values():
            revalidate_separator_model_root(runtime)

    On separator_model_root_invalid:

    1. restore the deep-copied derived views;
    2. set snapshot["overall_status"] = "failed";
    3. write only that restored non-attributed mutable snapshot for future resume;
    4. return _fatal_outcome("separator_model_root_invalid");
    5. do not call derived scoring or comparison.

    Do not delete immutable cache or prediction bytes. Do not remove prior valid evidence that existed before this invocation.

- [ ] **Step 4: Cover exceptional cleanup without masking the original failure**

    Initialize runtime/preimage/attempt state before the outer try boundary. In the outer cleanup route, if separator_invocation_attempted is true and normal postflight did not finish, call the same revalidator before letting the original exception or ordinary fatal path proceed.

    If cleanup discovers separator_model_root_invalid and a snapshot exists, restore its preimages and persist only that restored state. Preserve an already-raised non-attestation exception rather than replacing it; when the pilot would otherwise return a normal fatal outcome, attach the postflight native code through _fatal_outcome().

    Keep backend closing in the existing finally path. Do not introduce a second backend lifecycle or retry a poison/fatal OaF backend.

- [ ] **Step 5: Run postflight plus comparison regressions**

    Run:

        uv run pytest tests/benchmark/test_separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py tests/benchmark/test_separation_comparison.py -q

    Expected: PASS. Successful runs still publish reports/comparisons; a postflight root change publishes neither new derived report nor comparison and preserves resume recovery.

- [ ] **Step 6: Commit**

    Run:

        git add src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py tests/benchmark/test_separation_comparison.py
        git commit -m "fix: hold separator evidence pending postflight"

---

### Task 7: Prove portable-output and downstream-schema preservation

**Files:**

- Modify: tests/benchmark/test_separation_pilot_acceptance.py
- Modify: tests/benchmark/test_separation_handoff.py
- Modify: tests/test_cli_benchmark.py
- Modify: docs/superpowers/specs/2026-08-18-hpa-328-portable-runtime-attestation-design.md
- Verify unchanged: src/benchmark/separation_comparison.py
- Verify unchanged: src/benchmark/separation_handoff.py

**Interfaces:**

    # Persisted schemas remain unchanged:
    SEPARATION_RUN_SCHEMA == "crux.oaf-separation-run/v1"
    # failure_code is only OafSeparationPilotOutcome/CLI diagnostic data.

- Consumes: the completed v2 lock, attester, pilot, CLI, and postflight paths from Tasks 1-6.
- Produces: an offline acceptance proof that host-local runtime inputs do not escape into persistent HPA-328 artifacts and existing finalizer consumers still work.

- [ ] **Step 1: Add a red portable-output regression**

    Use a successful synthetic pilot where request.spleeter_python, request.demucs_python, request.spleeter_model_root, and request.demucs_model_root have distinct absolute temporary paths. After completion, read run.json, every generated derived report, comparison JSON, and the finalized HPA-396 handoff manifest:

        forbidden = (
            str(request.spleeter_python),
            str(request.demucs_python),
            str(request.spleeter_model_root),
            str(request.demucs_model_root),
        )
        for content in published_bytes:
            assert all(value.encode() not in content for value in forbidden)

    Assert the run schema remains v1, no top-level run failure_code was added, the CLI normal payload contains failure_code: null, and a fatal CLI fixture exposes only the closed native code with no exception text/path.

- [ ] **Step 2: Verify red**

    Run:

        uv run pytest tests/benchmark/test_separation_pilot_acceptance.py tests/benchmark/test_separation_handoff.py tests/test_cli_benchmark.py -k "portable or failure_code or separation" -q

    Expected: FAIL until the complete implementation consistently keeps runtime inputs out of persisted output and adds the CLI carrier.

- [ ] **Step 3: Add only assertions and status documentation**

    Do not change a persisted schema or alter source reports in this task. Make the test enumerate only files actually produced by the successful synthetic run, comparison, and handoff fixture. Update the spec Status section to state that code implementation is complete only after all verification in Step 4 passes; retain the explicit Task 11 operational-input hold.

- [ ] **Step 4: Run the focused stack and static checks**

    Run:

        uv run pytest tests/benchmark/test_separator_environment_probe.py tests/benchmark/test_separators.py tests/benchmark/test_separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py tests/benchmark/test_separation_comparison.py tests/benchmark/test_separation_handoff.py tests/test_cli_benchmark.py -q
        uv run ruff check src/benchmark/separators.py src/benchmark/separator_environment_probe.py src/benchmark/separation_pilot.py src/cli/benchmark.py scripts/freeze_separator_runtime.py tests/benchmark/test_separators.py tests/benchmark/test_separator_environment_probe.py tests/benchmark/test_separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py tests/test_cli_benchmark.py
        uv run black --check src/benchmark/separators.py src/benchmark/separator_environment_probe.py src/benchmark/separation_pilot.py src/cli/benchmark.py scripts/freeze_separator_runtime.py tests/benchmark/test_separators.py tests/benchmark/test_separator_environment_probe.py tests/benchmark/test_separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py tests/test_cli_benchmark.py
        uv run pylint src/benchmark/separators.py src/benchmark/separation_pilot.py src/cli/benchmark.py scripts/freeze_separator_runtime.py

    Expected: PASS. If a formatter changes files, apply it, rerun the affected focused tests, then rerun the formatter check.

- [ ] **Step 5: Run whole-suite verification**

    Run:

        uv run pytest

    Expected: PASS. Record any environment-only test isolation issue separately from product failures; do not weaken lazy-import or runtime-attestation tests to make an order-dependent suite pass.

- [ ] **Step 6: Commit**

    Run:

        git add docs/superpowers/specs/2026-08-18-hpa-328-portable-runtime-attestation-design.md tests/benchmark/test_separation_pilot_acceptance.py tests/benchmark/test_separation_handoff.py tests/test_cli_benchmark.py
        git commit -m "test: verify portable separator attestation output"

## Subagent-Driven Execution Gates

1. Use this plan and its spec as the only task authority. Before Task 1, create the plan-scoped SDD workspace and ledger, record the merge base, scan all shared files/interfaces, and record any ruling in the ledger.
2. Dispatch one fresh implementation worker for each task in sequence. Do not run implementation workers in parallel because Tasks 2-7 consume prior interfaces and several share separators.py or separation_pilot.py.
3. Each worker writes its task report and commits its work. Run the SDD review-package script for that task's base-to-head range, then dispatch a task-scoped reviewer for spec compliance and code quality.
4. Resolve every Critical/Important task-review finding through the SDD fix/re-review loop before beginning the next task. Preserve any ruling or deferred minor in the ledger.
5. After Task 7, run the whole-branch review against the merge base with the highest-capability reviewer. If it finds issues, dispatch one consolidated fix worker and exactly one scoped re-review as required by the SDD protocol.
6. Do not generate production locks or run Task 11 during any automated task. The final handoff must state that Task 11 remains blocked by unavailable immutable upstream inputs.
