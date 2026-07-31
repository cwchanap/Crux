# HPA-481/HPA-482 Work-Job Artifact Attestation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace HPA-320's cross-runner native evidence with same-job GitHub-hosted evidence, deterministic work artifacts, signed provenance, and strict outside-in acceptance for bootstrap, measurement, and candidate production.

**Architecture:** Each native phase becomes one GitHub-hosted work job that validates its own runner, produces a strict v2 host bundle and phase payload, emits a canonical manifest plus deterministic ustar archive, and signs both subjects with a pinned GitHub attestation action. Shared Python code owns schema parsing, no-follow publication, phase allowlists, canonical packing, and GitHub CLI verification; workflow YAML only orders those reviewed operations. The migration is atomic across all v2 containing schemas and ends by regenerating the source/build-context identities and reissuing the still-pre-dispatch bootstrap request.

**Tech Stack:** Python 3.13, pytest, canonical JSON/SHA-256, POSIX descriptor APIs, manual POSIX ustar serialization, GitHub Actions YAML, `actions/attest` v4.2.1, GitHub CLI artifact-attestation verification, UV, Ruff, Black, and Pylint.

## Global Constraints

- Production accepts only `runner.environment == "github-hosted"`, `runner.os == "Linux"`, `runner.arch == "X64"`, `uname -m == "x86_64"`, Docker `OSType == "linux"`, and Docker `Architecture == "x86_64"`.
- The three exact job keys are `native-bootstrap`, `native-measurement`, and `native-candidate`; `GITHUB_JOB` must equal the phase's key.
- `RUNNER_ENVIRONMENT_CONTEXT`, `RUNNER_OS_CONTEXT`, `RUNNER_ARCH_CONTEXT`, and `WORKFLOW_SOURCE_SHA` come only from the observation step's YAML `env:` mapping. Do not assign or override `RUNNER_ENVIRONMENT`, `RUNNER_OS`, or `RUNNER_ARCH`.
- `github.workflow_sha == GITHUB_SHA == checked-out HEAD == COMMIT_SHA`; divergent workflow and source commits are rejected.
- Production performs no GitHub Jobs REST request, uses no `GH_TOKEN`, writes no host observation to `$GITHUB_OUTPUT`, and grants no `actions: read`.
- New host authority is `crux.oaf-native-host-attestation-bundle/v2` containing exactly three sibling files: `attestation-bundle.json`, `native-host-evidence.json`, and `native-host-observation.json`.
- `kind == "github_hosted"` accepts only `crux.github-hosted-native-evidence/v2`; there is no v1 production compatibility reader or inferred `runner_environment`.
- The containing evidence schemas move atomically to `crux.backend-seal-evidence/v2`, `crux.oaf-base-system-package-evidence/v2`, `crux.oaf-calibration-bootstrap-evidence/v2`, `crux.oaf-calibration-measurement-evidence/v2`, and `crux.oaf-seal-candidate/v2`.
- Tasks 1–3 are one non-dispatchable schema-migration sequence. Their focused RED/GREEN commits are review checkpoints only; do not publish native authority from an intermediate commit, and require the complete migrated boundary suite to pass in Task 3 before continuing.
- The v2 seal candidate and native work manifests contain no `native_host_api_record` role and no `github-job-api-record.json.hex` path.
- The work manifest schema is `crux.oaf-native-work-artifact-manifest/v1`; `status` is exactly `success`, rows are strict and sorted by UTF-8 bytes of `(role, path)`, and every payload file must match one immutable phase `(path, role)` mapping.
- The manifest, sibling archive, and sibling Sigstore bundle never list themselves in a manifest row. The Sigstore bundle remains detached from both signed subjects.
- The archive is uncompressed POSIX ustar with one root `hpa320-native-<phase>-<workflow_commit>/`, UTF-8 byte-sorted members, file mode `0644`, directory mode `0755`, UID/GID `0`, empty owner/group names, mtime `0`, and no PAX, symlink, hard-link, device, socket, or FIFO member.
- Hash and size every payload or archive file through no-follow descriptors in bounded chunks; reject symlinked roots, pre/post-`fstat` identity drift, unsafe paths, and non-regular members.
- GitHub Actions uses `actions/attest@508db95dd578ae2727ebd6217d5ba78e4fbda05d` (`v4.2.1`) with default SLSA provenance and exactly two explicit subjects.
- Workflow permissions are exactly `contents: read`, `id-token: write`, and `attestations: write`; omit `push-to-registry`, `create-storage-record`, and `artifact-metadata: write`.
- Acceptance supports `2.68.0 <= gh < 3.0.0` and runs `gh attestation verify` separately for the manifest and archive with the same local bundle, trusted root, exact repository, workflow, source commit, signer commit, SLSA predicate, GitHub issuer, SHA-256 algorithm, and `--deny-self-hosted-runners`.
- Both GitHub CLI invocations must return the same verified statement, and that statement must name both expected SHA-256 subjects.
- Each phase has exactly one success upload named `hpa320-native-<phase>-<workflow_commit>`, three literal non-overlapping paths, `if-no-files-found: error`, and `retention-days: 30`.
- A failure upload, when present, is named `hpa320-native-<phase>-diagnostic-<workflow_commit>-run-<github_run_id>-attempt-<github_run_attempt>`, contains only allowlisted sanitized diagnostics, and never contains a success manifest, archive, or Sigstore bundle.
- Keep request schema identities at `/v1`; after all covered source bytes are final, regenerate the runner source manifest, regenerate the build-context manifest from the exact authenticated wheelhouse, and then reissue the calibration-bootstrap request with the new cross-hashes.
- The vendored upstream source manifest, checkpoint request, base-system request, distribution-build manifest, and instrumentation patch remain byte-identical unless an independently frozen input actually changes.
- Prefix every shell command segment with `rtk`, preserve unrelated worktree changes, and use Conventional Commit subjects under 72 characters.

---

### Task 1: Strict v2 GitHub-hosted Evidence

**Files:**
- Modify: `src/benchmark/backend_process.py`
- Modify: `tools/hpa320/github_host_evidence.py`
- Modify: `tests/benchmark/test_backend_process.py`
- Modify: `tests/benchmark/test_hpa320_github_host_evidence.py`

**Interfaces:**
- Consumes: `HostNumericFingerprint`, `JsonValue`, `canonical_json_bytes(value, *, trailing_newline=False)`, and `sha256_hex(content)`.
- Produces:

```python
def build_github_hosted_evidence(
    *,
    github_job: str,
    github_repository: str,
    github_run_attempt: int,
    github_run_id: int,
    github_workflow_ref: str,
    github_workflow_sha: str,
    host_numeric_fingerprint: HostNumericFingerprint,
    run_url: str,
    runner_arch: str,
    runner_environment: str,
    runner_os: str,
    workflow_commit: str,
) -> dict[str, JsonValue]:
    """Build and self-validate one strict v2 GitHub-hosted evidence record."""
```

- `NativeHostEvidence` with `kind="github_hosted"` accepts only the exact v2 payload produced above, verifies its canonical digest, and exposes `host_numeric_fingerprint`.

- [ ] **Step 1: Add failing parser tests for the exact v2 payload and run-level URL**

Add an exact success record and parameterized negative mutations in `tests/benchmark/test_backend_process.py`:

```python
GITHUB_V2_PAYLOAD = {
    "schema": "crux.github-hosted-native-evidence/v2",
    "github_job": "native-bootstrap",
    "github_repository": "cwchanap/Crux",
    "github_run_attempt": 2,
    "github_run_id": 123456789,
    "github_workflow_ref": (
        "cwchanap/Crux/.github/workflows/hpa320-native-bootstrap.yml@refs/tags/native-v2"
    ),
    "github_workflow_sha": "a" * 40,
    "host_numeric_fingerprint": HOST_NUMERIC_FINGERPRINT,
    "run_url": "https://github.com/cwchanap/Crux/actions/runs/123456789",
    "runner_arch": "X64",
    "runner_environment": "github-hosted",
    "runner_os": "Linux",
    "workflow_commit": "a" * 40,
}


def github_v2_record(payload: dict[str, object] | None = None) -> dict[str, object]:
    selected = GITHUB_V2_PAYLOAD if payload is None else payload
    return {
        "kind": "github_hosted",
        "official_execution_allowed": True,
        "payload": selected,
        "sha256": sha256_hex(canonical_json_bytes(selected)),
    }


def test_github_hosted_native_evidence_accepts_exact_v2_run_identity() -> None:
    record = github_v2_record()
    evidence = NativeHostEvidence(**record)
    assert evidence.payload == GITHUB_V2_PAYLOAD
    assert evidence.host_numeric_fingerprint.as_json() == HOST_NUMERIC_FINGERPRINT


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "crux.github-hosted-native-evidence/v1"),
        ("runner_environment", "self-hosted"),
        ("runner_environment", ""),
        ("runner_os", "Windows"),
        ("runner_arch", "ARM64"),
        ("github_job", ""),
        ("github_run_attempt", 0),
        ("github_run_id", 0),
        ("run_url", "https://github.com/cwchanap/Crux/actions/runs/123456789/job/7"),
        ("run_url", "https://github.com/other/Crux/actions/runs/123456789"),
        ("run_url", "https://github.com/cwchanap/Crux/actions/runs/9"),
    ],
)
def test_github_hosted_native_evidence_rejects_invalid_v2_identity(
    field: str,
    value: object,
) -> None:
    payload = dict(GITHUB_V2_PAYLOAD)
    payload[field] = value
    with pytest.raises(ValueError):
        NativeHostEvidence(**github_v2_record(payload))
```

Also retain one named negative fixture using the old exact v1 key set and assert it is rejected rather than upgraded.

- [ ] **Step 2: Run the parser tests to verify RED**

Run:

```bash
rtk uv run pytest \
  tests/benchmark/test_backend_process.py::test_github_hosted_native_evidence_accepts_exact_v2_run_identity \
  tests/benchmark/test_backend_process.py::test_github_hosted_native_evidence_rejects_invalid_v2_identity \
  -v
```

Expected: FAIL because the current parser still requires `api_record_sha256`, `approved_labels`, `job_id`, and a job-level URL.

- [ ] **Step 3: Replace the GitHub-hosted parser contract**

In `src/benchmark/backend_process.py`, replace the v1 GitHub payload key set and URL expression with:

```python
_GITHUB_HOSTED_SCHEMA = "crux.github-hosted-native-evidence/v2"
_GITHUB_HOSTED_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "github_job",
        "github_repository",
        "github_run_attempt",
        "github_run_id",
        "github_workflow_ref",
        "github_workflow_sha",
        "host_numeric_fingerprint",
        "run_url",
        "runner_arch",
        "runner_environment",
        "runner_os",
        "workflow_commit",
    }
)
_GITHUB_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_LOWERCASE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _validate_github_hosted_payload(payload: Mapping[str, JsonValue]) -> None:
    if set(payload) != _GITHUB_HOSTED_PAYLOAD_KEYS:
        raise ValueError("github_hosted native evidence must contain the exact v2 key set")
    repository = payload["github_repository"]
    run_id = payload["github_run_id"]
    run_attempt = payload["github_run_attempt"]
    workflow_commit = payload["workflow_commit"]
    workflow_sha = payload["github_workflow_sha"]
    if (
        payload["schema"] != _GITHUB_HOSTED_SCHEMA
        or not isinstance(repository, str)
        or _GITHUB_REPOSITORY.fullmatch(repository) is None
        or not isinstance(run_id, int)
        or isinstance(run_id, bool)
        or run_id <= 0
        or not isinstance(run_attempt, int)
        or isinstance(run_attempt, bool)
        or run_attempt <= 0
        or payload["run_url"] != f"https://github.com/{repository}/actions/runs/{run_id}"
        or not isinstance(payload["github_job"], str)
        or not payload["github_job"]
        or not isinstance(payload["github_workflow_ref"], str)
        or not payload["github_workflow_ref"]
        or not isinstance(workflow_commit, str)
        or _LOWERCASE_COMMIT.fullmatch(workflow_commit) is None
        or not isinstance(workflow_sha, str)
        or _LOWERCASE_COMMIT.fullmatch(workflow_sha) is None
        or workflow_sha != workflow_commit
        or payload["runner_environment"] != "github-hosted"
        or payload["runner_os"] != "Linux"
        or payload["runner_arch"] != "X64"
    ):
        raise ValueError("github_hosted native evidence identity is invalid")
    parse_host_numeric_fingerprint(payload["host_numeric_fingerprint"])
```

Keep the existing wrapper validation: exact wrapper keys, `official_execution_allowed is True`, and `sha256 == sha256_hex(canonical_json_bytes(payload))`.

- [ ] **Step 4: Add failing builder tests**

Replace the API-record-oriented tests in `tests/benchmark/test_hpa320_github_host_evidence.py` with:

```python
def test_builder_emits_the_exact_v2_record() -> None:
    evidence = build_github_hosted_evidence(
        github_job="native-bootstrap",
        github_repository="cwchanap/Crux",
        github_run_attempt=2,
        github_run_id=123456789,
        github_workflow_ref=(
            "cwchanap/Crux/.github/workflows/hpa320-native-bootstrap.yml@refs/tags/native-v2"
        ),
        github_workflow_sha="a" * 40,
        host_numeric_fingerprint=FINGERPRINT,
        run_url="https://github.com/cwchanap/Crux/actions/runs/123456789",
        runner_arch="X64",
        runner_environment="github-hosted",
        runner_os="Linux",
        workflow_commit="a" * 40,
    )
    assert set(evidence) == {
        "kind",
        "official_execution_allowed",
        "payload",
        "sha256",
    }
    assert set(evidence["payload"]) == {
        "schema",
        "github_job",
        "github_repository",
        "github_run_attempt",
        "github_run_id",
        "github_workflow_ref",
        "github_workflow_sha",
        "host_numeric_fingerprint",
        "run_url",
        "runner_arch",
        "runner_environment",
        "runner_os",
        "workflow_commit",
    }
    assert evidence["payload"]["schema"] == "crux.github-hosted-native-evidence/v2"
    assert evidence["sha256"] == sha256_hex(canonical_json_bytes(evidence["payload"]))
```

Delete tests that require raw API bytes, Jobs API labels, or numeric job IDs.

- [ ] **Step 5: Run the builder test to verify RED**

Run:

```bash
rtk uv run pytest tests/benchmark/test_hpa320_github_host_evidence.py -v
```

Expected: FAIL because `build_github_hosted_evidence` still exposes the v1 API-record signature.

- [ ] **Step 6: Implement the v2 evidence builder**

Replace the builder body in `tools/hpa320/github_host_evidence.py` with:

```python
def build_github_hosted_evidence(
    *,
    github_job: str,
    github_repository: str,
    github_run_attempt: int,
    github_run_id: int,
    github_workflow_ref: str,
    github_workflow_sha: str,
    host_numeric_fingerprint: HostNumericFingerprint,
    run_url: str,
    runner_arch: str,
    runner_environment: str,
    runner_os: str,
    workflow_commit: str,
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "schema": "crux.github-hosted-native-evidence/v2",
        "github_job": github_job,
        "github_repository": github_repository,
        "github_run_attempt": github_run_attempt,
        "github_run_id": github_run_id,
        "github_workflow_ref": github_workflow_ref,
        "github_workflow_sha": github_workflow_sha,
        "host_numeric_fingerprint": host_numeric_fingerprint.as_json(),
        "run_url": run_url,
        "runner_arch": runner_arch,
        "runner_environment": runner_environment,
        "runner_os": runner_os,
        "workflow_commit": workflow_commit,
    }
    digest = sha256_hex(canonical_json_bytes(payload))
    record: dict[str, JsonValue] = {
        "kind": "github_hosted",
        "official_execution_allowed": True,
        "payload": payload,
        "sha256": digest,
    }
    NativeHostEvidence(
        kind="github_hosted",
        payload=payload,
        sha256=digest,
        official_execution_allowed=True,
    )
    return record
```

Do not retain an overload or compatibility shim for `api_record_bytes` or `job_id`.

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
rtk uv run pytest \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_hpa320_github_host_evidence.py \
  -v
rtk uv run ruff check \
  src/benchmark/backend_process.py \
  tools/hpa320/github_host_evidence.py \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_hpa320_github_host_evidence.py
rtk git add \
  src/benchmark/backend_process.py \
  tools/hpa320/github_host_evidence.py \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_hpa320_github_host_evidence.py
rtk git commit -m "feat: require v2 GitHub-hosted evidence"
```

Expected: focused pytest and Ruff pass; the commit contains only the strict v2 evidence cutover.

### Task 2: Same-Job Host Observation and Atomic v2 Bundle

**Files:**
- Modify: `src/benchmark/backend_publication.py`
- Modify: `src/benchmark/backend_prepare.py`
- Modify: `tools/hpa320/oaf_host_attestation.py`
- Modify: `tests/benchmark/test_backend_publication.py`
- Modify: `tests/benchmark/test_oaf_host_attestation.py`
- Rename: `tests/benchmark/schema_goldens/crux.oaf-native-host-attestation-bundle-v1.json` to `tests/benchmark/schema_goldens/crux.oaf-native-host-attestation-bundle-v2.json`

**Interfaces:**
- Consumes: Task 1's `build_github_hosted_evidence(...)` and `NativeHostEvidence`.
- Produces:

```python
def rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory only when destination is absent."""


@dataclass(frozen=True)
class NativeHostAttestationBundle:
    native_host_evidence: CheckpointIdentity
    native_host_observation: CheckpointIdentity
    phase: str
    sha256: str
    evidence: NativeHostEvidence
    observation: Mapping[str, JsonValue]


def publish_github_host_attestation(
    *,
    phase: str,
    output_directory: Path,
) -> NativeHostAttestationBundle:
    """Validate this exact work job and atomically publish its v2 host bundle."""
```

- CLI: `python -m tools.hpa320.oaf_host_attestation publish-github --phase <phase> --output <directory>`.
- Removed interfaces: `observe_github_host`, `finalize_github_host_attestation`, `observe-github`, and `finalize-github`.

- [ ] **Step 1: Add RED tests for reusable no-replace directory publication**

In `tests/benchmark/test_backend_publication.py`, add:

```python
def test_rename_directory_no_replace_publishes_once(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "published"

    rename_directory_no_replace(source, destination)

    assert not source.exists()
    assert (destination / "evidence.json").read_bytes() == b"{}\n"


def test_rename_directory_no_replace_never_replaces_existing_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "published"
    source.mkdir()
    destination.mkdir()
    (destination / "sentinel").write_text("keep", encoding="utf-8")

    with pytest.raises(ArtifactPublicationError):
        rename_directory_no_replace(source, destination)

    assert (destination / "sentinel").read_text(encoding="utf-8") == "keep"
    assert source.is_dir()
```

- [ ] **Step 2: Run the publication tests to verify RED**

Run:

```bash
rtk uv run pytest \
  tests/benchmark/test_backend_publication.py::test_rename_directory_no_replace_publishes_once \
  tests/benchmark/test_backend_publication.py::test_rename_directory_no_replace_never_replaces_existing_target \
  -v
```

Expected: collection fails because `rename_directory_no_replace` is not public.

- [ ] **Step 3: Move the syscall implementation to shared publication code**

Move the existing macOS `renameatx_np(..., RENAME_EXCL)` and Linux `renameat2(..., RENAME_NOREPLACE)` implementation from `src/benchmark/backend_prepare.py` into `src/benchmark/backend_publication.py` behind:

```python
def rename_directory_no_replace(source: Path, destination: Path) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    try:
        source_metadata = os.lstat(source_path)
    except OSError as error:
        raise ArtifactPublicationError("publication source is unavailable") from error
    if not stat.S_ISDIR(source_metadata.st_mode):
        raise ArtifactPublicationError(
            "publication source must be a non-symlink directory"
        )
    if source_path.parent != destination_path.parent:
        raise ArtifactPublicationError(
            "publication rename must stay within one parent directory"
        )
    try:
        _rename_no_replace_syscall(source_path, destination_path)
    except FileExistsError as error:
        raise ArtifactPublicationError("publication destination already exists") from error
    except OSError as error:
        raise ArtifactPublicationError(
            "atomic no-replace directory publication failed"
        ) from error
```

Import and call this public helper from `backend_prepare.py`; delete its private duplicate and preserve its existing exception translation.

- [ ] **Step 4: Add RED same-job host tests**

Create a shared test environment helper in `tests/benchmark/test_oaf_host_attestation.py`:

```python
def set_same_job_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase: str = "bootstrap",
    runner_environment: str = "github-hosted",
) -> None:
    job = f"native-{phase}"
    workflow = f"hpa320-native-{phase}.yml"
    values = {
        "COMMIT_SHA": "a" * 40,
        "GITHUB_JOB": job,
        "GITHUB_REPOSITORY": "cwchanap/Crux",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_WORKFLOW_REF": f"cwchanap/Crux/.github/workflows/{workflow}@refs/tags/native-v2",
        "RUNNER_ARCH": "X64",
        "RUNNER_ARCH_CONTEXT": "X64",
        "RUNNER_ENVIRONMENT": runner_environment,
        "RUNNER_ENVIRONMENT_CONTEXT": runner_environment,
        "RUNNER_OS": "Linux",
        "RUNNER_OS_CONTEXT": "Linux",
        "WORKFLOW_SOURCE_SHA": "a" * 40,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def stub_native_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    def run_text(command: tuple[str, ...], _label: str) -> str:
        if command == ("git", "rev-parse", "HEAD"):
            return "a" * 40
        if command == ("uname", "-m"):
            return "x86_64"
        raise AssertionError(f"unexpected native command: {command!r}")

    docker_values = {
        "OSType": "linux",
        "Architecture": "x86_64",
        "ServerVersion": "28.3.2",
    }
    monkeypatch.setattr(host_module, "_run_text", run_text)
    monkeypatch.setattr(host_module, "_docker_info", docker_values.__getitem__)
    monkeypatch.setattr(
        host_module,
        "collect_host_numeric_fingerprint",
        lambda: HostNumericFingerprint(
            architecture="x86_64",
            cpu_vendor_id="GenuineIntel",
            cpu_family="6",
            cpu_model="143",
            cpu_stepping="8",
        ),
    )
```

Add success, mismatch, and atomicity tests:

```python
def test_publish_github_host_attestation_binds_the_current_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_same_job_environment(monkeypatch)
    stub_native_commands(monkeypatch)
    output = tmp_path / "bootstrap-host-attestation"

    bundle = publish_github_host_attestation(phase="bootstrap", output_directory=output)

    assert {path.name for path in output.iterdir()} == {
        "attestation-bundle.json",
        "native-host-evidence.json",
        "native-host-observation.json",
    }
    assert bundle.observation["github_job"] == "native-bootstrap"
    assert bundle.observation["github_workflow_sha"] == "a" * 40
    assert bundle.observation["runner_environment"] == "github-hosted"
    assert bundle.evidence.payload["run_url"].endswith("/actions/runs/123456789")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RUNNER_ENVIRONMENT", "self-hosted"),
        ("RUNNER_ENVIRONMENT_CONTEXT", "self-hosted"),
        ("RUNNER_OS_CONTEXT", "Windows"),
        ("RUNNER_ARCH_CONTEXT", "ARM64"),
        ("GITHUB_JOB", "observe-native-host"),
        ("WORKFLOW_SOURCE_SHA", "b" * 40),
    ],
)
def test_publish_github_host_attestation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    set_same_job_environment(monkeypatch)
    monkeypatch.setenv(name, value)
    stub_native_commands(monkeypatch)
    output = tmp_path / "bootstrap-host-attestation"

    with pytest.raises(HostAttestationError):
        publish_github_host_attestation(phase="bootstrap", output_directory=output)

    assert not output.exists()
```

Also assert missing/empty values, context/default disagreement for all three runner fields, wrong repository/ref/run/commit, wrong Docker facts, an existing output directory, and a simulated mid-write exception all leave no newly published directory.

- [ ] **Step 5: Run the host tests to verify RED**

Run:

```bash
rtk uv run pytest tests/benchmark/test_oaf_host_attestation.py -v
```

Expected: FAIL because the current code still exposes two cross-job commands, requires `GH_TOKEN`, and publishes four v1 files directly.

- [ ] **Step 6: Implement strict same-job observation**

In `tools/hpa320/oaf_host_attestation.py`, define exact phase ownership:

```python
BUNDLE_SCHEMA = "crux.oaf-native-host-attestation-bundle/v2"
PHASE_WORKFLOWS = MappingProxyType(
    {
        "bootstrap": ("native-bootstrap", "hpa320-native-bootstrap.yml"),
        "measurement": ("native-measurement", "hpa320-native-measurement.yml"),
        "candidate": ("native-candidate", "hpa320-native-candidate.yml"),
    }
)
_IDENTITY_NAMES = MappingProxyType(
    {
        "native_host_evidence": "native-host-evidence.json",
        "native_host_observation": "native-host-observation.json",
    }
)
_BUNDLE_KEYS = frozenset(
    {"native_host_evidence", "native_host_observation", "phase", "schema"}
)
_OBSERVATION_KEYS = frozenset(
    {
        "docker_architecture",
        "docker_os_type",
        "docker_server_version",
        "github_job",
        "github_repository",
        "github_run_attempt",
        "github_run_id",
        "github_run_url",
        "github_sha",
        "github_workflow_ref",
        "github_workflow_sha",
        "host_numeric_fingerprint",
        "runner_arch",
        "runner_environment",
        "runner_os",
        "uname_architecture",
    }
)
```

Build the observation only after requiring:

```python
runner_environment = _matching_runner_value(
    context_name="RUNNER_ENVIRONMENT_CONTEXT",
    default_name="RUNNER_ENVIRONMENT",
    expected="github-hosted",
)
runner_os = _matching_runner_value(
    context_name="RUNNER_OS_CONTEXT",
    default_name="RUNNER_OS",
    expected="Linux",
)
runner_arch = _matching_runner_value(
    context_name="RUNNER_ARCH_CONTEXT",
    default_name="RUNNER_ARCH",
    expected="X64",
)
job, workflow_file = PHASE_WORKFLOWS[phase]
if _required_environment("GITHUB_JOB") != job:
    raise HostAttestationError("current GitHub job does not own this native phase")
commit = _required_environment("COMMIT_SHA")
if {
    commit,
    _required_environment("GITHUB_SHA"),
    _required_environment("WORKFLOW_SOURCE_SHA"),
    _run_text(("git", "rev-parse", "HEAD"), "checked-out commit"),
} != {commit}:
    raise HostAttestationError("workflow commit does not match the exact checkout")
```

Validate the exact `GITHUB_WORKFLOW_REF` prefix for the selected workflow and a nonempty ref suffix. Include the old Docker, uname, repository, run, fingerprint, and server checks, plus `github_job`, `github_workflow_sha`, and `runner_environment`. Remove every Jobs API HTTP helper and API-record decoder.

- [ ] **Step 7: Implement private staging and no-replace publication**

Generate observation and evidence bytes, then publish through:

```python
parent = output_directory.parent
parent.mkdir(parents=True, exist_ok=True)
staging = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=parent))
try:
    observation_path = staging / "native-host-observation.json"
    evidence_path = staging / "native-host-evidence.json"
    observation_content = canonical_json_bytes(observation, trailing_newline=True)
    evidence_content = canonical_json_bytes(evidence_record, trailing_newline=True)
    _write_new_regular_file(observation_path, observation_content)
    _write_new_regular_file(evidence_path, evidence_content)
    bundle_payload: JsonValue = {
        "native_host_evidence": _identity_payload(evidence_path.name, evidence_content),
        "native_host_observation": _identity_payload(
            observation_path.name,
            observation_content,
        ),
        "phase": phase,
        "schema": BUNDLE_SCHEMA,
    }
    bundle_path = staging / "attestation-bundle.json"
    bundle_content = canonical_json_bytes(bundle_payload, trailing_newline=True)
    _write_new_regular_file(bundle_path, bundle_content)
    load_native_host_attestation_bundle(bundle_path, expected_phase=phase)
    rename_directory_no_replace(staging, output_directory)
    return load_native_host_attestation_bundle(
        output_directory / "attestation-bundle.json",
        expected_phase=phase,
    )
except Exception:
    shutil.rmtree(staging, ignore_errors=True)
    raise
```

Use this exclusive writer inside the private staging directory:

```python
def _write_new_regular_file(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)
```

The strict loader performs all final sibling reads through the shared no-follow reader. `rename_directory_no_replace` verifies the source binding after the syscall and `fsync`s the common parent before returning.

- [ ] **Step 8: Replace the CLI and v2 golden**

The parser must expose only:

```python
publish = subparsers.add_parser("publish-github")
publish.add_argument("--phase", required=True, choices=tuple(PHASE_WORKFLOWS))
publish.add_argument("--output", required=True, type=Path)
```

Dispatch directly to `publish_github_host_attestation`. Remove `--observation`, `--github-output`, API inputs, and both old command names. Rename the golden with `git mv`, remove its API identity, change its schema to `/v2`, and update its two remaining `CheckpointIdentity` rows.

- [ ] **Step 9: Run focused tests and commit**

Run:

```bash
rtk uv run pytest \
  tests/benchmark/test_backend_publication.py \
  tests/benchmark/test_oaf_host_attestation.py \
  -v
rtk uv run ruff check \
  src/benchmark/backend_publication.py \
  src/benchmark/backend_prepare.py \
  tools/hpa320/oaf_host_attestation.py \
  tests/benchmark/test_backend_publication.py \
  tests/benchmark/test_oaf_host_attestation.py
rtk git add \
  src/benchmark/backend_publication.py \
  src/benchmark/backend_prepare.py \
  tools/hpa320/oaf_host_attestation.py \
  tests/benchmark/test_backend_publication.py \
  tests/benchmark/test_oaf_host_attestation.py \
  tests/benchmark/schema_goldens/crux.oaf-native-host-attestation-bundle-v2.json
rtk git add -u tests/benchmark/schema_goldens
rtk git commit -m "feat: publish same-job native host evidence"
```

Expected: focused tests pass, the old v1 golden is gone, and no REST/API-record production path remains.

### Task 3: Atomic v2 Containing-Evidence Migration

**Files:**
- Create: `tools/hpa320/oaf_native_artifacts.py`
- Modify: `src/benchmark/backend_lock.py`
- Modify: `tools/hpa320/oaf_system_packages.py`
- Modify: `tools/hpa320/seal_oaf_backend.py`
- Modify: `tools/hpa320/oaf_native_calibration.py`
- Modify: `tools/hpa320/oaf_candidate_builder.py`
- Modify: `runtime/oaf_tf1/calibration_entrypoint.py`
- Modify: `runtime/oaf_tf1/oaf_backend.py`
- Modify: `runtime/oaf_tf1/tests/test_calibration_protocol.py`
- Modify: `tests/benchmark/test_backend_lock.py`
- Modify: `tests/benchmark/test_oaf_backend.py`
- Modify: `tests/benchmark/test_oaf_seal.py`
- Modify: `tests/benchmark/schema_goldens/crux.oaf-calibration-measurement-request-v1.json`
- Rename: the five containing-schema goldens from `*-v1.json` to `*-v2.json`

**Interfaces:**
- Consumes: Task 2's `load_native_host_attestation_bundle(..., expected_phase=...)`.
- Produces these exact public constants in `tools/hpa320/oaf_native_artifacts.py`:

```python
CANDIDATE_ARTIFACTS: tuple[tuple[str, str], ...]
CANDIDATE_ARTIFACT_PATHS: Mapping[str, str]
```

- All five production schema readers accept only their new `/v2` identity. Existing request schemas remain `/v1`.

- [ ] **Step 1: Add RED schema-cutover tests**

In the owning test files, parameterize the five replacement identities:

```python
V2_SCHEMA_REPLACEMENTS = {
    "crux.backend-seal-evidence/v1": "crux.backend-seal-evidence/v2",
    "crux.oaf-base-system-package-evidence/v1": (
        "crux.oaf-base-system-package-evidence/v2"
    ),
    "crux.oaf-calibration-bootstrap-evidence/v1": (
        "crux.oaf-calibration-bootstrap-evidence/v2"
    ),
    "crux.oaf-calibration-measurement-evidence/v1": (
        "crux.oaf-calibration-measurement-evidence/v2"
    ),
    "crux.oaf-seal-candidate/v1": "crux.oaf-seal-candidate/v2",
}
```

For each owning loader, assert its valid fixture uses the replacement and that replacing only `schema` with the former `/v1` value raises its existing strict domain error. In `tests/benchmark/test_oaf_seal.py`, add:

```python
def test_v2_candidate_inventory_removes_the_jobs_api_record() -> None:
    roles = tuple(role for role, _path in CANDIDATE_ARTIFACTS)
    assert "native_host_api_record" not in roles
    assert all(
        not path.endswith("github-job-api-record.json.hex")
        for path in CANDIDATE_ARTIFACT_PATHS.values()
    )
    assert set(roles) == {
        "conversion_audit",
        "native_host_attestation_bundle",
        "native_host_evidence",
        "native_host_observation",
        "host_adapter_source_manifest",
        "tensor_coverage",
        "advisory_snapshot",
        "security_scan",
        "oci_layout_archive",
        "oci_layout_manifest",
        "smoke_audio",
        "smoke_prediction",
        "smoke_oracle",
        "seal_evidence",
        "runtime_lock",
        "backend_lock",
    }
```

- [ ] **Step 2: Run the migrated schema tests to verify RED**

Run:

```bash
rtk uv run pytest \
  tests/benchmark/test_backend_lock.py \
  tests/benchmark/test_oaf_backend.py \
  tests/benchmark/test_oaf_seal.py \
  runtime/oaf_tf1/tests/test_calibration_protocol.py \
  -v
```

Expected: FAIL on `/v2` fixtures and on the still-present API-record candidate row.

- [ ] **Step 3: Create one immutable owner for candidate paths and roles**

Move `_CANDIDATE_ARTIFACTS` and `_CANDIDATE_ARTIFACT_PATHS` out of `seal_oaf_backend.py` into `tools/hpa320/oaf_native_artifacts.py`, rename them without a leading underscore, remove `native_host_api_record`, and freeze both:

```python
from types import MappingProxyType

CANDIDATE_ARTIFACTS = (
    (
        "conversion_audit",
        "docs/superpowers/evidence/hpa-320/legacy-conversion-audit.json",
    ),
    (
        "native_host_attestation_bundle",
        "docs/superpowers/evidence/hpa-320/native/"
        "candidate-host-attestation/attestation-bundle.json",
    ),
    (
        "native_host_evidence",
        "docs/superpowers/evidence/hpa-320/native/"
        "candidate-host-attestation/native-host-evidence.json",
    ),
    (
        "native_host_observation",
        "docs/superpowers/evidence/hpa-320/native/"
        "candidate-host-attestation/native-host-observation.json",
    ),
    (
        "host_adapter_source_manifest",
        "runtime/oaf_tf1/host-adapter-source-manifest.json",
    ),
    (
        "tensor_coverage",
        "docs/superpowers/evidence/hpa-320/oaf-tensor-coverage.json",
    ),
    (
        "advisory_snapshot",
        "docs/superpowers/evidence/hpa-320/oaf-advisory-snapshot.json",
    ),
    (
        "security_scan",
        "docs/superpowers/evidence/hpa-320/oaf-security-scan.json",
    ),
    (
        "oci_layout_archive",
        "artifacts/benchmark/backends/oaf-tf1/runtime.oci.tar",
    ),
    (
        "oci_layout_manifest",
        "docs/superpowers/evidence/hpa-320/oaf-oci-layout-manifest.json",
    ),
    ("smoke_audio", "tests/fixtures/oaf_tf1_smoke/canonical.wav"),
    (
        "smoke_prediction",
        "docs/superpowers/evidence/hpa-320/oaf-smoke-prediction.jsonl",
    ),
    ("smoke_oracle", "tests/fixtures/oaf_tf1_smoke/smoke-oracle.json"),
    (
        "seal_evidence",
        "config/benchmark/backends/"
        "magenta-egmd-tf1-94529798-8hit-v1.seal-evidence.json",
    ),
    (
        "runtime_lock",
        "config/benchmark/backends/"
        "magenta-egmd-tf1-94529798-8hit-v1.runtime-lock.json",
    ),
    (
        "backend_lock",
        "config/benchmark/backends/"
        "magenta-egmd-tf1-94529798-8hit-v1.backend-lock.json",
    ),
)
CANDIDATE_ARTIFACT_PATHS = MappingProxyType(dict(CANDIDATE_ARTIFACTS))
```

Import these constants in `seal_oaf_backend.py`, `oaf_candidate_builder.py`, and tests. The 16 rows are the candidate manifest's hashed dependencies; the enclosing `candidate-manifest.json` becomes the seventeenth outer native-work row in Task 4. There must be no duplicate candidate mapping.

- [ ] **Step 4: Switch every producer and strict consumer to v2**

Change only the five containing schema constants:

```python
BACKEND_SEAL_EVIDENCE_SCHEMA = "crux.backend-seal-evidence/v2"
BASE_SYSTEM_PACKAGE_EVIDENCE_SCHEMA = "crux.oaf-base-system-package-evidence/v2"
CALIBRATION_BOOTSTRAP_EVIDENCE_SCHEMA = "crux.oaf-calibration-bootstrap-evidence/v2"
CALIBRATION_MEASUREMENT_EVIDENCE_SCHEMA = "crux.oaf-calibration-measurement-evidence/v2"
SEAL_CANDIDATE_SCHEMA = "crux.oaf-seal-candidate/v2"
```

Update imports and exact schema comparisons in the host tools, native calibration producer, candidate builder, calibration entrypoint, runtime backend, schema registry, and tests. Keep request schema constants unchanged. Do not add a tuple of supported versions or an upgrade branch.

Keep `crux.oaf-calibration-measurement-request/v1`, but change its exact `output_schemas` expectation from `crux.oaf-calibration-measurement-evidence/v1` to `crux.oaf-calibration-measurement-evidence/v2`. Seal-profile request fields remain hash references and require no schema rename.

- [ ] **Step 5: Rename and rewrite the five success goldens**

Use:

```bash
rtk git mv \
  tests/benchmark/schema_goldens/crux.backend-seal-evidence-v1.json \
  tests/benchmark/schema_goldens/crux.backend-seal-evidence-v2.json
rtk git mv \
  tests/benchmark/schema_goldens/crux.oaf-base-system-package-evidence-v1.json \
  tests/benchmark/schema_goldens/crux.oaf-base-system-package-evidence-v2.json
rtk git mv \
  tests/benchmark/schema_goldens/crux.oaf-calibration-bootstrap-evidence-v1.json \
  tests/benchmark/schema_goldens/crux.oaf-calibration-bootstrap-evidence-v2.json
rtk git mv \
  tests/benchmark/schema_goldens/crux.oaf-calibration-measurement-evidence-v1.json \
  tests/benchmark/schema_goldens/crux.oaf-calibration-measurement-evidence-v2.json
rtk git mv \
  tests/benchmark/schema_goldens/crux.oaf-seal-candidate-v1.json \
  tests/benchmark/schema_goldens/crux.oaf-seal-candidate-v2.json
```

Canonicalize each file after changing only its schema identity, nested v2 host record, dependent digests, and removed API-record candidate row. Preserve old v1 bytes only in explicitly named rejection fixtures local to tests.

- [ ] **Step 6: Run the complete schema boundary suite**

Run:

```bash
rtk uv run pytest \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_backend_lock.py \
  tests/benchmark/test_hpa320_github_host_evidence.py \
  tests/benchmark/test_oaf_host_attestation.py \
  tests/benchmark/test_oaf_backend.py \
  tests/benchmark/test_oaf_seal.py \
  runtime/oaf_tf1/tests/test_calibration_protocol.py \
  -v
```

Expected: PASS, including explicit v1 rejection at each migrated production boundary.

- [ ] **Step 7: Commit the atomic schema migration**

Run:

```bash
rtk uv run ruff check \
  src/benchmark/backend_lock.py \
  tools/hpa320/oaf_system_packages.py \
  tools/hpa320/seal_oaf_backend.py \
  tools/hpa320/oaf_native_calibration.py \
  tools/hpa320/oaf_candidate_builder.py \
  tools/hpa320/oaf_native_artifacts.py \
  runtime/oaf_tf1/calibration_entrypoint.py \
  runtime/oaf_tf1/oaf_backend.py
rtk git add \
  src/benchmark/backend_lock.py \
  tools/hpa320/oaf_system_packages.py \
  tools/hpa320/seal_oaf_backend.py \
  tools/hpa320/oaf_native_calibration.py \
  tools/hpa320/oaf_candidate_builder.py \
  tools/hpa320/oaf_native_artifacts.py \
  runtime/oaf_tf1/calibration_entrypoint.py \
  runtime/oaf_tf1/oaf_backend.py \
  runtime/oaf_tf1/tests/test_calibration_protocol.py \
  tests/benchmark/test_backend_lock.py \
  tests/benchmark/test_oaf_backend.py \
  tests/benchmark/test_oaf_seal.py \
  tests/benchmark/schema_goldens
rtk git commit -m "feat: migrate native seal evidence to v2"
```

Expected: one commit moves all five containing schemas together; no production success reader accepts v1.

### Task 4: Canonical Native Work Manifest and Exact Phase Mappings

**Files:**
- Modify: `tools/hpa320/oaf_native_artifacts.py`
- Create: `tests/benchmark/test_oaf_native_artifacts.py`
- Modify: `tests/benchmark/test_oaf_seal.py`

**Interfaces:**
- Consumes: Task 2's `NativeHostAttestationBundle` and Task 3's candidate mappings.
- Produces:

```python
@dataclass(frozen=True)
class ArtifactReference:
    path: str
    role: str
    sha256: str
    size: int


@dataclass(frozen=True)
class NativeWorkArtifactManifest:
    path: Path
    phase: str
    files: tuple[ArtifactReference, ...]
    github_job: str
    github_repository: str
    github_run_attempt: int
    github_run_id: int
    github_workflow_ref: str
    github_workflow_sha: str
    native_host_attestation_bundle_sha256: str
    status: str
    workflow_commit: str
    sha256: str


def publish_native_work_manifest(
    *,
    phase: str,
    payload_root: Path,
    host_bundle_path: Path,
) -> NativeWorkArtifactManifest:
    """Scan the exact phase allowlist and atomically publish artifact-manifest.json."""


def load_native_work_manifest(
    path: Path,
    *,
    expected_phase: str,
) -> NativeWorkArtifactManifest:
    """Strict-load one canonical native work manifest."""


def verify_native_work_payload(
    *,
    payload_root: Path,
    manifest: NativeWorkArtifactManifest,
) -> None:
    """Reproduce every manifest row through no-follow descriptors."""


@dataclass(frozen=True)
class BootstrapPayload:
    root: Path
    host_bundle: Path
```

- [ ] **Step 1: Freeze the role vocabulary and exact phase mappings**

Extend `oaf_native_artifacts.py` with:

```python
MANIFEST_SCHEMA = "crux.oaf-native-work-artifact-manifest/v1"
MANIFEST_NAME = "artifact-manifest.json"
ROLES = frozenset(
    {
        "advisory_snapshot",
        "backend_lock",
        "base_system_package_evidence",
        "calibration_bootstrap_evidence",
        "calibration_measurement_evidence",
        "checkpoint_acquisition_evidence",
        "conversion_audit",
        "diagnostic",
        "host_adapter_source_manifest",
        "native_host_attestation_bundle",
        "native_host_evidence",
        "native_host_observation",
        "oci_layout_archive",
        "oci_layout_manifest",
        "runtime_lock",
        "seal_candidate_manifest",
        "seal_evidence",
        "security_scan",
        "smoke_audio",
        "smoke_oracle",
        "smoke_prediction",
        "tensor_coverage",
    }
)

BOOTSTRAP_FILES = MappingProxyType(
    {
        "bootstrap-host-attestation/attestation-bundle.json": "native_host_attestation_bundle",
        "bootstrap-host-attestation/native-host-evidence.json": "native_host_evidence",
        "bootstrap-host-attestation/native-host-observation.json": "native_host_observation",
        "checkpoint-acquisition-evidence.json": "checkpoint_acquisition_evidence",
        "calibration-image/calibration-bootstrap-evidence.json": "calibration_bootstrap_evidence",
        "calibration-image/oci-layout-manifest.json": "oci_layout_manifest",
        "calibration-image/runtime.oci.tar": "oci_layout_archive",
        "base-system-package-evidence.json": "base_system_package_evidence",
    }
)

MEASUREMENT_FILES = MappingProxyType(
    {
        "measurement-host-attestation/attestation-bundle.json": "native_host_attestation_bundle",
        "measurement-host-attestation/native-host-evidence.json": "native_host_evidence",
        "measurement-host-attestation/native-host-observation.json": "native_host_observation",
        "operational-checkpoint-acquisition-evidence.json": "checkpoint_acquisition_evidence",
        "operational-image/calibration-bootstrap-evidence.json": "calibration_bootstrap_evidence",
        "operational-image/oci-layout-manifest.json": "oci_layout_manifest",
        "operational-image/runtime.oci.tar": "oci_layout_archive",
        "calibration-measurement-evidence.json": "calibration_measurement_evidence",
    }
)
```

Define the candidate mapping exactly:

```python
CANDIDATE_FILES = MappingProxyType(
    {
        "candidate-host-attestation/attestation-bundle.json": (
            "native_host_attestation_bundle"
        ),
        "candidate-host-attestation/native-host-evidence.json": "native_host_evidence",
        "candidate-host-attestation/native-host-observation.json": (
            "native_host_observation"
        ),
        "operational-checkpoint-acquisition-evidence.json": (
            "checkpoint_acquisition_evidence"
        ),
        "operational-image/calibration-bootstrap-evidence.json": (
            "calibration_bootstrap_evidence"
        ),
        "operational-image/oci-layout-manifest.json": "oci_layout_manifest",
        "operational-image/runtime.oci.tar": "oci_layout_archive",
        "seal-candidate/candidate-manifest.json": "seal_candidate_manifest",
        **{
            f"seal-candidate/{path}": role
            for role, path in CANDIDATE_ARTIFACTS
        },
    }
)
assert len(CANDIDATE_FILES) == 24
```

Define immutable `PHASE_FILES` and `PHASE_HOST_BUNDLE_PATHS`; successful diagnostic mappings start empty until a reviewed deterministic path is added.

- [ ] **Step 2: Add RED mapping and canonical-manifest tests**

In `tests/benchmark/test_oaf_native_artifacts.py`, add:

```python
@pytest.fixture
def bootstrap_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> BootstrapPayload:
    commit = "a" * 40
    environment = {
        "COMMIT_SHA": commit,
        "GITHUB_JOB": "native-bootstrap",
        "GITHUB_REPOSITORY": "cwchanap/Crux",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_SHA": commit,
        "GITHUB_WORKFLOW_REF": (
            "cwchanap/Crux/.github/workflows/"
            "hpa320-native-bootstrap.yml@refs/tags/native-v2"
        ),
        "RUNNER_ARCH": "X64",
        "RUNNER_ARCH_CONTEXT": "X64",
        "RUNNER_ENVIRONMENT": "github-hosted",
        "RUNNER_ENVIRONMENT_CONTEXT": "github-hosted",
        "RUNNER_OS": "Linux",
        "RUNNER_OS_CONTEXT": "Linux",
        "WORKFLOW_SOURCE_SHA": commit,
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        host_module,
        "_run_text",
        lambda command, _label: commit if command[0] == "git" else "x86_64",
    )
    monkeypatch.setattr(
        host_module,
        "_docker_info",
        {
            "OSType": "linux",
            "Architecture": "x86_64",
            "ServerVersion": "28.3.2",
        }.__getitem__,
    )
    monkeypatch.setattr(
        host_module,
        "collect_host_numeric_fingerprint",
        lambda: HostNumericFingerprint(
            architecture="x86_64",
            cpu_vendor_id="GenuineIntel",
            cpu_family="6",
            cpu_model="143",
            cpu_stepping="8",
        ),
    )
    root = tmp_path / "hpa320-bootstrap"
    host_directory = root / "bootstrap-host-attestation"
    publish_github_host_attestation(
        phase="bootstrap",
        output_directory=host_directory,
    )
    for relative in BOOTSTRAP_FILES:
        path = root / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            canonical_json_bytes({"fixture_path": relative}, trailing_newline=True)
        )
    return BootstrapPayload(
        root=root,
        host_bundle=host_directory / "attestation-bundle.json",
    )


def test_candidate_mapping_is_the_exact_prefixed_candidate_inventory() -> None:
    expected_candidate_rows = {
        "seal-candidate/candidate-manifest.json": "seal_candidate_manifest",
        **{
            f"seal-candidate/{path}": role
            for role, path in CANDIDATE_ARTIFACTS
        },
    }
    actual_candidate_rows = {
        path: role
        for path, role in CANDIDATE_FILES.items()
        if path.startswith("seal-candidate/")
    }
    assert actual_candidate_rows == expected_candidate_rows
    assert len(actual_candidate_rows) == 17
    assert all("github-job-api-record" not in path for path in CANDIDATE_FILES)


def test_publish_manifest_hashes_every_allowlisted_payload_file(
    bootstrap_payload: BootstrapPayload,
) -> None:
    manifest = publish_native_work_manifest(
        phase="bootstrap",
        payload_root=bootstrap_payload.root,
        host_bundle_path=bootstrap_payload.host_bundle,
    )
    payload = strict_json_loads(
        (bootstrap_payload.root / "artifact-manifest.json").read_bytes()[:-1],
        require_canonical=True,
    )
    assert payload["schema"] == MANIFEST_SCHEMA
    assert payload["status"] == "success"
    assert [(row["role"], row["path"]) for row in payload["files"]] == sorted(
        [(role, path) for path, role in BOOTSTRAP_FILES.items()],
        key=lambda item: (item[0].encode(), item[1].encode()),
    )
    assert manifest.sha256 == sha256_hex(
        (bootstrap_payload.root / "artifact-manifest.json").read_bytes()
    )
```

Build fixture helpers by writing minimal canonical files for every mapping row and a real Task 2 v2 host bundle. Do not bypass the host parser with a mock.

- [ ] **Step 3: Run the manifest success tests to verify RED**

Run:

```bash
rtk uv run pytest \
  tests/benchmark/test_oaf_native_artifacts.py::test_candidate_mapping_is_the_exact_prefixed_candidate_inventory \
  tests/benchmark/test_oaf_native_artifacts.py::test_publish_manifest_hashes_every_allowlisted_payload_file \
  -v
```

Expected: FAIL because manifest types and publication do not exist.

- [ ] **Step 4: Implement strict manifest parsing and publication**

Use exact key sets:

```python
_MANIFEST_KEYS = frozenset(
    {
        "files",
        "github_job",
        "github_repository",
        "github_run_attempt",
        "github_run_id",
        "github_workflow_ref",
        "github_workflow_sha",
        "native_host_attestation_bundle_sha256",
        "phase",
        "schema",
        "status",
        "workflow_commit",
    }
)
_REFERENCE_KEYS = frozenset({"path", "role", "sha256", "size"})
```

The publisher must:

1. reject a missing, symlinked, or non-directory payload root;
2. strict-load the phase's v2 host bundle;
3. recursively scan with directory descriptors and no-follow metadata;
4. require the discovered regular-file paths to equal the phase mapping exactly;
5. stream each file in 1 MiB chunks while checking pre/post `fstat`;
6. sort rows with `key=lambda row: (row.role.encode("utf-8"), row.path.encode("utf-8"))`;
7. derive all run/workflow metadata from the validated host bundle;
8. write canonical JSON plus one trailing newline to a private sibling temporary file; and
9. publish only when `artifact-manifest.json` is absent.

The loader rejects noncanonical JSON, extra/missing keys, invalid schema/status/phase, an empty file array, unknown roles, invalid hash/size, duplicates, wrong order, `.`/`..`/backslash/absolute paths, self-reference, phase-disallowed `(path, role)` pairs, and host-bundle hash disagreement.

- [ ] **Step 5: Add RED tamper and path-policy tests**

Add direct tamper tests:

```python
def test_manifest_publisher_rejects_an_unlisted_payload_file(
    bootstrap_payload: BootstrapPayload,
) -> None:
    (bootstrap_payload.root / "unexpected.json").write_bytes(b"{}\n")
    with pytest.raises(NativeArtifactError):
        publish_native_work_manifest(
            phase="bootstrap",
            payload_root=bootstrap_payload.root,
            host_bundle_path=bootstrap_payload.host_bundle,
        )


def test_manifest_loader_rejects_a_valid_role_at_the_wrong_path(
    bootstrap_payload: BootstrapPayload,
) -> None:
    published = publish_native_work_manifest(
        phase="bootstrap",
        payload_root=bootstrap_payload.root,
        host_bundle_path=bootstrap_payload.host_bundle,
    )
    payload = strict_json_loads(
        read_regular_file_no_follow(published.path)[:-1],
        require_canonical=True,
    )
    payload["files"][0]["role"] = "security_scan"
    published.path.write_bytes(canonical_json_bytes(payload, trailing_newline=True))
    with pytest.raises(NativeArtifactError):
        load_native_work_manifest(published.path, expected_phase="bootstrap")
```

Add equally direct tests named for missing file, changed file, symlinked file, symlinked directory, unknown role, duplicate path, wrong row order, manifest/archive/Sigstore self-reference, dot segment, parent segment, absolute path, and backslash path. Each test performs the named single mutation and asserts the loader or publisher raises `NativeArtifactError`. Add a race test that substitutes a file between pre/post `fstat` using an injected chunk hook and asserts publication fails without leaving `artifact-manifest.json`.

- [ ] **Step 6: Implement payload reproduction and tamper rejection**

Implement `verify_native_work_payload` by rescanning through the same private scanner used by publication:

```python
expected = tuple((row.path, row.role, row.sha256, row.size) for row in manifest.files)
actual = tuple(
    (row.path, row.role, row.sha256, row.size)
    for row in _scan_phase_payload(
        phase=manifest.phase,
        payload_root=payload_root,
        include_manifest=False,
    )
)
if actual != expected:
    raise NativeArtifactError("native work payload does not reproduce its manifest")
```

Separately require the literal manifest path to be a no-follow regular file and its full-file digest to equal `manifest.sha256`.

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
rtk uv run pytest \
  tests/benchmark/test_oaf_native_artifacts.py \
  tests/benchmark/test_oaf_seal.py \
  -v
rtk uv run ruff check \
  tools/hpa320/oaf_native_artifacts.py \
  tests/benchmark/test_oaf_native_artifacts.py \
  tests/benchmark/test_oaf_seal.py
rtk git add \
  tools/hpa320/oaf_native_artifacts.py \
  tests/benchmark/test_oaf_native_artifacts.py \
  tests/benchmark/test_oaf_seal.py
rtk git commit -m "feat: manifest native work artifacts"
```

Expected: exact path/role mappings, canonical serialization, no-follow scanning, and tamper cases pass.

### Task 5: Deterministic ustar Archive and Structural CLI

**Files:**
- Modify: `tools/hpa320/oaf_native_artifacts.py`
- Modify: `tests/benchmark/test_oaf_native_artifacts.py`

**Interfaces:**
- Consumes: Task 4's manifest loader and payload verifier.
- Produces:

```python
def pack_native_work_archive(
    *,
    phase: str,
    payload_root: Path,
    manifest_path: Path,
    archive_path: Path,
) -> CheckpointIdentity:
    """Write, self-verify, and no-replace publish the canonical ustar archive."""


def verify_native_work_archive(
    *,
    phase: str,
    payload_root: Path,
    manifest_path: Path,
    archive_path: Path,
) -> CheckpointIdentity:
    """Require the archive to be the byte-exact canonical packing."""


def copy_attestation_bundle(
    *,
    source: Path,
    destination: Path,
) -> CheckpointIdentity:
    """Copy one no-follow regular Sigstore bundle to a previously absent path."""


@dataclass(frozen=True)
class PackedBootstrap:
    payload_root: Path
    manifest: Path
    archive: Path
```

- CLI commands:

```text
publish --phase PHASE --payload-root PAYLOAD_ROOT --host-bundle HOST_BUNDLE --archive ARCHIVE
verify --phase PHASE --payload-root PAYLOAD_ROOT --archive ARCHIVE [--bundle BUNDLE]
copy-bundle --source ACTION_BUNDLE --destination STABLE_BUNDLE
```

- [ ] **Step 1: Add RED deterministic archive tests**

Add:

```python
def test_packer_reproduces_byte_identical_ustar_archives(
    bootstrap_payload: BootstrapPayload,
    tmp_path: Path,
) -> None:
    manifest = publish_native_work_manifest(
        phase="bootstrap",
        payload_root=bootstrap_payload.root,
        host_bundle_path=bootstrap_payload.host_bundle,
    )
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    pack_native_work_archive(
        phase="bootstrap",
        payload_root=bootstrap_payload.root,
        manifest_path=manifest.path,
        archive_path=first,
    )
    pack_native_work_archive(
        phase="bootstrap",
        payload_root=bootstrap_payload.root,
        manifest_path=manifest.path,
        archive_path=second,
    )
    assert first.read_bytes() == second.read_bytes()

    with tarfile.open(first, mode="r:") as archive:
        members = archive.getmembers()
    assert all(member.uid == member.gid == member.mtime == 0 for member in members)
    assert all(member.uname == member.gname == "" for member in members)
    assert all(member.pax_headers == {} for member in members)
    assert all(
        member.mode == (0o755 if member.isdir() else 0o644)
        for member in members
    )
```

Also assert one canonical root directory, explicit required intermediate directories, UTF-8 byte ordering, two terminal 512-byte zero blocks, and only regular-file/directory member types.

- [ ] **Step 2: Run the packer test to verify RED**

Run:

```bash
rtk uv run pytest \
  tests/benchmark/test_oaf_native_artifacts.py::test_packer_reproduces_byte_identical_ustar_archives \
  -v
```

Expected: FAIL because no archive packer exists.

- [ ] **Step 3: Implement bounded manual ustar serialization**

Use a manual 512-byte header writer so Python's `tarfile` cannot silently add PAX metadata. The core header API is:

```python
def _ustar_header(*, name: bytes, mode: int, size: int, typeflag: bytes) -> bytes:
    prefix, leaf = _split_ustar_path(name)
    block = bytearray(512)
    _write_bytes(block, 0, 100, leaf)
    _write_octal(block, 100, 8, mode)
    _write_octal(block, 108, 8, 0)
    _write_octal(block, 116, 8, 0)
    _write_octal(block, 124, 12, size)
    _write_octal(block, 136, 12, 0)
    block[148:156] = b"        "
    block[156:157] = typeflag
    block[257:263] = b"ustar\0"
    block[263:265] = b"00"
    _write_bytes(block, 345, 155, prefix)
    checksum = sum(block)
    block[148:156] = f"{checksum:06o}\0 ".encode("ascii")
    return bytes(block)
```

Reject NULs, non-UTF-8 names, paths not exactly representable by ustar's `name`/`prefix` fields, and numeric values outside their octal fields. Emit the root and all intermediate directories once, then the manifest and mapped files in UTF-8 member-path order. Stream regular file bodies in 1 MiB chunks, zero-pad each body to a 512-byte boundary, append exactly two zero blocks, `fsync`, strict-re-read, and no-replace rename.

- [ ] **Step 4: Add RED archive tamper and race tests**

Create the shared packed fixture, then add cases for:

```python
@pytest.fixture
def packed_bootstrap(
    bootstrap_payload: BootstrapPayload,
    tmp_path: Path,
) -> PackedBootstrap:
    published = publish_native_work_manifest(
        phase="bootstrap",
        payload_root=bootstrap_payload.root,
        host_bundle_path=bootstrap_payload.host_bundle,
    )
    archive = tmp_path / f"hpa320-native-bootstrap-{'a' * 40}.tar"
    pack_native_work_archive(
        phase="bootstrap",
        payload_root=bootstrap_payload.root,
        manifest_path=published.path,
        archive_path=archive,
    )
    return PackedBootstrap(
        payload_root=bootstrap_payload.root,
        manifest=published.path,
        archive=archive,
    )


def test_archive_verifier_rejects_changed_archive_bytes(
    packed_bootstrap: PackedBootstrap,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "changed.tar"
    content = bytearray(read_regular_file_no_follow(packed_bootstrap.archive))
    content[520] ^= 1
    archive.write_bytes(content)
    with pytest.raises(NativeArtifactError):
        verify_native_work_archive(
            phase="bootstrap",
            payload_root=packed_bootstrap.payload_root,
            manifest_path=packed_bootstrap.manifest,
            archive_path=archive,
        )
```

Add direct tests named for a missing member, duplicate member, wrong root, wrong mode, wrong UID, wrong GID, wrong mtime, nonempty owner/group, PAX member, symlink, hard link, device, FIFO, wrong order, and trailing bytes. Each test builds exactly that malformed archive and asserts `NativeArtifactError`. Add a pre/post-`fstat` substitution race for a large fake OCI member and assert the temporary archive is removed and the destination remains absent.

- [ ] **Step 5: Implement canonical archive verification**

Strictly parse every 512-byte header without extracting. Require the exact expected member sequence and metadata, stream-compare each regular member to its no-follow payload descriptor, validate padding and terminal blocks, and require EOF immediately after the second zero block. Then independently repack to a private temporary archive and compare its SHA-256 and size to the supplied archive:

```python
expected_identity = _repack_for_comparison(
    phase=phase,
    payload_root=payload_root,
    manifest_path=manifest_path,
    parent=archive_path.parent,
)
actual_identity = _stream_regular_file_identity(archive_path)
if (
    actual_identity.sha256,
    actual_identity.size,
) != (
    expected_identity.sha256,
    expected_identity.size,
):
    raise NativeArtifactError("native work archive is not the canonical packing")
```

Delete the comparison archive in `finally`.

- [ ] **Step 6: Add and implement detached bundle copy tests**

Test no-follow source validation, destination absence, byte identity, and refusal to replace:

```python
def test_copy_attestation_bundle_is_no_replace_and_byte_exact(tmp_path: Path) -> None:
    source = tmp_path / "action-bundle.json"
    destination = tmp_path / "stable.sigstore.json"
    source.write_bytes(b'{"mediaType":"application/vnd.dev.sigstore.bundle+json;version=0.3"}\n')

    identity = copy_attestation_bundle(source=source, destination=destination)

    assert destination.read_bytes() == source.read_bytes()
    assert identity.sha256 == sha256_hex(source.read_bytes())
    with pytest.raises(NativeArtifactError):
        copy_attestation_bundle(source=source, destination=destination)
```

Copy through no-follow source and new-destination descriptors, stream in bounded chunks, `fsync`, and verify the destination identity before returning.

- [ ] **Step 7: Add the structural CLI and workflow-order tests**

Build subcommands with literal arguments:

```python
publish_parser = subparsers.add_parser("publish")
publish_parser.add_argument("--phase", required=True, choices=tuple(PHASE_FILES))
publish_parser.add_argument("--payload-root", required=True, type=Path)
publish_parser.add_argument("--host-bundle", required=True, type=Path)
publish_parser.add_argument("--archive", required=True, type=Path)

verify_parser = subparsers.add_parser("verify")
verify_parser.add_argument("--phase", required=True, choices=tuple(PHASE_FILES))
verify_parser.add_argument("--payload-root", required=True, type=Path)
verify_parser.add_argument("--archive", required=True, type=Path)
verify_parser.add_argument("--bundle", type=Path)

copy_parser = subparsers.add_parser("copy-bundle")
copy_parser.add_argument("--source", required=True, type=Path)
copy_parser.add_argument("--destination", required=True, type=Path)
```

`publish` creates the manifest, packs the archive, and performs structural verification. `verify` strict-loads the manifest, reproduces the payload and archive, and, when `--bundle` is given, requires it to be a no-follow regular file. `copy-bundle` invokes only `copy_attestation_bundle`.

- [ ] **Step 8: Run focused tests and commit**

Run:

```bash
rtk uv run pytest tests/benchmark/test_oaf_native_artifacts.py -v
rtk uv run ruff check \
  tools/hpa320/oaf_native_artifacts.py \
  tests/benchmark/test_oaf_native_artifacts.py
rtk uv run ruff format --check \
  tools/hpa320/oaf_native_artifacts.py \
  tests/benchmark/test_oaf_native_artifacts.py
rtk git add \
  tools/hpa320/oaf_native_artifacts.py \
  tests/benchmark/test_oaf_native_artifacts.py
rtk git commit -m "feat: pack deterministic native evidence"
```

Expected: deterministic byte identity, unsafe-member rejection, large-file streaming, and CLI ordering pass.

### Task 6: GitHub Attestation Policy and Phase Acceptance Gates

**Files:**
- Modify: `tools/hpa320/oaf_native_artifacts.py`
- Modify: `tools/hpa320/seal_oaf_backend.py`
- Modify: `tests/benchmark/test_oaf_native_artifacts.py`
- Modify: `tests/benchmark/test_oaf_seal.py`

**Interfaces:**
- Consumes: Task 5's structural verifier and Task 3's v2 phase loaders.
- Produces:

```python
@dataclass(frozen=True)
class VerifiedGitHubAttestation:
    gh_version: str
    statement: Mapping[str, JsonValue]
    statement_sha256: str
    subjects: tuple[CheckpointIdentity, ...]


def verify_github_attestations(
    *,
    phase: str,
    workflow_commit: str,
    manifest_path: Path,
    archive_path: Path,
    sigstore_bundle_path: Path,
    trusted_root_path: Path,
    command_runner: Callable[[tuple[str, ...]], bytes] = _run_checked_command,
) -> VerifiedGitHubAttestation:
    """Verify both signed subjects under the frozen GitHub policy."""


def validate_native_work_phase(
    *,
    phase: str,
    payload_root: Path,
    repository_root: Path,
) -> None:
    """Apply the selected phase's existing request/evidence/content validation."""


@dataclass(frozen=True)
class SignedBootstrap:
    manifest: Path
    archive: Path
    bundle: Path
    trusted_root: Path
    verification_json: bytes


@dataclass(frozen=True)
class NativePhasePayload:
    root: Path
    repository_root: Path
    mutate: Callable[[str], None]
```

- CLI: `verify-attestation --phase PHASE --workflow-commit COMMIT --payload-root PAYLOAD_ROOT --archive ARCHIVE --bundle BUNDLE --trusted-root TRUSTED_ROOT --repository-root REPOSITORY_ROOT`.

- [ ] **Step 1: Add RED GitHub CLI policy tests**

Inject a fake command runner and freeze both invocations:

```python
@pytest.fixture
def signed_bootstrap(
    packed_bootstrap: PackedBootstrap,
    tmp_path: Path,
) -> SignedBootstrap:
    bundle = tmp_path / "bootstrap.sigstore.json"
    trusted_root = tmp_path / "trusted-root.json"
    bundle.write_bytes(b'{"mediaType":"application/vnd.dev.sigstore.bundle+json;version=0.3"}\n')
    trusted_root.write_bytes(b'{"mediaType":"application/vnd.dev.sigstore.trustedroot+json;version=0.1"}\n')
    subjects = [
        {
            "name": (
                "artifacts/benchmark/backends/"
                "hpa320-bootstrap/artifact-manifest.json"
            ),
            "digest": {
                "sha256": sha256_hex(
                    read_regular_file_no_follow(packed_bootstrap.manifest)
                )
            },
        },
        {
            "name": (
                "artifacts/benchmark/backends/"
                f"hpa320-native-bootstrap-{'a' * 40}.tar"
            ),
            "digest": {
                "sha256": sha256_hex(
                    read_regular_file_no_follow(packed_bootstrap.archive)
                )
            },
        },
    ]
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "predicate": {},
        "predicateType": "https://slsa.dev/provenance/v1",
        "subject": subjects,
    }
    verification_json = json.dumps(
        [{"verificationResult": {"statement": statement}}],
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return SignedBootstrap(
        manifest=packed_bootstrap.manifest,
        archive=packed_bootstrap.archive,
        bundle=bundle,
        trusted_root=trusted_root,
        verification_json=verification_json,
    )


def test_github_verifier_runs_the_exact_policy_for_both_subjects(
    signed_bootstrap: SignedBootstrap,
) -> None:
    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...]) -> bytes:
        commands.append(command)
        if command[:2] == ("gh", "version"):
            return b"gh version 2.68.1 (2026-01-01)\n"
        return signed_bootstrap.verification_json

    verified = verify_github_attestations(
        phase="bootstrap",
        workflow_commit="a" * 40,
        manifest_path=signed_bootstrap.manifest,
        archive_path=signed_bootstrap.archive,
        sigstore_bundle_path=signed_bootstrap.bundle,
        trusted_root_path=signed_bootstrap.trusted_root,
        command_runner=run,
    )

    verify_commands = [command for command in commands if command[:3] == ("gh", "attestation", "verify")]
    assert [command[3] for command in verify_commands] == [
        str(signed_bootstrap.manifest),
        str(signed_bootstrap.archive),
    ]
    for command in verify_commands:
        assert command[command.index("--repo") + 1] == "cwchanap/Crux"
        assert command[command.index("--signer-workflow") + 1] == (
            "cwchanap/Crux/.github/workflows/hpa320-native-bootstrap.yml"
        )
        assert command[command.index("--source-digest") + 1] == "a" * 40
        assert command[command.index("--signer-digest") + 1] == "a" * 40
        assert "--deny-self-hosted-runners" in command
        assert command[command.index("--digest-alg") + 1] == "sha256"
        assert command[command.index("--predicate-type") + 1] == (
            "https://slsa.dev/provenance/v1"
        )
        assert command[command.index("--cert-oidc-issuer") + 1] == (
            "https://token.actions.githubusercontent.com"
        )
        assert command[-2:] == ("--format", "json")
    assert verified.gh_version == "2.68.1"
```

The fixture's JSON array must contain `verificationResult.statement` with exactly the manifest and archive subjects.

- [ ] **Step 2: Run the GitHub policy test to verify RED**

Run:

```bash
rtk uv run pytest \
  tests/benchmark/test_oaf_native_artifacts.py::test_github_verifier_runs_the_exact_policy_for_both_subjects \
  -v
```

Expected: FAIL because the policy verifier does not exist.

- [ ] **Step 3: Implement version and command policy**

Parse only the first `gh version` line with:

```python
_GH_VERSION = re.compile(
    r"gh version ([0-9]+)\.([0-9]+)\.([0-9]+)(?:\s.*)?\Z"
)


def _supported_gh_version(output: bytes) -> str:
    first_line = output.decode("utf-8", errors="strict").splitlines()[0]
    match = _GH_VERSION.fullmatch(first_line)
    if match is None:
        raise NativeArtifactError("GitHub CLI version output is invalid")
    version = tuple(int(part) for part in match.groups())
    if version < (2, 68, 0) or version >= (3, 0, 0):
        raise NativeArtifactError("GitHub CLI must satisfy 2.68.0 <= gh < 3.0.0")
    return ".".join(match.groups())
```

Build each verify command from one shared tuple:

```python
policy = (
    "--repo",
    "cwchanap/Crux",
    "--signer-workflow",
    f"cwchanap/Crux/.github/workflows/hpa320-native-{phase}.yml",
    "--source-digest",
    workflow_commit,
    "--signer-digest",
    workflow_commit,
    "--deny-self-hosted-runners",
    "--digest-alg",
    "sha256",
    "--predicate-type",
    "https://slsa.dev/provenance/v1",
    "--cert-oidc-issuer",
    "https://token.actions.githubusercontent.com",
    "--bundle",
    str(sigstore_bundle_path),
    "--custom-trusted-root",
    str(trusted_root_path),
    "--format",
    "json",
)
```

Invoke `("gh", "attestation", "verify", str(subject), *policy)` once per subject.

- [ ] **Step 4: Add RED statement and tamper tests**

Add direct rejection tests:

```python
def test_github_verifier_rejects_an_unsupported_cli(
    signed_bootstrap: SignedBootstrap,
) -> None:
    def run(command: tuple[str, ...]) -> bytes:
        if command[:2] == ("gh", "version"):
            return b"gh version 2.67.0 (2025-01-01)\n"
        raise AssertionError("verification must not run with unsupported gh")

    with pytest.raises(NativeArtifactError):
        verify_github_attestations(
            phase="bootstrap",
            workflow_commit="a" * 40,
            manifest_path=signed_bootstrap.manifest,
            archive_path=signed_bootstrap.archive,
            sigstore_bundle_path=signed_bootstrap.bundle,
            trusted_root_path=signed_bootstrap.trusted_root,
            command_runner=run,
        )


def test_github_verifier_rejects_different_verified_statements(
    signed_bootstrap: SignedBootstrap,
) -> None:
    invocation = 0

    def run(command: tuple[str, ...]) -> bytes:
        nonlocal invocation
        if command[:2] == ("gh", "version"):
            return b"gh version 2.68.1 (2026-01-01)\n"
        invocation += 1
        if invocation == 1:
            return signed_bootstrap.verification_json
        value = json.loads(signed_bootstrap.verification_json)
        value[0]["verificationResult"]["statement"]["predicate"]["changed"] = True
        return json.dumps(value).encode("utf-8")

    with pytest.raises(NativeArtifactError):
        verify_github_attestations(
            phase="bootstrap",
            workflow_commit="a" * 40,
            manifest_path=signed_bootstrap.manifest,
            archive_path=signed_bootstrap.archive,
            sigstore_bundle_path=signed_bootstrap.bundle,
            trusted_root_path=signed_bootstrap.trusted_root,
            command_runner=run,
        )
```

Add equally direct tests for `gh` 3.0.0, malformed version text, command failure, malformed JSON shape, missing manifest subject, missing archive subject, wrong manifest digest, wrong archive digest, an extra subject, a signed diagnostic/failed result presented under the success name, and a v1 raw-API host bundle presented as v2 authority. Simulate wrong-workflow, wrong-commit, and self-hosted provenance as a nonzero `gh attestation verify` result, while the Step 1 command assertions prove the rejecting flags are always present. Even though `gh` enforces signer policy, parse `verificationResult.statement` and assert both outputs are deep-equal and their subjects are exactly the two canonical workflow subject names and local SHA-256 identities.

- [ ] **Step 5: Implement strict verification-result parsing**

Require a JSON array with one result and one statement:

```python
def _verification_statement(content: bytes) -> dict[str, JsonValue]:
    value = json.loads(content)
    if not isinstance(value, list) or len(value) != 1:
        raise NativeArtifactError("GitHub attestation verification result must have one entry")
    result = value[0]
    if not isinstance(result, dict) or "verificationResult" not in result:
        raise NativeArtifactError("GitHub attestation verification result fields are invalid")
    verification = result["verificationResult"]
    if not isinstance(verification, dict) or "statement" not in verification:
        raise NativeArtifactError("GitHub attestation statement is missing")
    statement = verification["statement"]
    if not isinstance(statement, dict):
        raise NativeArtifactError("GitHub attestation statement is invalid")
    return cast(dict[str, JsonValue], statement)
```

Preserve but do not treat other CLI result fields as authority; the required authority is `verificationResult.statement`. Compare canonical statement bytes from both invocations and validate the exact two `name`/`digest.sha256` subjects against local no-follow identities.

- [ ] **Step 6: Add RED phase acceptance tests**

In `tests/benchmark/test_oaf_seal.py`, create one valid payload fixture per phase and then mutate one internal edge at a time:

```python
@pytest.mark.parametrize(
    ("phase", "mutation"),
    [
        ("bootstrap", "checkpoint_request_hash"),
        ("bootstrap", "base_system_request_hash"),
        ("bootstrap", "bootstrap_image_identity"),
        ("measurement", "bootstrap_evidence_hash"),
        ("measurement", "measurement_request_hash"),
        ("measurement", "host_bundle_hash"),
        ("candidate", "candidate_manifest_hash"),
        ("candidate", "runtime_lock_hash"),
        ("candidate", "backend_lock_hash"),
        ("candidate", "host_bundle_hash"),
    ],
)
def test_native_work_phase_gate_rejects_internal_inconsistency(
    native_phase_payload: Callable[[str], NativePhasePayload],
    phase: str,
    mutation: str,
) -> None:
    payload = native_phase_payload(phase)
    payload.mutate(mutation)
    with pytest.raises(SealError):
        validate_native_work_phase(
            phase=phase,
            payload_root=payload.root,
            repository_root=payload.repository_root,
        )
```

The `native_phase_payload` fixture is a factory over three real valid roots: bootstrap invokes the existing checkpoint/bootstrap/base-system test producers; measurement extends that root with the existing measurement request and evidence producer; candidate extends measurement with the existing `calibrate` candidate builder. Its `mutate` callable changes exactly the named canonical hash or identity and reserializes only the owning record. Assert the unmutated root passes `validate_native_work_phase` before running every negative mutation.

- [ ] **Step 7: Implement phase-specific content gates**

Expose existing strict loaders through one dispatcher:

```python
_PHASE_VALIDATORS = {
    "bootstrap": _validate_bootstrap_work_payload,
    "measurement": _validate_measurement_work_payload,
    "candidate": _validate_candidate_work_payload,
}


def validate_native_work_phase(
    *,
    phase: str,
    payload_root: Path,
    repository_root: Path,
) -> None:
    try:
        validator = _PHASE_VALIDATORS[phase]
    except KeyError as error:
        raise SealError("native work phase is invalid") from error
    validator(
        payload_root=Path(payload_root),
        repository_root=Path(repository_root),
    )
```

Bootstrap validates checkpoint acquisition, bootstrap image evidence/OCI identity, base-system evidence, their requests, and host binding. Measurement validates the operational checkpoint/image, accepted bootstrap/base/checkpoint inputs, measurement request/evidence, and host binding. Candidate validates the operational checkpoint/image, the exact repository-shaped candidate through the existing `validate_seal_candidate`, all lock/evidence hashes, and host binding; it exact-compares the three phase-level candidate host files with their repository-shaped copies below `seal-candidate/`. Every branch requires v2 schemas and the same host-bundle identity named by the outer manifest.

- [ ] **Step 8: Add the outside-in acceptance CLI**

The command must order verification as:

```python
verified = verify_github_attestations(
    phase=args.phase,
    workflow_commit=args.workflow_commit,
    manifest_path=args.payload_root / MANIFEST_NAME,
    archive_path=args.archive,
    sigstore_bundle_path=args.bundle,
    trusted_root_path=args.trusted_root,
)
manifest = load_native_work_manifest(
    args.payload_root / MANIFEST_NAME,
    expected_phase=args.phase,
)
verify_native_work_archive(
    phase=args.phase,
    payload_root=args.payload_root,
    manifest_path=args.payload_root / MANIFEST_NAME,
    archive_path=args.archive,
)
validate_native_work_phase(
    phase=args.phase,
    payload_root=args.payload_root,
    repository_root=args.repository_root,
)
```

Only after all four calls pass may the CLI emit a canonical review report containing the recorded `gh_version`, command arguments, statement digest, and two subjects. The report is review evidence, not signed authority.

- [ ] **Step 9: Run focused tests and commit**

Run:

```bash
rtk uv run pytest \
  tests/benchmark/test_oaf_native_artifacts.py \
  tests/benchmark/test_oaf_seal.py \
  -v
rtk uv run ruff check \
  tools/hpa320/oaf_native_artifacts.py \
  tools/hpa320/seal_oaf_backend.py \
  tests/benchmark/test_oaf_native_artifacts.py \
  tests/benchmark/test_oaf_seal.py
rtk git add \
  tools/hpa320/oaf_native_artifacts.py \
  tools/hpa320/seal_oaf_backend.py \
  tests/benchmark/test_oaf_native_artifacts.py \
  tests/benchmark/test_oaf_seal.py
rtk git commit -m "feat: verify native artifact provenance"
```

Expected: both-subject GitHub policy, tamper rejection, and phase-specific acceptance all pass.

### Task 7: Single-Job Native Workflows and Pinned Signing

**Files:**
- Modify: `.github/workflows/hpa320-native-bootstrap.yml`
- Modify: `.github/workflows/hpa320-native-measurement.yml`
- Modify: `.github/workflows/hpa320-native-candidate.yml`
- Delete: `.github/workflows/hpa320-native-host-evidence.yml`
- Modify: `tests/benchmark/test_hpa320_native_bootstrap_workflow.py`
- Modify: `tests/benchmark/test_hpa320_native_measurement_workflow.py`
- Modify: `tests/benchmark/test_hpa320_native_candidate_workflow.py`

**Interfaces:**
- Consumes: Task 2's `publish-github` CLI and Task 5's `publish`, `verify`, and `copy-bundle` CLIs.
- Produces: one signed success artifact per phase with exact payload-directory, tar, and Sigstore-bundle paths.

- [ ] **Step 1: Revalidate the pinned action before changing YAML**

Run:

```bash
rtk gh api repos/actions/attest/git/ref/tags/v4.2.1 --jq '.object.sha'
rtk gh api \
  -H 'Accept: application/vnd.github.raw+json' \
  'repos/actions/attest/contents/action.yml?ref=508db95dd578ae2727ebd6217d5ba78e4fbda05d'
```

Expected: the tag resolves to `508db95dd578ae2727ebd6217d5ba78e4fbda05d`; `action.yml` declares `subject-path` input and `bundle-path` output. If either differs, stop and amend the approved design instead of substituting another action version.

- [ ] **Step 2: Rewrite workflow tests to assert the one-job contract**

Each phase test should parse YAML and assert:

```python
def assert_native_workflow_contract(
    workflow: dict[str, object],
    *,
    phase: str,
) -> None:
    job_id = f"native-{phase}"
    assert set(workflow["jobs"]) == {job_id}
    job = workflow["jobs"][job_id]
    assert "needs" not in job
    assert workflow["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }
    serialized = yaml.safe_dump(workflow)
    for forbidden in (
        "observe-native-host",
        "observe-github",
        "finalize-github",
        "--github-output",
        "GH_TOKEN",
        "actions: read",
        "github-job-api-record",
        "artifact-sha256s.txt",
    ):
        assert forbidden not in serialized
    assert serialized.count("publish-github") == 1
    assert serialized.count("actions/attest@508db95dd578ae2727ebd6217d5ba78e4fbda05d") == 1
```

Also inspect ordered step indexes to require host preflight before acquisition/build/native work, manifest/archive publication after native work, structural verification before attestation, bundle copy followed by structural verification, and success upload last.

- [ ] **Step 3: Add RED exact attestation/upload tests**

For each phase, assert:

```python
attest = only_step_using(job, "actions/attest")
assert attest["with"] == {
    "subject-path": (
        f"artifacts/benchmark/backends/hpa320-native-{phase}-"
        "${{ inputs.commit_sha }}.tar\n"
        f"artifacts/benchmark/backends/hpa320-{phase}/artifact-manifest.json\n"
    )
}

upload = only_success_upload(job)
assert upload["with"] == {
    "name": f"hpa320-native-{phase}-${{{{ inputs.commit_sha }}}}",
    "path": (
        f"artifacts/benchmark/backends/hpa320-{phase}/\n"
        f"artifacts/benchmark/backends/hpa320-native-{phase}-"
        "${{ inputs.commit_sha }}.tar\n"
        f"artifacts/benchmark/backends/hpa320-native-{phase}-"
        "${{ inputs.commit_sha }}.sigstore.json\n"
    ),
    "if-no-files-found": "error",
    "retention-days": 30,
}
```

Assert the tar and Sigstore paths are siblings of the payload root, have no wildcard, and appear in no manifest mapping. Assert any diagnostic upload has `if: failure()`, the exact non-colliding run/attempt name, only the literal phase `diagnostics/` path, `if-no-files-found: ignore`, and `retention-days: 30`.

- [ ] **Step 4: Run workflow tests to verify RED**

Run:

```bash
rtk uv run pytest \
  tests/benchmark/test_hpa320_native_bootstrap_workflow.py \
  tests/benchmark/test_hpa320_native_measurement_workflow.py \
  tests/benchmark/test_hpa320_native_candidate_workflow.py \
  -v
```

Expected: FAIL because the workflows still contain two jobs, REST finalization, and unsigned checksum artifacts.

- [ ] **Step 5: Convert each workflow to the exact sole work job**

At workflow scope use:

```yaml
permissions:
  contents: read
  id-token: write
  attestations: write

jobs:
  native-<phase>:
    name: native-<phase>
    runs-on: ubuntu-24.04
```

Preserve checkout at `${{ inputs.commit_sha }}` and the existing pinned UV setup. Remove `observe-native-host`, `needs`, cross-job outputs, `GH_TOKEN`, and `actions: read`.

The first native gate is:

```yaml
- name: Observe and validate the current native work job
  env:
    RUNNER_ENVIRONMENT_CONTEXT: ${{ runner.environment }}
    RUNNER_OS_CONTEXT: ${{ runner.os }}
    RUNNER_ARCH_CONTEXT: ${{ runner.arch }}
    WORKFLOW_SOURCE_SHA: ${{ github.workflow_sha }}
  run: >-
    uv run python -m tools.hpa320.oaf_host_attestation publish-github
    --phase <phase>
    --output artifacts/benchmark/backends/hpa320-<phase>/<phase>-host-attestation
```

Do not interpolate any context directly in shell source and do not use `$GITHUB_ENV` or `$GITHUB_OUTPUT`.

- [ ] **Step 6: Add ordered manifest, archive, attestation, and revalidation steps**

After the existing phase work:

```yaml
- name: Publish canonical native work subjects
  run: >-
    uv run python -m tools.hpa320.oaf_native_artifacts publish
    --phase <phase>
    --payload-root artifacts/benchmark/backends/hpa320-<phase>
    --host-bundle
    artifacts/benchmark/backends/hpa320-<phase>/<phase>-host-attestation/attestation-bundle.json
    --archive
    artifacts/benchmark/backends/hpa320-native-<phase>-${{ inputs.commit_sha }}.tar

- name: Verify native work subjects before attestation
  run: >-
    uv run python -m tools.hpa320.oaf_native_artifacts verify
    --phase <phase>
    --payload-root artifacts/benchmark/backends/hpa320-<phase>
    --archive
    artifacts/benchmark/backends/hpa320-native-<phase>-${{ inputs.commit_sha }}.tar

- name: Attest native phase evidence
  id: attest
  uses: actions/attest@508db95dd578ae2727ebd6217d5ba78e4fbda05d
  with:
    subject-path: |
      artifacts/benchmark/backends/hpa320-native-<phase>-${{ inputs.commit_sha }}.tar
      artifacts/benchmark/backends/hpa320-<phase>/artifact-manifest.json

- name: Preserve the local Sigstore bundle
  run: >-
    uv run python -m tools.hpa320.oaf_native_artifacts copy-bundle
    --source '${{ steps.attest.outputs.bundle-path }}'
    --destination
    artifacts/benchmark/backends/hpa320-native-<phase>-${{ inputs.commit_sha }}.sigstore.json

- name: Reverify the complete upload set
  run: >-
    uv run python -m tools.hpa320.oaf_native_artifacts verify
    --phase <phase>
    --payload-root artifacts/benchmark/backends/hpa320-<phase>
    --archive
    artifacts/benchmark/backends/hpa320-native-<phase>-${{ inputs.commit_sha }}.tar
    --bundle
    artifacts/benchmark/backends/hpa320-native-<phase>-${{ inputs.commit_sha }}.sigstore.json
```

Keep `push-to-registry`, `create-storage-record`, `sbom-path`, `predicate`, and `predicate-type` absent.

- [ ] **Step 7: Add exact success and diagnostic uploads**

Use one success upload:

```yaml
- name: Upload signed native phase evidence
  uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02
  with:
    name: hpa320-native-<phase>-${{ inputs.commit_sha }}
    path: |
      artifacts/benchmark/backends/hpa320-<phase>/
      artifacts/benchmark/backends/hpa320-native-<phase>-${{ inputs.commit_sha }}.tar
      artifacts/benchmark/backends/hpa320-native-<phase>-${{ inputs.commit_sha }}.sigstore.json
    if-no-files-found: error
    retention-days: 30
```

Preserve the workflow's already reviewed immutable `actions/upload-artifact` SHA. When a phase has sanitized failure diagnostics, add exactly:

```yaml
- name: Upload sanitized native phase diagnostics
  if: failure()
  uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02
  with:
    name: >-
      hpa320-native-<phase>-diagnostic-${{ inputs.commit_sha }}-run-${{ github.run_id }}-attempt-${{ github.run_attempt }}
    path: artifacts/benchmark/backends/hpa320-<phase>/diagnostics/
    if-no-files-found: ignore
    retention-days: 30
```

Do not manufacture a diagnostics directory or add an open wildcard solely to make this upload succeed.

- [ ] **Step 8: Delete the obsolete collector and finish workflow tests**

Delete `.github/workflows/hpa320-native-host-evidence.yml`. Add a repository-wide assertion:

```python
def test_legacy_native_host_collector_is_absent() -> None:
    assert not (
        REPOSITORY_ROOT / ".github/workflows/hpa320-native-host-evidence.yml"
    ).exists()
```

Run:

```bash
rtk uv run pytest \
  tests/benchmark/test_hpa320_native_bootstrap_workflow.py \
  tests/benchmark/test_hpa320_native_measurement_workflow.py \
  tests/benchmark/test_hpa320_native_candidate_workflow.py \
  -v
```

Expected: PASS for one-job structure, exact ordering, pinned attestation, least privilege, and upload isolation.

- [ ] **Step 9: Commit the workflow cutover**

Run:

```bash
rtk git add \
  .github/workflows/hpa320-native-bootstrap.yml \
  .github/workflows/hpa320-native-measurement.yml \
  .github/workflows/hpa320-native-candidate.yml \
  tests/benchmark/test_hpa320_native_bootstrap_workflow.py \
  tests/benchmark/test_hpa320_native_measurement_workflow.py \
  tests/benchmark/test_hpa320_native_candidate_workflow.py
rtk git add -u .github/workflows
rtk git commit -m "ci: attest native work-job artifacts"
```

Expected: all three workflows have one work job and the legacy collector is deleted.

### Task 8: Drift-Proof Freeze Regeneration, Documentation, and Final Verification

**Files:**
- Modify: `tools/hpa320/generate_runner_source_manifest.py`
- Modify: `tools/hpa320/oaf_build_context.py`
- Modify: `tools/hpa320/seal_oaf_backend.py`
- Modify: `tests/benchmark/test_oaf_build_context.py`
- Create: `tests/benchmark/test_hpa320_freeze_artifact_drift.py`
- Modify: `docs/superpowers/specs/2026-07-26-hpa-320-freeze-oaf-drums-backend-design.md`
- Modify: `docs/superpowers/specs/2026-07-30-hpa-481-work-job-artifact-attestation-design.md`
- Regenerate: `runtime/oaf_tf1/runner-source-manifest.json`
- Regenerate: `runtime/oaf_tf1/build-context-manifest.json`
- Reissue: `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json`

**Interfaces:**
- Consumes: all final workflow/tool/runtime source bytes from Tasks 1–7.
- Produces:

```text
generate_runner_source_manifest.py --check
oaf_build_context.py generate --check
oaf_build_context.py generate --replace
seal_oaf_backend.py reissue-bootstrap-request --request REQUEST_PATH
```

- [ ] **Step 1: Add RED source-manifest and build-context drift checks**

In `tests/benchmark/test_hpa320_freeze_artifact_drift.py`, compare checked files to fresh generation:

```python
def test_checked_runner_source_manifest_matches_final_sources() -> None:
    generated = canonical_manifest_bytes(
        build_runner_source_manifest(REPOSITORY_ROOT)
    )
    checked = read_regular_file_no_follow(
        REPOSITORY_ROOT / "runtime/oaf_tf1/runner-source-manifest.json"
    )
    assert generated == checked


def test_build_context_repository_rows_match_final_checkout() -> None:
    manifest = load_build_context_manifest(
        REPOSITORY_ROOT / "runtime/oaf_tf1/build-context-manifest.json"
    )
    repository_rows = tuple(
        row
        for row in manifest.files
        if not row.path.startswith("runtime/oaf_tf1/wheelhouse/")
    )
    for row in repository_rows:
        path = REPOSITORY_ROOT / row.path
        assert path.is_file() and not path.is_symlink()
        content = read_regular_file_no_follow(path)
        assert len(content) == row.byte_length
        assert sha256_hex(content) == row.sha256
```

Add an exact bootstrap request cross-hash test:

```python
def test_bootstrap_request_cross_hashes_every_current_input() -> None:
    request = load_calibration_bootstrap_request(
        REPOSITORY_ROOT
        / "config/benchmark/backends/"
        "magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json"
    )
    expected_paths = {
        "runner_source_manifest_sha256": "runtime/oaf_tf1/runner-source-manifest.json",
        "build_context_manifest_sha256": "runtime/oaf_tf1/build-context-manifest.json",
        "upstream_source_manifest_sha256": (
            "runtime/oaf_tf1/source-manifest.json"
        ),
        "checkpoint_acquisition_request_sha256": (
            "config/benchmark/backends/"
            "magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json"
        ),
        "base_system_package_request_sha256": (
            "runtime/oaf_tf1/base-system-package-request.json"
        ),
        "distribution_build_manifest_sha256": (
            "runtime/oaf_tf1/distribution-build-manifest.json"
        ),
        "instrumentation_patch_sha256": (
            "runtime/oaf_tf1/patches/capture-emitted-frame.patch"
        ),
    }
    for field, relative_path in expected_paths.items():
        content = read_regular_file_no_follow(REPOSITORY_ROOT / relative_path)
        assert request.payload[field] == sha256_hex(content)
```

- [ ] **Step 2: Run drift tests to observe the known RED state**

Run:

```bash
rtk uv run pytest tests/benchmark/test_hpa320_freeze_artifact_drift.py -v
```

Expected: FAIL. Before HPA-481 changes, these eight covered paths were already stale in both checked manifests; Tasks 1–7 add further intentional drift:

```text
.github/workflows/hpa320-native-bootstrap.yml
runtime/oaf_tf1/entrypoint.py
runtime/oaf_tf1/oaf_backend.py
runtime/oaf_tf1/tests/test_calibration_protocol.py
runtime/oaf_tf1/tests/test_tensor_coverage.py
tools/hpa320/oaf_host_attestation.py
tools/hpa320/oaf_native_runner.py
tools/hpa320/oaf_system_packages.py
```

The failing rows must be reported, not silently ignored.

- [ ] **Step 3: Add non-mutating generator checks and explicit replacement**

For `generate_runner_source_manifest.py`, add:

```python
parser.add_argument(
    "--check",
    action="store_true",
    help="compare generated canonical bytes with --output without writing",
)
```

Insert `"tools/hpa320/oaf_native_artifacts.py"` directly between `"tools/hpa320/oaf_native_calibration.py"` and `"tools/hpa320/oaf_native_runner.py"` in the existing literal `SOURCE_PATHS` tuple. Add a test that this new Docker-copied tool is covered exactly once. Generate canonical bytes in memory. Under `--check`, no-follow read `--output`, exact-compare bytes, print a sanitized mismatch, and exit nonzero without writing. Without `--check`, preserve atomic replacement behavior.

For `oaf_build_context.py generate`, add mutually exclusive:

```python
mode = generate_parser.add_mutually_exclusive_group()
mode.add_argument("--check", action="store_true")
mode.add_argument("--replace", action="store_true")
```

Default generation still requires an absent output. `--check` exact-compares without writing. `--replace` writes a sibling temporary file, validates it, and atomically replaces only the named manifest. Add tests proving all three modes and proving `--check` leaves mtime/content unchanged.

- [ ] **Step 4: Add a narrowly scoped bootstrap-request reissue command**

In `seal_oaf_backend.py`, reuse the existing exact `_BOOTSTRAP_HASH_FIELDS` tuple; it already names all seven request cross-hashes. Add:

```python
def reissue_calibration_bootstrap_request(
    *,
    request_path: Path,
    repository_root: Path,
) -> str:
    payload, original = _read_canonical_object(
        request_path,
        "calibration bootstrap request",
    )
    reissued = dict(payload)
    for relative, field in _BOOTSTRAP_HASH_FIELDS:
        content = _read_regular(repository_root / relative, relative)
        reissued[field] = sha256_hex(content)
    _validate_calibration_bootstrap_request_payload(reissued)
    content = canonical_json_bytes(reissued, trailing_newline=True)
    if strict_json_loads(original[:-1], require_canonical=True) != payload:
        raise SealError("existing calibration bootstrap request is not canonical")
    _atomic_replace_regular_file(request_path, content)
    loaded = load_calibration_bootstrap_request(request_path)
    return loaded.sha256
```

`_atomic_replace_regular_file` writes a sibling exclusive temporary file, mode `0644`, flushes and `fsync`s it, atomically replaces only the named request, and `fsync`s the parent directory. The parser adds `reissue-bootstrap-request --repository-root --request`. Tests assert every non-hash field is identical at the parsed-value level and all seven hashes are current.

- [ ] **Step 5: Restore the exact authenticated ignored wheelhouses**

Materialize from the retained HPA-320 worktree's already verified offline cache:

```bash
rtk mkdir -p runtime/oaf_tf1/wheelhouse
rtk /Users/chanwaichan/workspace/Crux/.venv/bin/python \
  -m tools.hpa320.resolve_oaf_runtime \
  --materialize-wheelhouse \
  --lock runtime/oaf_tf1/requirements.lock \
  --wheelhouse runtime/oaf_tf1/wheelhouse/runtime \
  --offline-cache \
  /Users/chanwaichan/workspace/Crux/.worktrees/hpa-320-freeze-backend/runtime/oaf_tf1/wheelhouse/runtime
rtk /Users/chanwaichan/workspace/Crux/.venv/bin/python \
  -m tools.hpa320.resolve_oaf_runtime \
  --materialize-wheelhouse \
  --lock runtime/oaf_tf1/requirements-test.lock \
  --wheelhouse runtime/oaf_tf1/wheelhouse/test \
  --offline-cache \
  /Users/chanwaichan/workspace/Crux/.worktrees/hpa-320-freeze-backend/runtime/oaf_tf1/wheelhouse/test
```

Expected: runtime verifies 71 distributions against lock SHA-256 `9e00c42066a72c673051e65404d28c0eb7fe2833b1266db6e451395af3fa1457`; test verifies 9 distributions against lock SHA-256 `574406e35b7c226b1f4dbd3decd9fe6cf7a2ccd75b98b28923cc466e63fcb193`. Stop if either count or digest differs.

- [ ] **Step 6: Amend the HPA-320 design at the six exact headings**

Add a short HPA-481/HPA-482 amendment under each affected heading. Each amendment points to `2026-07-30-hpa-481-work-job-artifact-attestation-design.md`, states that all unrelated HPA-320 contracts remain in force, and names these literal paragraph openings:

| Heading | Paragraph opening named by the amendment |
| --- | --- |
| `Native-amd64 calibration and sealing prerequisite` | “Every newly accepted bootstrap, measurement, and candidate execution preserves its own `crux.oaf-native-host-attestation-bundle/v1`.” |
| `Calibration-bootstrap authority` | “`bootstrap-image` materializes two fresh copies of the exact request-authenticated build context and performs one `linux/amd64` build from each.” |
| `Calibration, candidate publication, and seal` | “After those probes, `calibrate` generates exact 130/78/52 tensor coverage, uninitialized-variable evidence, active-dropout evidence, patched/unmodified `NoteSequence` parity, the nonempty raw smoke oracle and calibration-native event payload, security/advisory evidence, OCI archive/manifest/config/layer evidence, candidate runtime/backend/seal payloads, and a manifest of every candidate artifact.” |
| `Verification and Execution Flow` | “`verify-backend` performs the following steps in order:” |
| `Normative Schema Key Reference` > `Provenance and report schemas` | “The host-attestation bundle's `phase` is exactly one of `bootstrap`, `measurement`, or `candidate`.” |
| `Continuous Integration` | “A manual native Linux `amd64` bootstrap workflow accepts one exact commit SHA, checks out that commit, and records it in regenerated authenticated native-host evidence.”; “A manual native Linux `amd64` measurement workflow accepts that exact measurement-request commit, regenerates authenticated native-host evidence, obtains the exact accepted OCI archive or rebuilds and exact-compares it, regenerates and exact-compares all five fixture rows, runs exactly three fresh-container repetitions per row, validates the exact 15 healthy measurement rows, and uploads a canonical artifact manifest plus the complete measurement host-attestation bundle and every measurement and sanitized diagnostic artifact needed for review.”; and “A separate manual candidate phase accepts that exact profile commit, regenerates native-host evidence for it, strict-validates the committed evidence/profile chain, runs `calibrate`, and uploads the complete candidate, its host-attestation bundle, and a canonical artifact manifest for review.” |

Preserve the Markdown code formatting around identifiers when copying these openings into the amendments. The `Verification and Execution Flow` amendment explicitly says the 13-step post-seal `verify-backend` behavior remains unchanged; only pre-consumption acceptance of bootstrap, measurement, and candidate artifacts now uses the signed outside-in v2 gate. Do not use line numbers or rewrite unrelated frozen behavior.

- [ ] **Step 7: Regenerate in the only valid identity order**

After all code, workflow, tests, and covered documentation bytes are final, run:

```bash
rtk uv run python -m tools.hpa320.generate_runner_source_manifest \
  --repository-root . \
  --output runtime/oaf_tf1/runner-source-manifest.json
rtk uv run python -m tools.hpa320.oaf_build_context generate \
  --repository-root . \
  --wheelhouse-root runtime/oaf_tf1/wheelhouse \
  --output runtime/oaf_tf1/build-context-manifest.json \
  --replace
rtk uv run python -m tools.hpa320.seal_oaf_backend \
  reissue-bootstrap-request \
  --repository-root . \
  --request \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json
```

Then prove idempotence without writing:

```bash
rtk uv run python -m tools.hpa320.generate_runner_source_manifest \
  --repository-root . \
  --output runtime/oaf_tf1/runner-source-manifest.json \
  --check
rtk uv run python -m tools.hpa320.oaf_build_context generate \
  --repository-root . \
  --wheelhouse-root runtime/oaf_tf1/wheelhouse \
  --output runtime/oaf_tf1/build-context-manifest.json \
  --check
rtk uv run pytest tests/benchmark/test_hpa320_freeze_artifact_drift.py -v
```

Expected: both generator checks and all drift tests pass. Do not regenerate again after changing any covered byte.

- [ ] **Step 8: Mark the design implemented and run focused integration tests**

Change the HPA-481/HPA-482 design status from `Proposed` to `Implemented` only now. Run:

```bash
rtk uv run pytest \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_backend_publication.py \
  tests/benchmark/test_backend_lock.py \
  tests/benchmark/test_hpa320_github_host_evidence.py \
  tests/benchmark/test_oaf_host_attestation.py \
  tests/benchmark/test_oaf_native_artifacts.py \
  tests/benchmark/test_oaf_build_context.py \
  tests/benchmark/test_hpa320_freeze_artifact_drift.py \
  tests/benchmark/test_oaf_backend.py \
  tests/benchmark/test_oaf_seal.py \
  tests/benchmark/test_hpa320_native_bootstrap_workflow.py \
  tests/benchmark/test_hpa320_native_measurement_workflow.py \
  tests/benchmark/test_hpa320_native_candidate_workflow.py \
  runtime/oaf_tf1/tests/test_calibration_protocol.py \
  -v
```

Expected: PASS.

- [ ] **Step 9: Run the complete repository verification contract**

Run the current CI commands exactly:

```bash
rtk uv run pytest
rtk uv run ruff check .
rtk uv run ruff format --check src tests
rtk uv run pylint --errors-only --disable=E1120,E0401 src
```

Run the additional repository commands exactly:

```bash
rtk uv run ruff check src tests
rtk uv run black --check src tests
rtk uv run pylint src/app src/cli
```

Run the focused tool checks exactly:

```bash
rtk uv run ruff format --check \
  tools/hpa320/oaf_host_attestation.py \
  tools/hpa320/github_host_evidence.py \
  tools/hpa320/oaf_native_artifacts.py
rtk uv run pylint --errors-only --disable=E1120,E0401 \
  tools/hpa320/oaf_host_attestation.py \
  tools/hpa320/github_host_evidence.py \
  tools/hpa320/oaf_native_artifacts.py
```

Expected: every command passes. Record any platform-unavailable check explicitly; do not substitute a narrower scope.

- [ ] **Step 10: Review the final diff and commit the regenerated freeze**

Run:

```bash
rtk git diff --check
rtk git status --short
rtk git diff --stat
rtk git diff -- \
  runtime/oaf_tf1/runner-source-manifest.json \
  runtime/oaf_tf1/build-context-manifest.json \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json
```

Confirm the runner manifest and build context changed only for final covered source bytes, the request changed only its current cross-hashes, and the independently frozen inputs stayed byte-identical. Then:

```bash
rtk git add \
  tools/hpa320/generate_runner_source_manifest.py \
  tools/hpa320/oaf_build_context.py \
  tools/hpa320/seal_oaf_backend.py \
  tests/benchmark/test_oaf_build_context.py \
  tests/benchmark/test_hpa320_freeze_artifact_drift.py \
  docs/superpowers/specs/2026-07-26-hpa-320-freeze-oaf-drums-backend-design.md \
  docs/superpowers/specs/2026-07-30-hpa-481-work-job-artifact-attestation-design.md \
  runtime/oaf_tf1/runner-source-manifest.json \
  runtime/oaf_tf1/build-context-manifest.json \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json
rtk git commit -m "docs: reissue native bootstrap authority"
```

Expected: the final commit contains the drift guards, supersession notes, deterministic regenerated manifests, and reissued pre-dispatch request. HPA-423 remains blocked until this branch is reviewed, merged with HPA-482, the legacy workflow is disabled or its trigger branch retired, and the exact merged commit is confirmed.
