# HPA-320 Phase A Seal Closure Implementation Plan

> **Historical snapshot:** This plan captures the design intent at the time of writing. Checkbox states and version references reflect the original plan, not the current repository state.
> **Status:** Tasks 1-7 were implemented through commit `78b2424`. The remaining
> native image-bootstrap, calibration, and final-seal work is superseded by
> [the deterministic OCI seal-closure plan](2026-07-29-hpa-320-deterministic-oci-seal-closure.md).
> Keep this document as the historical implementation record; do not execute its
> Tasks 8-9 against the amended design.

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Complete the native OaF bootstrap, exact runtime protocol, deterministic
verification, and acyclic seal so the released E-GMD checkpoint becomes a genuinely
frozen Crux backend.

**Architecture:** Extend the existing Phase A host/runner implementation instead of
rebuilding it. Four checked-in request authorities drive checkpoint acquisition,
base-system attestation, diagnostic measurement, and reviewed seal-profile
selection. Two producer stages (`measure`, then `calibrate`) create immutable
evidence without choosing values automatically. Final locks are published only after
native `linux/amd64` verification, then the registry flips from `preseal` to `sealed`
in the same commit.

**Tech Stack:** Python 3.12 host, Python 3.7.17 isolated runner,
TensorFlow 1.15.5 CPU, Docker/OCI `linux/amd64`, Click 8, Pytest, SHA-256,
canonical JSON/JSONL, `/usr/bin/env -i`, standard-library `argparse`, `struct`,
`decimal`, `subprocess`, `threading`, `wave`, and immutable filesystem publication.

## Global Constraints

- Prerequisite: complete
  `docs/superpowers/plans/2026-07-28-hpa-320-phase-b-contract-closure.md`.
- Normative design:
  `docs/superpowers/specs/2026-07-26-hpa-320-freeze-oaf-drums-backend-design.md`
  at or after commit `d2ca20c`.
- Backend ID: `magenta-egmd-tf1-94529798-8hit-v1`.
- Model ID: `magenta-egmd-ckpt-569400-v1`.
- Architecture ID: `magenta-oaf-model-tpu-drums-v1`.
- Native output space: `magenta-oaf-midi88-a0-v1`.
- Training map: `magenta-egmd-data-8hit-94529798-v1`.
- Native metadata schema: `magenta-oaf-native-metadata-v1`, containing exactly
  `upstream_8hit_group_id`.
- Magenta source commit:
  `94529798dfbbb14c27ddfd76f23027dc8e2ce185`.
- Checkpoint archive SHA-256:
  `09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0`.
- Exact archive members and component hashes remain those in the committed design;
  only the three authenticated TensorFlow components enter the model cache.
- Runtime base is `python:3.7.17-slim-bullseye` for `linux/amd64`, pinned by manifest
  digest
  `sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673`.
- TensorFlow is exactly `1.15.5`; official inference is CPU-only and native
  `linux/amd64`.
- Runner execution is network-free, non-root, read-only, capabilities-dropped,
  `no-new-privileges`, resource-bounded, and one request at a time.
- The entrypoint uses `/usr/bin/env -i`. Its image-baked environment constant is the
  earliest execution authority; the mounted runtime lock is the sealed record; host,
  image, and runner copies must exact-match.
- The seven runtime environment variables and bootstrap-only
  `PYTHONCOERCECLOCALE=0` remain exactly as specified in the design.
- The host numeric fingerprint contains only `architecture`, `cpu_vendor_id`,
  `cpu_family`, `cpu_model`, and `cpu_stepping`. It is diagnostic, not an acceptance
  partition.
- Every authenticated smoke mismatch is `failed`, exit `2`, on every CPU
  fingerprint. `environment_unsupported`, exit `1`, is pre-inference platform or
  evidence failure only.
- Wire time and confidence are exactly 16 lowercase hexadecimal digits encoding
  big-endian IEEE-754 binary64. JSON decimal wire values are invalid.
- Raw velocity `+inf`, `-inf`, and `NaN` are all rejected even though upstream would
  produce `127`, `0`, and `0` respectively.
- Full verification performs exactly five inference calls: two startup raw-oracle
  checks, one per process, plus three post-ready artifact requests.
- No training, tuning, weight modification, architecture replacement, HDF5
  substitution, hidden chunking, partial restore, network inference, or heuristic
  fallback.
- No final resource, deadline, UID/GID, package, frame-bound, or smoke value may be
  guessed. Missing native evidence stops the plan at Task 8.
- Python lines remain within 100 characters.

## Execution Order

Tasks 1-7 are host-testable with fake native evidence, fake runners, and immutable
fixtures. Task 8 requires a newly generated accepted native `linux/amd64` evidence
record and is the only task allowed to publish final locks, oracle, seal evidence,
OCI evidence, or change the registry to `sealed`. Task 9 is the final acceptance
gate.

---

## File Map

### New checked-in authorities and evidence

- `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json`
- `runtime/oaf_tf1/base-system-package-request.json`
- `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-measurement-request.json`
- `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-profile-request.json`
- Final Phase A locks/evidence named in the design, created only in Task 8.

### New source and tests

- `src/benchmark/checkpoint_acquisition.py` — strict request/evidence parsing and
  request-to-cache cross-checking.
- `tests/benchmark/test_checkpoint_acquisition.py`
- `tests/benchmark/schema_goldens/` — Phase A rows and canonical payloads appended to
  the Phase B manifest.

### Modified source

- `src/benchmark/backend_prepare.py`
- `src/benchmark/backend_lock.py`
- `src/benchmark/backend_process.py`
- `src/benchmark/backend_attestation.py`
- `src/benchmark/backend_registry.py`
- `src/benchmark/backends/oaf_tf1.py`
- `src/cli/benchmark.py`
- `runtime/oaf_tf1/Dockerfile`
- `runtime/oaf_tf1/entrypoint.py`
- `runtime/oaf_tf1/protocol.py`
- `runtime/oaf_tf1/oaf_backend.py`
- `tools/hpa320/oaf_system_packages.py`
- `tools/hpa320/seal_oaf_backend.py`
- `tools/hpa320/generate_runner_source_manifest.py`
- `.github/workflows/hpa320-native-host-evidence.yml`
- `.github/workflows/oaf-backend-integration.yml`

### Cross-task interfaces

- Acquisition:
  `CheckpointAcquisitionRequest`, `CheckpointAcquisitionEvidence`,
  `load_checkpoint_acquisition_request`,
  `load_checkpoint_acquisition_evidence`.
- Base system:
  `BaseSystemPackageRequest`, `BaseSystemPackageEvidence`,
  `attest_base_system`.
- Calibration:
  `CalibrationMeasurementRequest`, `CalibrationMeasurementEvidence`,
  `SealProfileRequest`, `SealCandidate`, `measure`, `calibrate`.
- Locks:
  existing `LoadedBackendLock`, `LoadedRuntimeLock`, `LoadedSealEvidence`,
  and `validate_oaf_lock_set` with final v1 key sets.
- Protocol:
  `encode_binary64`, `decode_binary64`, `TranscribeRequest`,
  `NativeProtocolEvent`, `serve_requests`.
- Verification:
  existing `OafTf1Backend.verify`, `OafTf1Backend.transcribe`,
  `run_verify_backend`, and `run_transcribe_one`.

---

### Task 1: Make checkpoint acquisition independently authoritative

**Files:**

- Create: `src/benchmark/checkpoint_acquisition.py`
- Create: `tests/benchmark/test_checkpoint_acquisition.py`
- Create:
  `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json`
- Modify: `src/benchmark/backend_prepare.py`
- Modify: `src/cli/benchmark.py`
- Modify: `tests/benchmark/test_backend_prepare.py`
- Modify: `tests/test_cli_benchmark.py`

**Interfaces:**

- Consumes: the exact released archive/member identities in the design.
- Produces:

```python
@dataclass(frozen=True)
class CheckpointIdentity:
    name: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ArchiveMemberIdentity(CheckpointIdentity):
    role: Literal["pointer", "published_component"]


@dataclass(frozen=True)
class CheckpointAcquisitionRequest:
    backend_id: str
    checkpoint_url: str
    archive: CheckpointIdentity
    archive_members: tuple[ArchiveMemberIdentity, ...]
    published_component_names: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class CheckpointAcquisitionEvidence:
    request_sha256: str
    acquisition_mode: Literal["download", "archive", "cache_verify"]
    model_artifact_set_sha256: str
    cache_path: PurePosixPath
    sha256: str
```

- [ ] **Step 1: Write strict request/evidence tests**

```python
def test_checkpoint_request_is_complete_without_final_backend_lock(
    request_path: Path,
) -> None:
    request = load_checkpoint_acquisition_request(request_path)

    assert request.backend_id == OFFICIAL_BACKEND_ID
    assert request.archive.sha256 == EXPECTED_ARCHIVE_SHA256
    assert len(request.archive_members) == 4
    assert len(request.published_component_names) == 3
    assert {member.role for member in request.archive_members} == {
        "pointer",
        "published_component",
    }
```

Parameterize unknown, duplicate, missing, renamed, unsafe, and extra member cases.
Assert the pointer payload contains exactly the two expected checkpoint lines.

- [ ] **Step 2: Run tests and verify the current final-lock dependency**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_checkpoint_acquisition.py \
  tests/benchmark/test_backend_prepare.py \
  tests/test_cli_benchmark.py \
  -k 'checkpoint or prepare_backend'
```

Expected: FAIL because `prepare-backend` currently requires the absent final backend
lock.

- [ ] **Step 3: Implement strict request/evidence loaders**

Use `strict_json_loads(..., require_canonical=True)` and exact key sets. Evidence
reproduces the request hash, all four member identities, the three published
components, artifact-set hash, acquisition mode, and repository-relative cache path.
Both loaders reject symlinks and unstable reads.

- [ ] **Step 4: Extend the preparation request**

Change the host request to:

```python
@dataclass(frozen=True)
class PrepareBackendRequest:
    backend_id: str
    cache_root: Path
    archive_path: Path | None
    download: bool
    acquisition_request_path: Path
    evidence_output_path: Path | None
    backend_lock_path: Path | None
```

Pre-seal mode requires the acquisition request and evidence output. Post-seal mode
loads the fixed request and final lock, then exact-compares them before touching the
cache.

- [ ] **Step 5: Implement the three setup modes and exits**

`download` and `archive` remain mutually exclusive. With neither:

```python
if not model_cache_path.exists():
    return PrepareBackendOutcome(
        status="acquisition_failed",
        exit_code=1,
        code="model_cache_missing",
        evidence_artifact=None,
    )
```

This path never opens the network. Operational acquisition/staging inability returns
exit `1`; authenticated request/byte/cache/publication contradiction returns exit
`2`. Publish evidence immutably only after the cache has been reverified.

- [ ] **Step 6: Update the Click command**

Add:

```text
--acquisition-request FILE
--evidence-output FILE
--backend-lock FILE
```

Emit the common four-key summary from Phase B. When evidence is published,
`report_path` and `report_sha256` identify that evidence. A Click usage error emits no
summary.

- [ ] **Step 7: Run focused tests**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_checkpoint_acquisition.py \
  tests/benchmark/test_backend_prepare.py \
  tests/test_cli_benchmark.py
```

Expected: PASS.

- [ ] **Step 8: Commit acquisition authority**

```bash
rtk git add \
  src/benchmark/checkpoint_acquisition.py \
  src/benchmark/backend_prepare.py \
  src/cli/benchmark.py \
  tests/benchmark/test_checkpoint_acquisition.py \
  tests/benchmark/test_backend_prepare.py \
  tests/test_cli_benchmark.py \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json
rtk git commit -m "feat: authenticate OaF checkpoint acquisition"
```

---

### Task 2: Freeze base-system package requests and evidence

**Files:**

- Create: `runtime/oaf_tf1/base-system-package-request.json`
- Modify: `tools/hpa320/oaf_system_packages.py`
- Modify: `tools/hpa320/seal_oaf_backend.py`
- Modify: `runtime/oaf_tf1/Dockerfile`
- Modify: `tests/benchmark/test_oaf_smoke_generator.py`
- Modify: `tests/benchmark/test_oaf_seal.py`

**Interfaces:**

- Consumes: pinned base-image manifest, base archive keyring hash, native host
  evidence, and exact required probes.
- Produces:

```python
@dataclass(frozen=True)
class BaseSystemPackageRequest:
    base_image: str
    base_image_manifest_digest: str
    base_image_archive_keyring_sha256: str
    platform: Literal["linux/amd64"]
    required_probes: tuple[str, ...]
    additional_system_packages: tuple[()]
    sha256: str


@dataclass(frozen=True)
class BaseSystemPackageEvidence:
    request_sha256: str
    base_image_manifest_digest: str
    package_inventory: tuple[SystemPackage, ...]
    package_inventory_sha256: str
    probes: tuple[ProbeResult, ...]
    native_host_evidence: NativeHostEvidence
    sha256: str
```

- [ ] **Step 1: Write failing request/evidence tests**

```python
def test_base_system_evidence_reproduces_exact_inventory(
    request_path: Path,
    evidence_path: Path,
) -> None:
    request = load_base_system_package_request(request_path)
    evidence = load_base_system_package_evidence(evidence_path)

    assert evidence.request_sha256 == request.sha256
    assert evidence.base_image_manifest_digest == request.base_image_manifest_digest
    assert evidence.package_inventory_sha256 == inventory_sha256(
        evidence.package_inventory
    )
    assert request.additional_system_packages == ()
```

Reject duplicate or unsorted `name/version/architecture` rows, a nonempty additional
package array, an unapproved probe, a base-manifest mismatch, or a copied keyring in
the bundle.

- [ ] **Step 2: Run the test and expose the current bundle-only contract**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_oaf_smoke_generator.py \
  tests/benchmark/test_oaf_seal.py \
  -k 'base_system or system_package'
```

Expected: FAIL because current tooling authenticates a package bundle but has no
checked-in base-system request/evidence pair.

- [ ] **Step 3: Implement `attest-base-system`**

Add this producer signature:

```python
def attest_base_system(
    *,
    request_path: Path,
    host_evidence_path: Path,
    image: str,
    output_path: Path,
) -> PublishedArtifact:
    """Publish immutable base-system evidence after exact native probes."""
```

It verifies the image manifest, archive keyring hash, full sorted
`Package<TAB>Version<TAB>Architecture` inventory, empty additional package set, and
all request-named probes. It stages, fsyncs, and no-replace publishes canonical
evidence.

- [ ] **Step 4: Remove obsolete runtime-lock package fields**

Do not write `debian_snapshot_repository`, `debian_release_sha256`, or
`system_packages` into candidate runtime locks. Retain Python distribution identity
separately.

- [ ] **Step 5: Run focused package tests**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_oaf_smoke_generator.py \
  tests/benchmark/test_oaf_seal.py
```

Expected: PASS.

- [ ] **Step 6: Commit base-system authority**

```bash
rtk git add \
  runtime/oaf_tf1/base-system-package-request.json \
  runtime/oaf_tf1/Dockerfile \
  tools/hpa320/oaf_system_packages.py \
  tools/hpa320/seal_oaf_backend.py \
  tests/benchmark/test_oaf_smoke_generator.py \
  tests/benchmark/test_oaf_seal.py
rtk git commit -m "feat: attest OaF base system"
```

---

### Task 3: Implement two-stage measurement and calibration

**Files:**

- Create:
  `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-measurement-request.json`
- Create:
  `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-profile-request.json`
- Modify: `tools/hpa320/seal_oaf_backend.py`
- Modify: `tests/benchmark/test_oaf_seal.py`
- Modify: `tests/test_cli_benchmark.py`

**Interfaces:**

- Consumes: Tasks 1-2, provisional image, fake/native host evidence, and the exact
  canonical fixture set.
- Produces:

```python
@dataclass(frozen=True)
class MeasurementRow:
    input_frame_count: int
    repetition: int
    process_instance_id: str
    peak_cpu_millis: int
    peak_rss_bytes: int
    peak_tmp_bytes: int
    peak_shm_bytes: int
    peak_pid_count: int
    startup_millis: int
    request_millis: int
    stdout_max_line_bytes: int
    stderr_max_line_bytes: int
    exit_code: int
    signal: int | None
    oom_killed: bool
    prediction_sha256: str | None
```

`measure(...) -> PublishedArtifact` and
`calibrate(...) -> PublishedSealCandidate`.

- [ ] **Step 1: Write failing request-separation tests**

```python
def test_measure_cannot_publish_seal_candidate(
    measurement_request: Path,
    output_path: Path,
) -> None:
    artifact = measure(
        request_path=measurement_request,
        host_evidence_path=fake_native_evidence(),
        image="sha256:" + "a" * 64,
        model_cache=model_cache(),
        checkpoint_evidence_path=checkpoint_evidence(),
        base_system_evidence_path=base_system_evidence(),
        output_path=output_path,
    )

    assert artifact.path == output_path
    assert json.loads(output_path.read_bytes())["schema"] == (
        "crux.oaf-calibration-measurement-evidence/v1"
    )
    assert not (output_path.parent / "candidate-manifest.json").exists()
```

Add tests that `calibrate` rejects missing, unrelated, sentinel-bearing, or
underprovisioned profile requests.

- [ ] **Step 2: Run tests and verify current `calibrate` is a blocker stub**

```bash
rtk uv run pytest -q tests/benchmark/test_oaf_seal.py -k 'measure or calibrate'
```

Expected: FAIL because current `calibrate` always raises
`native calibration output contract is not specified`.

- [ ] **Step 3: Implement diagnostic `measure`**

Strict-load the measurement request. For every requested frame count and repetition,
run the exact container restrictions and collect every `MeasurementRow`. Sort by
frame count, process instance, and repetition. Publish only
`crux.oaf-calibration-measurement-evidence/v1`.

Do not choose a bound, resource limit, UID/GID, or deadline in code.

- [ ] **Step 4: Implement reviewed `calibrate`**

Strict-load the seal-profile request and exact-compare all acquisition, base-system,
measurement request, and measurement evidence hashes. Require every chosen bound to
be explicit and strictly above the corresponding observed in-bound peak after unit
conversion.

Run:

- `max_input_audio_frames - 1`;
- exact `max_input_audio_frames`;
- one frame over the bound;
- two requests in one persistent process; and
- one request in a fresh process.

Over-bound input must fail before inference. In-bound OOM, signal, allocator, or
timeout invalidates the candidate.

- [ ] **Step 5: Standardize producer summaries and exits**

Add:

```python
@dataclass(frozen=True)
class ProducerOutcome:
    status: str
    exit_code: Literal[0, 1, 2]
    report_path: Path | None
    report_sha256: str | None
```

Operational inability returns exit `1`; integrity/authentication/publication
contradiction returns exit `2`; success returns `0`. Argparse usage remains exit `2`
before a typed summary. Every parsed command emits exactly the common four summary
keys.

- [ ] **Step 6: Run producer tests**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_oaf_seal.py \
  tests/test_cli_benchmark.py \
  -k 'measure or calibrate or producer'
```

Expected: PASS.

- [ ] **Step 7: Commit the two-stage gate**

```bash
rtk git add \
  tools/hpa320/seal_oaf_backend.py \
  tests/benchmark/test_oaf_seal.py \
  tests/test_cli_benchmark.py \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-measurement-request.json \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-profile-request.json
rtk git commit -m "feat: measure OaF seal profile"
```

---

### Task 4: Migrate backend, runtime, and seal schemas

**Files:**

- Modify: `src/benchmark/backend_lock.py`
- Modify: `runtime/oaf_tf1/oaf_backend.py`
- Modify: `tests/benchmark/test_backend_lock.py`
- Modify: `runtime/oaf_tf1/tests/test_tensor_coverage.py`
- Modify: `tests/benchmark/test_oaf_seal.py`
- Modify: `tests/benchmark/schema_goldens/manifest.json`
- Create: Phase A JSON goldens under `tests/benchmark/schema_goldens/`

**Interfaces:**

- Consumes: Phase B golden harness and Tasks 1-3 request/evidence types.
- Produces: final strict v1 lock/evidence loaders matching the normative appendix.

- [ ] **Step 1: Add failing exact-key tests**

```python
def test_runtime_lock_uses_base_system_evidence_not_legacy_package_fields(
    runtime_payload: dict[str, object],
) -> None:
    runtime_payload.update(
        {
            "additional_system_packages": [],
            "base_image_archive_keyring_sha256": "a" * 64,
            "base_system_package_request_sha256": "b" * 64,
            "base_system_package_evidence_sha256": "c" * 64,
            "base_system_package_inventory": [
                {
                    "architecture": "amd64",
                    "name": "libc6",
                    "version": "2.31-13+deb11u11",
                }
            ],
            "base_system_package_inventory_sha256": "d" * 64,
        }
    )
    for field in (
        "debian_snapshot_repository",
        "debian_release_sha256",
        "system_packages",
    ):
        runtime_payload.pop(field)

    loaded = load_runtime_lock(write_canonical(runtime_payload))
    assert loaded.payload["additional_system_packages"] == ()
```

Add backend-lock tests for checkpoint request/evidence hashes and seal-evidence tests
for all request/evidence/profile/candidate/fingerprint fields.

- [ ] **Step 2: Run lock tests and verify exact-key failure**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_lock.py \
  tests/benchmark/test_oaf_seal.py \
  runtime/oaf_tf1/tests/test_tensor_coverage.py
```

Expected: FAIL because current exact key sets still require legacy Debian fields and
lack request/evidence hashes.

- [ ] **Step 3: Replace host and runner key sets together**

Update both validators from the same design appendix in one commit. Add
`checkpoint_acquisition_request_sha256` and
`checkpoint_acquisition_evidence_sha256` to the backend lock. Replace runtime
package fields with base-system request/evidence/inventory fields and
`additional_system_packages`.

Extend seal evidence with:

```text
checkpoint_acquisition_request_sha256
checkpoint_acquisition_evidence_sha256
base_system_package_request_sha256
base_system_package_evidence_sha256
base_system_package_inventory
base_system_package_inventory_sha256
base_image_archive_keyring_sha256
additional_system_packages
reference_host_numeric_fingerprint
calibration_measurement_request_sha256
calibration_measurement_evidence_sha256
seal_profile_request_sha256
seal_candidate_sha256
```

Remove only the three obsolete package-distribution fields named above.

- [ ] **Step 4: Fix native metadata lock identity**

The exact lock metadata field list is:

```python
(
    MappingProxyType(
        {
            "name": "upstream_8hit_group_id",
            "nullable": True,
            "type": "string",
        }
    ),
)
```

`frame_index` remains a protocol field, not persisted native metadata.

- [ ] **Step 5: Add Phase A schema goldens**

Append rows and canonical payloads for every Phase A schema named in the design,
including requests, evidence, locks, runner protocol support, smoke oracle, tensor
coverage, OCI manifest, and source manifests. List both host and runner validators
for schemas enforced across the isolation boundary.

- [ ] **Step 6: Run lock and schema suites**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_lock.py \
  tests/benchmark/test_oaf_seal.py \
  tests/benchmark/test_schema_goldens.py \
  runtime/oaf_tf1/tests/test_tensor_coverage.py
```

Expected: PASS.

- [ ] **Step 7: Commit schema migration**

```bash
rtk git add \
  src/benchmark/backend_lock.py \
  runtime/oaf_tf1/oaf_backend.py \
  tests/benchmark/test_backend_lock.py \
  tests/benchmark/test_oaf_seal.py \
  tests/benchmark/test_schema_goldens.py \
  tests/benchmark/schema_goldens \
  runtime/oaf_tf1/tests/test_tensor_coverage.py
rtk git commit -m "refactor: align OaF seal schemas"
```

---

### Task 5: Authenticate native fingerprints and environment authority

**Files:**

- Modify: `src/benchmark/backend_process.py`
- Modify: `src/benchmark/backend_attestation.py`
- Modify: `src/benchmark/backends/oaf_tf1.py`
- Modify: `runtime/oaf_tf1/entrypoint.py`
- Modify: `runtime/oaf_tf1/Dockerfile`
- Modify: `tools/hpa320/seal_oaf_backend.py`
- Modify: `tests/benchmark/test_backend_process.py`
- Modify: `tests/benchmark/test_backend_attestation.py`
- Modify: `tests/benchmark/test_oaf_backend.py`
- Modify: `runtime/oaf_tf1/tests/test_protocol.py`

**Interfaces:**

- Consumes: Phase B `HostNumericFingerprint`.
- Produces: exact fingerprint-bearing native evidence and three-way environment
  agreement.

- [ ] **Step 1: Write failing native-evidence tests**

```python
def test_github_native_evidence_requires_diagnostic_fingerprint(
    github_payload: dict[str, object],
) -> None:
    github_payload["host_numeric_fingerprint"] = {
        "architecture": "x86_64",
        "cpu_vendor_id": "GenuineIntel",
        "cpu_family": "6",
        "cpu_model": "143",
        "cpu_stepping": "8",
    }

    evidence = NativeHostEvidence(
        kind="github_hosted",
        payload=github_payload,
        sha256=identity_sha256(github_payload),
        official_execution_allowed=True,
    )

    assert evidence.host_numeric_fingerprint.cpu_vendor_id == "GenuineIntel"
```

Require the same object inside all three evidence payload variants. Reject
`cpu_microcode`, `kernel_release`, mixed visible logical CPU tuples, or a fingerprint
outside the authenticated payload hash.

- [ ] **Step 2: Add environment-agreement tests**

Test that:

- Docker ENTRYPOINT starts `/usr/bin/env -i`;
- `PYTHONCOERCECLOCALE=0` exists only until entrypoint bootstrap removes it;
- the post-bootstrap process environment has exactly seven fields;
- the image constant, runtime lock, and host launcher environment exact-match;
- missing, extra, or mismatched values fail before importing TensorFlow.

- [ ] **Step 3: Run focused tests and record current gaps**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_backend_attestation.py \
  tests/benchmark/test_oaf_backend.py \
  runtime/oaf_tf1/tests/test_protocol.py
```

Expected: fingerprint cases fail; existing `/usr/bin/env -i` tests continue to pass.

- [ ] **Step 4: Implement fingerprint capture and validation**

Add host-side collection from trusted native evidence. The workflow may read
`/proc/cpuinfo`, but it must normalize only the five stable fields and require all
visible logical CPUs to match. Copy the seal host object to
`reference_host_numeric_fingerprint`; copy the current object to every execution
attestation.

- [ ] **Step 5: Enforce three-way environment equality**

At host preflight:

```python
if dict(runtime_lock.environment) != EXPECTED_ENVIRONMENT:
    raise BackendFatalFailure(
        BackendError(
            code="runtime_environment_mismatch",
            message="Runtime environment identity does not match the image.",
        )
    )
```

The runner independently performs the same comparison after parsing the mounted
runtime lock. No copy overrides another.

- [ ] **Step 6: Run focused tests**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_backend_attestation.py \
  tests/benchmark/test_oaf_backend.py \
  runtime/oaf_tf1/tests/test_protocol.py
```

Expected: PASS.

- [ ] **Step 7: Commit native environment evidence**

```bash
rtk git add \
  src/benchmark/backend_process.py \
  src/benchmark/backend_attestation.py \
  src/benchmark/backends/oaf_tf1.py \
  runtime/oaf_tf1/entrypoint.py \
  runtime/oaf_tf1/Dockerfile \
  tools/hpa320/seal_oaf_backend.py \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_backend_attestation.py \
  tests/benchmark/test_oaf_backend.py \
  runtime/oaf_tf1/tests/test_protocol.py
rtk git commit -m "fix: authenticate OaF execution environment"
```

---

### Task 6: Replace decimal wire numbers with exact binary64 hex

**Files:**

- Modify: `runtime/oaf_tf1/protocol.py`
- Modify: `runtime/oaf_tf1/oaf_backend.py`
- Modify: `src/benchmark/backends/oaf_tf1.py`
- Modify: `runtime/oaf_tf1/tests/test_protocol.py`
- Modify: `runtime/oaf_tf1/tests/test_instrumentation.py`
- Modify: `tests/benchmark/test_oaf_backend.py`
- Modify: `tests/benchmark/test_oaf_real_checkpoint.py`
- Modify: `tests/fixtures/fake_oaf_runner.py`

**Interfaces:**

- Consumes: existing `NativeEvent` and exact frame-time expression.
- Produces:

```python
def encode_binary64(value: float) -> str:
    if not math.isfinite(value):
        raise ProtocolFailure("nonfinite_event", "Event value is nonfinite.", fatal=False)
    return struct.pack(">d", value).hex()


def decode_binary64(value: object, field: str) -> float:
    """Require 16 lowercase hex digits and return one finite IEEE-754 binary64."""
```

Runner event keys become exactly:

```text
frame_index
time_sec_binary64
native_class_id
model_output_bin
native_midi_note
upstream_8hit_group_id
confidence_binary64
velocity_midi
```

- [ ] **Step 1: Write failing bit-identity tests**

```python
@pytest.mark.parametrize(
    "value",
    [0.0, -0.0, 1.0 / (44100.0 / 512.0), 0.625, 1.0],
)
def test_binary64_wire_round_trip_preserves_bits(value: float) -> None:
    encoded = encode_binary64(value)

    assert re.fullmatch(r"[0-9a-f]{16}", encoded)
    assert struct.pack(">d", decode_binary64(encoded, "value")) == struct.pack(
        ">d",
        value,
    )
```

Reject uppercase, prefixes, wrong length, non-hex, JSON numbers, decimal strings,
NaN, and infinities.

- [ ] **Step 2: Add exact frame-expression tests**

Require:

```python
frames_per_second = 44100 / 512
frame_length_seconds = 1.0 / frames_per_second
expected = frame_index * frame_length_seconds
```

The host re-evaluates this expression and compares the binary64 bits to
`time_sec_binary64`. It never substitutes its value.

- [ ] **Step 3: Add full nonfinite velocity divergence tests**

```python
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_velocity_rejects_every_nonfinite_value(value: float) -> None:
    with pytest.raises(AdapterItemFailure, match="finite"):
        velocity_to_midi(value)
```

Keep finite upstream clamp/scale/truncate cases unchanged.

- [ ] **Step 4: Run protocol tests and verify current decimal fields fail**

```bash
rtk uv run pytest -q \
  runtime/oaf_tf1/tests/test_protocol.py \
  runtime/oaf_tf1/tests/test_instrumentation.py \
  tests/benchmark/test_oaf_backend.py
```

Expected: FAIL because current wire events use `time_sec_raw`,
`confidence_raw`, `upstream_group_id`, and `velocity`.

- [ ] **Step 5: Update runner and host validators together**

Encode once in the runner. Decode once in the host. Pass the reconstructed floats to
the existing artifact quantizer, which alone performs
`Decimal.from_float(...).quantize(Decimal("0.000001"), ROUND_HALF_EVEN)`.

Do not send `frame_index` inside `native_metadata`; preserve it as protocol evidence
only. Map `upstream_8hit_group_id` to the one-key OaF metadata object.

- [ ] **Step 6: Update fake runner and oracle fixtures**

Regenerate fake runner payloads and smoke-oracle event rows to use exact binary64
strings. Do not generate or change the final native oracle in this host-only task.

- [ ] **Step 7: Run focused protocol and artifact tests**

```bash
rtk uv run pytest -q \
  runtime/oaf_tf1/tests/test_protocol.py \
  runtime/oaf_tf1/tests/test_instrumentation.py \
  tests/benchmark/test_oaf_backend.py \
  tests/benchmark/test_prediction_artifact.py
```

Expected: PASS.

- [ ] **Step 8: Commit wire migration**

```bash
rtk git add \
  runtime/oaf_tf1/protocol.py \
  runtime/oaf_tf1/oaf_backend.py \
  src/benchmark/backends/oaf_tf1.py \
  runtime/oaf_tf1/tests/test_protocol.py \
  runtime/oaf_tf1/tests/test_instrumentation.py \
  tests/benchmark/test_oaf_backend.py \
  tests/benchmark/test_oaf_real_checkpoint.py \
  tests/fixtures/fake_oaf_runner.py
rtk git commit -m "fix: preserve OaF binary64 event identity"
```

---

### Task 7: Enforce five-call smoke verification and failure policy

**Files:**

- Modify: `src/benchmark/backends/oaf_tf1.py`
- Modify: `src/benchmark/backend_process.py`
- Modify: `src/benchmark/transcription.py`
- Modify: `runtime/oaf_tf1/oaf_backend.py`
- Modify: `tests/benchmark/test_oaf_backend.py`
- Modify: `tests/benchmark/test_backend_process.py`
- Modify: `tests/benchmark/test_transcription.py`
- Modify: `tests/benchmark/test_oaf_real_checkpoint.py`

**Interfaces:**

- Consumes: Task 6 wire protocol and existing persistent runner controller.
- Produces:

```python
@dataclass(frozen=True)
class SmokeVerificationArtifacts:
    persistent_first: PublishedArtifact
    persistent_second: PublishedArtifact
    fresh_first: PublishedArtifact
```

- [ ] **Step 1: Write a failing five-call accounting test**

```python
def test_verify_backend_performs_two_startup_and_three_post_ready_calls(
    runner_factory: RecordingRunnerFactory,
) -> None:
    result = verify_oaf_with_repeatability(runner_factory)

    assert runner_factory.process_count == 2
    assert runner_factory.startup_smoke_calls == [1, 1]
    assert runner_factory.post_ready_calls == [2, 1]
    assert len(result.artifacts) == 3
```

Assert startup checks are handshake evidence and are not published as the three
comparable artifacts.

- [ ] **Step 2: Add cross-fingerprint failure tests**

Run the same authenticated smoke mismatch with reference and non-reference
fingerprints. Both must produce `status: "failed"`, exit `2`, code
`smoke_mismatch`, and no prediction. Add a pre-inference platform/evidence case that
alone produces `environment_unsupported`, exit `1`.

- [ ] **Step 3: Run focused tests and capture current accounting**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_oaf_backend.py \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_transcription.py \
  -k 'smoke or fingerprint or unsupported'
```

Expected: new five-call and cross-fingerprint cases fail.

- [ ] **Step 4: Implement explicit repeatability orchestration**

Use two explicit process lifetimes:

```python
with backend.start_verified_process() as persistent:
    first = backend.transcribe_smoke(persistent)
    second = backend.transcribe_smoke(persistent)
with backend.start_verified_process() as fresh:
    third = backend.transcribe_smoke(fresh)
```

Each `start_verified_process` already performs one raw-oracle startup inference before
`ready`. Compare the three post-ready JSONL artifacts byte-for-byte and against bytes
generated from oracle + descriptor + mounted locks.

- [ ] **Step 5: Keep `environment_unsupported` pre-inference only**

Remove any fingerprint comparison that selects unsupported status after smoke. A
completed mismatch always raises `BackendFatalFailure`.

- [ ] **Step 6: Run smoke and real-checkpoint test harnesses**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_oaf_backend.py \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_transcription.py \
  tests/benchmark/test_oaf_real_checkpoint.py
```

Expected: host/fake tests PASS. The real-checkpoint module skips only when its
explicit native integration flag is absent.

- [ ] **Step 7: Commit verification policy**

```bash
rtk git add \
  src/benchmark/backends/oaf_tf1.py \
  src/benchmark/backend_process.py \
  src/benchmark/transcription.py \
  runtime/oaf_tf1/oaf_backend.py \
  tests/benchmark/test_oaf_backend.py \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_transcription.py \
  tests/benchmark/test_oaf_real_checkpoint.py
rtk git commit -m "fix: prove OaF smoke repeatability"
```

---

### Task 8: Regenerate native evidence, calibrate, and publish the final seal

**Files:**

- Modify: `.github/workflows/hpa320-native-host-evidence.yml`
- Create: `.github/workflows/oaf-backend-integration.yml`
- Replace scratch only through a new run:
  `.github/hpa320-native-evidence/<new-job-id>/`
- Generate: final locks, seal evidence, acquisition/base-system/measurement
  evidence, tensor coverage, smoke oracle, security evidence, and OCI manifest.
- Modify: `src/benchmark/backend_registry.py` only after final verification succeeds.

**Interfaces:**

- Consumes: Tasks 1-7 and a newly completed native `linux/amd64` GitHub job or other
  accepted evidence form.
- Produces: the final immutable backend/runtime identity and required CI gate.

- [ ] **Step 1: Update the native evidence workflow**

Collect the exact diagnostic fingerprint in the authenticated payload. Do not reuse
`.github/hpa320-native-evidence/job-90329445692/` as final evidence; it predates this
contract.

The workflow must derive job/run identity dynamically from the completed job API
record rather than hard-code the old IDs.

- [ ] **Step 2: Run the four-authority bootstrap**

On the accepted native worker:

```bash
rtk uv run crux benchmark prepare-backend \
  --backend magenta-egmd-tf1-94529798-8hit-v1 \
  --acquisition-request \
    config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json \
  --evidence-output \
    artifacts/benchmark/backends/hpa320-bootstrap/checkpoint-acquisition-evidence.json \
  --cache-root artifacts/benchmark/model-cache \
  --download
rtk uv run python -m tools.hpa320.seal_oaf_backend \
  attest-base-system \
  --request runtime/oaf_tf1/base-system-package-request.json \
  --host-evidence /workspace/hpa320/native-host-evidence.json \
  --image crux-oaf-tf1:hpa320-seal \
  --output \
    artifacts/benchmark/backends/hpa320-bootstrap/base-system-package-evidence.json
rtk uv run python -m tools.hpa320.seal_oaf_backend \
  measure \
  --request \
    config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-measurement-request.json \
  --host-evidence /workspace/hpa320/native-host-evidence.json \
  --image crux-oaf-tf1:hpa320-seal \
  --model-cache artifacts/benchmark/model-cache \
  --checkpoint-evidence \
    artifacts/benchmark/backends/hpa320-bootstrap/checkpoint-acquisition-evidence.json \
  --base-system-evidence \
    artifacts/benchmark/backends/hpa320-bootstrap/base-system-package-evidence.json \
  --output artifacts/benchmark/backends/hpa320-bootstrap/calibration-measurements.json
```

Expected: each command exits `0` and publishes one immutable evidence artifact.

- [ ] **Step 3: Review measurement evidence and commit the profile request**

Review every row and choose explicit headroom values. Write the seal-profile request
with exact reviewed request/evidence hashes and no sentinel/default values. The
chosen `max_input_audio_frames` must be one measured frame count.

Commit that request separately:

```bash
rtk git add \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-profile-request.json
rtk git commit -m "docs: approve OaF seal profile"
```

- [ ] **Step 4: Calibrate and inspect the candidate**

```bash
rtk uv run python -m tools.hpa320.seal_oaf_backend \
  calibrate \
  --request \
    config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-profile-request.json \
  --measurement-evidence \
    artifacts/benchmark/backends/hpa320-bootstrap/calibration-measurements.json \
  --host-evidence /workspace/hpa320/native-host-evidence.json \
  --image crux-oaf-tf1:hpa320-seal \
  --model-cache artifacts/benchmark/model-cache \
  --checkpoint-evidence \
    artifacts/benchmark/backends/hpa320-bootstrap/checkpoint-acquisition-evidence.json \
  --base-system-evidence \
    artifacts/benchmark/backends/hpa320-bootstrap/base-system-package-evidence.json \
  --output artifacts/benchmark/backends/hpa320-seal-candidate
```

Expected: exit `0`; exact 130/78/52 inventories; patched/unmodified `NoteSequence`
parity; no active dropout; finite nonempty smoke; preserved OCI archive; no final
lock hashes inside candidate evidence.

- [ ] **Step 5: Seal without a content cycle**

Use the seal command from the design. Validate all candidate artifacts in memory,
publish immutable evidence/oracle/manifests first, runtime lock second, backend lock
last. Existing different bytes are integrity failure; never overwrite.

- [ ] **Step 6: Run final five-call verification**

```bash
rtk env CRUX_RUN_OAF_INTEGRATION=1 uv run pytest -q \
  tests/benchmark/test_oaf_real_checkpoint.py
rtk uv run crux benchmark verify-backend \
  --backend magenta-egmd-tf1-94529798-8hit-v1
```

Expected: real-checkpoint test PASS and verifier summary exit `0`, status
`verified`. Any `environment_unsupported`, skip, missing report, or smoke mismatch
fails the gate.

- [ ] **Step 7: Flip the registry and publish final outputs in one commit**

Change the OaF registration literal from `preseal` to `sealed`. Stage it with both
locks and all exact dependencies:

```bash
rtk git add \
  src/benchmark/backend_registry.py \
  config/benchmark/backends \
  runtime/oaf_tf1/host-adapter-source-manifest.json \
  docs/superpowers/evidence/hpa-320 \
  tests/fixtures/oaf_tf1_smoke/smoke-oracle.json \
  .github/hpa320-native-evidence
rtk git commit -m "feat: seal frozen OaF backend"
```

No commit may contain `seal_state: "sealed"` without both final locks.

- [ ] **Step 8: Add the required native integration workflow**

The job runs on native Linux X64 when inference-relevant files change, on schedule,
and on manual dispatch. It:

1. performs exact acquisition;
2. verifies base-system evidence;
3. builds or pulls by immutable digest;
4. reruns measurement/profile calibration;
5. runs `verify-backend`;
6. runs real-checkpoint tests; and
7. uploads every required report.

Pass only when verifier exit is `0`, report status is `verified`, all tests ran, and
all evidence uploads exist.

- [ ] **Step 9: Commit required CI**

```bash
rtk git add \
  .github/workflows/hpa320-native-host-evidence.yml \
  .github/workflows/oaf-backend-integration.yml
rtk git commit -m "ci: verify sealed OaF backend"
```

---

### Task 9: Prove Phase A acceptance

**Files:**

- Modify: `tests/benchmark/test_backend_contract_acceptance.py`
- Modify: `docs/drumery-dtx-midi-benchmarking-reference.md`
- Test: every Phase A path above.

**Interfaces:**

- Consumes: Tasks 1-8.
- Produces: HPA-320's official sealed OaF acceptance evidence.

- [ ] **Step 1: Add the sealed-registry acceptance case**

```python
def test_sealed_default_backend_requires_complete_authority_set(
    repository_root: Path,
) -> None:
    registry = default_backend_registry()
    registration = registry.registrations[OFFICIAL_BACKEND_ID]

    assert registration.seal_state == "sealed"
    locks = load_complete_oaf_lock_set(repository_root)
    assert locks.backend.backend_id == OFFICIAL_BACKEND_ID
    assert locks.backend.payload["checkpoint_acquisition_request_sha256"]
    assert locks.runtime.payload["base_system_package_evidence_sha256"]
    assert locks.seal.payload["reference_host_numeric_fingerprint"]
```

- [ ] **Step 2: Run host, runner, and CLI suites**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_checkpoint_acquisition.py \
  tests/benchmark/test_backend_prepare.py \
  tests/benchmark/test_backend_lock.py \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_backend_attestation.py \
  tests/benchmark/test_oaf_backend.py \
  tests/benchmark/test_oaf_seal.py \
  tests/benchmark/test_legacy_conversion_audit.py \
  tests/benchmark/test_oaf_smoke_generator.py \
  tests/benchmark/test_backend_contract_acceptance.py \
  tests/benchmark/test_schema_goldens.py \
  tests/test_cli_benchmark.py \
  runtime/oaf_tf1/tests
```

Expected: PASS.

- [ ] **Step 3: Run complete repository quality gates**

```bash
rtk uv run pytest -q
rtk uv run ruff check src tests tools
rtk uv run black --check src tests tools
rtk uv run pylint src/app src/cli src/benchmark
rtk git diff --check
```

Expected: every command exits `0`.

- [ ] **Step 4: Verify immutable files and source manifests**

```bash
rtk uv run python tools/hpa320/generate_runner_source_manifest.py --check
rtk git status --short
rtk git diff origin/main...HEAD --check
```

Expected: manifests reproduce, worktree is clean, and no whitespace errors exist.

- [ ] **Step 5: Update operator documentation and commit acceptance**

Document the final backend ID, preflight/setup commands, offline inference rule,
five-call smoke behavior, status/exit matrix, recovery limitation for lost OCI bytes,
and the exact paths of published locks/evidence.

```bash
rtk git add \
  tests/benchmark/test_backend_contract_acceptance.py \
  docs/drumery-dtx-midi-benchmarking-reference.md
rtk git commit -m "docs: publish frozen OaF operations"
```

Phase A is complete only after the native required workflow passes. A local or
emulated diagnostic run cannot satisfy this gate.
