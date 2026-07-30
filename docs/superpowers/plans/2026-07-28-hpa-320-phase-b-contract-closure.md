# HPA-320 Phase B Contract Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Bring the existing common transcription, provenance, artifact, report, and
registry implementation into exact agreement with the final HPA-320 v1 contracts
before native OaF sealing or compatibility-adapter completion.

**Architecture:** Preserve the existing backend-agnostic Phase B modules and make the
committed design's schema appendix executable through strict golden fixtures. Add an
explicit registry seal state so the checked-in pre-seal OaF entry is operationally
unavailable without looking corrupt, then extend execution attestations and report
validation with diagnostic host evidence. Phase A and Phase C append their own schema
goldens through the same manifest and validator interface.

**Tech Stack:** Python 3.12, Click 8, Pytest, standard-library `dataclasses`,
`decimal`, `hashlib`, `json`, `pathlib`, canonical UTF-8 JSON/JSONL, SHA-256, Ruff,
Black, and Pylint.

## Global Constraints

- Normative design:
  `docs/superpowers/specs/2026-07-26-hpa-320-freeze-oaf-drums-backend-design.md`
  at or after commit `d2ca20c`.
- This is a delta plan. The historical Phase B plan produced the existing modules;
  do not recreate or rename those public APIs without a failing contract test.
- The common layer must not import TensorFlow, Magenta, Librosa, or the application
  `DrumTranscriber`.
- OaF backend ID is
  `magenta-egmd-tf1-94529798-8hit-v1`; heuristic ID is
  `heuristic-onset-v1`; legacy compatibility ID is `legacy-tf2-h5-v0`.
- The checked-in OaF registry entry is `preseal` until final Phase A locks and
  evidence are committed. `preseal` returns `backend_not_sealed`, exit `1`, and no
  prediction. A `sealed` entry with missing or contradictory authorities is
  integrity failure, exit `2`.
- `environment_unsupported` is limited to platform or host-evidence preflight before
  inference. It cannot classify a completed smoke mismatch.
- Every post-Click backend command emits one canonical one-line summary with exactly
  `status`, `exit_code`, `report_path`, and `report_sha256`. Click usage errors occur
  before a summary.
- Operational reports retain canonical sorted error facts; causal sequence remains
  in bounded sanitized diagnostics.
- Prediction JSONL is the authoritative artifact. MIDI remains optional and
  derivative.
- All header fields duplicated in the embedded descriptor must exact-match it.
- Python lines remain within 100 characters.

## Execution Order

Complete Tasks 1-4 before Phase A changes any persisted lock or wire schema. Task 5
is the Phase B handoff gate. Phase A and Phase C then extend the schema-golden
manifest without changing the harness established here.

---

## File Map

### New files

- `tests/benchmark/schema_goldens/manifest.json` — canonical registry of schema IDs,
  validator modules, and checked-in golden paths.
- `tests/benchmark/schema_goldens/*.json` — Phase B canonical JSON goldens.
- `tests/benchmark/schema_goldens/drum-prediction-events-v1.jsonl` — canonical
  three-record JSONL golden.
- `tests/benchmark/test_schema_goldens.py` — manifest and validator drift tests.

### Modified files

- `src/benchmark/backend_registry.py` — explicit `preseal`/`sealed` state and typed
  creation failures.
- `src/benchmark/backend_attestation.py` — exact diagnostic host fingerprint.
- `src/benchmark/backend_reports.py` — pre-seal failure validation and attestation
  agreement.
- `src/benchmark/prediction_artifact.py` — explicit descriptor/header agreement
  tests remain the only authority for duplicated fields.
- `src/benchmark/transcription.py` — maps registry failures without opening item
  outputs.
- `src/cli/benchmark.py` — preserves canonical post-parse summaries.
- `tests/benchmark/test_backend_registry.py`
- `tests/benchmark/test_backend_attestation.py`
- `tests/benchmark/test_backend_reports.py`
- `tests/benchmark/test_prediction_artifact.py`
- `tests/benchmark/test_transcription.py`
- `tests/test_cli_benchmark.py`
- `docs/drumery-dtx-midi-benchmarking-reference.md`

### Cross-task interfaces

- Schema harness:
  `SchemaGoldenEntry`, `load_schema_golden_manifest`,
  `validate_schema_golden_entry`.
- Registry:
  `SealState = Literal["preseal", "sealed"]`,
  `BackendRegistration`, `BackendNotSealed`, `BackendIntegrityUnavailable`.
- Attestation:
  `HostNumericFingerprint`, `ExecutionAttestation`,
  `publish_execution_attestation`, `validate_execution_attestation`.
- Existing public orchestration remains:
  `run_verify_backend`, `run_transcribe_one`, `publish_operational_report`,
  `render_prediction_artifact`, and `read_prediction_artifact`.

---

### Task 1: Establish the strict schema-golden harness

**Files:**

- Create: `tests/benchmark/schema_goldens/manifest.json`
- Create: `tests/benchmark/schema_goldens/crux.transcription-backend-descriptor-v1.json`
- Create: `tests/benchmark/schema_goldens/crux.heuristic-backend-descriptor-v1.json`
- Create: `tests/benchmark/schema_goldens/crux.input-view-manifest-v1.json`
- Create: `tests/benchmark/schema_goldens/crux.backend-execution-item-id-v1.json`
- Create: `tests/benchmark/schema_goldens/crux.backend-execution-attestation-v1.json`
- Create: `tests/benchmark/schema_goldens/crux.backend-verification-report-v1.json`
- Create: `tests/benchmark/schema_goldens/crux.backend-execution-report-v1.json`
- Create: `tests/benchmark/schema_goldens/crux.legacy-score-report-v1.json`
- Create: `tests/benchmark/schema_goldens/crux.drum-prediction-events-v1.jsonl`
- Create: `tests/benchmark/test_schema_goldens.py`
- Modify: `src/benchmark/backend_identity.py`
- Modify: `src/benchmark/input_view.py`
- Modify: `src/benchmark/backend_attestation.py`
- Modify: `src/benchmark/backend_reports.py`
- Modify: `src/benchmark/prediction_artifact.py`

**Interfaces:**

- Consumes: the exact key sets in the design's normative schema appendix.
- Produces:

```python
@dataclass(frozen=True)
class SchemaGoldenEntry:
    schema: str
    validator_modules: tuple[str, ...]
    golden_path: PurePosixPath


def load_schema_golden_manifest(repository_root: Path) -> tuple[SchemaGoldenEntry, ...]:
    """Strict-load the canonical manifest and reject duplicate schema IDs or paths."""


def validate_schema_golden_entry(
    entry: SchemaGoldenEntry,
    repository_root: Path,
) -> None:
    """Import every listed validator and require every validator to accept the golden."""
```

- [ ] **Step 1: Write the failing manifest and golden tests**

Create a manifest fixture with exactly `schema`, `validator_modules`, and
`golden_path` per row. Add this test:

```python
def test_phase_b_schema_goldens_are_complete_and_strict(
    repository_root: Path,
) -> None:
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
    }.issubset(schemas)
    for entry in entries:
        validate_schema_golden_entry(entry, repository_root)
```

Parameterize mutations that remove one key, add `unexpected`, duplicate one JSON
key in raw bytes, and replace one typed value. Require every listed validator module
to reject every mutation.

- [ ] **Step 2: Run the test and verify the harness is absent**

Run:

```bash
rtk uv run pytest -q tests/benchmark/test_schema_goldens.py
```

Expected: collection fails because the manifest loader and validator wrappers do not
exist.

- [ ] **Step 3: Implement the manifest loader and validator wrappers**

Keep the loader in the test module because the manifest is a drift detector, not
runtime identity. Expose one uniform wrapper from each production module:

```python
def validate_schema_golden(schema: str, content: bytes) -> None:
    if schema == INPUT_VIEW_SCHEMA:
        validate_input_view_manifest_bytes(content)
        return
    raise ValueError("unsupported schema golden")
```

Use existing strict loaders rather than duplicating schema logic. For report and
descriptor schemas, provide representative valid identity values and call the
existing normalizers. For JSONL, call `read_prediction_artifact`.

- [ ] **Step 4: Add exact canonical goldens**

Each JSON golden must use sorted keys, no insignificant whitespace, UTF-8, and one
final Unix newline. The JSONL golden must contain exactly one header, one OaF event,
and one terminal record whose `prefix_sha256` reproduces the preceding bytes.

The execution attestation golden includes:

```json
{
  "host_numeric_fingerprint": {
    "architecture": "x86_64",
    "cpu_family": "6",
    "cpu_model": "143",
    "cpu_stepping": "8",
    "cpu_vendor_id": "GenuineIntel"
  }
}
```

Do not include `cpu_microcode` or `kernel_release`.

- [ ] **Step 5: Run focused schema and canonicalization tests**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_schema_goldens.py \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_input_view.py \
  tests/benchmark/test_backend_attestation.py \
  tests/benchmark/test_backend_reports.py \
  tests/benchmark/test_prediction_artifact.py
```

Expected: PASS.

- [ ] **Step 6: Commit the schema harness**

```bash
rtk git add \
  src/benchmark/backend_identity.py \
  src/benchmark/input_view.py \
  src/benchmark/backend_attestation.py \
  src/benchmark/backend_reports.py \
  src/benchmark/prediction_artifact.py \
  tests/benchmark/schema_goldens \
  tests/benchmark/test_schema_goldens.py
rtk git commit -m "test: freeze HPA-320 common schemas"
```

---

### Task 2: Add explicit pre-seal registry state

**Files:**

- Modify: `src/benchmark/backend_registry.py`
- Modify: `src/benchmark/transcription.py`
- Modify: `tests/benchmark/test_backend_registry.py`
- Modify: `tests/benchmark/test_transcription.py`
- Modify: `tests/test_cli_benchmark.py`

**Interfaces:**

- Consumes: existing lazy backend factories and typed operational reports.
- Produces:

```python
SealState = Literal["preseal", "sealed"]


@dataclass(frozen=True)
class BackendRegistration:
    backend_id: str
    seal_state: SealState
    factory: Callable[..., TranscriptionBackend]


class BackendNotSealed(RuntimeError):
    backend_id: str


class BackendIntegrityUnavailable(RuntimeError):
    backend_id: str
```

- [ ] **Step 1: Write failing registry-state tests**

```python
def test_preseal_official_backend_returns_typed_not_sealed() -> None:
    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        registrations={
            OFFICIAL_BACKEND_ID: BackendRegistration(
                backend_id=OFFICIAL_BACKEND_ID,
                seal_state="preseal",
                factory=unexpected_factory,
            )
        },
    )

    with pytest.raises(BackendNotSealed):
        registry.create(None)
    assert unexpected_factory.calls == 0
```

Add separate tests proving:

- `sealed` invokes the factory;
- missing locks from a `sealed` factory become `BackendIntegrityUnavailable`;
- an unknown ID remains `BackendUnavailable`;
- neither pre-seal nor sealed-integrity failure invokes the heuristic.

- [ ] **Step 2: Verify current behavior fails**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_transcription.py \
  tests/test_cli_benchmark.py \
  -k 'preseal or not_sealed or sealed_integrity'
```

Expected: FAIL because the current registry maps missing OaF locks to
`backend_unavailable`, exit `2`.

- [ ] **Step 3: Implement state-aware registry creation**

Replace the `factories` mapping with immutable registrations. The checked-in default
registry uses:

```python
BackendRegistration(
    backend_id=OFFICIAL_BACKEND_ID,
    seal_state="preseal",
    factory=_create_official_backend,
)
```

Phase A changes only this literal to `sealed` in the same commit that publishes both
final locks. Do not infer state from file existence.

- [ ] **Step 4: Map typed failures before output creation**

In `run_verify_backend` and `run_transcribe_one`:

```python
except BackendNotSealed as error:
    return publish_registry_failure(
        backend_id=error.backend_id,
        code="backend_not_sealed",
        exit_code=1,
    )
except BackendIntegrityUnavailable as error:
    return publish_registry_failure(
        backend_id=error.backend_id,
        code="backend_integrity_unavailable",
        exit_code=2,
    )
```

The pre-seal execution report has `status: "failed"`, contains the stable error code,
and has no prediction item. Resolve registry state before `_resolve_output_path` or
any prediction publication call.

- [ ] **Step 5: Run the focused registry and CLI suite**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_transcription.py \
  tests/test_cli_benchmark.py
```

Expected: PASS, including default OaF invocation returning exit `1` while the
registry remains pre-seal.

- [ ] **Step 6: Commit the registry state**

```bash
rtk git add \
  src/benchmark/backend_registry.py \
  src/benchmark/transcription.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_transcription.py \
  tests/test_cli_benchmark.py
rtk git commit -m "feat: expose OaF pre-seal state"
```

---

### Task 3: Add diagnostic host fingerprints to execution attestations

**Files:**

- Modify: `src/benchmark/backend_attestation.py`
- Modify: `src/benchmark/backend_process.py`
- Modify: `src/benchmark/backends/base.py`
- Modify: `src/benchmark/transcription.py`
- Modify: `tests/benchmark/test_backend_attestation.py`
- Modify: `tests/benchmark/test_backend_process.py`
- Modify: `tests/benchmark/test_backend_reports.py`
- Modify: `tests/benchmark/test_transcription.py`

**Interfaces:**

- Consumes: authenticated native-host evidence from Phase A or diagnostic host
  preflight.
- Produces:

```python
@dataclass(frozen=True)
class HostNumericFingerprint:
    architecture: str
    cpu_vendor_id: str
    cpu_family: str
    cpu_model: str
    cpu_stepping: str

    def as_json(self) -> dict[str, str]: ...
```

`ExecutionAttestation` gains
`host_numeric_fingerprint: HostNumericFingerprint | None`.

- [ ] **Step 1: Write failing strict fingerprint tests**

```python
def test_execution_attestation_accepts_exact_diagnostic_fingerprint(
    attestation_payload: dict[str, object],
) -> None:
    attestation_payload["host_numeric_fingerprint"] = {
        "architecture": "x86_64",
        "cpu_vendor_id": "GenuineIntel",
        "cpu_family": "6",
        "cpu_model": "143",
        "cpu_stepping": "8",
    }

    loaded = validate_execution_attestation(
        canonical_json_bytes(attestation_payload),
        expected_backend_id=OFFICIAL_BACKEND_ID,
        expected_descriptor_sha256="a" * 64,
    )

    assert loaded.host_numeric_fingerprint.cpu_model == "143"
```

Parameterize missing/extra keys and specifically reject `cpu_microcode` and
`kernel_release` inside the stable object.

- [ ] **Step 2: Run the test and verify the schema is stale**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_attestation.py \
  tests/benchmark/test_backend_process.py
```

Expected: FAIL because the current attestation exact key set lacks
`host_numeric_fingerprint`.

- [ ] **Step 3: Implement one parser shared by host evidence and attestations**

Add:

```python
def parse_host_numeric_fingerprint(value: object) -> HostNumericFingerprint:
    payload = require_exact_object(
        value,
        {
            "architecture",
            "cpu_vendor_id",
            "cpu_family",
            "cpu_model",
            "cpu_stepping",
        },
        "host numeric fingerprint",
    )
    return HostNumericFingerprint(**cast(dict[str, str], payload))
```

Native evidence payloads must contain this object. Diagnostic preflight may return
null only when the platform/evidence check fails before one can be measured.

- [ ] **Step 4: Thread the fingerprint into publication**

Extend `BackendVerification` and the execution-attestation publisher so official OaF
verification/execution always publishes the measured object. Do not compare it to
the seal reference to choose status. It is diagnostic evidence only.

- [ ] **Step 5: Run attestation and report tests**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_attestation.py \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_backend_reports.py \
  tests/benchmark/test_transcription.py
```

Expected: PASS.

- [ ] **Step 6: Commit fingerprint evidence**

```bash
rtk git add \
  src/benchmark/backend_attestation.py \
  src/benchmark/backend_process.py \
  src/benchmark/backends/base.py \
  src/benchmark/transcription.py \
  tests/benchmark/test_backend_attestation.py \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_backend_reports.py \
  tests/benchmark/test_transcription.py
rtk git commit -m "feat: attest backend host fingerprint"
```

---

### Task 4: Lock report, artifact-header, and CLI summary behavior

**Files:**

- Modify: `src/benchmark/backend_reports.py`
- Modify: `src/benchmark/prediction_artifact.py`
- Modify: `src/cli/benchmark.py`
- Modify: `tests/benchmark/test_backend_reports.py`
- Modify: `tests/benchmark/test_prediction_artifact.py`
- Modify: `tests/test_cli_benchmark.py`
- Modify: `docs/drumery-dtx-midi-benchmarking-reference.md`

**Interfaces:**

- Consumes: Tasks 1-3.
- Produces: exact post-parse summaries and strict artifact/report agreement.

- [ ] **Step 1: Add failing disagreement and summary tests**

```python
@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model_id", "other-model"),
        ("architecture_id", "other-architecture"),
        ("backend_lock_sha256", "f" * 64),
        ("runtime_lock_sha256", "e" * 64),
        ("native_output_space_id", "other-space"),
    ],
)
def test_prediction_header_rejects_descriptor_disagreement(
    valid_prediction_bytes: bytes,
    field: str,
    replacement: str,
) -> None:
    mutated = replace_jsonl_header(valid_prediction_bytes, field, replacement)
    with pytest.raises(PredictionArtifactError, match="descriptor"):
        read_prediction_artifact(mutated)
```

Add CLI tests proving:

- Click usage exit `2` emits no JSON summary;
- every parsed operational exit `0`, `1`, or `2` emits one summary with exactly four
  keys;
- report-publication failure emits no summary;
- `errors` sorting does not claim causal ordering.

- [ ] **Step 2: Run focused tests and capture current failures**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_reports.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/test_cli_benchmark.py
```

Expected: at least the new pre-seal and summary-matrix cases fail.

- [ ] **Step 3: Centralize summary rendering at the CLI boundary**

Add a private helper:

```python
def _emit_backend_summary(
    *,
    status: str,
    exit_code: int,
    report_path: Path | None,
    report_sha256: str | None,
) -> None:
    payload = {
        "exit_code": exit_code,
        "report_path": None if report_path is None else str(report_path),
        "report_sha256": report_sha256,
        "status": status,
    }
    click.get_binary_stream("stdout").write(
        canonical_json_bytes(payload, trailing_newline=True)
    )
```

Use it for `prepare-backend`, `verify-backend`, `transcribe-one`, and the Phase C
legacy score command. Call it only after Click parsing succeeds.

- [ ] **Step 4: Preserve normalized error facts**

Keep deterministic `(code, message)` sorting in `backend_reports.py`. Add a docstring
and test that explicitly treats the array as unordered normalized facts. Bounded
stderr remains the causal diagnostic channel.

- [ ] **Step 5: Update operator documentation**

Document the `preseal`/`sealed` states, four-key summary, report-presence
discriminator, and descriptor/header exact-agreement rule. State that current main
remains pre-seal until the Phase A native gate succeeds.

- [ ] **Step 6: Run focused and quality checks**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_reports.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/test_cli_benchmark.py
rtk uv run ruff check src/benchmark src/cli tests/benchmark tests/test_cli_benchmark.py
rtk uv run black --check src/benchmark src/cli tests/benchmark tests/test_cli_benchmark.py
rtk git diff --check
```

Expected: PASS.

- [ ] **Step 7: Commit report and summary closure**

```bash
rtk git add \
  src/benchmark/backend_reports.py \
  src/benchmark/prediction_artifact.py \
  src/cli/benchmark.py \
  tests/benchmark/test_backend_reports.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/test_cli_benchmark.py \
  docs/drumery-dtx-midi-benchmarking-reference.md
rtk git commit -m "fix: close common backend contracts"
```

---

### Task 5: Prove the Phase B closure gate

**Files:**

- Modify: `tests/benchmark/test_backend_contract_acceptance.py`
- Test: all Phase B files above.

**Interfaces:**

- Consumes: Tasks 1-4.
- Produces: stable common interfaces for Phase A and Phase C.

- [ ] **Step 1: Add an end-to-end pre-seal acceptance test**

```python
def test_default_oaf_is_typed_preseal_without_prediction(
    tmp_path: Path,
) -> None:
    outcome = run_transcribe_one(
        direct_request(tmp_path),
        registry=preseal_registry(),
    )

    assert outcome.exit_code == 1
    assert outcome.status == "failed"
    assert not direct_request(tmp_path).output_path.exists()
    report = read_report(outcome.report_artifact)
    assert report["errors"] == [
        {
            "code": "backend_not_sealed",
            "message": "Backend is not sealed.",
        }
    ]
```

Add the complementary sealed-integrity test with exit `2`.

- [ ] **Step 2: Run the complete Phase B suite**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_backend_types.py \
  tests/benchmark/test_input_view.py \
  tests/benchmark/test_backend_publication.py \
  tests/benchmark/test_backend_attestation.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_backend_reports.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_transcription.py \
  tests/benchmark/test_backend_contract_acceptance.py \
  tests/benchmark/test_schema_goldens.py \
  tests/test_cli_benchmark.py
```

Expected: PASS.

- [ ] **Step 3: Prove the import boundary**

```bash
rtk rg -n 'tensorflow|librosa|DrumTranscriber' \
  src/benchmark/backend_identity.py \
  src/benchmark/backends/base.py \
  src/benchmark/input_view.py \
  src/benchmark/prediction_artifact.py \
  src/benchmark/scorer_input.py \
  src/benchmark/backend_attestation.py \
  src/benchmark/backend_reports.py \
  src/benchmark/backend_registry.py \
  src/benchmark/transcription.py
```

Expected: only lazy adapter factory imports in `backend_registry.py`; no common path
imports heavy libraries or application fallback code.

- [ ] **Step 4: Run repository quality gates**

```bash
rtk uv run pytest -q
rtk uv run ruff check src tests tools
rtk uv run black --check src tests tools
rtk uv run pylint src/app src/cli src/benchmark
rtk git diff --check
```

Expected: every command exits `0`.

- [ ] **Step 5: Commit acceptance coverage**

```bash
rtk git add tests/benchmark/test_backend_contract_acceptance.py
rtk git commit -m "test: prove HPA-320 common contract closure"
```

Phase B is complete only after this task passes. Phase A may then migrate OaF locks,
wire protocol, and sealing evidence; Phase C may add its schema rows and adapters
without redefining the common types.
