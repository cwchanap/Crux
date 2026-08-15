# HPA-326 OaF Full-Mix Corpus Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the validated OaF backend over the exact HPA-324-eligible authoritative full mixes with resumable immutable predictions, deterministic execution evidence, measured runtime projection, and HPA-325 reports.

**Architecture:** Add one OaF-specific `src/benchmark/oaf_corpus_run.py` that composes the existing manifest/cache/backend/prediction/scoring contracts. Reuse a promoted canonical-manifest reader and shared atomic-replace helper; keep canonical audio temporary; use one persistent backend with a 3600-second request ceiling and a separate 30-second worker-close deadline. Prediction paths are source-hash keyed, but resume still re-materializes and validates exact canonical input bytes before reuse. Backend errors are classified through a closed policy; poison stops further inference and `--resume` is recovery.

**Tech Stack:** Python 3.13, Click, existing `librosa`/`soundfile`, existing benchmark canonical JSON/cache/backend/scoring helpers, pytest, Ruff, Pylint.

## Global Constraints

- One OaF-specific runner only; no generic backend/plugin pipeline.
- One sequential persistent `OafBackend`; no pool, batching, queue, distributed execution, restart framework, or retry/backoff engine.
- `OAF_CORPUS_REQUEST_TIMEOUT_SECONDS = 3600.0` for the first real pilot request/readiness ceiling.
- The 3600-second request ceiling is intentionally not reduced to 900 seconds before the pilot: the only real pre-HPA-326 timing evidence is the HPA-423 one-second smoke, whose first lazy-backend call took and reported RTF `24.818518`; it includes startup, but does not establish a safe 900-second full-song ceiling.
- `WorkerProcess.close()` must use a separate `OAF_WORKER_CLOSE_TIMEOUT_SECONDS = 30.0` default rather than the request timeout.
- `--cache-dir` is the HPA-321 corpus/audio cache only. Keep `create_backend()`'s existing checkpoint-cache default; no `--checkpoint-dir`.
- No R2/network fill from HPA-326.
- Prediction artifact v2 stays unchanged and reference-independent.
- Prediction paths are keyed by authoritative `source_audio_sha256`, backend descriptor, and inference-config identity.
- Resume re-materializes current canonical input and requires exact `input_audio_sha256` equality before artifact reuse.
- Canonicalization explicitly uses `res_type="soxr_hq"` and revision `librosa-soxr-hq-mono44k1-soundfile-pcm16/v1`.
- Canonical WAVs remain temporary; no durable derived-audio cache.
- HPA-325 `CohortFailureReason` stays unchanged. Detailed runner errors map through one closed table.
- Unknown include/exclude IDs and include/exclude overlap are fatal preflight errors before backend creation.
- HPA-324 loading reuses the existing canonical JSONL core from `reference_timing_manifest.py`.
- Native reference artifacts resolve relative to `timing_manifest_path.parent.parent`.
- Every HPA-324 eligible reference artifact is reconstructed before inference, so scoring cannot discover a broken eligible reference only after the corpus run.
- `run.json` is one mutable atomic snapshot written through a shared durability primitive.
- No Python `float` reaches `canonical_json_bytes()`; float-derived run/CLI fields cross `quantize_six()`.
- `config.max_input_audio_frames` from the loaded OaF model config is the source of the canonical-input frame limit.
- Worker close failure is run-level evidence: reports still finalize, overall status becomes partial/exit 1.
- Concurrency waits for measured pilot RTF and projected wall time.

## Hard Gates

### Gate A — lineage and scope

Exact HPA-324 -> supplied HPA-323 hash/version match, valid eligible reference artifacts, known filters, and disjoint include/exclude sets are proven before backend creation.

### Gate B — timeout separation

Unit tests prove HPA-326 passes exactly `3600.0` as the OaF request timeout and `WorkerProcess.close()` uses an independent 30-second deadline.

### Gate C — backend error disposition

Every known `OafBackendError.code` is explicitly classified. Unknown codes fail closed as poison. Only item-local codes may continue to the next song.

### Gate D — authoritative source

Only locally verified source bytes matching the original remote identity and HPA-323 source digest may be inferred.

### Gate E — resume identity

Source-hash path lookup never silently turns canonicalizer drift into an ordinary miss. Resume re-materializes and compares current `input_audio_sha256`; mismatch is an explicit artifact failure with no overwrite.

### Gate F — canonical numerics

Run snapshot and CLI canonical JSON contain no Python floats. Exact tests pin Decimal parsing/round-trip.

### Gate G — scorer non-success shape

Failed/skipped/quarantined rows satisfy HPA-325 artifact-nullability and reference-coverage balance before the real pilot.

### Gate H — shared durability

HPA-326 does not add a third private atomic writer.

### Gate I — pilot before broad run

Pilot evidence records exact IDs, request/close deadlines, per-song/aggregate RTF, eligible duration coverage, projected sequential wall time, persistent-worker behavior, and resume behavior before unfiltered execution.

---

## File Map

### Create

- `src/benchmark/oaf_corpus_run.py` — HPA-326 request/outcome, identities, preflight, source resolution, canonical materialization, backend policy, sequential execution, resume, run snapshot, projection, scorer adaptation.
- `tests/benchmark/test_oaf_corpus_run.py` — unit/policy/path/preflight/timeout/error/resume tests.
- `tests/benchmark/test_oaf_corpus_run_acceptance.py` — multi-song local prediction + poison/resume + score/report acceptance with fake backend.

### Modify

- `src/benchmark/reference_timing_manifest.py` — promote shared canonical manifest reader; expose HPA-323 -> HPA-322 source/chart view.
- `src/benchmark/reference_set_manifest.py` — public HPA-324 loader via shared reader; promote native-reference reader.
- `tests/benchmark/test_reference_timing_manifest.py`
- `tests/benchmark/test_reference_set_manifest.py`
- `src/benchmark/durability.py` — shared `atomic_replace_bytes()`.
- `tests/benchmark/test_durability.py`
- `src/benchmark/corpus_manifest.py` — delegate JSON replacement to shared bytes helper while retaining domain error translation.
- `tests/benchmark/test_corpus_manifest.py`
- `src/benchmark/r2_corpus_sync.py` — use shared bytes helper and remove private duplicate.
- `tests/benchmark/test_r2_corpus_sync.py`
- `src/benchmark/worker_process.py` — separate close timeout from request timeout.
- `tests/benchmark/test_worker_process.py`
- `src/benchmark/input_view.py` — `load_materialized_audio()` preserving source/input hash distinction.
- `tests/benchmark/test_input_view.py`
- `src/benchmark/backends/oaf.py` — add `OAF_ADAPTER_REVISION` only.
- `tests/benchmark/test_task_d_contract.py`
- `src/cli/benchmark.py` — thin `run-oaf-corpus` command.
- `tests/test_cli_benchmark.py`

### Explicitly unchanged

- `runtime/oaf_tf1/model.py` inference semantics.
- `src/benchmark/prediction_artifact.py` schema v2.
- `src/benchmark/mapping.py` taxonomy/map semantics.
- `src/benchmark/cohort_scoring.py` failure enum/scoring semantics.
- `src/benchmark/reports.py` report schema.
- HPA-321/HPA-323 R2 fill behavior.
- HPA-320 seal/attestation estate.

---

## Task 1: Reuse the shared manifest core and expose frozen reference inputs

**Files:**
- Modify: `src/benchmark/reference_timing_manifest.py`
- Modify: `src/benchmark/reference_set_manifest.py`
- Modify: `tests/benchmark/test_reference_timing_manifest.py`
- Modify: `tests/benchmark/test_reference_set_manifest.py`

**Interfaces:**

```python
def read_canonical_manifest_core(
    path: Path,
    *,
    schema_version: str,
    validate_rows: Callable[[tuple[Mapping[str, object], ...]], None] | None = None,
) -> CanonicalManifestRead: ...


def load_reference_set_manifest(path: Path) -> LoadedReferenceSetManifest: ...


def reference_chart_view_from_timing_row(
    loaded: LoadedReferenceTimingRow,
) -> ReferenceChartRowView: ...


def read_native_reference_events(
    loaded: LoadedReferenceTimingRow,
    *,
    timing_output_root: Path,
) -> tuple[NativeReferenceEvent, ...]: ...
```

- [ ] **Step 1: Pin the current private canonical reader before renaming**

Keep/add tests proving the existing core rejects malformed framing, wrong schema, and invalid derived corpus version while returning the exact file SHA-256.

Run:

```bash
uv run pytest tests/benchmark/test_reference_timing_manifest.py -q
```

Expected: PASS before the rename.

- [ ] **Step 2: Promote `_read_canonical_manifest_core()` without semantic change**

Rename the current return dataclass/helper to public names and update HPA-322/HPA-323 call sites. Do not add a second abstraction layer.

```python
@dataclass(frozen=True)
class CanonicalManifestRead:
    manifest_sha256: str
    corpus_version: str
    rows: tuple[Mapping[str, object], ...]
```

Run the same focused file again; expected PASS.

- [ ] **Step 3: Write RED tests for the HPA-324 loader**

Use this narrow public shape:

```python
@dataclass(frozen=True)
class ReferenceSetRowView:
    simfile_id: int
    eligibility_status: ReferenceEligibilityStatus
    eligibility_reason_codes: tuple[EligibilityReasonCode, ...]
    eligibility_warnings: tuple[str, ...]
    mapped_event_count: int
    common_scored_event_count: int
    ignored_event_count: int
    unmapped_event_count: int
    duplicate_common_event_count: int


@dataclass(frozen=True)
class LoadedReferenceSetRow:
    source_row: Mapping[str, object]
    view: ReferenceSetRowView


@dataclass(frozen=True)
class LoadedReferenceSetManifest:
    manifest_sha256: str
    corpus_version: str
    source_reference_timing_manifest_sha256: str
    source_reference_timing_version: str
    rows: tuple[LoadedReferenceSetRow, ...]
```

Tests must fail until `load_reference_set_manifest()` exists and must cover exact hash, duplicate IDs, mixed HPA-323 identity, malformed HPA-324 row, and byte-identical canonical round-trip.

Run:

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py -q
```

Expected: FAIL because the new loader is absent.

- [ ] **Step 4: Implement the loader through `read_canonical_manifest_core()`**

The implementation must call:

```python
canonical = read_canonical_manifest_core(
    path,
    schema_version=BENCHMARK_REFERENCE_MANIFEST_SCHEMA,
    validate_rows=validate_rows,
)
```

Inside `validate_rows`, call `_validate_reference_set_row()` and enforce one source timing identity plus unique simfile IDs. Do not parse file bytes separately.

- [ ] **Step 5: Expose the timing-row chart/source view and native event reader**

Promote `_read_native_reference_events()` unchanged. Add `reference_chart_view_from_timing_row()` by reconstructing the HPA-322 row exactly as `_validate_timing_manifest_row()` already does and delegating to `reference_chart_row_view_from_row()`.

Pin the HPA-326 root calculation in a direct regression:

```python
timing_output_root = timing_manifest_path.parent.parent
events = read_native_reference_events(
    loaded_row,
    timing_output_root=timing_output_root,
)
```

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_reference_set_manifest.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/reference_timing_manifest.py \
        src/benchmark/reference_set_manifest.py \
        tests/benchmark/test_reference_timing_manifest.py \
        tests/benchmark/test_reference_set_manifest.py
git commit -m "feat: expose benchmark reference inputs"
```

---

## Task 2: Share atomic replacement and separate worker close timeout

**Files:**
- Modify: `src/benchmark/durability.py`
- Modify: `tests/benchmark/test_durability.py`
- Modify: `src/benchmark/corpus_manifest.py`
- Modify: `tests/benchmark/test_corpus_manifest.py`
- Modify: `src/benchmark/r2_corpus_sync.py`
- Modify: `tests/benchmark/test_r2_corpus_sync.py`
- Modify: `src/benchmark/worker_process.py`
- Modify: `tests/benchmark/test_worker_process.py`

**Interfaces:**

```python
def atomic_replace_bytes(path: Path, content: bytes) -> None: ...


class WorkerProcess:
    @classmethod
    def start(
        cls,
        command: Sequence[str] | str | Path,
        *,
        timeout_seconds: float = 30.0,
        close_timeout_seconds: float = 30.0,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> "WorkerProcess": ...
```

- [ ] **Step 1: Write RED durability tests**

Add tests that `atomic_replace_bytes()`:

```python
def test_atomic_replace_bytes_replaces_and_fsyncs_parent(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    path.write_bytes(b"old")
    calls = []
    monkeypatch.setattr(durability, "fsync_directory", lambda p: calls.append(p))

    atomic_replace_bytes(path, b"new")

    assert path.read_bytes() == b"new"
    assert calls == [tmp_path]
    assert list(tmp_path.glob(".run.json.*.tmp")) == []
```

Also inject `os.replace` failure and require `OSError("artifact publication failed")` plus temporary cleanup.

Run:

```bash
uv run pytest tests/benchmark/test_durability.py -q
```

Expected: FAIL before helper exists.

- [ ] **Step 2: Promote the existing R2 byte-replacement implementation**

Move the generic implementation into `durability.py` with its existing OSError contract. Replace the private `r2_corpus_sync._atomic_replace_bytes` call sites with the import.

Refactor `corpus_manifest._atomic_replace_json()` to:

```python
def _atomic_replace_json(path: Path, payload: dict[str, object]) -> None:
    try:
        atomic_replace_bytes(path, canonical_json_line(payload))
    except OSError:
        raise ManifestPublicationError(_PUBLICATION_ERROR) from None
```

Keep existing corpus/R2 tests green; this is behavior-preserving reuse.

- [ ] **Step 3: Run durability/caller tests**

```bash
uv run pytest \
  tests/benchmark/test_durability.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_r2_corpus_sync.py -q
```

Expected: PASS.

- [ ] **Step 4: Write RED worker close-timeout tests**

Construct a fake `Popen` whose `wait()` records timeout arguments. Instantiate:

```python
worker = WorkerProcess(
    process,
    timeout_seconds=3600.0,
    close_timeout_seconds=30.0,
    ready={"type": "ready"},
)
worker.close()
```

Assert every timed `wait()`/stderr `join()` uses `30.0`, never `3600.0`. Keep existing request-timeout tests proving `_read_record()` still uses `_timeout_seconds`.

Run:

```bash
uv run pytest tests/benchmark/test_worker_process.py -q
```

Expected: FAIL until the separate close timeout exists.

- [ ] **Step 5: Implement the separate close deadline**

Store:

```python
self._timeout_seconds = timeout_seconds
self._close_timeout_seconds = close_timeout_seconds
```

Validate both are positive. Use `_close_timeout_seconds` in `close()` for the pre-terminate wait, post-terminate wait, and stderr-thread join. Keep the final post-`kill()` `wait()` behavior unchanged.

`WorkerProcess.start()` passes both values into the constructor; default close timeout remains 30 seconds for existing callers.

- [ ] **Step 6: Run worker tests**

```bash
uv run pytest tests/benchmark/test_worker_process.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/durability.py \
        src/benchmark/corpus_manifest.py \
        src/benchmark/r2_corpus_sync.py \
        src/benchmark/worker_process.py \
        tests/benchmark/test_durability.py \
        tests/benchmark/test_corpus_manifest.py \
        tests/benchmark/test_r2_corpus_sync.py \
        tests/benchmark/test_worker_process.py
git commit -m "refactor: share durable replacement and close timeout"
```

---

## Task 3: Add canonical materialized-input identity and OaF adapter revision

**Files:**
- Modify: `src/benchmark/input_view.py`
- Modify: `tests/benchmark/test_input_view.py`
- Modify: `src/benchmark/backends/oaf.py`
- Modify: `tests/benchmark/test_task_d_contract.py`

**Interfaces:**

```python
def load_materialized_audio(
    path: Path,
    *,
    source_audio_id: str,
    source_audio_sha256: str,
    input_view_id: str,
    max_input_audio_frames: int | None,
) -> CanonicalAudio: ...


OAF_ADAPTER_REVISION = "crux.oaf-adapter/v1"
```

- [ ] **Step 1: Write RED `load_materialized_audio()` tests**

Use a tiny valid canonical WAV and an independent source digest. Assert:

```python
assert audio.source_audio_sha256 == source_digest
assert audio.input_audio_sha256 == sha256_hex(wav_bytes)
assert audio.input_view_id == "crux.oaf-full-mix-mono44k1-pcm16/v1"
assert audio.sample_rate == 44100
assert audio.channel_count == 1
assert audio.sample_width_bytes == 2
```

Also reject malformed source SHA and enforce the existing frame-limit path.

Run:

```bash
uv run pytest tests/benchmark/test_input_view.py -q
```

Expected: FAIL because helper is absent.

- [ ] **Step 2: Implement as a sibling of `load_direct_audio_bytes()`**

Read staged bytes, call `parse_canonical_wav()`, compute input digest, preserve supplied source digest, and build `CanonicalAudio`. Do not duplicate RIFF parsing.

- [ ] **Step 3: Pin the adapter revision**

Add/export:

```python
OAF_ADAPTER_REVISION = "crux.oaf-adapter/v1"
```

and assert the exact value in the existing OaF contract suite.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest \
  tests/benchmark/test_input_view.py \
  tests/benchmark/test_task_d_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/benchmark/input_view.py \
        src/benchmark/backends/oaf.py \
        tests/benchmark/test_input_view.py \
        tests/benchmark/test_task_d_contract.py
git commit -m "feat: define OaF corpus input identity"
```

---

## Task 4: Define the pure HPA-326 run contract, scope, error policies, and canonical snapshot

**Files:**
- Create: `src/benchmark/oaf_corpus_run.py`
- Create: `tests/benchmark/test_oaf_corpus_run.py`

**Interfaces:**

```python
OAF_CORPUS_RUN_SCHEMA = "crux.oaf-corpus-run/v1"
OAF_INFERENCE_CONFIG_SCHEMA = "crux.oaf-inference-config/v1"
OAF_FULL_MIX_INPUT_VIEW_ID = "crux.oaf-full-mix-mono44k1-pcm16/v1"
OAF_CANONICALIZATION_REVISION = "librosa-soxr-hq-mono44k1-soundfile-pcm16/v1"
OAF_CORPUS_REQUEST_TIMEOUT_SECONDS = 3600.0
OAF_WORKER_CLOSE_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class OafCorpusRunRequest:
    reference_manifest_path: Path
    timing_manifest_path: Path
    cache_dir: Path
    output_dir: Path
    include_simfile_ids: tuple[int, ...] = ()
    exclude_simfile_ids: tuple[int, ...] = ()
    resume: bool = False
    crux_commit: str | None = None


@dataclass(frozen=True)
class OafCorpusRunOutcome:
    overall_status: Literal["complete", "partial", "failed"]
    exit_code: Literal[0, 1, 2]
    run_id: str | None
    run_path: Path | None
    reports_path: Path | None
    success_count: int
    failed_count: int
    skipped_count: int
    quarantined_count: int
    aggregate_rtf: float | None
    projected_full_wall_time_sec: float | None
```

- [ ] **Step 1: Write RED identity/path tests**

Test exact model-lock hashing, deterministic inference-config hash, deterministic `run_id`, normalized sorted filters, and source-keyed prediction path:

```text
predictions/<simfile_id>/<source_audio_sha256>/<backend_descriptor_sha256>/<inference_config_sha256>.jsonl
```

Assert prediction path stays independent from reference manifest hash and does **not** contain `input_audio_sha256`.

Run:

```bash
uv run pytest tests/benchmark/test_oaf_corpus_run.py -q
```

Expected: FAIL because module is absent.

- [ ] **Step 2: Implement pure identity helpers**

Use only `canonical_json_bytes()`, `require_sha256()`, `sha256_hex()`, model config, `OAF_ADAPTER_REVISION`, and `OAF_PREDICTION_MAP_ID`.

Inference config keys are exactly:

```python
{
    "schema": OAF_INFERENCE_CONFIG_SCHEMA,
    "backend_descriptor_sha256": descriptor.sha256,
    "model_lock_sha256": model_lock_sha256,
    "checkpoint_archive_sha256": config.checkpoint.archive_sha256,
    "adapter_revision": OAF_ADAPTER_REVISION,
    "prediction_map_version": OAF_PREDICTION_MAP_ID,
    "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
    "canonicalization_revision": OAF_CANONICALIZATION_REVISION,
}
```

- [ ] **Step 3: Write RED scope-membership preflight tests**

Given loaded manifest IDs `{10, 20, 30}`, require:

```python
_validate_scope((10,), (), {10, 20, 30})       # PASS
_validate_scope((99,), (), {10, 20, 30})       # ValueError unknown include
_validate_scope((), (99,), {10, 20, 30})       # ValueError unknown exclude
_validate_scope((10,), (10,), {10, 20, 30})    # ValueError overlap
```

The validation must run before any backend factory call; add a fake factory that raises if touched.

- [ ] **Step 4: Write RED backend-error policy tests**

Pin:

```python
OAF_BACKEND_ERROR_POLICY = {
    "inference_failed": ("inference_failed", "item_local"),
    "invalid_request": ("inference_failed", "item_local"),
    "input_path_invalid": ("canonical_input_failed", "item_local"),
    "native_event_invalid": ("inference_failed", "item_local"),
    "worker_error": ("worker_protocol_failed", "poison"),
    "worker_start_failed": ("backend_unavailable", "poison"),
    "worker_ready_invalid": ("backend_unavailable", "poison"),
    "worker_identity_invalid": ("backend_unavailable", "poison"),
    "worker_response_invalid": ("worker_protocol_failed", "poison"),
    "backend_closed": ("worker_protocol_failed", "poison"),
    "descriptor_invalid": (None, "fatal_preflight"),
    "worker_close_failed": (None, "finalization"),
}
```

Test unknown code:

```python
assert classify_oaf_backend_error("future_code") == (
    "worker_protocol_failed",
    "poison",
)
```

Also pin the existing runner-code -> HPA-325 reason table and assert every target is in `COHORT_FAILURE_REASONS`.

- [ ] **Step 5: Write RED canonical run-snapshot tests**

Build a snapshot containing binary floats such as `1.25`, `0.5`, and `3600.0`. The renderer must convert them through `quantize_six()` before `canonical_json_bytes()`.

Assert:

```python
content = render_oaf_corpus_run(snapshot)
parsed = strict_json_loads(content)
assert parsed["request_timeout_seconds"] == Decimal("3600")
assert parsed["items"][0]["wall_time_sec"] == Decimal("1.25")
assert render_oaf_corpus_run(parse_oaf_corpus_run(content)) == content
```

Also assert direct Python float injection into the canonical payload is rejected in the test fixture, proving the renderer owns normalization.

- [ ] **Step 6: Implement the snapshot through `atomic_replace_bytes()`**

The private writer should only do:

```python
content = render_oaf_corpus_run(snapshot)
atomic_replace_bytes(run_path, content)
```

Do not write a local temp/replace implementation.

- [ ] **Step 7: Run pure tests**

```bash
uv run pytest tests/benchmark/test_oaf_corpus_run.py -q
```

Expected: PASS for identity, scope, policy, and snapshot tests.

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run.py
git commit -m "feat: define OaF corpus run contract"
```

---

## Task 5: Resolve exact source audio, preflight references, and materialize canonical full mix

**Files:**
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ResolvedSourceAudio:
    path: Path
    source_audio_id: str
    source_audio_sha256: str
    duration_sec: float


def _resolve_source_audio(...) -> ResolvedSourceAudio: ...

def _materialize_oaf_full_mix(...) -> CanonicalAudio: ...
```

- [ ] **Step 1: Write eligible-reference preflight tests**

Create HPA-324/HPA-323 fixtures where an HPA-324 eligible row's referenced event artifact is removed/corrupted after manifest creation. `run_oaf_corpus()` must return fatal exit 2 before backend creation.

For valid eligible rows, preflight stores `ReferenceMappingResult` for later success/failure/skip scorer assembly.

Quarantined `upstream_reference_unavailable` / `reference_event_artifact_invalid` rows are expected to lack a usable mapping and do not make the run fatal.

- [ ] **Step 2: Write source-cache resolution tests**

Cover:

1. carried verified remote/body;
2. stale manifest cache status + matching verified cache-index entry;
3. index entry with changed remote identity;
4. missing/corrupt body;
5. digest mismatch against HPA-323 `source_audio_content_hash`.

Only cases 1/2 return a path. Import no R2 store/config into `oaf_corpus_run.py`.

- [ ] **Step 3: Write duration-probe tests**

Use existing `inspect_source_audio()` and prove a source decode/header failure becomes item-local `source_audio_decode_failed` for an eligible row rather than a run-level crash.

- [ ] **Step 4: Write canonical materialization RED tests**

Use a stereo 22.05 kHz fixture. Monkeypatch `librosa.load` to assert the exact call includes:

```python
sr=44100
mono=True
res_type="soxr_hq"
```

Assert `soundfile.write(..., 44100, format="WAV", subtype="PCM_16")` and then real `load_materialized_audio()` validation.

Use:

```python
max_input_audio_frames=config.max_input_audio_frames
```

from the already loaded model config. Do not hardcode or invent another limit.

- [ ] **Step 5: Pin temporary cleanup**

Staged WAV must be below the one run `input_root` and removed in `finally` after success, item-local failure, or poison.

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest tests/benchmark/test_oaf_corpus_run.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run.py
git commit -m "feat: resolve authoritative OaF corpus inputs"
```

---

## Task 6: Implement one-worker sequential inference, exact resume, poison stop, and close evidence

**Files:**
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run.py`
- Create: `tests/benchmark/test_oaf_corpus_run_acceptance.py`

**Public entry point:**

```python
def run_oaf_corpus(
    request: OafCorpusRunRequest,
    *,
    backend_factory: Callable[..., OafBackend] = create_backend,
    perf_counter: Callable[[], float] = time.perf_counter,
    clock: Callable[[], datetime] = _utc_now,
) -> OafCorpusRunOutcome: ...
```

- [ ] **Step 1: Write backend-construction timeout test**

Fake factory records kwargs. For any invocation needing inference:

```python
assert calls == [
    {
        "input_root": expected_work_root,
        "timeout_seconds": OAF_CORPUS_REQUEST_TIMEOUT_SECONDS,
    }
]
assert OAF_CORPUS_REQUEST_TIMEOUT_SECONDS == 3600.0
```

Do not pass checkpoint dir or close timeout; existing defaults own those.

- [ ] **Step 2: Write healthy persistent-worker test**

Three eligible rows with successful fake predictions must produce:

```text
backend factory calls = 1
transcribe calls       = 3
close calls            = 1
```

Feed fake native predictions through real `map_oaf_prediction()` and real `publish_prediction_artifact()`.

- [ ] **Step 3: Implement the healthy lifecycle using the smoke sequence**

For each inferred item, preserve the existing single-song order:

```text
materialize CanonicalAudio
start timer
backend.transcribe(audio)
stop timer
map_oaf_prediction(native)
publish_prediction_artifact(path, mapped)
checkpoint run.json
```

RTF uses the authoritative decoded duration stored during source preflight. Mapping/publication time stays outside inference elapsed time, matching `smoke_backend`.

- [ ] **Step 4: Write exact resume RED tests**

Prediction path is source-hash keyed. Cases:

1. matching artifact/current canonical input -> no transcribe, `resumed`;
2. source path exists but current canonical `input_audio_sha256` differs -> `prediction_artifact_invalid`, no transcribe, no overwrite;
3. artifact missing under `--resume` -> infer/publish;
4. prior failed run row + artifact missing -> infer/publish;
5. wrong descriptor/model/map/view/source -> artifact invalid, no overwrite;
6. existing target without `--resume` -> `prediction_output_conflict`;
7. artifact exists but prior snapshot missed it due interruption -> resume validates/reuses it.

The test must assert the materializer is called for case 1, pinning exact-input validation rather than decode-free resume.

- [ ] **Step 5: Implement resume validation**

After source resolution and canonical materialization, call `read_prediction_artifact()` and compare:

```text
source_audio_id
source_audio_sha256
input_view_id
input_audio_sha256
backend_descriptor_sha256/model identity
prediction_map_version
```

The path already binds source hash/backend/config; content validation remains mandatory.

- [ ] **Step 6: Write item-local versus poison tests**

Parameterize the known OaF codes. For `inference_failed`, `invalid_request`, `input_path_invalid`, and `native_event_invalid`, fail the middle song and assert a later song is attempted.

For each poison code, fail the middle song and assert:

- the current failed row is checkpointed;
- no later `transcribe()` calls occur in that invocation;
- later rows remain missing/failed according to the snapshot contract;
- backend closes once;
- a second invocation with `--resume` processes outstanding rows.

Add one unknown code and assert it follows poison behavior.

- [ ] **Step 7: Write close-failure test**

Fake backend succeeds for all rows but raises:

```python
OafBackendError("worker close failed", code="worker_close_failed")
```

from `close()`.

Assert:

- prediction artifacts remain valid;
- run snapshot records bounded close failure evidence;
- reports still get produced;
- outcome is `partial`, exit 1;
- no successful item is reclassified as a prediction failure.

- [ ] **Step 8: Implement close handling and finalization order**

Use a structure equivalent to:

```python
close_error = None
try:
    _execute_rows(...)
finally:
    if backend is not None:
        try:
            backend.close()
        except OafBackendError as error:
            close_error = _bounded_close_error(error)

snapshot = replace(snapshot, close_error=close_error)
_write_run_snapshot(snapshot)
return _finalize_scoring_and_outcome(snapshot, ...)
```

Never let close failure erase already persisted predictions/reports.

- [ ] **Step 9: Add interrupted/resumed acceptance test**

Run 3–4 local rows twice:

- invocation 1 publishes earlier successes and hits one poison failure;
- invocation 2 with `--resume` reuses exact matches and executes only outstanding rows;
- previously successful prediction bytes/SHA are unchanged;
- final prediction artifacts parse as v2.

No real Docker/model in CI.

- [ ] **Step 10: Run focused tests**

```bash
uv run pytest \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add src/benchmark/oaf_corpus_run.py \
        tests/benchmark/test_oaf_corpus_run.py \
        tests/benchmark/test_oaf_corpus_run_acceptance.py
git commit -m "feat: run resumable OaF corpus inference"
```

---

## Task 7: Pin non-success scorer assembly, runtime projection, and HPA-325 reports

**Files:**
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run_acceptance.py`

- [ ] **Step 1: Write non-success `CohortItem` shape tests**

For an HPA-324 eligible reference mapping, failed and skipped rows must use:

```python
CohortItem(
    simfile_id=simfile_id,
    status="failed",  # or skipped
    reference_events=reference_to_benchmark_events(simfile_id, mapping.common_events),
    prediction_events=None,
    coverage=coverage_from_artifacts(mapping, None),
    failure_reason=RUNNER_FAILURE_TO_COHORT_REASON[runner_code],
    artifact_identity=None,
    reference_artifact=None,
    prediction_artifact=None,
)
```

For `unclassified_reference_lane` / `no_scored_drum_events` quarantine with readable native reference artifact, use the same mapping-derived reference events/coverage but `status="quarantined"`, `failure_reason="reference_quarantined"`, and all evidence fields `None`.

For `upstream_reference_unavailable` / `reference_event_artifact_invalid`:

```python
CohortCoverage(
    reference_native_event_count=0,
    reference_common_event_count=0,
    reference_ignored_event_count=0,
    reference_unmapped_event_count=0,
    reference_duplicate_collapsed_count=0,
    prediction_native_event_count=None,
    prediction_mapped_event_count=None,
    prediction_unmapped_event_count=None,
    prediction_native_class_counts=(),
)
```

with `reference_events=()`.

Pass every constructed item through real `validate_cohort_items()` in the tests.

- [ ] **Step 2: Pin `prediction_missing` adaptation**

Simulate a run row marked success whose prediction file is deleted before scoring. Adapt it to `status="failed"`, reason `prediction_missing`, with mapping-derived reference events/coverage and all artifact evidence fields `None`.

- [ ] **Step 3: Write projection tests**

Given inferred rows with known durations/times:

```python
aggregate_rtf = measured_wall_time_sec / measured_audio_duration_sec
projected_full_wall_time_sec = aggregate_rtf * full_eligible_audio_duration_sec
```

Assert resume hits without retained inference timing contribute neither zero time nor zero duration. If any eligible duration is unavailable, projection is `None` and coverage count remains explicit.

- [ ] **Step 4: Implement projection and canonical persistence**

Keep in-memory outcome fields as float if convenient, but when updating `run.json`, every binary float crosses `quantize_six()` via one private render helper. Do not store raw floats in snapshot payloads.

- [ ] **Step 5: Write HPA-325 end-to-end scorer tests**

Assemble success/resume/failure/skip/quarantine rows and assert:

- success uses `cohort_item_from_artifacts()`;
- all other rows satisfy the pinned table above;
- `CohortIdentity.reference_manifest_sha256` is the exact HPA-324 manifest hash;
- `reference_timing_version` is the exact HPA-323 version;
- model lock/descriptor/map/input-view/run ID are frozen identities;
- scorer population equals run population.

- [ ] **Step 6: Implement only existing scoring/report calls**

```python
score_result = score_cohort(identity, tuple(items), diagnostics_for=())
report_artifacts = write_cohort_reports(score_result, run_dir / "reports")
```

Do not copy matching, alignment, class aggregation, or report rendering into HPA-326.

- [ ] **Step 7: Prove persisted rescoring needs no inference**

After one successful run, invoke the same scope with `resume=True` and a backend fake that raises if `transcribe()` is called. Exact artifacts should validate, reports regenerate, and inference call count remain zero.

- [ ] **Step 8: Run focused tests**

```bash
uv run pytest \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_reports.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/benchmark/oaf_corpus_run.py \
        tests/benchmark/test_oaf_corpus_run.py \
        tests/benchmark/test_oaf_corpus_run_acceptance.py
git commit -m "feat: score OaF corpus control runs"
```

---

## Task 8: Add the thin `run-oaf-corpus` CLI

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`

- [ ] **Step 1: Write RED Click argument tests**

Invoke:

```bash
crux benchmark run-oaf-corpus \
  --manifest <hpa324> \
  --timing-manifest <hpa323> \
  --cache-dir <cache> \
  --output-dir <output> \
  --include-simfile-id 10 \
  --include-simfile-id 20 \
  --exclude-simfile-id 30 \
  --resume
```

Both repeated ID options use:

```python
type=click.IntRange(0, MAX_SIMFILE_ID)
```

Monkeypatch the domain runner so CLI tests never start Docker.

- [ ] **Step 2: Implement the thin lazy-import command**

The handler only constructs `OafCorpusRunRequest`, calls `run_oaf_corpus()`, builds a small stdout payload, quantizes any float fields, writes canonical JSON, and exits with `outcome.exit_code`.

Required stdout keys:

```text
status
exit_code
run_id
run_path
success_count
failed_count
skipped_count
quarantined_count
aggregate_rtf
projected_full_wall_time_sec
reports_path
```

- [ ] **Step 3: Pin domain preflight propagation**

CLI tests assert domain exit 2 is preserved for unknown/overlapping scope rather than being rewritten as a generic Click usage failure. Syntax-range violations remain Click errors.

- [ ] **Step 4: Pin exit semantics**

- exit 0: all selected eligible items successful, clean close, reports produced;
- exit 1: item/poison/close failure but trustworthy run/report produced;
- exit 2: fatal preflight/snapshot/setup failure.

- [ ] **Step 5: Run CLI tests**

```bash
uv run pytest tests/test_cli_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py
git commit -m "feat: add OaF corpus benchmark command"
```

---

## Task 9: Operational acceptance — real fixed-checkpoint pilot before broad run

**Files:**
- No production code unless the real pilot exposes a concrete defect.
- Record operational evidence in HPA-326 / PR discussion rather than adding a permanent pilot-data artifact solely for this gate.

- [ ] **Step 1: Freeze 4–6 technically diverse IDs before reading model scores**

Select from HPA-323/HPA-324 metadata only. Include shorter/longer audio, lower/higher reference density, and multiple packs/audio representations when available.

- [ ] **Step 2: Record the timeout evidence before execution**

Record:

```text
request_timeout_seconds = 3600
worker_close_timeout_seconds = 30
prior_hpa423_smoke_first_call_rtf = 24.818518
```

The HPA-423 smoke is only evidence that a pre-pilot 900-second full-song ceiling is not yet justified; because the first call includes lazy startup it is **not** used as the corpus projection.

- [ ] **Step 3: Run the fixed pilot**

```bash
uv run crux benchmark run-oaf-corpus \
  --manifest <exact-hpa324-manifest> \
  --timing-manifest <exact-hpa323-manifest> \
  --cache-dir <exact-r2-cache> \
  --output-dir artifacts/benchmark/oaf-corpus \
  --include-simfile-id <id-1> \
  --include-simfile-id <id-2> \
  --include-simfile-id <id-3> \
  --include-simfile-id <id-4>
```

Capture run ID, manifest hashes, model/descriptor/checkpoint identity, adapter/canonicalizer/config identity, each inferred song's duration/wall time/RTF, aggregate RTF, eligible-duration coverage, projected wall time, and report paths.

- [ ] **Step 4: Prove one healthy persistent worker**

Confirm worker/model ready occurs once and multiple pilot requests complete through that worker. Do not add new telemetry infrastructure solely for this proof.

- [ ] **Step 5: Prove real resume**

Immediately rerun the identical pilot with `--resume`. Require:

- matching artifacts reused;
- no inference request for resumed successes;
- canonical input is re-materialized and hash-validated;
- prediction bytes/SHA unchanged;
- reports regenerate.

- [ ] **Step 6: Review measured request ceiling and concurrency need**

Use measured steady-state per-song RTF to decide whether the 3600-second request ceiling should be tightened. If any legitimate song approaches the ceiling, increase/tune only from evidence. If projected sequential runtime is acceptable, keep one worker. If prohibitive, change scope explicitly before adding concurrency.

- [ ] **Step 7: Run the broad corpus**

```bash
uv run crux benchmark run-oaf-corpus \
  --manifest <exact-hpa324-manifest> \
  --timing-manifest <exact-hpa323-manifest> \
  --cache-dir <exact-r2-cache> \
  --output-dir artifacts/benchmark/oaf-corpus \
  --resume
```

No include/exclude flags.

- [ ] **Step 8: Reconcile the fixed control**

Require:

```text
manifest total = success + failed + skipped + quarantined
unfiltered broad-run explicit skips = 0
eligible = success + failed
```

A valid prediction or explicit failure exists for every eligible row. Record final run/report path for HPA-395/HPA-328/HPA-562.

---

## Task 10: Full verification and scope audit

- [ ] **Step 1: Run targeted suite**

```bash
uv run pytest \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_reference_set_manifest.py \
  tests/benchmark/test_durability.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_r2_corpus_sync.py \
  tests/benchmark/test_worker_process.py \
  tests/benchmark/test_input_view.py \
  tests/benchmark/test_task_d_contract.py \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_cohort_scoring_acceptance.py \
  tests/benchmark/test_reports.py \
  tests/test_cli_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full tests**

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run CI-equivalent static checks**

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check
```

Expected: all pass; `git diff --check` emits no output.

- [ ] **Step 4: Audit scope**

Verify the implementation still has:

- one OaF runner module, not a framework;
- one backend lifecycle;
- no runner R2/network dependency;
- no prediction schema change;
- no generic retry/queue/pool/restart abstraction;
- no HPA-320 compatibility machinery;
- no corpus-derived tuning;
- no durable canonical WAV corpus;
- shared manifest and durability helpers rather than third copies;
- closed backend error policy with unknown->poison;
- exact non-success HPA-325 item contract;
- no raw Python float in canonical run/CLI JSON.

If a new abstraction has one caller and is not necessary for a proven contract/test seam, collapse it before final review.

- [ ] **Step 5: Commit only verification fixes if needed**

```bash
git add <only-the-files-changed-by-real-verification-fixes>
git commit -m "fix: finalize HPA-326 corpus runner"
```

Do not create an empty cleanup commit.

---

## Expected Implementation Result

After execution of this plan, Crux has one command that:

1. validates exact HPA-324/HPA-323 lineage and closed filter scope;
2. proves eligible reference artifacts before expensive inference;
3. resolves authoritative source audio from the existing local cache only;
4. materializes the explicit OaF full-mix canonical view temporarily;
5. uses one persistent OaF worker with a measured request ceiling and bounded close deadline;
6. distinguishes item-local backend failures from poison through a closed fail-closed policy;
7. persists prediction v2 immutably under source/model/config identity;
8. resumes only after exact current canonical input validation;
9. checkpoints one canonical mutable run snapshot through the shared durability primitive;
10. serializes all float-derived evidence through `quantize_six()`;
11. assembles success/failure/skip/quarantine rows that already satisfy HPA-325 validation;
12. publishes pilot runtime projection and HPA-325 fixed-control reports without rerunning matching logic.

Concurrency, generic runners, additional models, separated inputs, and paired comparisons remain later-ticket work.
