# HPA-326 OaF Full-Mix Corpus Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the smallest resumable execution layer that runs the validated OaF backend sequentially over HPA-324-eligible authoritative full mixes, persists/reuses prediction artifact v2, records runtime/provenance, and produces the HPA-325 fixed-control reports.

**Architecture:** One OaF-specific `src/benchmark/oaf_corpus_run.py` composes landed HPA-321/HPA-323/HPA-324/HPA-423/HPA-325 seams. It uses the existing shared canonical-manifest reader, local corpus cache, temporary canonical audio, one persistent backend with an explicit corpus request deadline, immutable prediction v2 artifacts, one atomic mutable `run.json`, and the existing HPA-325 scorer/reporter. Worker protocol poison stops further inference in that invocation; `--resume` is the recovery path.

**Tech Stack:** Python 3.12, Click, existing `librosa`/`soundfile`, existing benchmark canonical JSON/cache/backend/scoring helpers, pytest, Ruff, Pylint.

## Global constraints

- Keep one OaF-specific runner. Do not introduce a generic model/plugin pipeline.
- Keep one sequential `OafBackend`; no pool, queue, batching, distributed execution, retry/backoff, or restart framework.
- Use `OAF_CORPUS_REQUEST_TIMEOUT_SECONDS = 3600.0`; do not expose a timeout CLI flag.
- `--cache-dir` is the HPA-321 corpus/audio cache only. Use `create_backend()`'s existing checkpoint-cache default; do not add `--checkpoint-dir`.
- Do not fetch/fill R2 from the inference runner.
- Prediction artifact v2 stays unchanged and reference-independent.
- Canonical WAVs are temporary; do not add a durable derived-audio cache.
- HPA-325 `CohortFailureReason` stays unchanged. Detailed runner errors map through one closed table.
- Unknown include/exclude IDs and include/exclude overlap are fatal preflight errors before inference.
- HPA-324 loading must reuse the existing canonical JSONL core from `reference_timing_manifest.py`; no forked manifest reader.
- Native reference artifacts are resolved relative to `timing_manifest_path.parent.parent`.
- `run.json` is a mutable atomic snapshot; do not publish it with immutable-artifact helpers.
- No threshold/model/mapping/scoring tuning from corpus results.
- Concurrency waits for measured pilot RTF/wall-time evidence.

## Risks / hard gates

### Gate A — real worker deadline

The existing 30-second backend default must never be used by HPA-326. Unit tests must prove the backend factory receives exactly `OAF_CORPUS_REQUEST_TIMEOUT_SECONDS`.

### Gate B — worker poison

A worker startup/readiness/timeout/protocol failure poisons the persistent process. HPA-326 must checkpoint the current run, stop issuing further inference requests in that invocation, close the backend, and return partial status. `--resume` retries missing/failed work later. Model-level inference errors returned through a valid worker response remain item-local and may continue to the next row.

### Gate C — shared manifest reader

`load_reference_set_manifest()` must call the promoted existing `read_canonical_manifest_core()` for framing, exact bytes, schema, SHA-256, and `render_manifest()` round-trip.

### Gate D — closed scope

Every include/exclude ID must exist in HPA-324, and the two sets must be disjoint. Failure is exit 2 before backend creation.

### Gate E — closed failure mapping

Each runner failure code maps to exactly one existing HPA-325 reason. No exception handler chooses an ad-hoc “closest” reason.

### Gate F — immutable prediction reuse

Resume validates source/input/backend/model/map identity through real prediction v2 parsing. Mismatches/conflicts never overwrite existing output.

### Gate G — offline source identity

Only a locally verified cache body matching the original remote identity and HPA-323 source hash may be inferred. No R2 substitution.

### Gate H — pilot before broad run

Operational evidence records exact pilot IDs, request timeout, per-song/aggregate RTF, eligible duration coverage, and projected sequential wall time before unfiltered execution.

---

## File map

### Create

- `src/benchmark/oaf_corpus_run.py` — HPA-326 identity/preflight/cache/materialization/orchestration/resume/run-snapshot/projection/scorer adaptation.
- `tests/benchmark/test_oaf_corpus_run.py` — pure/unit orchestration tests with fake backend.
- `tests/benchmark/test_oaf_corpus_run_acceptance.py` — multi-song local artifact + poison/resume + score/report acceptance without real Docker/model.

### Modify

- `src/benchmark/reference_timing_manifest.py` — promote shared canonical manifest reader and expose narrow HPA-323 -> HPA-322 source/chart view.
- `src/benchmark/reference_set_manifest.py` — public HPA-324 loader using shared reader; promote native-reference reader.
- corresponding manifest tests.
- `src/benchmark/input_view.py` / tests — materialized canonical-audio loader preserving source hash.
- `src/benchmark/backends/oaf.py` / existing contract tests — adapter revision only.
- `src/cli/benchmark.py` / CLI tests — thin `run-oaf-corpus` command.

### Explicitly unchanged

- `runtime/oaf_tf1/model.py` and worker inference semantics.
- `src/benchmark/prediction_artifact.py` schema v2.
- `src/benchmark/mapping.py` taxonomy/map behavior.
- `src/benchmark/cohort_scoring.py` failure enum/scoring behavior.
- `src/benchmark/reports.py` report schema.
- HPA-321/HPA-323 R2 fill logic.
- HPA-320 seal/attestation code.

---

## Task 1: Reuse the shared manifest core and expose frozen reference inputs

**Files:**
- Modify: `src/benchmark/reference_timing_manifest.py`
- Modify: `src/benchmark/reference_set_manifest.py`
- Modify: `tests/benchmark/test_reference_timing_manifest.py`
- Modify: `tests/benchmark/test_reference_set_manifest.py`

**Interfaces:**
- Produces `read_canonical_manifest_core(...)` by promoting the existing private implementation without semantic change.
- Produces `load_reference_set_manifest(path) -> LoadedReferenceSetManifest`.
- Produces `reference_chart_view_from_timing_row(loaded) -> ReferenceChartRowView`.
- Produces `read_native_reference_events(loaded, *, timing_output_root)`.

- [ ] **Step 1: Pin the existing canonical core before renaming it**

Add/retain tests proving the shared core rejects non-canonical framing, wrong schema, invalid derived corpus version, and returns the exact file SHA-256.

Run:

```bash
uv run pytest tests/benchmark/test_reference_timing_manifest.py -q
```

Expected: PASS before rename; these tests pin semantics.

- [ ] **Step 2: Promote `_read_canonical_manifest_core()` without changing behavior**

Rename it to:

```python
def read_canonical_manifest_core(
    path: Path,
    *,
    schema_version: str,
    validate_rows: Callable[[tuple[Mapping[str, object], ...]], None] | None = None,
) -> CanonicalManifestRead:
    ...
```

A public dataclass/name for the existing return value is acceptable; do not add callbacks/registries beyond the current validator hook.

Update existing HPA-322/HPA-323 call sites to use the public name.

- [ ] **Step 3: Write RED tests for the HPA-324 loader**

Add a narrow view:

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
class LoadedReferenceSetManifest:
    manifest_sha256: str
    corpus_version: str
    source_reference_timing_manifest_sha256: str
    source_reference_timing_version: str
    rows: tuple[LoadedReferenceSetRow, ...]
```

Tests must prove:

- exact HPA-324 bytes/hash are returned;
- loader uses the same canonical round-trip rules as HPA-323;
- duplicate simfile IDs fail;
- mixed HPA-323 hash/version fails;
- malformed HPA-324 rows fail through `_validate_reference_set_row()`;
- row order stays canonical input order.

Run:

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py -q
```

Expected: FAIL because loader/types do not exist.

- [ ] **Step 4: Implement `load_reference_set_manifest()` through the promoted core**

The implementation must call:

```python
canonical = read_canonical_manifest_core(
    path,
    schema_version=BENCHMARK_REFERENCE_MANIFEST_SCHEMA,
    validate_rows=validate_rows,
)
```

`validate_rows` performs only HPA-324 domain checks: `_validate_reference_set_row()`, unique simfile IDs, one source timing identity, narrow view construction.

Do **not** parse JSONL independently in `reference_set_manifest.py`.

- [ ] **Step 5: Expose HPA-323 source/chart reconstruction**

Add:

```python
def reference_chart_view_from_timing_row(
    loaded: LoadedReferenceTimingRow,
) -> ReferenceChartRowView:
    ...
```

Reconstruct the HPA-322 row exactly as `_validate_timing_manifest_row()` does and delegate to `reference_chart_row_view_from_row()`.

Tests: ready row returns original source inventory/selected chart; malformed timing row still fails existing validation.

- [ ] **Step 6: Promote native-reference reader and pin timing root**

Rename `_read_native_reference_events()` to `read_native_reference_events()` with no semantic change.

Direct caller regression:

```python
timing_output_root = timing_manifest_path.parent.parent
events = read_native_reference_events(
    loaded_timing_row,
    timing_output_root=timing_output_root,
)
```

The test fixture must fail if it incorrectly uses `timing_manifest_path.parent`.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_reference_set_manifest.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/reference_timing_manifest.py \
        src/benchmark/reference_set_manifest.py \
        tests/benchmark/test_reference_timing_manifest.py \
        tests/benchmark/test_reference_set_manifest.py
git commit -m "feat: expose benchmark reference inputs"
```

---

## Task 2: Add canonical materialized-input identity and OaF adapter revision

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
) -> CanonicalAudio:
    ...
```

```python
OAF_ADAPTER_REVISION = "crux.oaf-adapter/v1"
```

- [ ] **Step 1: Write RED materialized-audio tests**

Prove:

- supplied source SHA is retained;
- input SHA is computed from staged WAV bytes;
- mono/44.1k/PCM16 validation is still `parse_canonical_wav()`;
- frame limit remains enforced;
- invalid source SHA/IDs fail.

Run:

```bash
uv run pytest tests/benchmark/test_input_view.py -q
```

Expected: FAIL because helper is absent.

- [ ] **Step 2: Implement as a sibling of `load_direct_audio_bytes()`**

Read canonical WAV bytes, call `parse_canonical_wav()`, hash bytes, construct `CanonicalAudio`. Do not duplicate WAV parsing.

- [ ] **Step 3: Add adapter revision constant and pin it**

No seal/lock document. Export the constant from the existing OaF adapter module.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/benchmark/test_input_view.py tests/benchmark/test_task_d_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/benchmark/input_view.py src/benchmark/backends/oaf.py \
        tests/benchmark/test_input_view.py tests/benchmark/test_task_d_contract.py
git commit -m "feat: define OaF corpus input identity"
```

---

## Task 3: Define run identity, scope preflight, failure mapping, and atomic snapshot

**Files:**
- Create: `src/benchmark/oaf_corpus_run.py`
- Create: `tests/benchmark/test_oaf_corpus_run.py`

**Interfaces:**

```python
OAF_CORPUS_RUN_SCHEMA = "crux.oaf-corpus-run/v1"
OAF_INFERENCE_CONFIG_SCHEMA = "crux.oaf-inference-config/v1"
OAF_FULL_MIX_INPUT_VIEW_ID = "crux.oaf-full-mix-mono44k1-pcm16/v1"
OAF_CANONICALIZATION_REVISION = "librosa-mono44k1-soundfile-pcm16/v1"
OAF_CORPUS_REQUEST_TIMEOUT_SECONDS = 3600.0


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
```

Detailed failures are a closed literal:

```python
RunnerFailureCode = Literal[
    "source_audio_unavailable",
    "source_audio_decode_failed",
    "canonical_input_failed",
    "backend_unavailable",
    "worker_protocol_failed",
    "inference_failed",
    "prediction_artifact_invalid",
    "prediction_output_conflict",
    "prediction_publish_failed",
    "prediction_missing",
]
```

and one constant mapping:

```python
RUNNER_FAILURE_TO_COHORT_REASON = {
    "source_audio_unavailable": "inference_failed",
    "source_audio_decode_failed": "inference_failed",
    "canonical_input_failed": "inference_failed",
    "backend_unavailable": "backend_unavailable",
    "worker_protocol_failed": "backend_unavailable",
    "inference_failed": "inference_failed",
    "prediction_artifact_invalid": "prediction_artifact_invalid",
    "prediction_output_conflict": "prediction_artifact_invalid",
    "prediction_publish_failed": "prediction_artifact_invalid",
    "prediction_missing": "prediction_missing",
}
```

- [ ] **Step 1: RED identity tests**

Pin model-lock SHA, inference-config hash, deterministic run ID, deterministic prediction path, and reference-independent prediction path.

- [ ] **Step 2: RED closed failure-map test**

Assert:

```python
assert set(RUNNER_FAILURE_TO_COHORT_REASON) == set(get_args(RunnerFailureCode))
assert set(RUNNER_FAILURE_TO_COHORT_REASON.values()) <= COHORT_FAILURE_REASONS
```

Also pin the exact values above. No “closest family” helper.

- [ ] **Step 3: RED scope preflight tests**

After loading an HPA-324 manifest with IDs `{10, 20, 30}`:

- include `(10, 99)` -> fatal preflight;
- exclude `(99,)` -> fatal preflight;
- include `(10,)`, exclude `(10,)` -> fatal preflight;
- duplicate CLI/request values normalize or fail consistently before run identity;
- valid include/exclude produce deterministic sorted scope.

The check must happen before backend factory invocation.

- [ ] **Step 4: Implement pure identity/scope helpers**

Use existing `canonical_json_bytes()` / `require_sha256()`. Do not add another JSON encoder.

- [ ] **Step 5: RED `run.json` tests**

Pin:

- canonical JSON round-trip;
- exact run identity;
- item order by simfile ID;
- disposition/status/count reconciliation;
- timeout recorded as `3600.0`;
- `resumed` is success, not skipped;
- completion timestamp only after iteration finishes/halts cleanly;
- worker-poison partial run remains resumable.

- [ ] **Step 6: Implement atomic mutable snapshot writer**

Use the repository's existing mutable-snapshot pattern (same-directory temp, flush, fsync, `os.replace`, directory fsync where existing helper does it). Do not use `publish_immutable_file()`.

- [ ] **Step 7: Run tests**

```bash
uv run pytest tests/benchmark/test_oaf_corpus_run.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run.py
git commit -m "feat: define OaF corpus run contract"
```

---

## Task 4: Resolve authoritative cached audio and materialize full-mix input

**Files:**
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run.py`

- [ ] **Step 1: RED cache-resolution cases**

Cover:

1. carried remote already verified;
2. immutable manifest says not verified but matching cache-index entry/body exists;
3. cache index remote identity differs;
4. cache body missing/corrupt;
5. cache SHA differs from HPA-323 `source_audio_content_hash`.

Only 1/2 resolve.

- [ ] **Step 2: Implement by composing public HPA-321 helpers**

Use only:

```python
CacheIndexStore.get(...)
cache_entry_matches_remote(...)
validate_cached_body(...)
resolve_verified_cache_body(...)
```

Do not copy HPA-323 `_resolve_or_queue_audio()` and do not import R2 store/config code.

- [ ] **Step 3: RED source-duration probe tests**

Call existing header-only `inspect_source_audio()`. Probe failure records `source_audio_decode_failed`; it must not be mistaken for backend failure.

- [ ] **Step 4: RED canonical materialization tests**

A stereo 22.05k fixture becomes temporary mono 44.1k PCM16 under `work_root` using existing `librosa` + `soundfile` and `load_materialized_audio()`.

Assert source SHA != input SHA when appropriate and source identity remains HPA-323 identity.

- [ ] **Step 5: Pin cleanup**

Temporary canonical WAV is removed in `finally` after success or item-local failure.

- [ ] **Step 6: Run tests and commit**

```bash
uv run pytest tests/benchmark/test_oaf_corpus_run.py -q
git add src/benchmark/oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run.py
git commit -m "feat: materialize authoritative OaF inputs"
```

---

## Task 5: Implement one-worker sequential inference, timeout, poison semantics, publication, and resume

**Files:**
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run.py`
- Create: `tests/benchmark/test_oaf_corpus_run_acceptance.py`

**Public seam:**

```python
def run_oaf_corpus(
    request: OafCorpusRunRequest,
    *,
    backend_factory: Callable[..., OafBackend] = create_backend,
    perf_counter: Callable[[], float] = time.perf_counter,
    clock: Callable[[], datetime] = _utc_now,
) -> OafCorpusRunOutcome:
    ...
```

- [ ] **Step 1: RED backend-construction test including timeout**

The injected factory records kwargs. For any run requiring inference assert exactly:

```python
assert factory_calls == [
    {
        "input_root": work_root,
        "timeout_seconds": OAF_CORPUS_REQUEST_TIMEOUT_SECONDS,
    }
]
```

Do not pass `checkpoint_dir`; existing `create_backend()` resolves its model cache.

- [ ] **Step 2: RED normal persistent-worker test**

Three eligible songs with a healthy fake backend:

```text
backend objects = 1
transcribe calls = 3
close calls = 1
```

Flow predictions through real `map_oaf_prediction()` + `publish_prediction_artifact()`.

- [ ] **Step 3: Implement normal lifecycle**

Order:

1. load HPA-324/HPA-323;
2. lineage + scope preflight;
3. derive model/config/run identities;
4. build run/work dirs;
5. load cache index once;
6. reconstruct references;
7. preflight eligible durations;
8. create one backend with `input_root` and 3600-second timeout if inference is needed;
9. iterate selected eligible rows by simfile ID;
10. close backend once;
11. finalize run snapshot/reports.

- [ ] **Step 4: RED valid model-level inference failure continuation**

Fake `transcribe()` raises/returns the equivalent of an OaF model-level `inference_failed` for song 2 without poisoning the fake process; song 3 still runs. Mapping is `inference_failed -> CohortFailureReason("inference_failed")`.

- [ ] **Step 5: RED worker-poison stop test**

Simulate a `WorkerProcessError`/OaF `worker_error`/protocol timeout on song 2. Assert:

- song 1 prediction remains persisted;
- song 2 records `worker_protocol_failed` -> `backend_unavailable`;
- song 3 does **not** receive `transcribe()` in this invocation;
- `run.json` is checkpointed as partial/incomplete;
- backend closes once;
- outcome exit code is 1;
- no restart/backend replacement occurs.

This replaces the old false assumption that every backend error can continue to the next song.

- [ ] **Step 6: RED success publication**

Assert each inferred success records source/chart/input/prediction identity, prediction SHA/path, duration, wall time, RTF, and that prediction v2 itself contains no chart/reference fields.

- [ ] **Step 7: RED resume matrix**

1. matching artifact -> `resumed`, no transcribe;
2. missing artifact -> infer/publish;
3. prior failed row + missing artifact -> infer/publish;
4. source/input/descriptor/model/map mismatch -> `prediction_artifact_invalid`, no overwrite;
5. existing target with `resume=False` -> `prediction_output_conflict`, no overwrite;
6. artifact exists but previous snapshot missed it -> resume validates/reuses;
7. after a prior worker-poison run, next `--resume` invocation reuses prior success and attempts only missing/failed eligible work.

A missing target with `resume=False` is normal inference, **not** an output conflict.

- [ ] **Step 8: Pin publish-failure grouping**

A failed immutable publish is runner code `prediction_publish_failed` and maps exactly to `prediction_artifact_invalid`. Do not map it ambiguously to `prediction_missing`.

`prediction_missing` is reserved for scorer/adaptation evidence where run state expects a prediction but the file is absent.

- [ ] **Step 9: Acceptance interruption/resume test**

Two invocations over 3–4 local fixture rows:

- first persists at least one success then stops on simulated worker poison;
- second `--resume` reuses unchanged success bytes and runs only outstanding rows;
- final prediction files parse as v2;
- no successful output SHA changes.

- [ ] **Step 10: Run focused tests and commit**

```bash
uv run pytest \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: PASS.

```bash
git add src/benchmark/oaf_corpus_run.py \
        tests/benchmark/test_oaf_corpus_run.py \
        tests/benchmark/test_oaf_corpus_run_acceptance.py
git commit -m "feat: run resumable OaF corpus inference"
```

---

## Task 6: Add runtime projection and HPA-325 fixed-control adaptation

**Files:**
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run_acceptance.py`

- [ ] **Step 1: RED projection tests**

Pin:

```python
aggregate_rtf = measured_wall_time_sec / measured_audio_duration_sec
projected_full_wall_time_sec = aggregate_rtf * full_eligible_audio_duration_sec
```

Only actual inferred rows contribute measured timing. Resume hits without retained timing do not contribute zero time. Missing any eligible duration makes projection `None` and reports coverage count.

Also assert `request_timeout_seconds == 3600.0` is persisted with the pilot evidence.

- [ ] **Step 2: Implement projection summary**

No confidence intervals, memory telemetry, or parallel speedup estimate.

- [ ] **Step 3: RED scorer adaptation using exact timing root**

Construct rows for:

- inferred success;
- resumed success;
- filter skip;
- HPA-324 quarantine;
- source/canonical/inference failure;
- run-row success whose expected prediction artifact is removed before scoring.

Reference load must use:

```python
timing_output_root = request.timing_manifest_path.parent.parent
native = read_native_reference_events(
    timing_row,
    timing_output_root=timing_output_root,
)
reference = map_reference_events(native)
```

The removed expected prediction maps to runner `prediction_missing` -> HPA-325 `prediction_missing`.

- [ ] **Step 4: Assert the complete deterministic failure table**

For every `RunnerFailureCode`, build/derive a failed scorer row and assert the exact mapped HPA-325 reason. The test should fail if a new runner code is added without updating the table.

- [ ] **Step 5: Implement HPA-325 reuse only**

```python
result = score_cohort(identity, tuple(items), diagnostics_for=())
reports = write_cohort_reports(result, run_dir / "reports")
```

No matching/aggregate/report logic in HPA-326.

- [ ] **Step 6: Prove reports regenerate without inference**

After a completed run, invoke `--resume` with a backend fake that raises if `transcribe()` is called. Matching successes reuse artifacts and reports regenerate.

- [ ] **Step 7: Run tests and commit**

```bash
uv run pytest \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_reports.py -q
```

Expected: PASS.

```bash
git add src/benchmark/oaf_corpus_run.py \
        tests/benchmark/test_oaf_corpus_run.py \
        tests/benchmark/test_oaf_corpus_run_acceptance.py
git commit -m "feat: score OaF corpus control runs"
```

---

## Task 7: Add the thin `run-oaf-corpus` CLI and closed scope behavior

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`

- [ ] **Step 1: RED CLI parsing tests**

Use the existing integer range:

```python
click.IntRange(0, MAX_SIMFILE_ID)
```

for repeated `--include-simfile-id` and `--exclude-simfile-id`.

No `--backend`, `--checkpoint-dir`, or `--timeout`.

- [ ] **Step 2: RED preflight exit-2 CLI tests**

With a fake HPA-324 manifest:

- unknown include -> exit 2;
- unknown exclude -> exit 2;
- overlap -> exit 2;
- backend factory/domain execution is not reached.

- [ ] **Step 3: Implement thin handler**

Construct `OafCorpusRunRequest`, call `run_oaf_corpus()`, print canonical summary, exit with domain code. Keep heavy imports lazy like existing benchmark commands.

Stdout summary includes at minimum:

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
request_timeout_seconds
reports_path
```

- [ ] **Step 4: Pin exit semantics**

- 0: selected eligible predictions complete;
- 1: item failures or worker-poison partial run with trustworthy snapshot;
- 2: lineage/scope/setup/run-level fatal failure.

- [ ] **Step 5: Run and commit**

```bash
uv run pytest tests/test_cli_benchmark.py -q
```

Expected: PASS.

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py
git commit -m "feat: add OaF corpus benchmark command"
```

---

## Task 8: Operational acceptance — real fixed-checkpoint pilot before broad run

**Files:**
- No production changes unless the real run proves a concrete bug.
- Record evidence in HPA-326 / implementation PR discussion.

- [ ] **Step 1: Freeze 4–6 technically diverse IDs before model scores**

Use HPA-323/HPA-324 metadata only: short/long duration, lower/higher reference density, multiple source packs/audio representations where available.

- [ ] **Step 2: Record the corpus request deadline before execution**

Evidence must state:

```text
OAF_CORPUS_REQUEST_TIMEOUT_SECONDS = 3600.0
```

and confirm the real backend was constructed with that deadline rather than the HPA-423 30-second default.

- [ ] **Step 3: Run fixed pilot**

```bash
uv run crux benchmark run-oaf-corpus \
  --manifest <exact-hpa324-manifest> \
  --timing-manifest <exact-hpa323-manifest> \
  --cache-dir <exact-r2-corpus-cache> \
  --output-dir artifacts/benchmark/oaf-corpus \
  --include-simfile-id <id-1> \
  --include-simfile-id <id-2> \
  --include-simfile-id <id-3> \
  --include-simfile-id <id-4>
```

Do not pass a checkpoint path; `create_backend()` uses the existing OaF model-cache default/environment.

Capture:

- run ID + exact HPA-324/HPA-323 hashes;
- backend descriptor/model/checkpoint identity;
- adapter revision/config hash;
- request timeout;
- each inferred song duration/wall time/RTF;
- aggregate RTF;
- full eligible duration coverage;
- projected full sequential wall time;
- prediction/report paths and validation.

- [ ] **Step 4: Prove persistent worker in normal real execution**

Confirm model/worker ready occurs once and multiple song requests complete through the same worker. Do not add telemetry solely for this evidence.

- [ ] **Step 5: Prove real resume**

Rerun same pilot with `--resume`:

- matching predictions reused;
- no new inference request for matching successes;
- prediction SHAs unchanged;
- reports regenerate.

- [ ] **Step 6: If a protocol/timeout failure occurs, verify specified poison behavior**

Confirm no later inference requests are issued in that invocation, `run.json` is partial/checkpointed, and a subsequent `--resume` attempts outstanding rows. Fix only a concrete deviation from the specified behavior; do not add an automatic restart framework.

- [ ] **Step 7: Review projection before broad run**

If sequential runtime is acceptable, keep one worker. If not, record measured evidence and explicitly revise scope before concurrency work.

- [ ] **Step 8: Run full eligible corpus**

```bash
uv run crux benchmark run-oaf-corpus \
  --manifest <exact-hpa324-manifest> \
  --timing-manifest <exact-hpa323-manifest> \
  --cache-dir <exact-r2-corpus-cache> \
  --output-dir artifacts/benchmark/oaf-corpus \
  --resume
```

- [ ] **Step 9: Reconcile fixed control**

For unfiltered broad run:

```text
eligible = success + failed_or_missing
manifest total = success + failed + skipped + quarantined
explicit filter skips = 0
```

Every eligible row has a valid prediction or explicit failed/missing state. Record the final run/report paths for HPA-395/HPA-328/HPA-562.

---

## Task 9: Full verification and scope audit

- [ ] **Step 1: Targeted benchmark tests**

```bash
uv run pytest \
  tests/benchmark/test_reference_set_manifest.py \
  tests/benchmark/test_reference_timing_manifest.py \
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

- [ ] **Step 2: Full test suite**

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: CI-equivalent static checks**

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check
```

Expected: all PASS; `git diff --check` emits no errors.

- [ ] **Step 4: Scope audit**

Verify:

- one OaF corpus runner, not a framework;
- normal inference uses one backend/worker;
- backend created with explicit 3600-second request timeout;
- protocol poison stops later requests and relies on resume;
- no worker restart/pool/retry engine;
- no R2/network dependency in runner;
- `--cache-dir` and model checkpoint cache remain distinct;
- HPA-324 loader uses shared `read_canonical_manifest_core()`;
- unknown include/exclude and overlap fail before backend creation;
- native reference root is `timing_manifest_path.parent.parent`;
- failure mapping is one closed constant table; HPA-325 enum unchanged;
- prediction artifact v2 unchanged;
- no durable canonical WAV corpus;
- `run.json` uses mutable atomic replacement, not immutable publisher;
- HPA-325 scoring/report logic is reused rather than copied;
- no corpus-derived model/scoring tuning.

Collapse any new one-caller abstraction that is not required for testability or identity correctness.

- [ ] **Step 5: Commit verification fixes only if needed**

```bash
git add <only-files-changed-by-verification-fixes>
git commit -m "fix: finalize HPA-326 corpus runner"
```

Do not create an empty cleanup commit.

---

## Expected result

After execution Crux has one command that can:

1. consume exact HPA-324/HPA-323 reference lineage through the shared canonical manifest reader;
2. reject typoed/overlapping frozen scopes before inference;
3. resolve the exact authoritative full mix from the existing local cache;
4. canonicalize it into the fixed OaF input view;
5. run one validated sequential worker with a real corpus-scale request deadline;
6. stop safely on worker protocol poison and recover outstanding work with `--resume`;
7. persist/reuse immutable prediction v2 without reference coupling;
8. map detailed operational failures deterministically into the unchanged HPA-325 taxonomy;
9. record per-song runtime/RTF and publish the pilot-derived full-corpus projection;
10. reconcile success/failure/skipped/quarantined/missing state and produce HPA-325 reports without rerunning inference;
11. provide stable prediction/run/report paths to HPA-395, HPA-328, and HPA-562.

Concurrency, generic runners, extra models, separated inputs, derived-audio caching, and benchmark comparison remain later work.
