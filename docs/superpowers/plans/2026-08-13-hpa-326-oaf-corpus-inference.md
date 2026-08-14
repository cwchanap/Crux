# HPA-326 OaF Full-Mix Corpus Inference Implementation Plan

> Planning companion to `docs/superpowers/specs/2026-08-13-hpa-326-oaf-corpus-inference-design.md`.

**Goal:** Add the smallest resumable execution layer that runs the validated OaF backend sequentially over HPA-324-eligible authoritative full mixes, persists/reuses prediction artifact v2, records runtime/provenance, and produces the HPA-325 fixed-control reports.

**Architecture:** One OaF-specific `oaf_corpus_run.py` module composes existing HPA-321/HPA-323/HPA-324/HPA-423/HPA-325 contracts. It owns run identity, local cache resolution, temporary canonical input materialization, the one-backend lifecycle, immutable prediction reuse, an atomic `run.json`, runtime projection, and conversion into HPA-325 cohort items. The CLI stays thin.

**Tech stack:** Python 3.12, Click, existing `librosa`/`soundfile`, existing benchmark dataclasses/canonical JSON helpers, pytest, Ruff, Pylint.

## Scope lock

Do not add:

- generic backend/plugin execution;
- worker pools, batching, queues, distributed jobs, or retry policy;
- R2/network fetches in the runner;
- a new prediction schema;
- database/Parquet output;
- HPA-320 seal/lock/attestation compatibility;
- tuning/calibration code;
- a canonical-input cache unless profiling proves resume decode cost matters.

If implementation starts requiring any of the above, stop and re-check the design rather than expanding HPA-326.

---

## Task 1: Add public read seams for the frozen HPA-324/HPA-323 reference inputs

**Files:**

- Modify: `src/benchmark/reference_set_manifest.py`
- Modify: `src/benchmark/reference_timing_manifest.py`
- Modify: `tests/benchmark/test_reference_set_manifest.py`
- Modify: `tests/benchmark/test_reference_timing_manifest.py`

### Step 1.1 — Write failing HPA-324 loader tests

Add tests that write a real canonical HPA-324 fixture/manifest and assert a new public loader returns:

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

Required cases:

- exact input-byte SHA-256 is returned;
- canonical rendering round-trips byte-identically;
- duplicate simfile IDs fail;
- mixed HPA-323 source timing hashes/versions fail;
- mixed taxonomy/lane-map or malformed eligibility rows fail through the existing validator;
- row order is retained from canonical manifest input.

Run:

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py -q
```

Expected: FAIL because `load_reference_set_manifest()` and the public dataclasses do not exist.

### Step 1.2 — Implement the narrow HPA-324 loader

In `reference_set_manifest.py`:

- reuse `_validate_reference_set_row()` for domain validation;
- parse canonical JSONL with existing strict JSON helpers;
- verify exact `render_manifest()` round-trip;
- hash the exact bytes;
- enforce one HPA-323 timing identity and unique simfile IDs;
- expose only the narrow row fields HPA-326 needs plus the original immutable source row.

Do not add a generic manifest reader framework.

### Step 1.3 — Expose the already-validated HPA-323 chart/source view

HPA-326 needs the HPA-321 inventory and selected chart from a loaded HPA-323 row. Add one public helper to `reference_timing_manifest.py`:

```python
def reference_chart_view_from_timing_row(
    loaded: LoadedReferenceTimingRow,
) -> ReferenceChartRowView:
    ...
```

Implementation should reconstruct the HPA-322 payload exactly as `_validate_timing_manifest_row()` already does, then delegate to `reference_chart_row_view_from_row()`.

Add tests proving:

- a ready timing row returns the original source inventory and selected chart;
- a quarantined timing row returns the validated source view with no selected chart when appropriate;
- malformed timing payloads still fail via existing validators.

Do not expose private key sets or make HPA-326 reconstruct manifests itself.

### Step 1.4 — Promote the native reference artifact reader

Rename/promote:

```python
_read_native_reference_events(...)
```

to:

```python
read_native_reference_events(...)
```

without changing its safe-path, SHA-256, canonical event parsing, or identity checks. Update HPA-324 internal call sites to use the public name.

Add one direct regression test showing an HPA-326-style caller can load a ready row's native reference artifact by passing the timing output root.

### Step 1.5 — Re-run focused tests

```bash
uv run pytest \
  tests/benchmark/test_reference_set_manifest.py \
  tests/benchmark/test_reference_timing_manifest.py -q
```

Expected: PASS.

### Step 1.6 — Commit

```bash
git add src/benchmark/reference_set_manifest.py \
        src/benchmark/reference_timing_manifest.py \
        tests/benchmark/test_reference_set_manifest.py \
        tests/benchmark/test_reference_timing_manifest.py
git commit -m "feat: expose benchmark reference inputs"
```

---

## Task 2: Add the canonical materialized-input seam and OaF adapter revision

**Files:**

- Modify: `src/benchmark/input_view.py`
- Modify: `tests/benchmark/test_input_view.py`
- Modify: `src/benchmark/backends/oaf.py`
- Modify: `tests/benchmark/test_task_d_contract.py`

### Step 2.1 — Write failing `load_materialized_audio()` tests

Add a sibling to `load_direct_audio()` for a canonical WAV whose source hash differs from the derived WAV hash:

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

Tests must prove:

- `source_audio_sha256` is preserved from the authoritative source;
- `input_audio_sha256` is computed from canonical WAV bytes;
- canonical WAV shape is still exactly mono / 44.1 kHz / PCM16;
- frame limit is enforced using the existing validation path;
- malformed SHA/input-view/source ID fails explicitly.

Run:

```bash
uv run pytest tests/benchmark/test_input_view.py -q
```

Expected: FAIL before implementation.

### Step 2.2 — Implement with existing canonical WAV parsing

Do not duplicate WAV parsing. Read the file, call existing `parse_canonical_wav()`, hash the canonical bytes, then construct `CanonicalAudio` with the supplied source hash and computed input hash.

Keep `load_direct_audio()` unchanged for existing callers.

### Step 2.3 — Pin one semantic adapter revision

Add to `src/benchmark/backends/oaf.py`:

```python
OAF_ADAPTER_REVISION = "crux.oaf-adapter/v1"
```

This is host-semantics identity only. Do not add a seal, lock document, or Git-tree attestation.

Extend the existing OaF contract test to pin the exact constant.

### Step 2.4 — Re-run focused tests

```bash
uv run pytest \
  tests/benchmark/test_input_view.py \
  tests/benchmark/test_task_d_contract.py -q
```

Expected: PASS.

### Step 2.5 — Commit

```bash
git add src/benchmark/input_view.py \
        src/benchmark/backends/oaf.py \
        tests/benchmark/test_input_view.py \
        tests/benchmark/test_task_d_contract.py
git commit -m "feat: define OaF corpus input identity"
```

---

## Task 3: Define the pure HPA-326 run contract and deterministic identities

**Files:**

- Create: `src/benchmark/oaf_corpus_run.py`
- Create: `tests/benchmark/test_oaf_corpus_run.py`

### Step 3.1 — Start with identity/path tests

Create pure tests first for the following minimal public contract:

```python
OAF_CORPUS_RUN_SCHEMA = "crux.oaf-corpus-run/v1"
OAF_INFERENCE_CONFIG_SCHEMA = "crux.oaf-inference-config/v1"
OAF_FULL_MIX_INPUT_VIEW_ID = "crux.oaf-full-mix-mono44k1-pcm16/v1"
OAF_CANONICALIZATION_REVISION = "librosa-mono44k1-soundfile-pcm16/v1"


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

The module may use private/internal dataclasses for the full run header/item rows; avoid exporting a broad framework.

Test pure helpers for:

- exact `model_lock_sha256 = sha256(runtime/oaf_tf1/model.json bytes)`;
- canonical inference-config hash changes when adapter/input-view/checkpoint/descriptor identity changes;
- include/exclude IDs are normalized, unique, sorted, and cannot overlap;
- deterministic `run_id` changes when reference identity or scope changes;
- deterministic prediction path changes when input hash/backend/config identity changes;
- path remains independent from reference manifest hash, so a reference-only change can reuse the same prediction;
- invalid SHA/commit/filter values fail before filesystem/model work.

Run:

```bash
uv run pytest tests/benchmark/test_oaf_corpus_run.py -q
```

Expected: FAIL before the module exists.

### Step 3.2 — Implement only the pure identity helpers

Use existing `canonical_json_bytes()`, `require_sha256()`, and backend descriptor/model config APIs. Do not add another JSON encoder.

Inference config should contain exactly the design fields:

```python
{
    "schema": OAF_INFERENCE_CONFIG_SCHEMA,
    "backend_descriptor_sha256": ...,
    "model_lock_sha256": ...,
    "checkpoint_archive_sha256": ...,
    "adapter_revision": OAF_ADAPTER_REVISION,
    "prediction_map_version": OAF_PREDICTION_MAP_ID,
    "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
    "canonicalization_revision": OAF_CANONICALIZATION_REVISION,
}
```

Prediction path shape:

```text
predictions/<simfile_id>/<input_audio_sha256>/<backend_descriptor_sha256>/<inference_config_sha256>.jsonl
```

Run ID hashes reference identities + model/config identity + normalized include/exclude scope.

### Step 3.3 — Add canonical `run.json` rendering/parsing tests

Define the smallest internal run/item snapshots needed by HPA-326 and test:

- canonical JSON byte stability;
- round-trip parsing;
- exact schema/run identity validation;
- item sorting by simfile ID;
- count reconciliation;
- `resumed` is a successful execution disposition, not `skipped`;
- skipped rows require explicit filter exclusion;
- completed run timestamp is only present when iteration finished.

Use one JSON document, not JSONL/event sourcing.

### Step 3.4 — Implement atomic snapshot publication

Add a private writer that stages to the run directory then `os.replace()`s `run.json`. Preserve the original `started_at` on resume. A crash after prediction publication but before snapshot replacement must be recoverable from the immutable prediction artifact on the next `--resume` invocation.

Do not build journaling or a database.

### Step 3.5 — Run focused tests

```bash
uv run pytest tests/benchmark/test_oaf_corpus_run.py -q
```

Expected: PASS for pure contract/snapshot tests.

### Step 3.6 — Commit

```bash
git add src/benchmark/oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run.py
git commit -m "feat: define OaF corpus run contract"
```

---

## Task 4: Resolve authoritative source audio and materialize the full-mix input

**Files:**

- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run.py`

### Step 4.1 — Write cache-resolution tests

Build small HPA-323 fixture rows whose carried `RemoteObject` is:

1. already verified;
2. not marked verified in the immutable manifest but has a matching verified `CacheIndexStore` entry;
3. pointed at an index entry whose remote identity changed;
4. pointed at a missing/corrupt cache body;
5. pointed at a cache body whose SHA differs from `source_audio_content_hash`.

The helper should return a validated local path only for cases 1/2.

Implementation sketch:

```python
def _resolve_source_audio(
    timing_row: LoadedReferenceTimingRow,
    *,
    cache_dir: Path,
    cache_index: CacheIndexStore,
) -> ResolvedSourceAudio:
    chart_view = reference_chart_view_from_timing_row(timing_row)
    ...
```

Use only public HPA-321 cache primitives:

- `CacheIndexStore.get()`;
- `cache_entry_matches_remote()`;
- `validate_cached_body()`;
- `resolve_verified_cache_body()`.

There must be no R2 config/store import in `oaf_corpus_run.py`.

### Step 4.2 — Add source-duration preflight tests

For a resolved local audio fixture, call existing `inspect_source_audio()` and retain:

- decoded duration;
- sample rate/channels/frames as optional internal evidence if useful;
- exact HPA-323 content hash already verified by cache resolution.

Test that duration probe failure becomes an item-local source/decode failure rather than an exception that discards other rows.

### Step 4.3 — Write canonical materialization tests

Use a tiny stereo 22.05 kHz WAV as authoritative source input. Test the private materializer produces a temporary file beneath the configured worker input root and that `load_materialized_audio()` returns:

- 44,100 Hz;
- mono;
- 16-bit PCM;
- source SHA equal to the authoritative source hash;
- input SHA equal to the staged WAV bytes;
- same source ID / fixed input-view ID.

Use existing `librosa.load(..., sr=44100, mono=True)` and `soundfile.write(..., format="WAV", subtype="PCM_16")`.

Do not add ffmpeg or another audio dependency.

### Step 4.4 — Ensure temporary input cleanup

Add a test that the staged WAV is removed in `finally` after success or item-local inference failure. Keep only the source cache + prediction/run metadata durable.

### Step 4.5 — Run focused tests

```bash
uv run pytest tests/benchmark/test_oaf_corpus_run.py -q
```

Expected: PASS.

### Step 4.6 — Commit

```bash
git add src/benchmark/oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run.py
git commit -m "feat: materialize authoritative OaF inputs"
```

---

## Task 5: Implement sequential inference, immutable prediction publication, and resume

**Files:**

- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run.py`
- Create: `tests/benchmark/test_oaf_corpus_run_acceptance.py`

### Step 5.1 — Write a multi-song persistent-backend test

Inject a fake backend factory that records:

- number of backend objects created;
- `transcribe()` calls and input paths;
- `close()` calls.

Run three eligible items and assert:

```text
backend factory calls = 1
transcribe calls       = 3
close calls            = 1
```

The fake prediction must flow through real `map_oaf_prediction()` and real `publish_prediction_artifact()` so the test pins prediction v2 reuse.

### Step 5.2 — Implement the one-backend lifecycle

Public entry point:

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

Implementation order:

1. load/validate HPA-324 + HPA-323 manifests;
2. verify exact lineage before model work;
3. load model config/descriptor and derive frozen identities;
4. build run ID/directory + work input root;
5. load cache index once;
6. reconstruct reference mappings for scoreable rows;
7. preflight eligible source durations for projection coverage;
8. create one backend using the fixed run work root;
9. process rows in simfile-ID order;
10. close backend once in `finally`;
11. finalize run snapshot + reports.

Keep progress callbacks/printing out of the domain module unless the existing CLI pattern already requires one; the CLI can report coarse progress.

### Step 5.3 — Write success publication assertions

For each inferred success assert:

- `map_oaf_prediction()` is used;
- prediction path matches the deterministic identity path;
- `publish_prediction_artifact()` writes canonical v2;
- run row records prediction SHA/path, source key/hash, chart key/hash, input hash/view, wall time, duration, RTF;
- prediction artifact itself remains free of selected-chart/reference identity;
- run snapshot checkpoints after the row.

Time only the item execution window needed for HPA-326's wall-time/RTF evidence. Keep the definition explicit and stable in a docstring/test.

### Step 5.4 — Write `--resume` tests before implementation

Cases:

1. matching existing prediction -> no `transcribe()`, disposition `resumed`, scorer status success;
2. missing prediction -> transcribe once and publish;
3. prior failed run row + no artifact -> transcribe once;
4. existing prediction has wrong source hash -> explicit artifact failure, no overwrite;
5. wrong input hash -> explicit artifact failure, no overwrite;
6. wrong descriptor/model/map/view -> explicit artifact failure, no overwrite;
7. existing target without `resume=True` -> explicit output conflict, no overwrite;
8. prediction file exists but previous `run.json` never recorded it -> `resume=True` still validates/reuses the artifact.

The resume path may re-materialize input to recompute `input_audio_sha256`; do not add a derived-input cache in this task.

### Step 5.5 — Implement resume validation

Use `read_prediction_artifact()` and compare its canonical fields with the current `CanonicalAudio`, backend descriptor/model, and map version. The deterministic path already binds adapter/config identity; still validate the artifact's actual header/event metadata before reuse.

Never call `update_file`/overwrite an existing prediction path at runtime; immutable publication is the guardrail.

### Step 5.6 — Write independent-failure tests

Make the fake backend fail one middle song and succeed the next. Assert:

- earlier success remains persisted;
- failed row has stable grouped failure + detailed run error code/message;
- later row still runs;
- final counts balance.

Keep runner-level detailed codes small and operational, e.g. source unavailable/decode, canonical input invalid, backend/inference, prediction invalid/conflict/publish. Map them to HPA-325's existing grouped failure reasons rather than expanding scorer taxonomy.

### Step 5.7 — Add acceptance fixture for interrupted/resumed publication

`tests/benchmark/test_oaf_corpus_run_acceptance.py` should exercise two invocations over 3–4 local fixture rows:

- first invocation publishes some predictions and records one failure/interruption-shaped state;
- second invocation with resume reuses valid outputs and runs only missing/failed rows;
- final prediction files are readable v2 artifacts;
- no successful output changed bytes across resume.

No real Docker/model in CI; the real-checkpoint pilot is an operational gate later.

### Step 5.8 — Run focused tests

```bash
uv run pytest \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: PASS.

### Step 5.9 — Commit

```bash
git add src/benchmark/oaf_corpus_run.py \
        tests/benchmark/test_oaf_corpus_run.py \
        tests/benchmark/test_oaf_corpus_run_acceptance.py
git commit -m "feat: run resumable OaF corpus inference"
```

---

## Task 6: Add runtime projection and HPA-325 fixed-control scoring

**Files:**

- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run_acceptance.py`

### Step 6.1 — Write projection tests

Given a filtered pilot with known durations/timings, assert:

```python
aggregate_rtf == measured_wall_time_sec / measured_audio_duration_sec
projected_full_wall_time_sec == aggregate_rtf * full_eligible_audio_duration_sec
```

Also assert:

- only actual measured inference rows contribute timing;
- resume hits with no retained timing do not contribute zero duration/time;
- full eligible duration is header-probed across the HPA-324 eligible population, independent of include/exclude inference scope;
- missing duration for any eligible source makes full projection `None` and reports projection coverage count instead of inventing a number.

Do not add confidence intervals or parallel-runtime estimates.

### Step 6.2 — Implement projection summary

Persist the measurement/projection in `run.json` and expose it in `OafCorpusRunOutcome` for CLI output.

The operator's fixed include list is already part of run identity, so no extra pilot schema/config file is needed.

### Step 6.3 — Write scorer-adaptation tests

Use real HPA-323 native reference artifacts + real HPA-324 eligibility rows and construct HPA-325 items for:

- successful inferred prediction;
- successful resumed prediction;
- explicit filter skip;
- upstream HPA-324 quarantine;
- inference failure.

Assertions:

- success uses `cohort_item_from_artifacts()`;
- references are mapped with `map_reference_events()`;
- failed/skipped/quarantined coverage balances;
- resumed stays `success`;
- scorer population counts equal run counts;
- `CohortIdentity` contains exact HPA-324 manifest hash, HPA-323 timing version, frozen taxonomy/lane map, model lock, descriptor, prediction map, input view, and run ID.

### Step 6.4 — Implement HPA-325 reuse

After execution, call only existing APIs:

```python
result = score_cohort(identity, tuple(items), diagnostics_for=())
artifacts = write_cohort_reports(result, run_dir / "reports")
```

Do not add matching/alignment/aggregate code to HPA-326.

### Step 6.5 — Prove scoring does not require inference

In the acceptance test:

1. finish a run and persist predictions;
2. invoke the HPA-326 assembly/scoring path again with a backend fake that would fail if `transcribe()` is called;
3. use `resume=True`;
4. assert reports are regenerated from persisted artifacts and `transcribe()` count is zero for matching successes.

This pins the acceptance criterion that later comparison can score/pair OaF without rerunning it.

### Step 6.6 — Run focused tests

```bash
uv run pytest \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_reports.py -q
```

Expected: PASS.

### Step 6.7 — Commit

```bash
git add src/benchmark/oaf_corpus_run.py \
        tests/benchmark/test_oaf_corpus_run.py \
        tests/benchmark/test_oaf_corpus_run_acceptance.py
git commit -m "feat: score OaF corpus control runs"
```

---

## Task 7: Add the thin `run-oaf-corpus` CLI

**Files:**

- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`

### Step 7.1 — Write failing CLI tests

Add Click tests for:

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

Use monkeypatch/injection so CLI tests do not start Docker.

Assert request construction, repeated include/exclude parsing, and canonical stdout JSON containing at least:

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

### Step 7.2 — Implement thin command

The Click handler should:

- construct `OafCorpusRunRequest`;
- call `run_oaf_corpus()`;
- print the returned summary as JSON;
- exit with the domain outcome code.

Keep heavy imports inside the command/module path consistently with the existing benchmark CLI so ordinary CLI startup does not eagerly import TensorFlow/model runtime.

Do **not** add `--backend`.

### Step 7.3 — Pin exit semantics

CLI tests:

- exit 0 when all selected eligible rows succeed (upstream quarantines/filter skips allowed);
- exit 1 when selected eligible rows have item-local failures but run/report artifacts exist;
- exit 2 on fatal manifest/identity/setup failure.

### Step 7.4 — Run CLI tests

```bash
uv run pytest tests/test_cli_benchmark.py -q
```

Expected: PASS.

### Step 7.5 — Commit

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py
git commit -m "feat: add OaF corpus benchmark command"
```

---

## Task 8: Operational acceptance — fixed real-checkpoint pilot before broad run

**Files:**

- No production code required unless the real run exposes a concrete bug.
- Record evidence in HPA-326 / PR discussion rather than adding a permanent benchmark-data file solely for the pilot.

This task is intentionally not a CI test because it requires the real HPA-423 Docker image/checkpoint and authoritative local corpus cache.

### Step 8.1 — Freeze a technically diverse pilot set before looking at model scores

Choose a small set (target 4–6 songs) from HPA-324/HPA-323 metadata only. Include at least:

- a relatively short and long track;
- lower and higher reference-event density;
- more than one source pack/audio representation when available.

Record the exact IDs in the command; the normalized include set becomes part of `run_id`/`run.json`.

Do not choose/revise the set based on OaF F1.

### Step 8.2 — Run the fixed pilot with real checkpoint

Example shape:

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

Capture:

- run ID + exact manifest hashes;
- backend descriptor/model/checkpoint identity;
- adapter revision + inference-config hash;
- each song's duration/wall time/RTF;
- aggregate RTF;
- full eligible duration coverage;
- projected full sequential wall time;
- prediction artifact validation;
- score/report paths.

### Step 8.3 — Prove persistent worker on real execution

Confirm worker/model ready is observed once for the process and multiple song requests complete through that same worker. Use existing logs/process evidence; do not add telemetry infrastructure solely for this proof.

### Step 8.4 — Prove real resume

Immediately rerun the same pilot with `--resume` and assert:

- all matching pilot predictions are reused;
- no OaF inference requests run for those successes;
- prediction bytes/SHA remain unchanged;
- reports regenerate/read successfully.

### Step 8.5 — Review projection before broad run

This is the HPA-326 decision gate. If the sequential projection is acceptable, continue with one worker. If it is unexpectedly prohibitive, capture the measured evidence and adjust HPA-326 scope explicitly before adding concurrency.

Do not preemptively parallelize.

### Step 8.6 — Run the full eligible corpus

Same command, no include/exclude flags:

```bash
uv run crux benchmark run-oaf-corpus \
  --manifest <exact-hpa324-manifest> \
  --timing-manifest <exact-hpa323-manifest> \
  --cache-dir <exact-r2-cache> \
  --output-dir artifacts/benchmark/oaf-corpus \
  --resume
```

Using `--resume` is safe even if pilot predictions exist under a different cohort run ID because the deterministic prediction path is reference/scope-independent and exact identity is revalidated.

### Step 8.7 — Reconcile the fixed control

Before marking HPA-326 done, assert from `run.json` + HPA-325 reports:

```text
eligible = successful + failed + explicitly skipped-within-scope-policy
manifest total = success + failed + skipped + quarantined
```

For the unfiltered broad run there should be no explicit filter skips. Every eligible row must therefore have either a valid prediction or explicit failure.

Record the final run/report path as the OaF full-mix control input for HPA-395/HPA-328/HPA-562.

---

## Task 9: Full verification and scope audit

### Step 9.1 — Run targeted benchmark suite

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

### Step 9.2 — Run the full test suite

```bash
uv run pytest -q
```

Expected: PASS.

### Step 9.3 — Run CI-equivalent static checks

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check
```

Expected: all PASS/no output from `git diff --check`.

### Step 9.4 — Audit architecture against scope lock

Before final review, verify the implementation still has:

- exactly one corpus runner module rather than a framework;
- exactly one backend instance/worker lifecycle;
- no R2/network dependency from the runner;
- prediction artifact v2 unchanged;
- no generic retry/queue/pool abstractions;
- no HPA-320 seal/attestation compatibility;
- no corpus-derived model/scoring tuning;
- no durable canonical WAV duplication;
- HPA-325 scoring/report code reused rather than copied.

If a new abstraction has only one caller and is not necessary for testability/identity correctness, collapse it before review.

### Step 9.5 — Final commit if verification fixes were required

```bash
git add <only-files-changed-by-verification-fixes>
git commit -m "fix: finalize HPA-326 corpus runner"
```

Do not create an empty cleanup commit.

---

## Expected implementation result

After this plan is executed, Crux has one command that can:

1. consume an exact HPA-324/HPA-323 reference lineage;
2. resolve the exact authoritative full mix from the existing local content cache;
3. canonicalize it into the fixed OaF input view;
4. reuse one validated OaF worker across sequential songs;
5. persist native + canonical prediction v2 artifacts immutably;
6. resume only exact matching outputs and retry only missing/failed items;
7. record per-song runtime/RTF and publish the pilot-derived full-corpus projection;
8. reconcile success/failure/skipped/quarantined population;
9. produce the HPA-325 OaF full-mix control reports without rerunning inference;
10. provide stable prediction/run paths to HPA-395, HPA-328, and HPA-562.

That is the complete HPA-326 requirement. Concurrency, generic runners, additional models, separated inputs, and benchmark comparison belong to later tickets.
