# HPA-326 OaF Full-Mix Corpus Inference Design

**Status:** Proposed  
**Date:** 2026-08-13  
**Linear:** HPA-326 — Run validated OaF Drums on the authoritative full-mix corpus and persist predictions

## Decision summary

Implement HPA-326 as one OaF-specific corpus orchestrator over the seams already landed:

- HPA-321 owns the authoritative R2 inventory and local content-addressed cache.
- HPA-323 owns source-audio selection and audio-relative reference timing.
- HPA-324 owns benchmark eligibility and the frozen DTX/common taxonomy.
- HPA-423 owns the reusable OaF backend, exact released checkpoint/source identity, and persistent sequential worker.
- HPA-423 prediction artifact v2 owns native + mapped prediction persistence.
- HPA-325 owns single-cohort scoring and deterministic reports.

HPA-326 adds only the missing execution layer: resolve the exact cached full mix, materialize the fixed OaF input view, run one persistent sequential worker with a corpus-scale request deadline, persist/reuse immutable prediction artifacts, keep an atomic mutable run snapshot, measure runtime, then adapt rows into HPA-325 scoring.

Do **not** introduce a generic multi-backend runner, job queue, worker pool, retry/backoff engine, model registry, database, second prediction schema, durable derived-audio cache, seal/attestation layer, or compatibility reader.

## Why HPA-326 is next

HPA-326 is High priority and its blockers HPA-321, HPA-323, HPA-324, and HPA-423 are complete. HPA-325 is also merged, so the OaF baseline can be scored immediately after inference.

Completing HPA-326 unlocks:

- HPA-395 — frozen MuScriptor comparison;
- HPA-328 — OaF drum-stem separation ablation;
- HPA-329 — final comparative benchmark findings.

The repository is missing orchestration, not another model/scoring abstraction.

## Keep the selected shape

The following decisions remain unchanged:

- one OaF-specific runner rather than a CLI loop or generic backend pipeline;
- prediction artifact v2 remains reference-independent;
- resume re-materializes and rehashes instead of adding a canonical-input cache;
- no R2 repair/fill from the inference runner;
- one sequential persistent `OafBackend`;
- concurrency only after measured pilot runtime justifies it;
- HPA-325 failure enum remains unchanged;
- no model/mapping/threshold tuning from corpus results.

## Existing contracts to reuse

### HPA-423 backend

`src/benchmark/backends/oaf.py` already provides:

- `OafBackend`;
- a lazily started `WorkerProcess` retained across requests;
- exact descriptor/model/upstream/native-output identity;
- default checkpoint lookup through `CRUX_OAF_CHECKPOINT_CACHE` or `artifacts/benchmark/model-cache`;
- one read-only `input_root` mount;
- `transcribe(CanonicalAudio) -> NativePrediction`;
- no heuristic fallback.

HPA-326 creates the backend through the existing `create_backend()` seam and must not add `--checkpoint-dir`. The CLI `--cache-dir` is **only** the HPA-321 corpus/audio cache.

### Worker timeout and poison semantics

The HPA-423 default `timeout_seconds=30.0` is suitable for smoke execution, not full songs. `WorkerProcess.request()` uses that value as the response deadline. A timeout/protocol error poisons and closes the worker process; `OafBackend` retains that closed process object, so later requests through the same backend cannot succeed.

HPA-326 therefore defines one non-CLI corpus deadline:

```python
OAF_CORPUS_REQUEST_TIMEOUT_SECONDS = 3600.0
```

and constructs the backend as:

```python
create_backend(
    input_root=work_root,
    timeout_seconds=OAF_CORPUS_REQUEST_TIMEOUT_SECONDS,
)
```

Use the existing checkpoint default. Do not expose the timeout as a user-facing tuning knob in this ticket.

The lifecycle semantics are intentionally simple:

- a model-level inference error returned as a valid worker response is item-local; record it and continue while the worker remains usable;
- a source/canonicalization/prediction-artifact error that never poisons the worker is item-local; record it and continue;
- a worker startup/readiness/timeout/protocol failure makes the backend unusable for later inference in that invocation; checkpoint `run.json`, stop issuing new inference requests, close the backend, and exit partial;
- `--resume` is the recovery path for missing/failed rows on the next invocation.

Do **not** build worker restart/retry machinery. If the pilot later proves that restarting one backend after poison is materially useful, that is a measured follow-up rather than part of the initial design.

### Prediction artifact v2

`src/benchmark/prediction_artifact.py` already persists canonical immutable JSONL containing:

- source/input audio identities;
- backend descriptor/model identity;
- native OaF class/bin/confidence/velocity;
- canonical/common mapping and prediction-map version.

HPA-326 reuses:

- `map_oaf_prediction()`;
- `publish_prediction_artifact()`;
- `read_prediction_artifact()`.

Reference/chart/run provenance stays in `run.json` and HPA-325 inputs. A later reference correction can therefore rescore an unchanged prediction without rerunning OaF.

### HPA-323/HPA-324 lineage

HPA-323 rows already carry:

- selected authoritative `source_audio_key`;
- `source_audio_content_hash`;
- immutable reference-event artifact path.

HPA-324 carries those fields forward with eligibility/mapping accounting.

The HPA-324 loader must reuse the existing canonical manifest core from `reference_timing_manifest.py`. Promote `_read_canonical_manifest_core()` to a public `read_canonical_manifest_core()` (or equivalent public name) and use it for HPA-322, HPA-323, and the new HPA-324 loader. Do not fork canonical JSONL framing/hash/round-trip logic.

Domain-specific HPA-324 work remains local to `reference_set_manifest.py`:

- `_validate_reference_set_row()`;
- unique simfile IDs;
- one HPA-323 source timing hash/version;
- narrow `ReferenceSetRowView` fields.

Promote `_read_native_reference_events()` to `read_native_reference_events()` without changing its safe-path/hash/identity validation.

When HPA-326 reads native references, derive:

```python
timing_output_root = timing_manifest_path.parent.parent
```

because HPA-323 event paths are relative to the timing output directory, not the manifest file directory.

### HPA-321 cache primitives

Compose the existing public cache operations:

- `CacheIndexStore.get()`;
- `cache_entry_matches_remote()`;
- `validate_cached_body()`;
- `resolve_verified_cache_body()`.

Do not copy `_resolve_or_queue_audio()` from HPA-323: its R2-fill behavior is explicitly outside HPA-326.

### HPA-325 scoring

`src/benchmark/cohort_scoring.py` already provides:

- `CohortIdentity`;
- success/failed/skipped/quarantined accounting;
- `cohort_item_from_artifacts()`;
- the closed `CohortFailureReason` set;
- 30/50/100 ms raw + aligned scoring;
- deterministic aggregates.

`src/benchmark/reports.py` writes the six report artifacts. HPA-326 builds items and calls those APIs. No scoring logic belongs in the runner.

## Goals

1. Run the validated OaF model over each eligible authoritative full-mix song with one persistent sequential worker.
2. Make restart safe by reusing only prediction artifacts whose complete input/model/config identity matches.
3. Persist per-song execution outcome, runtime, RTF, input/source/chart provenance, and prediction path in one atomic run snapshot.
4. Produce a fixed-batch full-corpus wall-time projection before broad inference.
5. Score the OaF cohort through HPA-325 without rerunning inference.
6. Keep prediction artifacts reusable by later paired comparisons/input-view ablations.

## Non-goals

- generic multi-model orchestration;
- parallel workers, batching, queues, distributed execution, or autoscaling;
- worker restart/retry/backoff policy;
- R2 download/fill in the runner;
- stem separation;
- threshold tuning, calibration, fine-tuning, or manual correction;
- MIDI as scoring input;
- changing prediction artifact v2;
- seal evidence/runtime locks/host attestation/HPA-320 compatibility;
- database or Parquet storage;
- durable canonical-input cache;
- mutable global “latest baseline” pointer.

## Approaches considered

### A. Loop over the existing single-song smoke command

Rejected. It repeatedly constructs execution state, does not own coherent run identity/resume, and cannot satisfy the persistent-worker requirement.

### B. Generic corpus pipeline with pluggable backends

Rejected under YAGNI. HPA-395 has not yet demonstrated that MuScriptor needs the same execution contract. Extract common orchestration only after a second concrete runner proves duplication.

### C. One OaF-specific corpus runner — selected

A narrow `oaf_corpus_run.py` owns exactly:

```text
manifests -> local cache -> canonical full mix -> OaF worker
          -> prediction v2 -> run.json -> HPA-325 reports
```

## Input contracts and preflight

CLI inputs:

```text
--manifest          exact HPA-324 benchmark-reference manifest
--timing-manifest   exact HPA-323 reference-timing manifest referenced by HPA-324
--cache-dir         HPA-321 local content-addressed corpus cache
--output-dir        HPA-326 predictions/runs/reports root
```

No `--backend`, `--checkpoint-dir`, or timeout flag.

Before any model inference the runner must validate:

- HPA-324 manifest through `load_reference_set_manifest()` built on `read_canonical_manifest_core()`;
- exact HPA-324 -> supplied HPA-323 hash/version lineage;
- unique/corresponding simfile IDs;
- frozen taxonomy/lane-map versions;
- valid OaF model config/descriptor;
- include/exclude sets contain only IDs present in HPA-324;
- include/exclude sets do not overlap.

Unknown include **or exclude** IDs and overlap are fatal preflight errors (exit 2). A typo must not silently create a smaller cohort/run identity.

The CLI should reuse `click.IntRange(0, MAX_SIMFILE_ID)` for repeated include/exclude options.

## Authoritative source-audio resolution

For a timing-ready row:

1. reconstruct the validated HPA-322/source inventory view from the HPA-323 row;
2. locate the `RemoteObject` whose key equals `source_audio_key`;
3. load `CacheIndexStore` once for the run;
4. if the carried remote is not already verified, look up exact `(endpoint, bucket, key)`;
5. require `cache_entry_matches_remote()`;
6. require the cache-entry SHA-256 to equal HPA-323 `source_audio_content_hash`;
7. require `validate_cached_body(...).state == "verified"`;
8. resolve through `resolve_verified_cache_body()`;
9. use existing `inspect_source_audio()` for header-only duration/sample metadata.

Missing/corrupt/mismatched content is an item failure. HPA-326 does not reach R2 to repair it.

## Canonical full-mix input view

The authoritative source is commonly compressed audio while OaF expects canonical WAV. Define:

```text
OAF_FULL_MIX_INPUT_VIEW_ID = "crux.oaf-full-mix-mono44k1-pcm16/v1"
OAF_CANONICALIZATION_REVISION = "librosa-mono44k1-soundfile-pcm16/v1"
```

Materialization v1:

- decode with existing `librosa`;
- resample to 44,100 Hz;
- mix to mono;
- write WAV PCM16 through existing `soundfile`;
- validate through a new `load_materialized_audio()` sibling of `load_direct_audio_bytes()`;
- `load_materialized_audio()` still calls `parse_canonical_wav()`;
- preserve the authoritative source hash as `source_audio_sha256`;
- compute staged WAV hash as `input_audio_sha256`.

The staged WAV lives under the worker `input_root` and is deleted in `finally`. Do not persist a second input-view manifest or duplicate the full audio corpus.

## Frozen model/inference identity

### Adapter revision

Add:

```python
OAF_ADAPTER_REVISION = "crux.oaf-adapter/v1"
```

Bump only when host-side inference semantics change.

### Model lock

`model_lock_sha256` is the exact SHA-256 of checked-in `runtime/oaf_tf1/model.json`. Record `checkpoint_archive_sha256` separately.

### Inference config

Canonical-hash a closed payload:

```text
schema
backend_descriptor_sha256
model_lock_sha256
checkpoint_archive_sha256
adapter_revision
prediction_map_version
input_view_id
canonicalization_revision
```

No corpus-derived thresholds belong here.

## Run identity

Hash a canonical payload containing:

```text
reference_manifest_sha256
reference_timing_manifest_sha256
backend_descriptor_sha256
model_lock_sha256
checkpoint_archive_sha256
adapter_revision
inference_config_sha256
input_view_id
sorted include IDs
sorted exclude IDs
```

Derive deterministic `run_id` such as `oaf-<16 hex>`. Reference/scope changes create a new run directory while prediction paths remain reusable when inference identity is unchanged.

## Prediction path and resume

After canonical materialization:

```text
<output>/predictions/
  <simfile_id>/
    <input_audio_sha256>/
      <backend_descriptor_sha256>/
        <inference_config_sha256>.jsonl
```

`publish_prediction_artifact()` remains immutable.

### `--resume`

1. materialize and hash current canonical input;
2. compute expected prediction path;
3. if path exists, read/validate v2;
4. verify source ID/hash, input-view ID/hash, descriptor/model, and map identity;
5. matching artifact -> execution disposition `resumed`, scorer status `success`, no `transcribe()`;
6. missing artifact -> run inference;
7. invalid/mismatched artifact -> explicit failure, never overwrite.

Without `--resume`, an **existing** target is `prediction_output_conflict`. A **missing** target is normal and proceeds to inference.

Failed/missing outputs rerun naturally under `--resume`; there is no retry counter.

## Persistent sequential worker lifecycle

Preflight/cache/source-duration work may happen before backend creation. Once inference is needed:

```text
create backend once with input_root + corpus timeout
for selected eligible rows in simfile-id order:
    resolve/materialize
    resume OR transcribe
    map/publish
    atomically checkpoint run.json
    if worker was poisoned: stop issuing inference requests
close backend once
score/report persisted state
```

A run containing only resume hits need not start the worker because `OafBackend` remains lazy.

## Closed runner failure mapping

Keep detailed operational codes in `run.json`, but map them through one closed table to the existing HPA-325 enum.

Define a small runner literal/table in `oaf_corpus_run.py`:

| Runner failure code | HPA-325 `CohortFailureReason` |
| --- | --- |
| `source_audio_unavailable` | `inference_failed` |
| `source_audio_decode_failed` | `inference_failed` |
| `canonical_input_failed` | `inference_failed` |
| `backend_unavailable` | `backend_unavailable` |
| `worker_protocol_failed` | `backend_unavailable` |
| `inference_failed` | `inference_failed` |
| `prediction_artifact_invalid` | `prediction_artifact_invalid` |
| `prediction_output_conflict` | `prediction_artifact_invalid` |
| `prediction_publish_failed` | `prediction_artifact_invalid` |
| `prediction_missing` | `prediction_missing` |

`prediction_missing` is reserved for scoring/adaptation evidence that a run state expects a prediction artifact but the artifact is actually absent. A failed publication is **not** ambiguously grouped; it is `prediction_artifact_invalid`.

Upstream HPA-324 quarantine and filter skips use their existing non-failure mappings:

- `quarantined` / `reference_quarantined`;
- `skipped` / `explicitly_skipped`.

Do not extend HPA-325's enum.

## Run snapshot

Persist one mutable canonical document:

```text
<output>/runs/<run_id>/run.json
```

Rewrite atomically after every completed row using the same repository pattern already used for mutable snapshots:

1. write a same-directory temporary file;
2. flush + `os.fsync()`;
3. `os.replace()` to `run.json`;
4. fsync the containing directory where the existing helper/pattern does so.

Do not use `publish_immutable_file()` for `run.json`; it is intentionally a mutable checkpoint.

### Run header

- schema/run ID;
- exact HPA-324/HPA-323 identities;
- model/backend/checkpoint identities;
- adapter revision/inference config;
- `OAF_CORPUS_REQUEST_TIMEOUT_SECONDS`;
- input-view identity;
- Crux commit;
- normalized include/exclude scope;
- started/completed timestamps;
- status/counts;
- runtime projection.

### Per-song row

- simfile ID;
- HPA-324 eligibility/reasons/warnings;
- chart key/hash;
- source audio key/hash/duration;
- input view/hash when materialized;
- disposition (`inferred`, `resumed`, `skipped`, `quarantined`, `failed`);
- detailed runner failure code + bounded detail;
- prediction path/SHA for success;
- measured wall time/RTF for actual inference.

Resume hits remain scorer-level success.

## Runtime measurement and pilot projection

Choose a 4–6 song pilot before inspecting model scores, using corpus metadata only (duration, density, source pack/audio representation).

Header-probe all eligible locally cached source audio with existing `inspect_source_audio()` before broad inference.

For actual pilot inference record:

```text
measured_item_count
measured_audio_duration_sec
measured_wall_time_sec
aggregate_rtf = measured_wall_time_sec / measured_audio_duration_sec
full_eligible_audio_duration_sec
projected_full_wall_time_sec = aggregate_rtf * full_eligible_audio_duration_sec
projection_coverage_count
request_timeout_seconds = OAF_CORPUS_REQUEST_TIMEOUT_SECONDS
```

Resume hits without retained measured timing do not count as zero-cost inference. If full eligible duration coverage is incomplete, projection is `null`.

Do not add confidence intervals, memory telemetry, parallel speedup models, or queues.

## Status and exit semantics

- `0`: every selected eligible item has a valid prediction; upstream quarantines/filter skips are reconciled;
- `1`: selected eligible work is incomplete/failed but a trustworthy run snapshot/report was produced, including worker-poison stop;
- `2`: fatal preflight/run-level setup/publication failure prevented trustworthy execution.

A worker protocol/start/readiness/timeout poison is exit 1 once the current run snapshot is checkpointed; subsequent not-yet-inferred rows remain missing for `--resume` rather than being falsely reported as successful.

## Scoring and fixed control

For each row:

- derive `timing_output_root = timing_manifest_path.parent.parent`;
- load native references through `read_native_reference_events()`;
- map with `map_reference_events()`;
- successes use `cohort_item_from_artifacts()`;
- non-success rows use the closed runner -> HPA-325 failure table;
- `CohortIdentity.cohort_id = run_id`;
- HPA-324 hash and HPA-323 timing version are exact;
- backend/model/map/input-view fields come from frozen run identity.

Then call only:

```python
result = score_cohort(identity, tuple(items), diagnostics_for=())
artifacts = write_cohort_reports(result, run_dir / "reports")
```

No broad event diagnostics by default.

A full unfiltered run with no eligible failures becomes the fixed OaF full-mix control. Later tickets receive explicit run/report paths; no global registry is needed.

## CLI

```bash
crux benchmark run-oaf-corpus \
  --manifest artifacts/benchmark/reference-set/manifests/<sha>.jsonl \
  --timing-manifest artifacts/benchmark/reference-timing/manifests/<sha>.jsonl \
  --cache-dir artifacts/benchmark/r2-corpus/cache \
  --output-dir artifacts/benchmark/oaf-corpus \
  [--include-simfile-id ID ...] \
  [--exclude-simfile-id ID ...] \
  [--resume]
```

Repeated ID options use `click.IntRange(0, MAX_SIMFILE_ID)`.

Do not add `--backend`, `--checkpoint-dir`, or timeout flags.

Stdout emits one small canonical JSON summary; progress may go to stderr.

## File-level design

### Create

- `src/benchmark/oaf_corpus_run.py` — identity, preflight, cache/source resolution, canonical materialization, sequential orchestration, closed failure mapping, run snapshot, resume, projection, HPA-325 adaptation.
- `tests/benchmark/test_oaf_corpus_run.py` — identity/preflight/cache/timeout/failure/resume/orchestration tests with injected fake backend.
- `tests/benchmark/test_oaf_corpus_run_acceptance.py` — multi-song persisted-artifact + poison/resume + score/report acceptance with fakes/local fixtures.

### Modify

- `src/benchmark/reference_timing_manifest.py` — promote `read_canonical_manifest_core()` and add narrow timing-row chart/source helper.
- `src/benchmark/reference_set_manifest.py` — HPA-324 loader via the shared core; promote native-reference reader.
- corresponding manifest tests;
- `src/benchmark/input_view.py` — `load_materialized_audio()` sibling using `parse_canonical_wav()`;
- `src/benchmark/backends/oaf.py` — semantic adapter revision constant only;
- `src/cli/benchmark.py` — thin command;
- CLI/OaF/input-view tests.

### Explicitly unchanged

- runtime OaF model/worker inference semantics;
- prediction artifact v2;
- taxonomy/mapping semantics;
- HPA-325 failure enum/scoring/report schemas;
- HPA-321/HPA-323 R2-fill path;
- old HPA-320 security/seal artifacts.

## Hard gates

### Gate A — shared canonical manifest reader

The HPA-324 loader calls the promoted existing canonical JSONL core; no forked reader.

### Gate B — reference lineage and scope

No inference before exact HPA-324 -> HPA-323 lineage passes. Unknown include/exclude IDs or overlap fail preflight with exit 2.

### Gate C — authoritative source body

Only a locally verified cache body matching original remote identity + HPA-323 SHA is used. No R2 substitution.

### Gate D — corpus request deadline and poison behavior

Tests prove backend factory receives `OAF_CORPUS_REQUEST_TIMEOUT_SECONDS`. Protocol/timeout poison stops further inference in that invocation; resume is recovery.

### Gate E — persistent sequential worker

Normal multi-song inference uses one backend object/worker and closes it once.

### Gate F — resume identity

Reuse only exact source/input/backend/model/map/config match; mismatches never overwrite.

### Gate G — closed failure mapping

Every detailed runner failure code maps deterministically to one existing HPA-325 reason; no ad-hoc mapping in exception handlers.

### Gate H — prediction schema reuse

Exactly prediction artifact v2 after `map_oaf_prediction()`; reference/run metadata remains outside it.

### Gate I — population balance

Run rows and HPA-325 reports reconcile exact HPA-324 population, including quarantine/skips/failures/missing-after-poison.

### Gate J — pilot evidence before broad run

Record pilot IDs, request timeout, RTF, full eligible duration coverage, and projected sequential wall time before unfiltered execution.

## Risks and deliberately deferred work

### Canonicalization cost on resume

Accepted. Decode/rehash is cheaper than OaF and avoids a second cache. Add one only if profiling later proves it matters.

### Worker poison

Known and specified, not speculative. The corpus deadline prevents the 30-second smoke default from killing normal tracks. Protocol poison stops further inference in that invocation; `--resume` recovers. No restart framework now.

### Storage volume

Only prediction JSONL and run/report metadata are durable. Canonical WAVs are temporary.

### Future backends

Refactor common execution only after a second concrete model runner demonstrates a stable common subset.

## Acceptance mapping

- Fixed diverse batch -> include-filtered run with frozen IDs.
- Intended checkpoint -> model lock + checkpoint archive + descriptor validation.
- Persistent worker -> one backend across normal sequential requests.
- Corpus-scale deadline -> explicit 3600-second backend request timeout recorded in run evidence.
- Poison recovery -> stop current invocation, resume missing/failed rows next invocation.
- Resume -> exact immutable artifact validation.
- Every eligible item accounted -> run snapshot + HPA-325 item ledger.
- Audio-relative predictions -> source starts at audio zero; HPA-323 references already audio-relative.
- Native + canonical outputs -> prediction v2 unchanged.
- No tuning -> frozen identity/config before inference.
- Count reconciliation -> run + scorer invariants.
- Full-corpus projection -> aggregate pilot RTF × full eligible header-probed duration.
- Pairing without rerun -> immutable prediction paths retained in `run.json`.
- No speculative frameworks -> OaF-only sequential module, one mutable JSON snapshot, no compatibility/security/distribution layers.
