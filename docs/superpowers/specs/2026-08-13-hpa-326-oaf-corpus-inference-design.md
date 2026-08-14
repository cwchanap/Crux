# HPA-326 OaF Full-Mix Corpus Inference Design

**Status:** Proposed  
**Date:** 2026-08-13  
**Linear:** HPA-326 — Run validated OaF Drums on the authoritative full-mix corpus and persist predictions

## Decision summary

Implement HPA-326 as one OaF-specific corpus orchestrator over the contracts already landed by HPA-321, HPA-323, HPA-324, HPA-423, and HPA-325.

The runner adds only the missing execution layer:

1. load the exact HPA-324/HPA-323 lineage;
2. resolve each authoritative full mix from the existing local HPA-321 content cache;
3. materialize the fixed OaF canonical input view temporarily;
4. run one persistent sequential `OafBackend`;
5. persist or resume immutable prediction artifact v2;
6. checkpoint one mutable atomic `run.json`;
7. measure per-song and aggregate runtime/RTF;
8. adapt the persisted state into the existing HPA-325 scorer and reports.

Do **not** add a generic backend/plugin runner, worker pool, retry engine, queue, database, new prediction schema, durable derived-audio cache, R2 repair path, HPA-320 compatibility layer, or corpus-driven tuning.

## Why HPA-326 is next

HPA-326 is the next unblocked high-priority benchmark task. HPA-321, HPA-323, HPA-324, and HPA-423 are complete, and HPA-325 is merged, so the OaF baseline can now be executed and scored without inventing another model or scoring primitive.

Completing HPA-326 unlocks:

- HPA-395 — MuScriptor comparison;
- HPA-328 — OaF separated-input ablation;
- HPA-329 — final comparative benchmark findings.

The repository is missing orchestration, not architecture.

## Keep the selected shape

These decisions remain unchanged:

- one OaF-only runner, no `--backend`;
- prediction artifact v2 stays reference-independent;
- chart/reference/run provenance lives in `run.json` and scorer inputs;
- deterministic `run_id`, no UUID and no mutable latest-baseline registry;
- no R2 fill/repair from inference;
- temporary canonical WAVs only;
- one persistent sequential OaF worker;
- HPA-325 owns matching, aggregation, and report rendering;
- HPA-325 `CohortFailureReason` remains unchanged;
- concurrency waits for measured pilot evidence.

## Existing seams to extend and reuse

### HPA-423 OaF backend

`src/benchmark/backends/oaf.py` already provides:

- `OafBackend`;
- one lazily started `WorkerProcess` retained across requests;
- validated backend/model/upstream/native-output identity;
- checkpoint resolution through `CRUX_OAF_CHECKPOINT_CACHE` or `artifacts/benchmark/model-cache`;
- `transcribe(CanonicalAudio) -> NativePrediction`;
- no heuristic fallback.

HPA-326 creates the backend with the existing `create_backend()` seam. `--cache-dir` is the HPA-321 corpus/audio cache only. Do **not** add `--checkpoint-dir`.

The single-song sequence in `src/cli/benchmark.py::smoke_backend` is the closest existing execution example: load canonical input -> transcribe -> map -> publish -> measure RTF. HPA-326 should reuse the underlying domain helpers and preserve that measurement idiom rather than extracting a generic pipeline from the CLI function.

### Prediction artifact v2

`src/benchmark/prediction_artifact.py` already persists canonical immutable JSONL with:

- source/input audio identity;
- backend descriptor and model identity;
- native OaF event class/bin/confidence/velocity;
- canonical/common mapping and prediction-map version.

HPA-326 reuses:

- `map_oaf_prediction()`;
- `publish_prediction_artifact()`;
- `read_prediction_artifact()`.

Selected-chart and reference-manifest identity stays outside this artifact. A later reference correction can therefore rescore an unchanged prediction.

### Shared canonical manifest reader

`src/benchmark/reference_timing_manifest.py` already contains `_read_canonical_manifest_core()`, which owns:

- canonical JSONL framing;
- exact input SHA-256;
- schema check;
- `render_manifest()` round-trip validation.

Promote it to `read_canonical_manifest_core()` without semantic change. HPA-322 and HPA-323 keep using it, and the new HPA-324 loader uses the same function. Do not create a third manifest reader.

`reference_set_manifest.py` still owns HPA-324-specific validation:

- `_validate_reference_set_row()`;
- unique simfile IDs;
- one shared HPA-323 source timing hash/version;
- narrow `ReferenceSetRowView` fields.

Promote `_read_native_reference_events()` to `read_native_reference_events()` without changing its path/hash/identity checks.

Native reference paths are relative to the HPA-323 timing output root, so HPA-326 derives:

```python
timing_output_root = timing_manifest_path.parent.parent
```

### HPA-321 cache primitives

Resolve source audio by composing the existing public cache API only:

- `CacheIndexStore.get()`;
- `cache_entry_matches_remote()`;
- `validate_cached_body()`;
- `resolve_verified_cache_body()`.

Do not copy `_resolve_or_queue_audio()` from HPA-323 because that helper intentionally queues R2 fill. HPA-326 is offline with respect to corpus data.

### Shared durability primitive

The repository currently has two private implementations of temp-file + file fsync + `os.replace()` + directory fsync:

- `_atomic_replace_json()` in `corpus_manifest.py`;
- `_atomic_replace_bytes()` in `r2_corpus_sync.py`.

Promote the generic bytes form to:

```python
src.benchmark.durability.atomic_replace_bytes(path: Path, content: bytes) -> None
```

with the existing `OSError` failure contract. Then:

- `r2_corpus_sync.py` calls the shared helper directly;
- `corpus_manifest.py` keeps its tiny JSON/domain-error wrapper and delegates the byte replacement;
- HPA-326 uses the shared helper for mutable `run.json`.

This is reuse, not a new persistence framework.

## Inputs and preflight

The CLI consumes:

```text
--manifest          exact HPA-324 benchmark-reference manifest
--timing-manifest   exact HPA-323 timing manifest referenced by HPA-324
--cache-dir         HPA-321 corpus/audio cache root
--output-dir        HPA-326 output root
[--include-simfile-id ID ...]
[--exclude-simfile-id ID ...]
[--resume]
```

Before backend creation, preflight must prove:

1. HPA-324 is canonical and schema-valid through the shared manifest core.
2. Every HPA-324 row shares one source HPA-323 hash/version.
3. That hash/version exactly matches the supplied HPA-323 manifest.
4. Simfile IDs are unique and correspond across manifests.
5. Taxonomy/lane-map versions are the current frozen versions.
6. Every include/exclude ID exists in HPA-324.
7. Include and exclude sets are disjoint.
8. Every HPA-324 **eligible** row can reconstruct its native reference artifact and `ReferenceMappingResult` before inference starts.
9. OaF model config/descriptor are valid.

Unknown scope IDs, overlapping filters, broken eligible reference artifacts, mixed lineage, or invalid descriptor/model identity are fatal exit-2 errors before inference.

The CLI uses the existing `click.IntRange(0, MAX_SIMFILE_ID)` for syntax-level ID bounds; domain preflight owns membership and overlap.

## Authoritative source audio

For a timing-ready row:

1. reconstruct the carried HPA-321 source inventory;
2. find the `RemoteObject` matching `source_audio_key`;
3. load `CacheIndexStore` once per run;
4. accept the carried verified remote when its body validates;
5. otherwise use the exact `(endpoint, bucket, key)` index entry only when `cache_entry_matches_remote()` succeeds;
6. require the resulting digest to equal HPA-323 `source_audio_content_hash`;
7. resolve the content-addressed local body;
8. call existing `inspect_source_audio()` for header-only duration metadata.

Missing, corrupt, identity-changed, or hash-mismatched bodies are item-local `source_audio_unavailable`/`source_audio_decode_failed` failures. The runner does not contact R2.

## Canonical full-mix input view

Define:

```python
OAF_FULL_MIX_INPUT_VIEW_ID = "crux.oaf-full-mix-mono44k1-pcm16/v1"
OAF_CANONICALIZATION_REVISION = "librosa-soxr-hq-mono44k1-soundfile-pcm16/v1"
```

Materialization v1 is explicit:

```python
samples, _ = librosa.load(
    source_path,
    sr=44100,
    mono=True,
    res_type="soxr_hq",
)
soundfile.write(output_path, samples, 44100, format="WAV", subtype="PCM_16")
```

Then `load_materialized_audio()`:

- reads the staged bytes;
- calls existing `parse_canonical_wav()`;
- preserves the authoritative source digest as `source_audio_sha256`;
- computes the staged WAV digest as `input_audio_sha256`;
- uses `config.max_input_audio_frames` from the already loaded OaF model config.

`runtime/oaf_tf1/model.json` currently sets `max_input_audio_frames` to `null`, so full mixes remain unbounded by the current model identity.

The staged WAV lives beneath the worker `input_root` and is removed in `finally`. Do not create a durable canonical-input corpus.

## Frozen inference identity

### Adapter revision

Add:

```python
OAF_ADAPTER_REVISION = "crux.oaf-adapter/v1"
```

Bump only when host-side inference semantics change.

### Model lock

`model_lock_sha256` is the exact SHA-256 of checked-in `runtime/oaf_tf1/model.json`. Record `checkpoint_archive_sha256` separately.

### Inference config

Canonical-hash exactly:

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

There are no corpus-derived thresholds.

## Request deadline and worker shutdown

The HPA-423 `timeout_seconds=30.0` default is a smoke deadline. HPA-326 needs a larger request ceiling, but that value must not become the shutdown deadline.

Define:

```python
OAF_CORPUS_REQUEST_TIMEOUT_SECONDS = 3600.0
OAF_WORKER_CLOSE_TIMEOUT_SECONDS = 30.0
```

The request ceiling remains 3600 seconds for the first pilot. The only real timing evidence available before HPA-326 is the HPA-423 one-second smoke, whose first lazy-backend call took 24.818518 seconds and reported `real_time_factor=24.818518`. That first call includes lazy worker/model startup, so it is not a steady-state full-song estimate, but it still does not support asserting that a 900-second ceiling is already safely above all plausible full-mix inference. The pilot exists to measure steady-state full-song RTF; adjust the request ceiling only from that evidence if needed.

`WorkerProcess` gains a separate close deadline:

```python
WorkerProcess.start(..., timeout_seconds=30.0, close_timeout_seconds=30.0)
```

and stores both values. `request()` continues using `_timeout_seconds`; `close()` uses only `_close_timeout_seconds` for process waits and stderr-thread join. `OafBackend` does not need a corpus-specific close parameter because the worker default stays 30 seconds.

HPA-326 creates the backend as:

```python
create_backend(
    input_root=work_root,
    timeout_seconds=OAF_CORPUS_REQUEST_TIMEOUT_SECONDS,
)
```

using the existing checkpoint default.

## OaF error disposition

`OafBackendError.code` must be classified explicitly because not every backend error poisons the worker.

Define one closed policy beside the runner->scorer failure table:

| `OafBackendError.code` | Runner code | Disposition |
| --- | --- | --- |
| `inference_failed` | `inference_failed` | item-local |
| `invalid_request` | `inference_failed` | item-local |
| `input_path_invalid` | `canonical_input_failed` | item-local |
| `native_event_invalid` | `inference_failed` | item-local |
| `worker_error` | `worker_protocol_failed` | poison |
| `worker_start_failed` | `backend_unavailable` | poison |
| `worker_ready_invalid` | `backend_unavailable` | poison |
| `worker_identity_invalid` | `backend_unavailable` | poison |
| `worker_response_invalid` | `worker_protocol_failed` | poison |
| `backend_closed` | `worker_protocol_failed` | poison |
| `descriptor_invalid` | — | fatal preflight |
| `worker_close_failed` | — | finalization evidence |

Any unknown `OafBackendError.code` defaults to **poison / `worker_protocol_failed`**. Never continue inference on an unclassified backend failure.

Rationale:

- worker error responses with `invalid_request` or `inference_failed` are valid line-protocol responses and do not poison `WorkerProcess`;
- `input_path_invalid` occurs before a request is sent;
- `native_event_invalid` occurs after a complete response is received;
- `WorkerProcessError` paths poison/close the subprocess before `OafBackend` wraps them as `worker_error`.

A poison disposition checkpoints the current run and stops issuing further inference requests in that invocation. Outstanding rows remain missing/failed and are recovered by a later `--resume`. Do not add automatic backend restart.

## Closed runner failure mapping

Keep operational detail in `run.json`, but every item failure maps through one fixed table to the unchanged HPA-325 enum:

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

`prediction_missing` is reserved for adaptation/scoring evidence that run state expects a persisted prediction but the file is absent. A failed publication is deterministically `prediction_artifact_invalid`.

Upstream quarantine and explicit filters use existing non-failure reasons:

```text
quarantined -> reference_quarantined
skipped     -> explicitly_skipped
```

## Prediction path and canonicalization drift

Use source identity, not derived WAV bytes, in the deterministic path:

```text
<output>/predictions/
  <simfile_id>/
    <source_audio_sha256>/
      <backend_descriptor_sha256>/
        <inference_config_sha256>.jsonl
```

This prevents a dependency-level canonicalization drift from looking like an ordinary missing prediction under a new input-hash directory.

However, **resume remains exact-input validated**. Do not make resume decode-free.

Resume flow:

1. resolve the exact source body and source SHA;
2. compute the source-keyed prediction path;
3. materialize the current canonical input and compute `input_audio_sha256`;
4. if `--resume` and the prediction exists, read canonical v2;
5. require exact source ID/hash, input-view ID/hash, descriptor/model, and map identity;
6. exact match -> `resumed`, scorer status `success`, no `transcribe()`;
7. current canonical input hash differs -> explicit `prediction_artifact_invalid`, no overwrite;
8. missing prediction -> run inference;
9. without `--resume`, an existing target -> `prediction_output_conflict`.

If canonicalization intentionally changes, bump `OAF_CANONICALIZATION_REVISION`; the inference-config hash then creates a new prediction path. If a dependency changes bytes without that revision bump, the existing source-keyed artifact is found and the input-hash mismatch fails visibly instead of silently re-inferring the corpus or silently reusing stale bytes.

Resume therefore still pays local decode/resample cost. That is accepted until profiling proves a canonical-input cache is worth its additional persistent state.

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

Derive `run_id` deterministically, for example `oaf-<16 hex>`. A reference/scope change gets a new run directory while an unchanged source/model/config prediction remains reusable.

## Run ledger

Persist one mutable canonical snapshot:

```text
<output>/runs/<run_id>/run.json
```

The run directory is made durable first. Rendering produces canonical bytes and delegates replacement to `durability.atomic_replace_bytes()`.

### Header

Persist:

- schema/run ID;
- exact HPA-324 and HPA-323 hashes/versions;
- backend/model/checkpoint identity;
- adapter revision;
- inference-config hash;
- input-view/canonicalization revision;
- Crux commit;
- normalized include/exclude scope;
- request timeout;
- started/completed timestamps;
- overall status/counts;
- run-level close error evidence when present.

### Item row

Persist:

- simfile ID;
- HPA-324 eligibility reason/warnings;
- selected chart key/hash;
- source key/hash/duration;
- input view/hash when materialized;
- execution disposition `inferred|resumed|skipped|quarantined|failed`;
- detailed runner failure code + bounded detail;
- prediction path/SHA for success;
- inference wall time + RTF for actually inferred rows.

Resume hits are scorer-level successes, not skips.

### Float serialization

`canonical_json_bytes()` rejects Python `float`, so **no float reaches it**.

At the run JSON boundary, every float-derived value uses `quantize_six()` first, including:

- `request_timeout_seconds`;
- source audio duration;
- per-item wall time;
- per-item RTF;
- aggregate RTF;
- measured audio/wall totals when represented as binary floats;
- full eligible duration;
- projected full wall time.

Persisted numeric values are `Decimal` at render time. `strict_json_loads()` also parses JSON decimals as `Decimal`; round-trip tests must not expect Python float values from the raw parsed document.

## Sequential lifecycle

Preflight/reference/cache-duration work occurs before backend creation.

When inference is needed:

```text
create backend once with corpus request timeout
for rows in simfile-ID order:
    resolve source
    materialize canonical input
    resume OR transcribe
    map/publish
    atomically checkpoint run.json
    on poison: stop issuing inference requests
attempt backend.close()
record close failure without discarding persisted predictions
finalize run.json
assemble HPA-325 cohort
write HPA-325 reports
```

A run containing only resume hits need not start the worker because `OafBackend` remains lazy.

`worker_close_failed` is run-level evidence, not an item failure. Catch it, record a bounded close error, finalize the snapshot/reports, and return partial/exit 1 rather than losing the deliverable after predictions are already persisted.

## Non-success HPA-325 item contract

HPA-325 forbids `reference_artifact`, `prediction_artifact`, and `artifact_identity` on every non-success `CohortItem`. Prediction events/coverage must also be absent. Reference coverage must still balance against the supplied `reference_events`.

HPA-326 pins construction before broad inference:

| Run/scorer row | `reference_events` | `coverage` | artifact evidence fields |
| --- | --- | --- | --- |
| success | built by `cohort_item_from_artifacts()` | built by success helper | success helper owns them |
| failed eligible | `reference_to_benchmark_events(mapping.common_events)` | `coverage_from_artifacts(mapping, None)` | all `None` |
| skipped eligible | same as failed eligible | same as failed eligible | all `None` |
| quarantined: `unclassified_reference_lane` or `no_scored_drum_events`, native artifact readable | common projection from `map_reference_events()`; may be empty | `coverage_from_artifacts(mapping, None)` | all `None` |
| quarantined: `upstream_reference_unavailable` or `reference_event_artifact_invalid` | `()` | all-zero reference counts; prediction counts `None` | all `None` |
| run row expected a success artifact but file is now missing | mapped reference events | `coverage_from_artifacts(mapping, None)` | all `None`; failure reason `prediction_missing` |

Do not copy HPA-324 manifest counts blindly into scorer coverage. Use the actual reconstructed mapping when available; use all-zero coverage only when the reference artifact itself is unavailable.

Because every HPA-324 eligible reference is reconstructed during preflight, eligible failed/skipped/success rows cannot discover a broken reference only after corpus inference finishes.

## Runtime measurement and pilot projection

The pilot include set is chosen from pre-inference metadata only and frozen into run identity.

Header-probe all eligible source audio with `inspect_source_audio()` before broad inference to compute eligible duration coverage.

For actually inferred pilot rows:

```text
measured_item_count
measured_audio_duration_sec
measured_wall_time_sec
aggregate_rtf = measured_wall_time_sec / measured_audio_duration_sec
full_eligible_audio_duration_sec
projected_full_wall_time_sec = aggregate_rtf * full_eligible_audio_duration_sec
projection_coverage_count
```

Resume hits without retained historical inference timing do not count as zero-cost inference. If any eligible duration is unavailable, projection is `null`/`None` with coverage count rather than an invented total.

The pilot evidence records `OAF_CORPUS_REQUEST_TIMEOUT_SECONDS` and the separate worker close timeout. Review the measured projection before launching the unfiltered run. Do not model parallel speedup or add concurrency before that review.

## Public outcome and exit semantics

Define the public result in Task 3, before the CLI consumes it:

```python
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

This follows the repository's existing `SyncOutcome` / reference-run convention without adding another generic result framework.

Exit rules:

- `0`: every selected eligible item has a valid prediction, worker close succeeded, and reports were produced;
- `1`: item-local/poison/close failures occurred but trustworthy `run.json` and reports were produced;
- `2`: fatal preflight/setup/run-snapshot failure prevented a trustworthy run/report.

## Scoring and fixed control

After execution, construct `CohortIdentity` from the frozen run identity and the exact HPA-324/HPA-323 lineage.

Use only existing HPA-325 APIs:

```python
result = score_cohort(identity, tuple(items), diagnostics_for=())
reports = write_cohort_reports(result, run_dir / "reports")
```

No broad event diagnostics by default. No matching/alignment/aggregate code is copied into HPA-326.

A full-scope run with zero eligible failures and clean worker close is the fixed OaF full-mix control. Later tasks consume explicit run/report paths; no mutable global baseline pointer is added.

## CLI

Add:

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

The CLI is a thin lazy-import wrapper. It constructs `OafCorpusRunRequest`, calls `run_oaf_corpus()`, canonical-renders a small summary with float fields quantized first, and exits with the domain outcome code.

No `--backend`, `--checkpoint-dir`, request-timeout flag, concurrency flag, or retry flag.

## File-level design

### Create

- `src/benchmark/oaf_corpus_run.py` — HPA-326 identity/preflight/cache/materialization/orchestration/resume/run snapshot/projection/scorer adaptation.
- `tests/benchmark/test_oaf_corpus_run.py` — unit/policy/path/timeout/error/resume tests with fake backend.
- `tests/benchmark/test_oaf_corpus_run_acceptance.py` — multi-song local prediction + poison/resume + report acceptance without real model.

### Modify

- `src/benchmark/reference_timing_manifest.py` — promote shared canonical reader; expose HPA-323 -> HPA-322 source/chart view.
- `src/benchmark/reference_set_manifest.py` — public HPA-324 loader via shared core; promote native reference reader.
- `src/benchmark/input_view.py` — `load_materialized_audio()` preserving source/input hash distinction.
- `src/benchmark/backends/oaf.py` — adapter revision only.
- `src/benchmark/worker_process.py` — separate close deadline from request deadline.
- `src/benchmark/durability.py` — shared `atomic_replace_bytes()`.
- `src/benchmark/corpus_manifest.py` — delegate existing JSON replacement to shared durability helper while preserving `ManifestPublicationError` translation.
- `src/benchmark/r2_corpus_sync.py` — use shared durability helper; delete private duplicate.
- `src/cli/benchmark.py` — thin `run-oaf-corpus` command.
- corresponding focused tests for each modified seam.

### Explicitly unchanged

- `runtime/oaf_tf1/model.py` inference semantics;
- `src/benchmark/prediction_artifact.py` schema v2;
- `src/benchmark/mapping.py` taxonomy/map semantics;
- `src/benchmark/cohort_scoring.py` failure enum/scoring semantics;
- `src/benchmark/reports.py` report schema;
- HPA-321/HPA-323 R2 fill behavior;
- HPA-320 seal/attestation estate.

## Hard gates

### Gate A — lineage and scope

Exact HPA-324 -> HPA-323 hash/version match, valid eligible reference artifacts, known include/exclude IDs, and disjoint filters are proven before backend creation.

### Gate B — authoritative source body

Only locally verified source bytes matching original remote identity and HPA-323 digest may be inferred. No R2 substitution.

### Gate C — timeout separation

HPA-326 passes the 3600s request ceiling; `WorkerProcess.close()` uses a separate 30s close deadline. Unit tests pin both.

### Gate D — backend error disposition

Every known `OafBackendError.code` is classified; unknown codes fail closed as poison. Item-local errors may continue; poison errors stop further inference in that invocation.

### Gate E — immutable prediction reuse

Path is source-hash keyed. Resume still re-materializes and requires exact current canonical input hash before reuse. Drift is an explicit conflict, not a silent miss or stale reuse.

### Gate F — canonical JSON numerics

No Python float reaches `canonical_json_bytes()`. Run and CLI float-derived values cross `quantize_six()`.

### Gate G — scorer population contract

Every non-success row satisfies HPA-325 artifact-nullability and reference-coverage balance before the real pilot.

### Gate H — shared durability

HPA-326 uses the promoted durability helper; no third private atomic writer.

### Gate I — pilot evidence before broad run

Fixed pre-score pilot IDs, per-song/aggregate RTF, eligible duration coverage, projected wall time, request/close deadlines, persistent-worker evidence, and resume evidence are reviewed before unfiltered execution.

## Risks and intentionally deferred work

### Worker request ceiling

The 3600s request ceiling is deliberately generous because full-song steady-state runtime is not measured yet. The separate close deadline prevents that ceiling from multiplying shutdown stalls. Tighten the request ceiling only from pilot evidence.

### Worker poison

Protocol/transport/start/readiness failures stop further inference in the invocation. This can leave outstanding work, but `--resume` is intentionally simpler than a speculative restart engine.

### Canonicalization cost on resume

Resume re-materializes canonical input to prove exact input-byte identity. This is local CPU/I/O and should be much cheaper than OaF inference. Add a derived-input cache only if profiling later shows this cost matters.

### Canonicalization implementation drift

Source-hash pathing makes drift visible: unchanged revision + changed canonical bytes conflicts with the existing artifact. An intentional canonicalizer change requires a revision bump and fresh inference under a new config path.

### Storage

Only prediction JSONL, `run.json`, and HPA-325 reports are durable HPA-326 outputs. Canonical WAVs remain temporary.

### Future backends

Refactor common orchestration only after a second concrete backend proves real duplication.

## Acceptance mapping

- fixed diverse batch -> include-filtered run with IDs frozen in `run.json`;
- intended checkpoint -> model lock + checkpoint archive + descriptor validation;
- persistent worker -> one backend object across healthy sequential requests;
- poison behavior -> closed backend-code policy + stop/`--resume` acceptance test;
- resume -> source-keyed path plus exact current canonical input validation;
- every eligible row accounted -> run ledger + HPA-325 population;
- native + canonical outputs -> unchanged prediction v2;
- no tuning -> config frozen before inference;
- count reconciliation -> run snapshot + scorer invariants;
- full-corpus projection -> measured pilot RTF × eligible header-probed duration;
- pairing without rerun -> immutable prediction paths retained in run state;
- no speculative framework -> OaF-only module, shared existing utilities, no compatibility/security/distribution layers.
