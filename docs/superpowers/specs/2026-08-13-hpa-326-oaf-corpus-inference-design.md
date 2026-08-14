# HPA-326 OaF Full-Mix Corpus Inference Design

**Status:** Proposed  
**Date:** 2026-08-13  
**Linear:** HPA-326 — Run validated OaF Drums on the authoritative full-mix corpus and persist predictions

## Decision summary

Implement HPA-326 as one OaF-specific corpus orchestrator around the seams that are already merged:

- HPA-323 owns authoritative source-audio selection and audio-relative reference timing.
- HPA-324 owns corpus eligibility and the frozen DTX/common taxonomy.
- HPA-423 owns the reusable OaF backend, exact model/checkpoint identity, and one persistent sequential worker.
- HPA-423 prediction artifact v2 owns native + mapped prediction persistence.
- HPA-325 owns cohort scoring and deterministic reports.

HPA-326 should add only the missing execution layer: resolve the authoritative cached full mix, materialize the canonical OaF input view, run one persistent worker sequentially, persist/reuse immutable prediction artifacts, keep an atomic run ledger, measure runtime, then adapt the resulting rows into HPA-325 scoring.

Do **not** introduce a generic multi-backend runner, job queue, worker pool, retry engine, model registry, database, seal/attestation layer, compatibility reader, or new prediction-event schema.

## Why HPA-326 is next

HPA-326 is High priority and its four blockers are complete: HPA-321, HPA-323, HPA-324, and HPA-423. HPA-325 is also now merged, so the full baseline can be scored immediately after inference. Completing HPA-326 unblocks HPA-395 (MuScriptor comparison), HPA-328 (OaF separated-input ablation), and HPA-329 (final benchmark findings).

The repository is therefore missing orchestration rather than another model or scoring primitive.

## Current landed seams

### HPA-423 backend

`src/benchmark/backends/oaf.py` already provides:

- `OafBackend` with one lazily started `WorkerProcess` retained across requests;
- validated backend/model/upstream/native-output descriptor identity;
- a fixed checkpoint cache and read-only input-root mount;
- `transcribe(CanonicalAudio) -> NativePrediction`;
- no heuristic fallback.

The corpus runner must create this backend once, send selected songs sequentially, and close it once. It must not wrap it in another pool/manager abstraction.

### Prediction artifact v2

`src/benchmark/prediction_artifact.py` already persists a canonical immutable JSONL artifact containing:

- source/input audio identities;
- backend descriptor + model identity;
- native OaF event identity, class/bin, confidence, and velocity;
- canonical/common mapping and map version.

HPA-326 must reuse `map_oaf_prediction()`, `publish_prediction_artifact()`, and `read_prediction_artifact()` rather than define a runner-specific event format.

### HPA-323 / HPA-324 reference lineage

HPA-323 rows contain the selected authoritative `source_audio_key`, its content hash, and the immutable reference-event artifact path. HPA-324 carries those fields forward while adding eligibility and mapping accounting.

One subtlety matters for HPA-326: the immutable upstream manifest can still show the selected audio object as not cached even though HPA-323 later filled it. HPA-323 already solves this by consulting `CacheIndexStore`, matching the original remote identity, and rebuilding a verified local cache reference. HPA-326 must use the same public cache primitives. It must not infer a filesystem path from the R2 object key and must not silently fetch a replacement body from R2.

### HPA-325 scoring

`src/benchmark/cohort_scoring.py` already provides:

- `CohortIdentity`;
- success/failure/skipped/quarantined item accounting;
- `cohort_item_from_artifacts()` for successful persisted reference/prediction pairs;
- 30/50/100 ms raw + aligned scoring;
- deterministic aggregate calculations.

`src/benchmark/reports.py` already writes the six report artifacts. HPA-326 should build the cohort items and call these APIs. No scoring logic belongs in the runner.

## Goals

1. Run the validated OaF model over each eligible authoritative full-mix song with one persistent sequential worker.
2. Make interruption/restart safe by reusing only prediction artifacts whose complete input/model/config identity matches.
3. Persist per-song inference outcome, runtime, RTF, input/source provenance, and prediction path in one simple run ledger.
4. Produce a fixed-batch runtime projection before broad inference.
5. Score the completed OaF cohort through HPA-325 without rerunning inference.
6. Leave prediction artifacts reusable by later paired comparisons and input-view ablations.

## Non-goals

- generic multi-model orchestration;
- parallel workers, batching, queues, distributed execution, or autoscaling;
- retry schedules/backoff policies;
- R2 download/fill in the inference runner;
- stem separation;
- threshold tuning, calibration, fine-tuning, or manual prediction correction;
- MIDI as a scoring input;
- changing prediction artifact v2;
- seal evidence, runtime locks, host attestation, HPA-320 compatibility, or release hardening;
- database/Parquet storage;
- a mutable “latest successful baseline” pointer.

## Approaches considered

### A. Loop over the existing single-song smoke command

This is superficially smallest, but it is the wrong execution boundary. A process-level CLI loop would repeatedly construct the backend/model environment, would not own a coherent run identity, and would make exact resume/reconciliation awkward. It also fails HPA-326's explicit persistent-worker requirement.

**Rejected.**

### B. Build a generic corpus pipeline with pluggable backends

A generic runner could eventually serve OaF, MuScriptor, separated inputs, and future models. However HPA-395 and HPA-562 do not yet prove that those models need the same execution contract, and HPA-423 intentionally removed unused backend abstractions. Building plugins, queues, job types, and generalized run schemas now would make HPA-326 larger without increasing baseline correctness.

**Rejected under YAGNI.** Extract a shared runner only after a second implementation demonstrates real duplication.

### C. Add one OaF-specific corpus runner that composes existing contracts

A narrow `oaf_corpus_run.py` can own exactly the missing lifecycle: manifests -> cache -> canonical input -> persistent OaF worker -> prediction v2 -> run ledger -> HPA-325 reports.

It preserves the existing stable boundaries and is easy to delete/refactor if a second backend later demands a different shape.

**Selected.**

## Proposed architecture

```text
HPA-324 benchmark-reference manifest
        |
        | lineage check
        v
HPA-323 reference-timing manifest -----> native reference artifacts
        |
        +---- source_audio_key/hash ----> HPA-321 cache index/body
                                           |
                                           v
                                  canonical full-mix WAV
                                           |
                                           v
                                  one OafBackend instance
                                           |
                                           v
                                   NativePrediction
                                           |
                                 map_oaf_prediction()
                                           |
                                           v
                              prediction artifact v2 (immutable)
                                           |
                         +-----------------+-----------------+
                         |                                   |
                         v                                   v
                  atomic run.json                    HPA-325 CohortItem
                                                             |
                                                             v
                                                    score_cohort()
                                                             |
                                                             v
                                                  write_cohort_reports()
```

The prediction artifact stays reference-independent. Reference/chart provenance belongs to the run row and the scorer inputs, not inside prediction artifact v2. This is important: a future reference correction should allow rescoring an unchanged prediction instead of forcing inference to rerun.

## Input contracts

The command consumes three explicit local inputs:

```text
--manifest          exact HPA-324 benchmark-reference manifest
--timing-manifest   exact HPA-323 reference-timing manifest referenced by HPA-324
--cache-dir         HPA-321 local content-addressed cache root
```

The runner validates before model execution that:

- the HPA-324 manifest is canonical and schema-valid;
- every HPA-324 row shares one `source_reference_timing_manifest_sha256` and version;
- that hash/version matches the supplied HPA-323 timing manifest exactly;
- simfile IDs are unique and correspond across the two manifests;
- taxonomy/lane-map versions remain the frozen current versions;
- the OaF backend descriptor and model config are valid.

Add a small public `load_reference_set_manifest()` in `reference_set_manifest.py`; do not teach the runner to parse canonical JSONL itself.

Promote the already-tested HPA-324 native-reference reader from private to public so HPA-326 can reconstruct `ReferenceMappingResult` from HPA-323 artifacts without duplicating event-path and hash validation.

## Authoritative source-audio resolution

For a timing-ready row:

1. Find the `RemoteObject` whose key equals `source_audio_key` in the carried HPA-321 inventory.
2. Load `CacheIndexStore` once for the run.
3. If the carried remote is not already verified, look up the exact `(source_endpoint_sha256, bucket, key)` cache entry.
4. Require `cache_entry_matches_remote()` so an older immutable manifest never binds to newer bytes at the same key.
5. Require the indexed SHA-256 to equal HPA-323 `source_audio_content_hash`.
6. Resolve/validate the content-addressed cache body.
7. Run `inspect_source_audio()` to obtain decoded duration metadata before inference.

If the body is absent/corrupt/mismatched, record a per-song source-audio failure. HPA-326 does **not** reach R2 to repair it. Re-running HPA-323/cache sync is the correct upstream repair.

This keeps the actual inference run offline with respect to corpus data and prevents an old cohort from silently consuming changed source audio.

## Canonical full-mix input view

The authoritative source is commonly `bgm.ogg`, while HPA-423's host boundary expects a canonical WAV (`44.1 kHz`, mono, `16-bit PCM`). The runner therefore materializes one temporary canonical view per selected song.

Define one semantic input-view constant, for example:

```text
crux.oaf-full-mix-mono44k1-pcm16/v1
```

The materialization policy is fixed for v1:

- decode the authoritative source with existing `librosa`;
- resample to 44,100 Hz;
- mix to mono;
- write WAV PCM16 with existing `soundfile`;
- immediately validate the result through `parse_canonical_wav()` / a small `load_materialized_audio()` sibling in `input_view.py`;
- retain the original source hash as `source_audio_sha256` and the canonical WAV hash as `input_audio_sha256`.

The staged WAV lives below the one `input_root` mounted into the OaF worker. It may be deleted after the song finishes because the source bytes remain in the content-addressed corpus cache and prediction v2 retains both source/input hashes.

Do not persist another input-view manifest or copy the original OGG into the run directory merely to use `load_derived_audio()`; that would duplicate large corpus data for no scoring benefit.

## Frozen model and inference identity

HPA-326 needs identities that are narrow enough to allow legitimate reuse but strong enough to prevent stale output reuse.

### Adapter revision

Add one explicit semantic constant to `backends/oaf.py`:

```text
OAF_ADAPTER_REVISION = "crux.oaf-adapter/v1"
```

Bump it only when host-side inference semantics change. The Crux Git commit is still recorded separately for provenance; using the entire repository commit as prediction identity would invalidate every artifact for unrelated documentation or scoring changes.

### Model lock hash

Use the exact SHA-256 of checked-in `runtime/oaf_tf1/model.json` as `model_lock_sha256`. The run row also records `checkpoint_archive_sha256` explicitly. This binds the model config, checkpoint identity, architecture/native-output metadata, and upstream source revision without reviving the abandoned HPA-320 seal machinery.

### Inference-config hash

Canonical-hash a small closed payload containing only inference-semantic fields:

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

There are no tunable thresholds in this payload because HPA-326 must not tune them against the corpus.

## Run identity

The run is a cohort execution, so its identity additionally includes reference/scope state:

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

Derive `run_id` deterministically from the canonical JSON of that payload (for example `oaf-<16 hex>`). A changed reference manifest or filter creates a new run directory, while the underlying prediction artifact can still be reused when its input/model/config identity is unchanged.

No random UUID and no “latest” pointer are required.

## Prediction paths and immutable reuse

After canonical materialization supplies the exact `input_audio_sha256`, use a deterministic path such as:

```text
<output>/predictions/
  <simfile_id>/
    <input_audio_sha256>/
      <backend_descriptor_sha256>/
        <inference_config_sha256>.jsonl
```

The combined config hash contains the input-view ID, checkpoint/model identity, adapter revision, map version, and canonicalization revision, so the path is keyed by all HPA-326-required inference dimensions without creating an unreadable directory hierarchy.

`publish_prediction_artifact()` remains immutable. The runner never overwrites an existing file.

### `--resume`

Resume is intentionally simple:

1. materialize and hash the current canonical input;
2. compute the expected prediction path;
3. if `--resume` and the file exists, read it through `read_prediction_artifact()`;
4. verify exact source ID/hash, input-view ID/hash, descriptor/model identity, and prediction-map version against current expectations;
5. if all match, mark the row `success` with execution disposition `resumed` and do not call `transcribe()`;
6. if missing, run inference;
7. if present but invalid/mismatched, record an explicit artifact failure and do not overwrite it.

Without `--resume`, an existing target artifact is an explicit output conflict rather than an implicit skip.

Failed or missing rows naturally rerun under `--resume` because they do not have a valid matching artifact. There is no retry counter/backoff framework.

Materializing/rehashing the canonical input during resume is deliberate: it keeps reuse correctness trivial and is much cheaper than OaF inference.

## Persistent sequential worker lifecycle

The runner creates one OaF backend instance for the run with the run work directory as `input_root`.

```text
create backend once
for each selected eligible row in simfile-id order:
    resolve source
    materialize canonical input
    resume or transcribe
    map + publish
    checkpoint run ledger
close backend once
```

`OafBackend` remains lazy, so a run containing only resume hits does not need to start the worker process. A single bad song is recorded and the next song is attempted. A fatal preflight identity failure aborts before inference.

Do not proactively restart the worker, maintain a worker pool, or add retry policy. If real corpus evidence later shows the worker cannot recover after a request-level failure, address that measured failure separately.

## Run ledger

Persist one canonical `run.json` snapshot under:

```text
<output>/runs/<run_id>/run.json
```

Rewrite it atomically after every completed row. A full event log/database is unnecessary for a few hundred songs.

The document contains:

### Run header

- schema and `run_id`;
- exact HPA-324 and HPA-323 manifest hashes/versions;
- backend/model/checkpoint identity;
- adapter revision;
- inference-config hash;
- input-view ID;
- Crux commit;
- normalized include/exclude scope;
- started/completed timestamps;
- run status and counts.

### Per-song row

- `simfile_id`;
- HPA-324 eligibility status/reason/warnings;
- selected chart key/hash when available;
- source audio key/hash and decoded duration;
- input-view ID and canonical input hash when materialized;
- execution disposition: `inferred`, `resumed`, `skipped`, `quarantined`, or `failed`;
- stable failure code + bounded detail string for failures;
- prediction artifact path + SHA-256 for successes;
- measured wall time and RTF for rows actually inferred.

Resume hits remain scorer-level **successes**, not skipped items. `skipped` is reserved for explicit include/exclude scope filtering.

The run ledger supplies the reference/chart and execution provenance that intentionally does not belong in prediction artifact v2.

## Runtime measurement and fixed-batch projection

The exact include set used for the pilot is frozen in the run identity and ledger. The technically diverse IDs must be selected **before inspecting model scores**, using corpus metadata only (for example duration, reference-event density, audio format/source pack), then passed through repeated `--include-simfile-id` flags.

Before any broad inference, the runner should header-probe all eligible locally cached source audio with `inspect_source_audio()`. This provides total eligible audio duration without waveform decode or model execution.

For inferred pilot rows record:

```text
measured_item_count
measured_audio_duration_sec
measured_wall_time_sec
aggregate_rtf = measured_wall_time_sec / measured_audio_duration_sec
full_eligible_audio_duration_sec
projected_full_wall_time_sec = aggregate_rtf * full_eligible_audio_duration_sec
projection_coverage_count
```

Only rows with an actual measured inference are included in the numerator/denominator; resume hits with no retained historical timing do not masquerade as zero-cost inference. If all eligible durations cannot be probed, the projection is `null` and coverage explains why.

This is deliberately a simple first-order estimate. Do not add confidence intervals, memory telemetry, parallel speedup modeling, or queue simulation until the measured sequential runtime justifies it.

The broad run is a human/operator gate: inspect the pilot artifact and projection first, then invoke the same command without include/exclude filters. No workflow engine is needed to enforce that ordering.

## Status and exit semantics

Run rows map into the HPA-325 status contract:

- upstream HPA-324 quarantine -> `quarantined` / `reference_quarantined`;
- explicit filter exclusion -> `skipped` / `explicitly_skipped`;
- valid persisted/reused prediction -> `success`;
- source/input/backend/inference/publish problems -> `failed` mapped to the closest existing HPA-325 failure family.

Keep detailed runner error codes in `run.json`; do not expand HPA-325's grouped failure enum merely to mirror every operational error.

CLI exit convention:

- `0`: every selected eligible item has a valid prediction; upstream quarantines and explicit filter skips are reconciled but do not fail inference;
- `1`: one or more selected eligible items failed while a run ledger/report was still produced;
- `2`: fatal run-level validation/setup/publication failure prevented a trustworthy run.

## Scoring and fixed-control publication

After row execution, reconstruct the HPA-325 cohort from persisted artifacts:

- timing-ready references are read through the promoted HPA-323 reader and `map_reference_events()`;
- successes use `cohort_item_from_artifacts()`;
- failed/skipped/quarantined rows use existing `CohortItem` / `coverage_from_artifacts()` semantics;
- `CohortIdentity.cohort_id` uses `run_id`;
- `reference_manifest_sha256` is the exact HPA-324 manifest hash;
- `reference_timing_version` is the HPA-323 timing manifest version;
- model/backend/map/input-view fields come from the frozen run identity.

Then call:

```text
score_cohort(..., diagnostics_for=())
write_cohort_reports(..., <run-dir>/reports)
```

No broad event diagnostics are generated by default.

A full-scope run with no eligible failures is the fixed OaF full-mix control. Later HPA-395/HPA-562 work receives the explicit `run.json`/report path; HPA-326 does not add a mutable global baseline registry.

## CLI

Add one OaF-specific command:

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

Do not add `--backend`; this ticket is explicitly the frozen OaF control. A generic backend option would imply unsupported semantics for later models.

Emit one small canonical JSON summary on stdout containing run ID/path, counts, measured RTF/projection, report path, status, and exit code. Progress can go to stderr.

## File-level design

### Create

- `src/benchmark/oaf_corpus_run.py` — identity, source resolution, canonical materialization, sequential orchestration, run snapshot, resume, projection, and HPA-325 adaptation.
- `tests/benchmark/test_oaf_corpus_run.py` — pure identity/path/status/resume/orchestration tests with injected fake backend.
- `tests/benchmark/test_oaf_corpus_run_acceptance.py` — multi-song persisted-artifact + resume + score/report acceptance using local fixtures/fakes, no real model.

### Modify

- `src/benchmark/reference_set_manifest.py` — public HPA-324 loader; promote existing native-reference artifact reader without semantic change.
- `tests/benchmark/test_reference_set_manifest.py` — loader/lineage tests.
- `src/benchmark/input_view.py` — add a small `load_materialized_audio()` helper that preserves distinct source/input hashes while reusing canonical WAV validation.
- `tests/benchmark/test_input_view.py` — helper tests.
- `src/benchmark/backends/oaf.py` — add semantic `OAF_ADAPTER_REVISION` constant only.
- `tests/benchmark/test_task_d_contract.py` — pin the adapter revision if this is the existing OaF adapter contract suite.
- `src/cli/benchmark.py` — add `run-oaf-corpus` command as a thin lazy-import wrapper.
- `tests/test_cli_benchmark.py` — CLI argument/summary/exit tests.

### Explicitly unchanged

- `runtime/oaf_tf1/model.py` and worker inference semantics;
- `src/benchmark/mapping.py` taxonomy/map behavior;
- `src/benchmark/prediction_artifact.py` schema v2;
- `src/benchmark/cohort_scoring.py` matcher/aggregate semantics;
- `src/benchmark/reports.py` report schema;
- HPA-321/HPA-323 R2 fill logic;
- old HPA-320 seal/lock artifacts.

## Hard gates

### Gate A — reference lineage

No inference starts unless the supplied HPA-324 rows point to the exact supplied HPA-323 manifest hash/version.

### Gate B — authoritative source body

Every inferred item uses a locally verified cache body whose original remote identity and SHA-256 match HPA-323. Missing/mismatched content is an explicit item failure; no R2 substitution occurs.

### Gate C — persistent worker

A multi-song non-resume test proves one backend object / worker process serves repeated `transcribe()` calls and is closed once.

### Gate D — resume identity

A resume hit must match source hash, canonical input hash, input-view ID, backend descriptor/model identity, prediction map, adapter/config-derived path, and canonical prediction artifact validation. Mismatch never overwrites.

### Gate E — prediction schema reuse

The runner publishes exactly prediction artifact v2 after `map_oaf_prediction()`. Reference/chart/run metadata stays in `run.json`.

### Gate F — population balance

Run rows reconcile to the exact HPA-324 manifest population, and HPA-325 reports retain explicit successful/failed/skipped/quarantined accounting.

### Gate G — no tuning

No threshold, mapping, weight, architecture, or scoring tolerance is derived from the corpus results.

### Gate H — pilot evidence before broad run

Operational acceptance records the fixed include set, batch RTF, full eligible duration coverage, and projected sequential wall time before launching the unfiltered corpus run.

## Risks and intentionally deferred work

### Canonicalization cost on resume

Resume re-materializes the canonical WAV to verify its input hash. That is extra CPU/I/O, but substantially cheaper than OaF inference and avoids a second derived-input cache/schema. If profiling later shows decode dominates resume, add a content-addressed canonical-input cache then — not now.

### Worker failure recovery

The selected design continues after row-local failures using the existing backend behavior. If a real worker error leaves the persistent process unusable for subsequent songs, the corpus evidence will make that visible. Do not add speculative restarts until observed.

### Storage volume

Only prediction JSONL and run/report metadata are durable HPA-326 outputs. Canonical WAVs are temporary, so the full mix is not duplicated on disk.

### Future backends

MuScriptor or separated-input work may reuse parts of this orchestration. Refactor only after the second concrete runner exists and the common subset is obvious.

## Acceptance mapping

- Fixed diverse batch: include-filtered run with frozen IDs in `run.json`.
- Intended checkpoint identity: model lock + checkpoint archive + descriptor validation.
- Persistent worker: one backend instance across sequential items.
- Resume: exact artifact validation, skip inference only on match.
- Every eligible item accounted: run ledger + HPA-325 item ledger.
- Audio-relative predictions: canonical input starts at source-audio zero; HPA-325 reference is already audio-relative.
- Native + canonical outputs: prediction v2 unchanged.
- No tuning: identities/config are frozen before inference.
- Count reconciliation: run schema + scorer population invariants.
- Full-corpus projection: aggregate pilot RTF × full eligible header-probed duration.
- Pairing without rerun: immutable prediction v2 paths retained in `run.json`.
- No speculative frameworks: OaF-only sequential module, one JSON ledger, no compatibility/security/distribution layers.
