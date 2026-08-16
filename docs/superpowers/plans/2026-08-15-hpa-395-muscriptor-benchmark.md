# HPA-395 MuScriptor Full-Mix Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze one official MuScriptor checkpoint, run hard drum-only native-event inference over the same canonical full mixes/references as OaF, score broad and HPA-327 reviewed populations through HPA-325, and publish narrow identity-matched OaF deltas plus runtime/license evidence.

**Architecture:** Keep MuScriptor comparator-only. Reuse the existing `TranscriptionBackend`, prediction-v2, HPA-324 reference, HPA-325 scorer/report, HPA-326 full-mix, and HPA-327 subset seams. Add one optional MuScriptor adapter and one model-specific corpus runner. Extend descriptor/prediction validation with backend-specific invariants rather than weakening OaF rules or creating a new prediction schema. Extract only seams with two real callers: canonical full-mix materialization and reviewed-subset filtering/scoring.

**Tech Stack:** Python 3.12, `muscriptor==0.3.0`, PyTorch through an optional MuScriptor extra, Hugging Face Hub for one-time gated checkpoint freezing, existing librosa/soundfile canonicalization, Click, pytest, existing canonical JSON/JSONL helpers, and HPA-323/HPA-324/HPA-325/HPA-326/HPA-327 artifacts.

## Global constraints

- Freeze MuScriptor `v0.3.0` at release commit `d73147e75e5b9b0c0a79ebe154587db4fd603e0c`.
- Start with the published medium checkpoint. Switching once to small is allowed only for demonstrated hardware impracticality before benchmark scores are inspected.
- Scored runs load a local safetensors file pinned by exact Hugging Face repo revision, SHA-256, byte size, companion config hash/size, package version, device/dtype, and code/weight licenses in `runtime/muscriptor/model.json`.
- Never load scored weights with bare `"medium"`, `"small"`, or an unqualified `hf://...` URL.
- Freeze `use_sampling=False`, `temperature=1.0`, `cfg_coef=1.0`, `instruments=["drums"]`, `batch_size=1`, `no_eos_is_ok=True`, `beam_size=1`, and `prelude_forcing=True`. No scored CLI overrides.
- Persist raw `NoteStartEvent.start_time` plus the native drum MIDI pitch. `NoteEndEvent` and `ProgressEvent` create no hit. Do not invent confidence or velocity and do not round-trip through MIDI.
- Reuse the exact HPA-326 44.1 kHz mono PCM16 `soxr_hq` full-mix bytes so matching OaF/MuScriptor songs share the same `input_view_id` and `input_audio_sha256`. MuScriptor's own 16 kHz preprocessing remains internal and frozen as model semantics.
- Keep `crux.drum-prediction-events/v2`; add backend-specific validation. Existing OaF v2 bytes and validation rules must remain unchanged.
- Keep HPA-323/HPA-324/HPA-327 identities, taxonomy, tolerances, modes, and HPA-325 scoring semantics frozen.
- Persist unsupported MuScriptor pitches as unmapped events rather than dropping them.
- Do not add MuScriptor to `default_backend_registry()` or production API selection.
- CI uses fake MuScriptor models/events. It never downloads gated weights or requires an accelerator.
- Do not add a generic experiment runner, plugin framework, database, queue, worker protocol, second scorer, tuning DSL, training, or stem separation.
- HPA-562 owns generalized multi-model reporting later; HPA-395 only needs OaF-vs-MuScriptor comparison.

---

### Task 1: Freeze the optional package/checkpoint/config contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/benchmark/muscriptor_model.py`
- Create: `scripts/freeze_muscriptor_model.py`
- Create during authenticated setup: `runtime/muscriptor/model.json`
- Create before first inference: `runtime/muscriptor/smoke.json`
- Create: `tests/benchmark/test_muscriptor_model.py`

- [ ] **Step 1: Write strict model-lock tests first**

Test the exact `crux.muscriptor-model/v1` key set, `muscriptor==0.3.0`, release commit, 40-character HF revision, lowercase SHA-256 values, positive sizes, nonempty separate code/weight license fields, exact drum-only decode settings, 16 kHz internal target, five-second chunk duration, and rejection of missing/edited checkpoint/config files.

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/benchmark/test_muscriptor_model.py -q
```

Expected: FAIL because the model-lock module does not exist.

- [ ] **Step 3: Add the optional dependency and lock it**

```toml
[project.optional-dependencies]
muscriptor = ["muscriptor==0.3.0"]
```

```bash
uv lock
```

If this cannot coexist with the current Python 3.12 dependency set, stop and use a tiny dedicated MuScriptor runtime only then. Do not preemptively add a subprocess boundary.

- [ ] **Step 4: Implement the strict local lock/checkpoint verifier**

Use existing `strict_json_loads()`, `require_sha256()`, and regular-file helpers. The verifier must hash the local safetensors and companion `config.json`; a scored run receives only the verified local path.

- [ ] **Step 5: Implement the one-time freeze script**

Resolve `MuScriptor/muscriptor-medium` to an exact HF commit, download `model.safetensors` and `config.json` at that revision, compute hashes/sizes, verify official code/weight license metadata, and write the proposed canonical lock. Expected package code license is MIT; the weight license must be copied from/verified against the official model source rather than guessed.

- [ ] **Step 6: Freeze the five pre-score smoke roles**

`runtime/muscriptor/smoke.json` contains exactly one unique simfile ID and pre-model reason for each role:

```text
short
long
dense
sparse
non_drum_heavy
```

Select them only from reference/audit/source-listening evidence, never model scores.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest tests/benchmark/test_muscriptor_model.py -q
```

Expected: PASS without network access.

---

### Task 2: Promote the canonical full-mix materializer with byte-for-byte OaF characterization

**Files:**
- Modify: `src/benchmark/input_view.py`
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_input_view.py`
- Modify: `tests/benchmark/test_oaf_corpus_run.py`
- Verify unchanged: `tests/benchmark/test_oaf_corpus_run_acceptance.py`

**Interface:**

```python
def materialize_full_mix_audio(
    source: Path | bytes,
    output_path: Path,
    *,
    source_audio_id: str,
    source_audio_sha256: str,
    input_view_id: str,
    max_input_audio_frames: int | None,
) -> CanonicalAudio:
    ...
```

The caller passes the existing persisted OaF full-mix input-view ID; do not rename that historical identity in HPA-395.

- [ ] **Step 1: Add a characterization test for current OaF output bytes**

Use a deterministic source fixture and record canonical WAV bytes/SHA, frame metadata, and input-view identity before extraction.

- [ ] **Step 2: Verify the characterization is green**

```bash
uv run pytest tests/benchmark/test_oaf_corpus_run.py -k materialize -q
```

- [ ] **Step 3: Move only the model-neutral librosa/soundfile conversion**

Preserve exactly:

```python
librosa.load(..., sr=44100, mono=True, res_type="soxr_hq")
soundfile.write(..., 44100, format="WAV", subtype="PCM_16")
```

then call existing `load_materialized_audio()`.

- [ ] **Step 4: Point OaF at the helper and delete the private materializer**

OaF still passes its existing max-frame guard. No compatibility alias.

- [ ] **Step 5: Prove unchanged OaF bytes/acceptance**

```bash
uv run pytest tests/benchmark/test_input_view.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: PASS and byte characterization unchanged.

---

### Task 3: Extend frozen descriptor/prediction-v2/scorer identity for a second backend

**Files:**
- Modify: `src/benchmark/backend_identity.py`
- Modify: `src/benchmark/prediction_artifact.py`
- Modify: `src/benchmark/cohort_scoring.py`
- Modify: `tests/benchmark/test_backend_identity.py`
- Modify: `tests/benchmark/test_prediction_artifact.py`
- Modify: `tests/benchmark/test_cohort_scoring.py`
- Verify unchanged: `tests/benchmark/schema_goldens/crux.drum-prediction-events-v2.jsonl`
- Verify unchanged: `tests/benchmark/schema_goldens/manifest.json`
- Verify unchanged: `tests/benchmark/test_schema_goldens.py`

**MuScriptor descriptor family:**

```text
backend_id = muscriptor-v0.3.0-drums-v1
architecture_id = muscriptor-transformer-v0.3.0
native_output_space_id = muscriptor-drums-midi128-v1
native_metadata_schema_id = muscriptor-note-start-metadata-v1
prediction_schema = crux.drum-prediction-events/v2
```

`model_id` binds the exact frozen checkpoint identity from the model lock. Use an opaque training-data identity that does not claim information the upstream release does not publish precisely.

- [ ] **Step 1: Add red descriptor tests**

`normalize_known_backend_descriptor()` must accept exactly the frozen OaF or MuScriptor family and reject mixed fields, unknown backends, edited release/source identity, and edited frozen model identity.

- [ ] **Step 2: Add red MuScriptor prediction-v2 round-trip tests**

Use an event with:

```python
NativeEvent(
    time_sec=1.25,
    native_class_id="drums:midi_38",
    model_output_bin=None,
    native_midi_note=38,
    native_metadata={"instrument_group": "drums"},
    confidence=None,
    velocity_midi=None,
)
```

Test render -> read -> render byte identity, and reject non-null output-bin/confidence/velocity, missing/out-of-range MIDI pitch, non-drum metadata, and mismatched class ID. Existing OaF nullability/identity tests stay unchanged.

- [ ] **Step 3: Verify red**

```bash
uv run pytest tests/benchmark/test_backend_identity.py tests/benchmark/test_prediction_artifact.py -q
```

- [ ] **Step 4: Dispatch validation by already-validated backend descriptor**

Keep descriptor schema v2 and prediction schema v2. Add explicit OaF and MuScriptor branches; never make OaF-required fields optional globally.

- [ ] **Step 5: Close the zero-hit cohort seam explicitly**

Change the internal artifact-identity helper to receive the expected cohort identity:

```python
_artifact_identity_from_artifacts(
    identity: CohortIdentity,
    simfile_id: str,
    reference: ReferenceMappingResult,
    prediction: PredictionArtifact,
) -> CohortArtifactIdentity
```

For an empty event artifact, use `identity.prediction_map_version` after descriptor/audio validation instead of the current OaF-only fallback. Add a zero-hit MuScriptor success test.

- [ ] **Step 6: Keep the existing schema golden untouched**

The schema-golden registry intentionally has one fixed prediction-v2 entry and the golden is one complete OaF artifact. Do **not** append a MuScriptor record or add a duplicate schema entry. MuScriptor canonical coverage lives in `test_prediction_artifact.py`; the existing golden proves backward OaF byte compatibility.

- [ ] **Step 7: Run focused regression**

```bash
uv run pytest tests/benchmark/test_backend_identity.py tests/benchmark/test_prediction_artifact.py tests/benchmark/test_cohort_scoring.py tests/benchmark/test_cohort_scoring_acceptance.py tests/benchmark/test_schema_goldens.py -q
```

Expected: PASS; OaF golden is unchanged.

---

### Task 4: Add the direct MuScriptor adapter and explicit pitch map

**Files:**
- Create: `src/benchmark/backends/muscriptor.py`
- Modify: `src/benchmark/taxonomy.py`
- Modify: `src/benchmark/mapping.py`
- Create: `tests/benchmark/test_muscriptor_backend.py`
- Modify: `tests/benchmark/test_mapping.py`

- [ ] **Step 1: Write fake upstream event/model fixtures in the backend test**

Model only the public v0.3.0 fields needed by Crux: `NoteStartEvent`, `NoteEndEvent`, `ProgressEvent`, `load_model()`, and `transcribe()`.

- [ ] **Step 2: Add red adapter tests**

Assert one model is loaded from the verified local safetensors path and `transcribe()` is called with exactly the frozen drum-only arguments. Only `NoteStartEvent` creates a `NativeEvent`; pitch/onset are preserved; no confidence/velocity/bin is fabricated. Reject non-drum events, nonfinite/negative onsets, and pitch outside 0..127.

- [ ] **Step 3: Add the explicit MuScriptor prediction map**

Add `MUSCRIPTOR_PREDICTION_MAP_ID = "crux.prediction-map/muscriptor-drums-v1"` plus an explicit MIDI-pitch table into the existing eight detailed/six common classes. Multiple native pitches may collapse to one class, but the native pitch remains in the artifact. Unsupported pitches map to `unmapped`.

Do not introduce a mapping DSL.

- [ ] **Step 4: Implement the lazy optional backend**

Import MuScriptor only inside backend construction. Verify installed package version, model lock, local checkpoint, device/dtype, and descriptor before first song. `close()` only releases local model state; do not add a process solely for teardown.

- [ ] **Step 5: Run focused tests**

```bash
uv run pytest tests/benchmark/test_muscriptor_backend.py tests/benchmark/test_mapping.py tests/benchmark/test_prediction_artifact.py -q
```

Expected: PASS.

---

### Task 5: Add a MuScriptor-specific persisted corpus runner

**Files:**
- Create: `src/benchmark/muscriptor_corpus_run.py`
- Modify only if needed for a true two-caller extraction: `src/benchmark/oaf_corpus_run.py`
- Modify only if needed for a true two-caller extraction: `src/benchmark/corpus_cache.py`
- Create: `tests/benchmark/test_muscriptor_corpus_run.py`
- Create: `tests/benchmark/test_muscriptor_corpus_run_acceptance.py`

**Contracts:**

```text
crux.muscriptor-corpus-run/v1
crux.muscriptor-inference-config/v1
```

- [ ] **Step 1: Add red run-ID/config/snapshot tests**

Bind reference/timing identities, model-lock SHA, checkpoint revision/SHA, descriptor/config hashes, adapter revision, shared input-view ID, decode settings, device/dtype, and include/exclude scope. Complete snapshots require reconciled item dispositions/counts.

- [ ] **Step 2: Add red execution/resume tests with an injected fake backend**

Prove all lineage/scope/model/checkpoint checks happen before backend creation; sources are re-pinned before materialization; the shared full-mix helper is used; the model is lazily created once; immutable prediction artifacts are persisted; item-local inference failures continue; identity/config failures stop; resume reuses exact artifacts and rejects edited evidence.

- [ ] **Step 3: Reuse verified source resolution with the smallest safe extraction**

HPA-326's `_resolve_source_audio()` is model-neutral but currently private. Promote only the resolver/value object if direct reuse would otherwise import OaF/TF1 code into MuScriptor. Prefer an existing cache/input module over a new framework. Keep OaF behavior characterized and unchanged.

- [ ] **Step 4: Implement the concrete runner**

Follow HPA-326's semantic order without extracting a generic base runner:

```text
preflight -> durable snapshot -> source pass -> per-item re-pin/materialize/resume-or-infer/map/publish/checkpoint -> close -> runtime evidence -> HPA-325 broad scoring/reports
```

Use the existing backend-descriptor/config/source-keyed prediction path if possible.

- [ ] **Step 5: Record cheap runtime evidence**

Persist device, dtype, measured wall time/audio duration, aggregate RTF, projected wall time when duration coverage is complete, and peak process RSS where cheaply available. Accelerator-specific peak memory is optional evidence when the selected PyTorch device exposes it directly; do not build telemetry infrastructure.

- [ ] **Step 6: Run MuScriptor + OaF regression tests**

```bash
uv run pytest tests/benchmark/test_muscriptor_corpus_run.py tests/benchmark/test_muscriptor_corpus_run_acceptance.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: PASS.

---

### Task 6: Share reviewed-subset filtering and add MuScriptor CLI

**Files:**
- Modify: `src/benchmark/reviewed_subset.py`
- Modify: `src/cli/benchmark.py`
- Modify: `tests/benchmark/test_reviewed_subset.py`
- Modify: `tests/benchmark/test_reviewed_subset_acceptance.py`
- Modify the existing benchmark CLI test module used by this repository

- [ ] **Step 1: Characterize existing `score_oaf_reviewed_subset()`**

Capture selected IDs, subset cohort ID, diagnostics IDs, exit codes, and report behavior for success plus failed selected rows.

- [ ] **Step 2: Extract only the model-neutral selection/scoring core**

```python
def score_reviewed_subset_cohort(
    parent_identity: CohortIdentity,
    parent_items: tuple[CohortItem, ...],
    subset: LoadedReviewedSubsetManifest,
    *,
    output_dir: Path,
) -> ScoreReviewedSubsetOutcome:
    ...
```

It filters exact HPA-327 membership, derives subset cohort ID from parent cohort + subset manifest hash, and calls HPA-325 with diagnostics for successful selected IDs. OaF keeps its existing run parsing/lineage checks and delegates to this core.

- [ ] **Step 3: Add `score_muscriptor_reviewed_subset()`**

Parse the MuScriptor run, validate the same HPA-323/HPA-324/HPA-327 lineage, reconstruct persisted MuScriptor `CohortItem`s, then delegate to the same core. No inference.

- [ ] **Step 4: Add CLI tests before commands**

Add only:

```text
crux benchmark run-muscriptor-corpus
crux benchmark score-muscriptor-reviewed-subset
```

`run-muscriptor-corpus` mirrors HPA-326 manifest/timing/cache/output/include/exclude/resume arguments. It exposes no model-size, temperature, beam, instrument, device, or dtype flags.

- [ ] **Step 5: Implement lazy-import CLI commands and run focused tests**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py tests/benchmark/test_muscriptor_corpus_run.py -q
```

Also run the benchmark CLI test module. Existing OaF command output remains unchanged.

---

### Task 7: Add narrow identity-matched OaF-vs-MuScriptor deltas

**Files:**
- Create: `src/benchmark/muscriptor_comparison.py`
- Modify: `src/cli/benchmark.py`
- Create: `tests/benchmark/test_muscriptor_comparison.py`
- Modify the existing benchmark CLI test module

- [ ] **Step 1: Add red exact-pairing tests**

A song is pairable only when both persisted items succeed and match:

```text
simfile_id
HPA-324 reference identity
HPA-323 timing identity
source_audio_sha256
input_view_id
input_audio_sha256
```

Mismatches are excluded/rejected explicitly; never pair by song ID alone.

- [ ] **Step 2: Add red delta-output tests**

Rebuild both HPA-325 cohorts from persisted runs, score with the frozen 30/50/100 ms raw/aligned settings, and emit deterministic per-song rows:

```text
simfile_id,tolerance_ms,mode,oaf_f1,muscriptor_f1,delta_f1
```

with `delta_f1 = muscriptor_f1 - oaf_f1`, plus paired count and mean/median delta per tolerance/mode. Optional reviewed-subset filtering uses the exact HPA-327 manifest.

No significance tests, rankings, bootstrap intervals, or generic N-model structures.

- [ ] **Step 3: Add `compare-oaf-muscriptor` CLI**

```text
--oaf-run
--muscriptor-run
--manifest
--timing-manifest
--output-dir
--subset-manifest (optional)
```

It never reruns inference.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest tests/benchmark/test_muscriptor_comparison.py tests/benchmark/test_cohort_scoring.py -q
```

Also run the benchmark CLI test module.

---

### Task 8: Run the frozen smoke gate, broad corpus, reviewed subset, and final regression

**Operational evidence:** local MuScriptor run/predictions/reports and paired reports; large model/artifact bytes remain outside Git.

- [ ] **Step 1: Install/authenticate and verify the exact frozen local checkpoint**

```bash
uv sync --extra muscriptor
uvx hf auth whoami
```

If gating/license acceptance or exact hash verification fails, stop rather than bypassing the official source.

- [ ] **Step 2: Run only the five committed smoke IDs**

Verify drum-only output, native onset/pitch validity, repeatability on at least one repeated song, sane behavior across five-second chunk boundaries on the long song, resume identity, wall time/RTF, and memory evidence.

- [ ] **Step 3: Freeze medium vs small from feasibility only**

If medium is stable/practical, keep it. If not, regenerate the lock once for small and rerun smoke from scratch. Do not inspect F1 to choose.

- [ ] **Step 4: Run the full technically eligible HPA-324 population**

Use the exact immutable HPA-324 manifest, matching HPA-323 timing manifest, and verified HPA-321 cache with `run-muscriptor-corpus`. Do not curate by MuScriptor outcome.

- [ ] **Step 5: Score the accepted HPA-327 subset from persisted predictions**

Use `score-muscriptor-reviewed-subset`; do not rerun inference.

- [ ] **Step 6: Produce broad and reviewed OaF pair deltas**

Run `compare-oaf-muscriptor` against the matching persisted HPA-326 OaF run once for the broad population and once with the HPA-327 subset filter.

- [ ] **Step 7: Verify acceptance evidence**

Confirm 30/50/100 ms event-micro, song-macro, class-macro and per-class results; population failures; native mapped/unmapped pitch coverage; reviewed event diagnostics; paired deltas; runtime/device/dtype/memory; exact model/package/checkpoint hashes; and separate code/weight licenses.

- [ ] **Step 8: Run full regression/static checks**

```bash
uv run pytest tests/benchmark -q
uv run pytest -q
```

Then run the repository's configured lint/type/static commands for changed Python files. Re-run the unchanged OaF prediction schema golden and HPA-325/HPA-326 acceptance tests before claiming completion.

## Stop conditions

Do not publish scored results while any of these is unresolved:

- checkpoint/license/revision/hash is not exact;
- medium/small choice is not frozen from pre-score feasibility evidence;
- hard `instruments=["drums"]` is unavailable in pinned v0.3.0;
- shared canonical full-mix bytes differ from the current OaF output for the same source;
- existing OaF prediction-v2 golden changes;
- HPA-323/HPA-324/HPA-327 lineage does not match;
- a proposed abstraction exists only for hypothetical future models rather than two current callers.

## Expected long-lived additions

Keep new production modules to the concrete ticket needs:

```text
src/benchmark/muscriptor_model.py
src/benchmark/backends/muscriptor.py
src/benchmark/muscriptor_corpus_run.py
src/benchmark/muscriptor_comparison.py
```

Existing modules receive narrow extensions only. If implementation starts creating a generic model registry, experiment object model, shared runner base class, or second report stack, stop and simplify.