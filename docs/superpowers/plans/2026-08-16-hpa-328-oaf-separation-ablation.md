# HPA-328 OaF Drum-Stem Separation Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one fixed HPA-327 reviewed-subset OaF input ablation comparing persisted full mix with Spleeter 4-stem drums and standard HTDemucs drums, then publish paired HPA-325 evidence and an immutable HTDemucs handoff for HPA-396.

**Architecture:** Start from current `main` (`d9a124835a020b630db088112fa733b33baee619` or newer). Keep the experiment concrete: two isolated one-shot separator commands, one HPA-328 run snapshot, the existing OaF transcribe→map→publish path, HPA-325 scoring, shared published-report comparison primitives, and one handoff manifest. Promote only existing machinery that gains a genuine third caller; do not build a generic run framework.

**Tech Stack:** Python 3.12 Crux host, Click, existing canonical JSON/JSONL/CSV helpers, NumPy/librosa/soundfile, isolated separator interpreters, existing OaF TF1 worker, pytest/Ruff/Pylint.

## Global Constraints

- Pilot membership is exactly the supplied HPA-327 `crux.reviewed-reference-subset/v1` population.
- Full-mix OaF is never rerun; full-mix reviewed reports come from `score_oaf_reviewed_subset()`.
- Required separators are exactly Spleeter 4-stem drums and standard `htdemucs` drums.
- Both separator runtimes must pass Task 0 feasibility before implementation begins.
- Spleeter/Demucs stay outside Crux's Python dependency graph.
- Production separator locks are generated and committed only at the final pre-score gate; unit tasks use fixture locks.
- Fixed stem QC: duration delta `<= max(0.5s, source_duration * 0.005)`, RMS `> -80 dBFS`, clipping evidence at absolute peak `>= 0.9999`.
- Fixed separator process timeout: `1800.0` seconds with `5.0` seconds terminate grace. Neither is a CLI flag.
- Retain exact separator-produced WAV bytes through HPA-396; canonical OaF WAVs are temporary.
- Derived input views are `crux.oaf-spleeter4-drums-mono44k1-pcm16/v1` and `crux.oaf-htdemucs-drums-mono44k1-pcm16/v1`.
- `input_view_id` must participate in OaF inference-config identity so prediction paths cannot collide.
- Do not extend `CohortFailureReason`.
- Persisted prediction artifacts retain authoritative source-object identity; scorer-local artifacts may normalize only `source_audio_id` after persisted bytes are validated.
- Pairing always requires the same `source_audio_sha256`; HPA-328 deliberately does **not** require equal `input_audio_sha256` across full mix and derived views.
- Headline paired event-micro metrics use the exact paired-success population and published per-song TP/FP/FN counts.
- Comparison keeps native HPA-328 failure-code histograms in addition to collapsed HPA-325 failure counts.
- Report wall time and retained bytes; do not add dollar-cost rates or cost fields.
- Do not add top-five helped/harmed artifacts; the complete paired CSVs are the inspection surface.
- No full-corpus separator mode, plugin registry, generic experiment runner, generic run snapshot framework, RPC, queue, DB, retry framework, SDR scorer, significance/bootstrap layer, model tuning, or compatibility shim.

---

## File Structure

### Shared seams modified narrowly

- `src/benchmark/cohort_scoring.py` — scorer-local persisted-artifact bridge and model-neutral non-success item constructor.
- `src/benchmark/input_view.py` — derived-stem canonicalization sharing the existing full-mix conversion body.
- `src/benchmark/oaf_corpus_run.py` — retarget scorer bridge/matchers; parameterize `build_inference_config(input_view_id=...)` without changing `build_run_id()`.
- `src/benchmark/muscriptor_corpus_run.py` — retarget scorer bridge and public prediction matchers only.
- `src/benchmark/prediction_artifact.py` — public model-neutral persisted-prediction matchers.
- `src/benchmark/reports.py` — typed reader for HPA-325 `summary.json`, `items.csv`, `per_song.csv`, and `per_class.csv` using the writer's existing field-name constants.
- `src/benchmark/published_comparison.py` — model-neutral pairable-set, delta/join, CSV, canonical summary, and Markdown primitives moved out of MuScriptor comparison.
- `src/benchmark/muscriptor_comparison.py` — consume shared reader/comparison primitives while preserving behavior/output.
- `src/cli/benchmark.py` — two thin HPA-328 commands.

### New HPA-328 modules

- `src/benchmark/separators.py` — separator lock validation, process execution/timeout, exact stem cache, and QC.
- `src/benchmark/separation_pilot.py` — fixed-subset preflight, HPA-328 run snapshot/resume, derived OaF execution, and derived HPA-325 reports.
- `src/benchmark/separation_comparison.py` — two-pair driver, paired event-micro aggregation, FP/FN-per-minute, native failure histograms, runtime/byte totals.
- `src/benchmark/separation_handoff.py` — canonical `crux.oaf-separation-pilot/v1` finalizer/loader.
- `scripts/freeze_separator_runtime.py` — mechanical real-runtime lock generator.

### Fixture/runtime files

Before Task 11, create only:

```text
tests/fixtures/separators/spleeter-model.json
tests/fixtures/separators/htdemucs-model.json
```

Task 11 creates and commits real:

```text
runtime/separators/spleeter/model.json
runtime/separators/htdemucs/model.json
```

---

### Task 0: Prove both isolated separator runtimes are feasible

**Files:** none committed.

**Goal:** Fail early if either required separator cannot be installed and run independently of Crux.

- [ ] **Step 1: Create an isolated Spleeter environment using a supported Python interpreter**

Do not install Spleeter into the Crux Python 3.12 environment. Record the resulting interpreter path as `SPLEETER_PYTHON`.

- [ ] **Step 2: Run one fixed known WAV through official 4-stem separation**

Stage the source as `input.wav` and run:

```bash
"$SPLEETER_PYTHON" -m spleeter separate -p spleeter:4stems -o /tmp/crux-spleeter-smoke input.wav
```

Expected: exit 0 and a decodable drums stem.

- [ ] **Step 3: Create an isolated Demucs environment and smoke standard HTDemucs**

Record interpreter as `DEMUCS_PYTHON`, then run:

```bash
"$DEMUCS_PYTHON" -m demucs -n htdemucs -o /tmp/crux-demucs-smoke input.wav
```

Expected: exit 0 and a decodable `drums.wav`.

- [ ] **Step 4: Gate implementation**

If either smoke fails, stop HPA-328 implementation and revise the two-separator design. Do not silently drop a required separator, substitute a different model, or continue to scored work.

No model hashes or production locks are created in Task 0.

---

### Task 1: Promote the scorer-local persisted-artifact bridge

**Files:**
- Modify: `src/benchmark/cohort_scoring.py`
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `src/benchmark/muscriptor_corpus_run.py`
- Test: `tests/benchmark/test_cohort_scoring.py`
- Test: existing OaF/MuScriptor corpus-run suites

**Interfaces:**

```python
def cohort_item_from_validated_prediction_artifact(
    identity: CohortIdentity,
    simfile_id: str,
    reference: ReferenceMappingResult,
    prediction: PredictionArtifact,
    *,
    warnings: tuple[str, ...] = (),
) -> CohortItem: ...


def cohort_item_without_prediction(
    identity: CohortIdentity,
    simfile_id: str,
    reference: ReferenceMappingResult | None,
    *,
    status: Literal["failed", "skipped", "quarantined"],
    failure_reason: CohortFailureReason,
    warnings: tuple[str, ...] = (),
) -> CohortItem: ...
```

- [ ] **Step 1: Characterize both existing runner adaptations**

```bash
uv run pytest -q \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_muscriptor_corpus_run.py
```

Expected: PASS.

- [ ] **Step 2: Add RED source-ID adaptation test**

Use a persisted artifact whose source object ID is not the simfile ID and assert the returned scorer item uses `simfile_id` while original persisted object identity/SHA/input SHA remain unchanged.

- [ ] **Step 3: Implement only the current duplicated adaptation**

```python
scorer_artifact = prediction
if prediction.prediction.audio.source_audio_id != simfile_id:
    scorer_audio = replace(prediction.prediction.audio, source_audio_id=simfile_id)
    scorer_prediction = replace(prediction.prediction, audio=scorer_audio)
    scorer_artifact = read_prediction_artifact(render_prediction_artifact(scorer_prediction))
return cohort_item_from_artifacts(
    identity,
    simfile_id,
    reference,
    scorer_artifact,
    warnings=warnings,
)
```

- [ ] **Step 4: Promote the duplicated non-success constructor**

Pin legal existing reason/status combinations. Do not change `CohortFailureReason`.

- [ ] **Step 5: Retarget OaF and MuScriptor after their persisted run-row validation**

Remove only duplicated scorer adaptation and non-success constructor bodies. Keep model/run identity checks local.

- [ ] **Step 6: Verify**

```bash
uv run pytest -q \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py \
  tests/benchmark/test_muscriptor_corpus_run.py \
  tests/benchmark/test_muscriptor_corpus_run_acceptance.py
```

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/cohort_scoring.py src/benchmark/oaf_corpus_run.py \
  src/benchmark/muscriptor_corpus_run.py tests/benchmark
git commit -m "refactor: share persisted cohort scoring bridge"
```

---

### Task 2: Support derived OaF input identity and share prediction matchers

**Files:**
- Modify: `src/benchmark/input_view.py`
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `src/benchmark/muscriptor_corpus_run.py`
- Modify: `src/benchmark/prediction_artifact.py`
- Test: input-view, prediction-artifact, OaF/MuScriptor corpus-run suites

**Interfaces:**

```python
def materialize_derived_audio(
    source_audio: ResolvedSourceAudio,
    derived_audio_path: Path,
    output_path: Path,
    *,
    input_root: Path,
    input_view_id: str,
    max_input_audio_frames: int | None,
) -> CanonicalAudio: ...


def build_inference_config(
    config: OafModelConfig,
    descriptor: BackendDescriptor,
    model_lock_sha256: str,
    *,
    input_view_id: str = OAF_FULL_MIX_INPUT_VIEW_ID,
) -> dict[str, str]: ...


def prediction_artifact_matches_audio(
    artifact: PredictionArtifact,
    *,
    source_audio_id: str,
    source_audio_sha256: str,
    audio: CanonicalAudio,
    descriptor: BackendDescriptor,
    prediction_map_version: str,
) -> bool: ...


def prediction_artifact_matches_run_row(
    artifact: PredictionArtifact,
    row: Mapping[str, object],
    *,
    expected_input_view_id: str,
) -> bool: ...
```

- [ ] **Step 1: Characterize current full-mix configuration and resume behavior**

Run current OaF/MuScriptor prediction resume tests before edits.

- [ ] **Step 2: Add derived inference-config test**

Assert only `input_view_id` differs from full mix and `inference_config_sha256()` therefore differs.

- [ ] **Step 3: Factor the existing canonicalization body**

Keep `materialize_full_mix_audio()` byte-identical. `materialize_derived_audio()` reads the retained stem but takes authoritative source identity from `ResolvedSourceAudio`.

- [ ] **Step 4: Delete the redundant full-mix-only matcher condition**

The shared audio matcher requires:

```python
prediction.audio.input_view_id == audio.input_view_id
prediction.audio.input_audio_sha256 == audio.input_audio_sha256
```

It must not separately require `OAF_FULL_MIX_INPUT_VIEW_ID`.

- [ ] **Step 5: Promote both artifact matchers to `prediction_artifact.py`**

The run-row matcher takes `expected_input_view_id` explicitly so OaF, MuScriptor, and HPA-328 do not duplicate/hardcode view policy.

- [ ] **Step 6: Retarget existing runners and verify behavior preservation**

```bash
uv run pytest -q \
  tests/benchmark/test_input_view.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py \
  tests/benchmark/test_muscriptor_corpus_run.py \
  tests/benchmark/test_muscriptor_corpus_run_acceptance.py
```

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/input_view.py src/benchmark/prediction_artifact.py \
  src/benchmark/oaf_corpus_run.py src/benchmark/muscriptor_corpus_run.py tests/benchmark
git commit -m "refactor: share prediction identity matching"
```

---

### Task 3: Define separator lock schema, fixture locks, and freeze script

**Files:**
- Create: `src/benchmark/separators.py`
- Create: `scripts/freeze_separator_runtime.py`
- Create: `tests/fixtures/separators/spleeter-model.json`
- Create: `tests/fixtures/separators/htdemucs-model.json`
- Create: `tests/benchmark/test_separators.py`

**Interfaces:**

```python
SEPARATOR_LOCK_SCHEMA = "crux.separator-lock/v1"
SPLEETER_SEPARATOR_ID = "spleeter4-drums-v1"
HTDEMUCS_SEPARATOR_ID = "htdemucs-drums-v1"

@dataclass(frozen=True)
class SeparatorModelFile:
    name: str
    sha256: str

@dataclass(frozen=True)
class SeparatorLock:
    separator_id: str
    repository_url: str
    repository_revision: str
    package_name: str
    package_version: str
    model_id: str
    model_files: tuple[SeparatorModelFile, ...]
    code_license: str
    model_license: str
    argv: tuple[str, ...]
    expected_drum_stem_relative_path: str
    output_container: str
    sha256: str
```

- [ ] **Step 1: RED lock parser tests**

Reject unknown/missing keys, noncanonical JSON, malformed hashes, duplicate/absolute model names, unsupported separator IDs, and command/model mismatch.

- [ ] **Step 2: Implement loader using existing canonical JSON/SHA helpers**

- [ ] **Step 3: Add synthetic canonical fixture locks**

Fixtures are test data only and must not be copied into runtime lock paths.

- [ ] **Step 4: Implement mechanical freeze script**

It receives an already-working isolated interpreter and explicit real model files, records package version, hashes files, writes canonical JSON, and round-trips through `load_separator_lock()`.

It does not install packages, run benchmark inference, or inspect scores.

- [ ] **Step 5: Verify/commit**

```bash
uv run pytest -q tests/benchmark/test_separators.py
uv run ruff check src/benchmark/separators.py scripts/freeze_separator_runtime.py \
  tests/benchmark/test_separators.py
git add src/benchmark/separators.py scripts/freeze_separator_runtime.py \
  tests/fixtures/separators tests/benchmark/test_separators.py
git commit -m "feat: define separator lock contract"
```

---

### Task 4: Execute separators with timeout, cache exact stems, and apply QC

**Files:**
- Modify: `src/benchmark/separators.py`
- Modify: `tests/benchmark/test_separators.py`

**Interfaces:**

```python
SEPARATOR_TIMEOUT_SECONDS = 1800.0
SEPARATOR_TERMINATE_GRACE_SECONDS = 5.0
STEM_NEAR_SILENT_DBFS = -80.0
STEM_CLIP_ABS = 0.9999
STEM_MAX_DURATION_DELTA_SECONDS = 0.5
STEM_MAX_DURATION_DELTA_RATIO = 0.005

@dataclass(frozen=True)
class StemQc: ...

@dataclass(frozen=True)
class SeparatedStem: ...

def run_spleeter_drums(...) -> SeparatedStem: ...
def run_htdemucs_drums(...) -> SeparatedStem: ...
```

- [ ] **Step 1: RED process/cache/QC tests**

Cover nonzero exit, missing output, decode failure, near silence, duration mismatch, >2 channels, nonfinite samples, warnings, cache hit, immutable conflict, and timeout.

The timeout test injects `subprocess.TimeoutExpired`; it does not actually sleep.

- [ ] **Step 2: Stage input under deterministic `input.wav`**

Run each separator in a private temp directory so output discovery is stable.

- [ ] **Step 3: Implement one-shot process timeout**

Use `subprocess.Popen(..., start_new_session=True)` and `communicate(timeout=SEPARATOR_TIMEOUT_SECONDS)`. On timeout terminate the process group, wait `SEPARATOR_TERMINATE_GRACE_SECONDS`, then kill if needed. Return/raise a stable HPA-328 detail code `separator_timeout`.

- [ ] **Step 4: Publish exact native stem bytes**

Use:

```text
derived/stems/<separator-id>/<source-sha>/<lock-sha>/drums.wav
```

and existing `publish_immutable_file()`.

- [ ] **Step 5: Implement fixed QC**

Use `soundfile.read(..., always_2d=True, dtype="float32")`; enforce finite samples/channel/duration/RMS rules and record clipping evidence.

- [ ] **Step 6: Prove cache hit bypasses subprocess but reruns QC**

- [ ] **Step 7: Verify/commit**

```bash
uv run pytest -q tests/benchmark/test_separators.py
git add src/benchmark/separators.py tests/benchmark/test_separators.py
git commit -m "feat: cache and validate separated drum stems"
```

---

### Task 5: Define HPA-328 preflight, run identity, snapshot, and full-mix control

**Files:**
- Create: `src/benchmark/separation_pilot.py`
- Create: `tests/benchmark/test_separation_pilot.py`

**Interfaces:**

```python
SEPARATION_RUN_SCHEMA = "crux.oaf-separation-run/v1"
SPLEETER_INPUT_VIEW_ID = "crux.oaf-spleeter4-drums-mono44k1-pcm16/v1"
HTDEMUCS_INPUT_VIEW_ID = "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1"

@dataclass(frozen=True)
class OafSeparationPilotRequest: ...

@dataclass(frozen=True)
class OafSeparationPilotOutcome: ...
```

- [ ] **Step 1: RED fatal-preflight tests**

Neither separator nor backend is touched for subset/reference/timing mismatch, parent run mismatch, missing subset member, bad lock, mixed OaF identity, or output alias.

- [ ] **Step 2: Exact-membership test**

The run contains every HPA-327 member exactly once, sorted by simfile ID. There is no sample/seed/count configuration.

- [ ] **Step 3: Deterministic run identity**

Hash the exact subset/reference/timing/parent-OaF/lock/view/scoring/Crux-commit identity. Keep this local to HPA-328.

- [ ] **Step 4: Implement HPA-328-specific canonical snapshot**

Use existing `canonical_json_bytes()`, `quantize_six()`, `atomic_replace_bytes()`, `format_manifest_timestamp()`, and SHA helpers. Do not create `run_snapshot_common.py` or copy include/exclude helpers from the flat corpus runners.

Rows retain native per-view status/failure code plus source/stem/input/prediction/runtime evidence.

- [ ] **Step 5: Produce full-mix reports through the public wrapper**

```python
score_oaf_reviewed_subset(
    ScoreReviewedSubsetRequest(
        reference_manifest_path=request.reference_manifest_path,
        timing_manifest_path=request.timing_manifest_path,
        subset_manifest_path=request.subset_manifest_path,
        run_path=request.oaf_run_path,
        output_dir=run_dir / "views" / "full_mix" / "reports",
    )
)
```

Parse parent `run.json` separately only for run-row identity evidence.

- [ ] **Step 6: Verify/commit**

```bash
uv run pytest -q tests/benchmark/test_separation_pilot.py
git add src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot.py
git commit -m "feat: define separation pilot run"
```

---

### Task 6: Execute and resume both derived views through frozen OaF

**Files:**
- Modify: `src/benchmark/separation_pilot.py`
- Modify: `tests/benchmark/test_separation_pilot.py`
- Create: `tests/benchmark/test_separation_pilot_acceptance.py`

- [ ] **Step 1: Prove full mix is never inferred**

Fake backend records calls; only the two derived view IDs may appear.

- [ ] **Step 2: RED resume matrix**

Cover exact stem+prediction hit, valid stem+missing prediction, wrong stem hash, run-row identity mismatch, current-audio identity mismatch, and immutable prediction conflict.

- [ ] **Step 3: Resolve all authoritative sources before expensive work**

Use `resolve_source_audio(..., load_body=False)`. Failure is fatal for fixed membership.

- [ ] **Step 4: Build view-specific OaF inference configs**

All fields except `input_view_id` equal the full-mix config.

- [ ] **Step 5: Reuse public prediction matchers**

For resume:

```python
assert prediction_artifact_matches_run_row(
    artifact,
    prior_row,
    expected_input_view_id=audio.input_view_id,
)
assert prediction_artifact_matches_audio(
    artifact,
    source_audio_id=source.source_audio_id,
    source_audio_sha256=source.source_audio_sha256,
    audio=audio,
    descriptor=descriptor,
    prediction_map_version=OAF_PREDICTION_MAP_ID,
)
```

No inline third matcher.

- [ ] **Step 6: Reuse exact OaF write path for misses**

```python
native = backend.transcribe(audio)
mapped, _ = map_oaf_prediction(native)
published = publish_prediction_artifact(prediction_target, mapped)
```

Use `classify_oaf_backend_error()` for backend errors.

- [ ] **Step 7: Checkpoint each durable boundary**

Stem/QC, input identity, prediction identity/runtime, timeout/failure status. A separator timeout is `separation_failed` with native code `separator_timeout`; continue to the other view.

- [ ] **Step 8: Verify/commit**

```bash
uv run pytest -q tests/benchmark/test_separation_pilot.py \
  tests/benchmark/test_separation_pilot_acceptance.py
git add src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot.py \
  tests/benchmark/test_separation_pilot_acceptance.py
git commit -m "feat: run OaF on separated drum views"
```

---

### Task 7: Score derived cohorts through HPA-325 with a frozen reason collapse

**Files:**
- Modify: `src/benchmark/separation_pilot.py`
- Modify: `tests/benchmark/test_separation_pilot_acceptance.py`

**Frozen mapping:**

```python
SEPARATION_FAILURE_TO_COHORT_REASON = {
    "separation_failed": "inference_failed",
    "stem_invalid": "inference_failed",
    "canonical_input_failed": "inference_failed",
    "inference_failed": "inference_failed",
    "prediction_invalid": "prediction_artifact_invalid",
    "prediction_output_conflict": "prediction_artifact_invalid",
    "prediction_publish_failed": "prediction_artifact_invalid",
}
```

- [ ] **Step 1: Test exact mapping equality**

Pin the dict so no third consumer grows a new HPA-325 reason family.

- [ ] **Step 2: Mixed-success population test**

Every derived `items.csv` contains all reviewed members even when one separator/OaF view fails.

- [ ] **Step 3: Success rows use scorer-local bridge only after persisted artifact validation**

- [ ] **Step 4: Non-success rows use shared constructor and frozen mapping**

Native HPA-328 details stay in `run.json`.

- [ ] **Step 5: Score/write only through HPA-325**

```python
result = score_cohort(identity, items, diagnostics_for=successful_ids)
write_cohort_reports(result, reports_path)
```

- [ ] **Step 6: Verify/commit**

```bash
uv run pytest -q tests/benchmark/test_separation_pilot_acceptance.py \
  tests/benchmark/test_cohort_scoring.py
git add src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py
git commit -m "feat: score separated OaF cohorts"
```

---

### Task 8A: Share HPA-325 report reading and model-neutral comparison primitives

**Files:**
- Modify: `src/benchmark/reports.py`
- Create: `src/benchmark/published_comparison.py`
- Modify: `src/benchmark/muscriptor_comparison.py`
- Test: existing report and MuScriptor comparison suites

**Report reader:**

```python
@dataclass(frozen=True)
class PublishedCohortReports:
    identity: CohortIdentity
    population: PopulationSummary
    aggregates: tuple[PublishedAggregate, ...]
    items: tuple[PublishedItemRow, ...]
    songs: tuple[PublishedSongRow, ...]
    classes: tuple[PublishedClassRow, ...]


def read_cohort_reports(
    report_dir: Path,
    *,
    expected_identity: CohortIdentity,
) -> PublishedCohortReports: ...
```

**Pairing primitives:**

```python
def pairable_success_ids(
    left: PublishedRunEvidence,
    right: PublishedRunEvidence,
    selected_ids: set[str] | None,
    *,
    require_identical_input_hash: bool = True,
) -> tuple[set[str], dict[str, int]]: ...
```

Also move the existing model-neutral metric-delta, paired song/class join, deterministic CSV, canonical summary, and Markdown rendering functions from `muscriptor_comparison.py` rather than rewriting them.

- [ ] **Step 1: Characterize MuScriptor comparison before refactor**

```bash
uv run pytest -q tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py
```

- [ ] **Step 2: RED `read_cohort_reports()` tests**

Reject bad schemas/numbers/identities/duplicates and score rows for non-success items. Parse independent `summary.json` event-micro aggregates.

- [ ] **Step 3: Move duplicated field schemas/reader logic beside writer**

Use `reports.py`'s existing `_ITEM_FIELDNAMES`, `_PER_SONG_FIELDNAMES`, `_PER_CLASS_FIELDNAMES` as the single source.

- [ ] **Step 4: Move model-neutral comparison primitives**

Parameterize only label names and `require_identical_input_hash`; do not move backend-family or subset-policy logic.

- [ ] **Step 5: Retarget MuScriptor with default identical-input policy**

Require existing output names/shape and characterization tests to remain unchanged.

- [ ] **Step 6: Verify/commit**

```bash
uv run pytest -q tests/benchmark/test_reports.py \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py
git add src/benchmark/reports.py src/benchmark/published_comparison.py \
  src/benchmark/muscriptor_comparison.py tests/benchmark
git commit -m "refactor: share published benchmark comparison"
```

---

### Task 8B: Publish only HPA-328-specific comparison evidence

**Files:**
- Create: `src/benchmark/separation_comparison.py`
- Create: `tests/benchmark/test_separation_comparison.py`
- Modify: `src/benchmark/separation_pilot.py`

**Outputs:**

```text
comparison/spleeter/paired_per_song.csv
comparison/spleeter/paired_per_class.csv
comparison/htdemucs/paired_per_song.csv
comparison/htdemucs/paired_per_class.csv
comparison/summary.json
comparison/summary.md
```

- [ ] **Step 1: Test HPA-328 pairing policy**

Same `source_audio_sha256` is mandatory. Different `input_audio_sha256` is expected and allowed. Pass `require_identical_input_hash=False`.

- [ ] **Step 2: Reuse shared paired row builders/writers**

No HPA-328 copy of metric delta, song/class join, CSV, or Markdown machinery.

- [ ] **Step 3: Add genuinely new paired event-micro aggregation**

For each tolerance/mode and exact pairable song set, sum published per-song `tp/fp/fn`, then derive precision/recall/F1 from those sums. Never subtract independent cohort summary F1 values.

- [ ] **Step 4: Add FP/FN per minute**

Use positive finite authoritative source duration from HPA-328 run rows:

```python
minutes = Decimal(str(source_duration_sec)) / Decimal("60")
fp_per_min = Decimal(fp) / minutes
fn_per_min = Decimal(fn) / minutes
```

- [ ] **Step 5: Add native failure-code histograms**

For each separator view, count native HPA-328 codes from `run.json` such as `separator_timeout`, `stem_invalid`, `canonical_input_failed`, `inference_failed`, and prediction failures. Publish alongside collapsed HPA-325 population and paired-set size.

- [ ] **Step 6: Add measured resource totals only**

Publish separator/OaF wall-time seconds and retained stem/prediction/report byte totals. No dollar rates or dollar fields.

- [ ] **Step 7: Do not build top-five lists**

Sort paired CSVs deterministically by score dimensions/simfile ID; complete <=30-song data is the inspection surface.

- [ ] **Step 8: Verify/commit**

```bash
uv run pytest -q tests/benchmark/test_separation_comparison.py \
  tests/benchmark/test_separation_pilot_acceptance.py
git add src/benchmark/separation_comparison.py src/benchmark/separation_pilot.py \
  tests/benchmark/test_separation_comparison.py
git commit -m "feat: compare OaF separation views"
```

---

### Task 9: Publish immutable HPA-396 handoff and operator decision

**Files:**
- Create: `src/benchmark/separation_handoff.py`
- Create: `tests/benchmark/test_separation_handoff.py`
- Create: `tests/benchmark/schema_goldens/oaf-separation-pilot-v1.jsonl`

**Interfaces:**

```python
SEPARATION_PILOT_SCHEMA = "crux.oaf-separation-pilot/v1"
SeparationDecision = Literal[
    "keep_full_mix",
    "use_spleeter",
    "use_htdemucs",
    "gather_more_evidence",
    "prioritize_another_model",
]

@dataclass(frozen=True)
class FinalizeSeparationPilotRequest:
    run_path: Path
    subset_manifest_path: Path
    output_manifest: Path
    decision: SeparationDecision
    rationale: str
```

- [ ] **Step 1: RED schema golden/loader tests**

Every HPA-327 member must appear; exact status/nullability rules are enforced.

- [ ] **Step 2: Bind exact row identity**

Include source-row/subset/reference/timing/OaF identities, per-view stem/input/prediction identities, comparison artifact hashes, decision/rationale, and Crux commit. Do not include cost rates/dollar fields.

- [ ] **Step 3: Re-read/hash retained artifacts before publication**

Edited/missing HTDemucs stem/prediction evidence prevents successful handoff publication.

- [ ] **Step 4: Reuse manifest rails**

Use `render_manifest()` / `publish_manifest()`; no new header protocol.

- [ ] **Step 5: HPA-396 consumer test**

Successful HTDemucs row provides `simfile_id`, `source_row_sha256`, source SHA, stem path/SHA, canonical input SHA, and OaF prediction SHA without mutable run state.

- [ ] **Step 6: Verify/commit**

```bash
uv run pytest -q tests/benchmark/test_separation_handoff.py
git add src/benchmark/separation_handoff.py tests/benchmark/test_separation_handoff.py \
  tests/benchmark/schema_goldens/oaf-separation-pilot-v1.jsonl
git commit -m "feat: publish separation pilot handoff"
```

---

### Task 10: Add two thin CLI commands

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Modify if required: `tests/test_cli_benchmark_coverage.py`

- [ ] **Step 1: Run-command signature test**

Only:

```text
--manifest
--timing-manifest
--subset-manifest
--oaf-run
--cache-dir
--output-dir
--spleeter-python
--demucs-python
--resume
```

- [ ] **Step 2: Finalize-command signature test**

Only:

```text
--run
--subset-manifest
--output-manifest
--decision
--rationale
```

Assert no model/QC/tolerance/full-corpus/third-separator/cost-rate flags.

- [ ] **Step 3: Lazy imports + stable exit codes**

Run: 0 complete, 1 partial, 2 fatal. Finalize: 0 only after immutable publication.

- [ ] **Step 4: Verify/commit**

```bash
uv run pytest -q tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py
git add src/cli/benchmark.py tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py
git commit -m "feat: add separation pilot commands"
```

---

### Task 11: Freeze real runtimes, run the fixed pilot, and close verification

**Files:**
- Create and commit before scoring:
  - `runtime/separators/spleeter/model.json`
  - `runtime/separators/htdemucs/model.json`
- Generated benchmark outputs remain outside git.

- [ ] **Step 1: Re-run both Task 0 smokes in the exact interpreters**

Do not proceed if either required runtime regressed.

- [ ] **Step 2: Generate real separator locks from exact model files**

Use `scripts/freeze_separator_runtime.py`; round-trip through production loader.

- [ ] **Step 3: Commit real locks before inspecting scored output**

```bash
git add runtime/separators/spleeter/model.json runtime/separators/htdemucs/model.json
git commit -m "chore: freeze separation benchmark runtimes"
git rev-parse HEAD
sha256sum runtime/separators/spleeter/model.json runtime/separators/htdemucs/model.json
```

- [ ] **Step 4: Run exact HPA-327 subset**

```bash
uv run crux benchmark run-oaf-separation-pilot \
  --manifest "$REFERENCE_MANIFEST" \
  --timing-manifest "$TIMING_MANIFEST" \
  --subset-manifest "$REVIEWED_SUBSET_MANIFEST" \
  --oaf-run "$OAF_RUN" \
  --cache-dir "$CORPUS_CACHE" \
  --output-dir "$SEPARATION_OUTPUT" \
  --spleeter-python "$SPLEETER_PYTHON" \
  --demucs-python "$DEMUCS_PYTHON"
```

If interrupted, rerun the exact command with `--resume`; do not alter locks/config after partial scores are visible.

- [ ] **Step 5: Review required evidence**

Require all reviewed members in populations; exact stems/predictions; both paired-success sizes; 30/50/100-ms song/class deltas; paired event-micro counts/F1; FP/FN/minute; native failure histograms; separator/OaF wall time; retained bytes.

- [ ] **Step 6: Finalize one closed decision**

```bash
uv run crux benchmark finalize-oaf-separation-pilot \
  --run "$SEPARATION_RUN" \
  --subset-manifest "$REVIEWED_SUBSET_MANIFEST" \
  --output-manifest "$SEPARATION_HANDOFF" \
  --decision "$DECISION" \
  --rationale "$RATIONALE"
```

- [ ] **Step 7: Repository-wide verification**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests scripts
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check
git status --short
```

- [ ] **Step 8: Scope audit**

```bash
git diff --stat origin/main...HEAD
```

Confirm no generic run framework, plugin system, full-corpus separator path, bundled audio/stem/result artifacts, dollar-cost configuration, or duplicate comparison engine.

---

## Plan Self-Review

**Reuse:** Task 1 shares the scorer bridge; Task 2 shares prediction matching instead of forking it; Task 8A moves the already-working HPA-325/MuScriptor comparison machinery rather than writing a second engine.

**YAGNI:** The proposed wholesale `run_snapshot_common.py` extraction is intentionally not included. HPA-328 does not consume several flat-run helpers, so its nested view snapshot stays local while public low-policy primitives are reused directly.

**Risk order:** Task 0 proves both required external runtimes before implementation. Task 11 freezes exact identities only after the implementation is ready and immediately before scored work.

**Scoring integrity:** HPA-325 reason vocabulary stays closed; independent cohort summaries remain independent; headline paired metrics aggregate only published per-song counts over exact success intersections.

**Operational evidence:** Native HPA-328 failure histograms remain visible, and measured wall time/bytes satisfy cost-awareness without introducing speculative rate configuration.