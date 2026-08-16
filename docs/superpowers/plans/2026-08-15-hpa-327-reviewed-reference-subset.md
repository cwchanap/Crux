# HPA-327 Reviewed Reference Subset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze and manually audit a deterministic 20–30-song reference subset before score-informed selection, publish it as a canonical manifest, and rescore the existing persisted OaF cohort on that exact membership with HPA-325 event diagnostics.

**Architecture:** Keep HPA-327 as one model-independent module for candidate selection, review CSV handling, subset publication/loading, and subset-score orchestration. First land two no-behavior-change extractions from HPA-326: shared reference/timing preflight moves to `reference_set_manifest.py`, then persisted cohort reconstruction becomes a public helper in `oaf_corpus_run.py`. New scoring remains a thin filter over persisted artifacts and delegates to `score_cohort()` plus `write_cohort_reports()`.

**Tech Stack:** Python 3.13, stdlib `csv`/`dataclasses`/`hashlib`, Click, pytest, existing Crux canonical JSON/JSONL helpers, HPA-323/HPA-324 reference artifacts, HPA-325 scorer/report writer, HPA-326 persisted OaF artifacts.

## Global Constraints

- `REVIEW_POLICY_VERSION = "hpa327-v1"`.
- `REVIEW_TARGET_COUNT = 30`, `REVIEW_MIN_COUNT = 20`, `REVIEW_MAX_COUNT = 30`.
- `REVIEW_SELECTION_SEED = "crux-hpa327-v1"`; no seed/count CLI flags.
- Candidate preparation accepts only HPA-323/HPA-324 inputs plus optional prior HPA-327 review evidence. It has no run/prediction/report/score/model parameter.
- Assign thirds with `min(2, (i * 3) // n)` after `(feature_value, simfile_id)` sorting.
- Order strata and members by seeded SHA-256, never lexical band labels.
- Reconcile each eligible mapping with HPA-324 `common_scored_event_count` before selection.
- Read `timing_warnings` from the matching HPA-323 row.
- Use `selects_real_or_full_chart`; do not call `real.dtx` / `full.dtx` nonstandard.
- CSV generated cells are reviewer hints/evidence, not authority. Finalization trusts current `simfile_id` membership plus the 12 manual fields and re-derives everything else.
- Continuation prepare/finalize receives the same prior ledger. Unchanged valid includes carry forward, unchanged excludes remain consumed, and replacements come only from the unused deterministic stream.
- Publish `crux.reviewed-reference-subset/v1` with source identities, six selection features, both band labels, `review_ledger_sha256`, and optional `prior_review_ledger_sha256`.
- `load_reviewed_subset_manifest()` uses `read_canonical_manifest_core()`; the schema is registered in the existing golden registry.
- HPA-325 owns score/report order. `candidate_rank` remains provenance only.
- Reviewed-subset scoring passes only successful selected IDs through `diagnostics_for`.
- Preserve HPA-326 item dispositions: persisted prediction missing/invalid becomes an item-level failed `CohortItem` and subset exit 1, not fatal exit 2.
- Do not instantiate `OafBackend` or rerun inference for subset scoring.
- Do not add a DB, reviewer UI, sampling DSL, experiment framework, generic runner, second scorer, automatic chart repair, training, or backward-compatibility layer.
- Automated acceptance stays offline. The real human audit is an operational completion gate; HPA-327 remains In Progress until that evidence exists.

---

### Task 1: Promote model-independent reference preflight out of the OaF runner

**Files:**
- Modify: `src/benchmark/reference_set_manifest.py`
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_reference_set_manifest.py`
- Verify unchanged: `tests/benchmark/test_oaf_corpus_run.py`
- Verify unchanged: `tests/benchmark/test_oaf_corpus_run_acceptance.py`

**Interface:**

```text
preflight_reference_mappings(
    reference_manifest: LoadedReferenceSetManifest,
    timing_manifest: LoadedReferenceTimingManifest,
    *,
    timing_output_root: Path,
) -> dict[int, ReferenceMappingResult | None]
```

- [ ] **Step 1: Add a red public-contract test**

```python
def test_reference_mapping_preflight_is_public_model_independent_contract() -> None:
    from src.benchmark.reference_set_manifest import preflight_reference_mappings

    assert preflight_reference_mappings.__module__ == "src.benchmark.reference_set_manifest"
```

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py::test_reference_mapping_preflight_is_public_model_independent_contract -q
```

Expected: FAIL because the public function does not exist.

- [ ] **Step 3: Move `_preflight_reference_mappings()` without semantic edits**

Move its body from `oaf_corpus_run.py` to `reference_set_manifest.py`. Preserve the current HPA-324/HPA-323 lineage check, the per-row equality checks for chart/audio/event identity, `None` mappings for legitimate quarantined rows, and fatal failure for broken eligible artifacts.

The lineage check stays:

```python
if (
    reference_manifest.source_reference_timing_manifest_sha256
    != timing_manifest.manifest_sha256
    or reference_manifest.source_reference_timing_version != timing_manifest.corpus_version
):
    raise ValueError("reference and timing manifests have different lineage")
```

The identity field tuple stays:

```python
(
    "selected_chart_key",
    "selected_chart_content_hash",
    "source_audio_key",
    "source_audio_content_hash",
    "source_endpoint_sha256",
    "source_bucket",
    "reference_events_cache_path",
)
```

- [ ] **Step 4: Point HPA-326 at the promoted helper and delete the private copy**

```python
mappings = preflight_reference_mappings(
    reference_manifest,
    timing_manifest,
    timing_output_root=request.timing_manifest_path.parent.parent,
)
```

Do not leave a compatibility alias in `oaf_corpus_run.py`.

- [ ] **Step 5: Run focused + unchanged acceptance**

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: PASS with no HPA-326 output changes.

- [ ] **Step 6: Commit**

```bash
git add src/benchmark/reference_set_manifest.py src/benchmark/oaf_corpus_run.py tests/benchmark/test_reference_set_manifest.py
git commit -m "refactor: share reference mapping preflight"
```

---

### Task 2: Extract persisted OaF cohort reconstruction with report-byte characterization

**Files:**
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run_acceptance.py`
- Verify unchanged: `tests/benchmark/test_cohort_scoring_acceptance.py`

**Interface:**

```text
build_oaf_cohort_from_snapshot(
    snapshot: Mapping[str, object],
    *,
    mappings: Mapping[int, ReferenceMappingResult | None],
    output_dir: Path,
) -> tuple[CohortIdentity, tuple[CohortItem, ...]]
```

- [ ] **Step 1: Add a red report-byte characterization path**

Using the existing persisted-run setup in `test_oaf_corpus_run_acceptance.py`:

```python
identity, cohort_items = build_oaf_cohort_from_snapshot(
    snapshot,
    mappings=mappings,
    output_dir=output_dir,
)
expected_result = score_cohort(identity, cohort_items, diagnostics_for=())
expected_reports = tmp_path / "expected-reports"
write_cohort_reports(expected_result, expected_reports)

outcome = _finalize_scoring_and_outcome(
    snapshot,
    run_id=run_id,
    run_path=run_path,
    reports_path=actual_reports,
    aggregate_rtf=None,
    projected_full_wall_time_sec=None,
    mappings=mappings,
    output_dir=output_dir,
)

for name in (
    "summary.json",
    "items.csv",
    "per_song.csv",
    "per_class.csv",
    "event_diagnostics.jsonl",
    "summary.md",
):
    assert (actual_reports / name).read_bytes() == (expected_reports / name).read_bytes()
assert outcome.success_count == expected_result.population.success_count
assert outcome.failed_count == expected_result.population.failed_count
assert outcome.skipped_count == expected_result.population.skipped_count
assert outcome.quarantined_count == expected_result.population.quarantined_count
```

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: FAIL because the helper is absent.

- [ ] **Step 3: Implement reconstruction only**

Place next to `_cohort_item_from_run_row()`:

```python
def build_oaf_cohort_from_snapshot(
    snapshot: Mapping[str, object],
    *,
    mappings: Mapping[int, ReferenceMappingResult | None],
    output_dir: Path,
) -> tuple[CohortIdentity, tuple[CohortItem, ...]]:
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    items = snapshot.get("items", [])
    if not isinstance(items, list):
        raise ValueError("run snapshot items must be a list")
    identity = _cohort_identity_from_snapshot(snapshot)
    cohort_items = tuple(
        _cohort_item_from_run_row(
            identity,
            row,
            mappings.get(int(row["simfile_id"])),
            output_dir=output_dir,
        )
        for row in items
        if isinstance(row, Mapping)
    )
    return identity, cohort_items
```

Do not move prediction parsing, runner failure mapping, scoring, or reporting.

- [ ] **Step 4: Replace broad finalization's inline reconstruction**

```python
resolved_output_dir = output_dir or run_path.parents[2]
try:
    identity, cohort_items = build_oaf_cohort_from_snapshot(
        snapshot,
        mappings=mappings or {},
        output_dir=resolved_output_dir,
    )
except (TypeError, ValueError):
    return _fatal_outcome()

score_result = score_cohort(identity, cohort_items, diagnostics_for=())
```

Keep the rest of `_finalize_scoring_and_outcome()` unchanged.

- [ ] **Step 5: Run characterization**

```bash
uv run pytest tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py tests/benchmark/test_cohort_scoring_acceptance.py -q
```

Expected: PASS and byte-identical broad reports.

- [ ] **Step 6: Commit**

```bash
git add src/benchmark/oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py
git commit -m "refactor: expose persisted OaF cohort reconstruction"
```

---

### Task 3: Add the reviewed-subset schema loader/golden and reusable synthetic reference fixture

**Files:**
- Create: `src/benchmark/reviewed_subset.py`
- Create: `tests/benchmark/test_reviewed_subset.py`
- Create: `tests/benchmark/reviewed_subset_fixtures.py`
- Create: `tests/benchmark/schema_goldens/crux.reviewed-reference-subset-v1.jsonl`
- Modify: `tests/benchmark/schema_goldens/manifest.json`
- Modify: `tests/benchmark/test_schema_goldens.py`

**Interfaces:**

```text
REVIEWED_REFERENCE_SUBSET_SCHEMA = "crux.reviewed-reference-subset/v1"
REVIEW_POLICY_VERSION = "hpa327-v1"
REVIEW_TARGET_COUNT = 30
REVIEW_MIN_COUNT = 20
REVIEW_MAX_COUNT = 30
REVIEW_SELECTION_SEED = "crux-hpa327-v1"

load_reviewed_subset_manifest(path: Path) -> LoadedReviewedSubsetManifest
validate_schema_golden(schema: str, content: bytes) -> None

build_reviewed_subset_reference_fixture(
    tmp_path: Path,
    *,
    eligible_count: int = 36,
    reverse_rows: bool = False,
) -> ReviewedSubsetReferenceFixture
```

- [ ] **Step 1: Register the schema golden before implementation**

```json
{"golden_path":"tests/benchmark/schema_goldens/crux.reviewed-reference-subset-v1.jsonl","schema":"crux.reviewed-reference-subset/v1","validator_modules":["src.benchmark.reviewed_subset"]}
```

Update `test_schema_goldens.py`'s expected list to place this after `crux.benchmark-reference-manifest/v1` and before the OaF smoke oracle.

- [ ] **Step 2: Create the one-row schema golden**

Use exactly these fields:

```text
schema_version
corpus_version
review_policy_version
review_ledger_sha256
prior_review_ledger_sha256
candidate_rank
simfile_id
source_reference_manifest_sha256
source_reference_manifest_version
source_timing_manifest_sha256
source_timing_manifest_version
source_row_sha256
selected_chart_key
selected_chart_content_hash
source_audio_key
source_audio_content_hash
common_event_count
reference_event_span_sec
common_event_density_per_sec
common_class_count
density_band
class_richness_band
has_timing_warning
selects_real_or_full_chart
reviewer
reviewed_at
musical_fidelity
drum_character
known_limitations
reason_codes
notes
```

Use candidate rank `1`, simfile `42`, valid lowercase 64-character hashes, `density_band="medium"`, `class_richness_band="low"`, `has_timing_warning=false`, `selects_real_or_full_chart=true`, `musical_fidelity="usable_with_limits"`, `drum_character="acoustic"`, `reason_codes=["chart_simplification"]`, and `prior_review_ledger_sha256=null`.

- [ ] **Step 3: Add a red one-row golden test and separate 20-row normal-loader test**

Golden shape test:

```python
def test_reviewed_subset_schema_golden_is_valid() -> None:
    content = (
        Path(__file__).parent
        / "schema_goldens"
        / "crux.reviewed-reference-subset-v1.jsonl"
    ).read_bytes()
    validate_schema_golden(REVIEWED_REFERENCE_SUBSET_SCHEMA, content)
```

Normal loading must enforce 20–30 rows. Build 20 canonical rows from the golden template:

```python
def test_reviewed_subset_loader_accepts_real_population(tmp_path: Path) -> None:
    golden = (
        Path(__file__).parent
        / "schema_goldens"
        / "crux.reviewed-reference-subset-v1.jsonl"
    ).read_bytes()
    source = strict_json_loads(golden[:-1], require_canonical=True)
    assert isinstance(source, dict)

    rows: list[dict[str, object]] = []
    for offset in range(20):
        row = dict(source)
        row.pop("corpus_version", None)
        row["simfile_id"] = 100 + offset
        row["candidate_rank"] = offset + 1
        rows.append(row)

    rendered = render_manifest(tuple(rows))
    path = tmp_path / "subset.jsonl"
    path.write_bytes(rendered.content)
    loaded = load_reviewed_subset_manifest(path)

    assert loaded.manifest_sha256 == sha256(rendered.content).hexdigest()
    assert len(loaded.rows) == 20
```

Also test duplicate simfile IDs/ranks, mixed source/timing identity, mixed ledger hash, invalid enums, fewer than 20 normal rows, more than 30 normal rows, and noncanonical JSONL.

- [ ] **Step 4: Verify red**

```bash
uv run pytest tests/benchmark/test_schema_goldens.py tests/benchmark/test_reviewed_subset.py -q
```

Expected: FAIL because `reviewed_subset.py` is absent.

- [ ] **Step 5: Implement closed types, exact row validation, loader, and golden validator**

```python
Band = Literal["low", "medium", "high"]
MusicalFidelity = Literal["close", "usable_with_limits", "not_representative"]
DrumCharacter = Literal["acoustic", "electronic", "hybrid", "unknown"]
ReviewReasonCode = Literal[
    "chart_selection_mismatch",
    "audio_revision_mismatch",
    "bgm_alignment_problem",
    "chart_audio_drift",
    "chart_simplification",
    "chart_authored_error",
    "unusual_lane_convention",
    "not_representative",
    "other",
]
```

Define `ReviewedSubsetRowView`, `LoadedReviewedSubsetRow`, and `LoadedReviewedSubsetManifest`. The loaded manifest exposes:

```text
manifest_sha256: str
corpus_version: str
review_policy_version: str
review_ledger_sha256: str
prior_review_ledger_sha256: str | None
source_reference_manifest_sha256: str
source_reference_manifest_version: str
source_timing_manifest_sha256: str
source_timing_manifest_version: str
rows: tuple[LoadedReviewedSubsetRow, ...]
```

Normal loading calls:

```python
canonical = read_canonical_manifest_core(
    path,
    schema_version=REVIEWED_REFERENCE_SUBSET_SCHEMA,
    validate_rows=validate_rows,
)
```

Require shared source-reference/timing identity, policy version, review-ledger hash, and prior-ledger hash; unique positive candidate ranks; unique simfile IDs; and 20–30 rows. Candidate ranks need not be contiguous because excluded review candidates are absent. `validate_schema_golden()` validates the same row shape without applying the normal 20-row minimum.

- [ ] **Step 6: Factor the reusable synthetic HPA-323/HPA-324 fixture now**

Create:

```python
@dataclass(frozen=True)
class ReviewedSubsetReferenceFixture:
    reference_manifest_path: Path
    timing_manifest_path: Path
    timing_output_root: Path
```

Factor the smallest already-working canonical timing/reference/event builders from existing HPA-323/HPA-324/HPA-326 tests into `reviewed_subset_fixtures.py`. Generate distinct simfile IDs and valid event artifacts. `eligible_count` controls population size; `reverse_rows=True` reverses source row input before publication. Existing owner tests may use the factored helper, but their expected bytes/semantics stay unchanged.

- [ ] **Step 7: Run schema + owner coverage**

```bash
uv run pytest tests/benchmark/test_schema_goldens.py tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reference_set_manifest.py tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/reviewed_subset.py tests/benchmark/test_reviewed_subset.py tests/benchmark/reviewed_subset_fixtures.py tests/benchmark/schema_goldens tests/benchmark/test_schema_goldens.py
git commit -m "feat: define reviewed subset manifest contract"
```

---

### Task 4: Implement deterministic candidate selection, review CSV preparation, and continuation

**Files:**
- Modify: `src/benchmark/reviewed_subset.py`
- Modify: `tests/benchmark/test_reviewed_subset.py`
- Modify: `tests/benchmark/reviewed_subset_fixtures.py`

**Interfaces:**

```text
PrepareReviewedSubsetRequest(
    reference_manifest_path: Path,
    timing_manifest_path: Path,
    output_file: Path,
    prior_ledger_path: Path | None = None,
)

PrepareReviewedSubsetOutcome(
    exit_code: Literal[0, 2],
    output_file: Path | None,
    candidate_count: int,
    carried_include_count: int,
    replacement_count: int,
)

build_candidate_stream(
    reference_manifest: LoadedReferenceSetManifest,
    timing_manifest: LoadedReferenceTimingManifest,
    *,
    mappings: Mapping[int, ReferenceMappingResult | None],
) -> tuple[ReviewCandidate, ...]

prepare_reviewed_subset(
    request: PrepareReviewedSubsetRequest,
) -> PrepareReviewedSubsetOutcome
```

- [ ] **Step 1: Add red population/API tests**

```python
def test_prepare_selects_exactly_30_without_model_inputs(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    output = tmp_path / "review.csv"
    request = PrepareReviewedSubsetRequest(
        reference_manifest_path=fixture.reference_manifest_path,
        timing_manifest_path=fixture.timing_manifest_path,
        output_file=output,
    )

    assert set(request.__dataclass_fields__) == {
        "reference_manifest_path",
        "timing_manifest_path",
        "output_file",
        "prior_ledger_path",
    }

    outcome = prepare_reviewed_subset(request)
    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    assert outcome.exit_code == 0
    assert len(rows) == 30
    assert [int(row["candidate_rank"]) for row in rows] == list(range(1, 31))
```

Add cases for 20–29 selecting all and fewer than 20 returning exit 2.

- [ ] **Step 2: Add feature/accounting tests**

```python
assert candidate.common_event_count == loaded_reference.view.common_scored_event_count
assert candidate.density_band in {"low", "medium", "high"}
assert candidate.class_richness_band in {"low", "medium", "high"}
assert candidate.source_audio_cache_path == (
    f"sha256/{candidate.source_audio_content_hash[:2]}/{candidate.source_audio_content_hash}"
)
```

Add fixtures for HPA-323 timing warnings and `real.dtx` / `full.dtx` / `mas.dtx`; only `real`/`full` sets `selects_real_or_full_chart`. Add a deliberate `common_scored_event_count` mismatch and require exit 2.

- [ ] **Step 3: Add seeded-stratum regression**

Build a population with more than 30 distinct nonempty strata/candidate positions. Compute expected stratum order:

```python
expected_order = sorted(
    nonempty_strata,
    key=lambda key: sha256(
        f"{REVIEW_SELECTION_SEED}:{canonical_stratum_key(key)}".encode()
    ).hexdigest(),
)
assert selected_first_round_strata == tuple(
    expected_order[: len(selected_first_round_strata)]
)
```

Reverse input manifest rows and require identical selected simfile order.

- [ ] **Step 4: Verify red**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py -k "prepare or candidate or stratum or band" -q
```

Expected: FAIL because selection/preparation is absent.

- [ ] **Step 5: Implement source hashing + feature extraction over shared preflight**

```python
def _source_row_sha256(source_row: Mapping[str, object]) -> str:
    payload = {
        key: value for key, value in source_row.items() if key != "corpus_version"
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()
```

Build the timing lookup once and call `preflight_reference_mappings()` once. Every HPA-324 eligible row must have a non-`None` mapping and satisfy:

```python
if len(mapping.common_events) != loaded.view.common_scored_event_count:
    raise ValueError("reference common event count does not match HPA-324")
```

Derive span/density/class richness from common events and warning state from HPA-323.

- [ ] **Step 6: Implement thirds + seeded round-robin**

```python
def _assign_bands(
    values: tuple[tuple[float | int, int], ...],
) -> dict[int, Band]:
    ordered = sorted(values)
    count = len(ordered)
    labels: tuple[Band, Band, Band] = ("low", "medium", "high")
    return {
        simfile_id: labels[min(2, (index * 3) // count)]
        for index, (_, simfile_id) in enumerate(ordered)
    }
```

```python
def canonical_stratum_key(candidate: ReviewCandidate) -> str:
    return (
        f"{candidate.density_band}|{candidate.class_richness_band}|"
        f"{int(candidate.has_timing_warning)}|"
        f"{int(candidate.selects_real_or_full_chart)}"
    )


def _seeded_hash(value: str) -> str:
    return sha256(f"{REVIEW_SELECTION_SEED}:{value}".encode()).hexdigest()
```

Produce a deterministic stream beyond row 30 for continuation replacements.

- [ ] **Step 7: Implement exact CSV boundary**

Generated fields are exactly:

```text
review_policy_version
selection_seed
candidate_rank
simfile_id
source_reference_manifest_sha256
source_reference_manifest_version
source_timing_manifest_sha256
source_timing_manifest_version
source_row_sha256
selected_chart_key
selected_chart_content_hash
selected_chart_cache_path
source_audio_key
source_audio_content_hash
source_audio_cache_path
common_event_count
reference_event_span_sec
common_event_density_per_sec
common_class_count
density_band
class_richness_band
has_timing_warning
selects_real_or_full_chart
```

Manual fields are exactly:

```text
reviewer
reviewed_at
chart_selection_confirmed
audio_revision_confirmed
bgm_alignment_confirmed
technical_mapping_confirmed
musical_fidelity
drum_character
known_limitations
decision
reason_codes
notes
```

Render numeric display values with:

```python
canonical_json_bytes(quantize_six(value)).decode("ascii")
```

Write rows with:

```python
writer = csv.DictWriter(
    handle,
    fieldnames=REVIEW_CSV_FIELDS,
    lineterminator="\n",
)
```

Leave manual cells blank for new candidates.

- [ ] **Step 8: Add red continuation tests**

Fill initial 30 rows with 24 valid includes and 6 valid excludes. Re-prepare with the prior ledger and assert:

```python
assert outcome.carried_include_count == 24
assert outcome.replacement_count == 6
assert outcome.candidate_count == 30
assert not excluded_ids & current_ids
assert carried_ids <= current_ids
```

Change one upstream row hash and assert its prior review is not carried. Replacements must equal the next unused IDs from `build_candidate_stream()`. Malformed completed prior manual fields return exit 2.

- [ ] **Step 9: Implement carry-forward rules**

Parse prior rows by canonical integer `simfile_id` and use prior `source_row_sha256` only as the guard:

```text
unchanged valid include -> carry manual fields
unchanged valid exclude -> consume and skip
changed source hash -> do not carry; row may reappear unreviewed
new/replacement row -> next unused deterministic stream candidate
```

Keep carried includes in previous relative order, append replacements in stream order, then assign fresh candidate ranks `1..N`.

- [ ] **Step 10: Run selection + continuation**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py -k "prepare or candidate or stratum or band or carry or prior" -q
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add src/benchmark/reviewed_subset.py tests/benchmark/test_reviewed_subset.py tests/benchmark/reviewed_subset_fixtures.py
git commit -m "feat: prepare reviewed reference subset"
```

---

### Task 5: Validate manual reviews and publish canonical subset + audit ledger

**Files:**
- Modify: `src/benchmark/reviewed_subset.py`
- Modify: `tests/benchmark/test_reviewed_subset.py`
- Create: `tests/benchmark/test_reviewed_subset_acceptance.py`

**Interfaces:**

```text
FinalizeReviewedSubsetRequest(
    reference_manifest_path: Path,
    timing_manifest_path: Path,
    review_file: Path,
    output_dir: Path,
    prior_ledger_path: Path | None = None,
)

FinalizeReviewedSubsetOutcome(
    exit_code: Literal[0, 2],
    manifest: PublishedManifest | None,
    review_ledger_path: Path | None,
    included_count: int,
    excluded_count: int,
)

finalize_reviewed_subset(
    request: FinalizeReviewedSubsetRequest,
) -> FinalizeReviewedSubsetOutcome
```

- [ ] **Step 1: Add red closed-review tests**

Cover blank reviewer, non-UTC/invalid timestamp, invalid confirmation, include with any false confirmation, include + `not_representative`, exclude with no reasons, `other` with no notes, unknown reason, fewer than 20 includes, and valid 20–30 includes.

CSV confirmations are exact `true` / `false`; decisions are `include` / `exclude`; CSV reasons are semicolon-separated closed tokens.

- [ ] **Step 2: Prove generated cells are non-authoritative**

Rewrite generated density/hash/boolean/rank display cells while preserving `simfile_id` and valid manual fields; finalization should still succeed. Change `simfile_id` to a stale/unknown member and require exit 2.

- [ ] **Step 3: Pin continuation reproducibility**

```python
without_prior = finalize_reviewed_subset(
    replace(finalize_request, prior_ledger_path=None)
)
assert without_prior.exit_code == 2

with_prior = finalize_reviewed_subset(finalize_request)
assert with_prior.exit_code == 0
```

- [ ] **Step 4: Verify red**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py -k "finalize or review" -q
```

Expected: FAIL because finalization is absent.

- [ ] **Step 5: Implement one completed-review parser**

Use a frozen review dataclass. Parse reasons to sorted unique tuples. Require RFC3339 UTC and canonicalize published timestamps to `Z`. Enforce closed fidelity/character/decision/reason sets and `other -> nonempty notes`. Ignore generated cells as authority.

- [ ] **Step 6: Reproduce membership from authoritative inputs**

Run the same in-memory candidate/continuation logic with the same optional prior ledger. Require exactly one submitted manual review per expected simfile and no extra/missing IDs. Use freshly derived rows for current hashes/features/ranks.

- [ ] **Step 7: Render canonical complete ledger and bind hashes**

Re-render `review-ledger.csv` from fresh generated + validated manual values using stable UTF-8/`\n`/header/CSV quoting. Publish it using the shared atomic byte-replacement helper.

```python
review_ledger_sha256 = sha256(review_ledger_bytes).hexdigest()
prior_review_ledger_sha256 = (
    sha256(request.prior_ledger_path.read_bytes()).hexdigest()
    if request.prior_ledger_path is not None
    else None
)
```

- [ ] **Step 8: Publish accepted rows on manifest rails**

For each valid include, emit the Task 3 row contract, turn CSV reasons into a JSON array, and include all six features + both bands.

```python
rendered = render_manifest(tuple(rows))
published = publish_manifest(request.output_dir, rendered)
publish_latest_manifest(request.output_dir, published, "complete", clock())
```

Every row carries the same current/prior ledger hashes.

- [ ] **Step 9: Add prepare -> finalize acceptance**

```python
assert outcome.manifest is not None
assert outcome.review_ledger_path is not None
loaded = load_reviewed_subset_manifest(outcome.manifest.path)
assert 20 <= len(loaded.rows) <= 30
assert loaded.review_ledger_sha256 == sha256(
    outcome.review_ledger_path.read_bytes()
).hexdigest()
assert loaded.prior_review_ledger_sha256 is None
```

Add a continuation variant and assert the prior-ledger SHA matches supplied bytes.

- [ ] **Step 10: Run finalization/schema acceptance**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py tests/benchmark/test_schema_goldens.py -q
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add src/benchmark/reviewed_subset.py tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py
git commit -m "feat: publish reviewed reference subset"
```

---

### Task 6: Rescore persisted OaF predictions on exact reviewed membership

**Files:**
- Modify: `src/benchmark/reviewed_subset.py`
- Modify: `tests/benchmark/reviewed_subset_fixtures.py`
- Modify: `tests/benchmark/test_reviewed_subset.py`
- Modify: `tests/benchmark/test_reviewed_subset_acceptance.py`
- Verify unchanged: `tests/benchmark/test_oaf_corpus_run_acceptance.py`
- Verify unchanged: `tests/benchmark/test_cohort_scoring_acceptance.py`

**Interfaces:**

```text
ScoreReviewedSubsetRequest(
    run_path: Path,
    reference_manifest_path: Path,
    timing_manifest_path: Path,
    subset_manifest_path: Path,
    output_dir: Path,
)

ScoreReviewedSubsetOutcome(
    exit_code: Literal[0, 1, 2],
    cohort_id: str | None,
    reports_path: Path | None,
    success_count: int,
    failed_count: int,
    skipped_count: int,
    quarantined_count: int,
)

score_oaf_reviewed_subset(
    request: ScoreReviewedSubsetRequest,
) -> ScoreReviewedSubsetOutcome
```

Import `parse_oaf_corpus_run` and `build_oaf_cohort_from_snapshot` locally inside the score function so candidate preparation has no import-time OaF execution dependency.

- [ ] **Step 1: Extend the synthetic fixture with a persisted OaF run**

```python
@dataclass(frozen=True)
class ReviewedSubsetOafFixture:
    reference_manifest_path: Path
    timing_manifest_path: Path
    timing_output_root: Path
    run_path: Path
    oaf_output_dir: Path
```

Factor the smallest existing persisted-run/prediction setup from HPA-326 acceptance. Include successes plus at least one selectable non-success item. No Docker, TensorFlow, network, or backend invocation.

- [ ] **Step 2: Add red no-inference test**

```python
def fail_backend(*args: object, **kwargs: object) -> object:
    raise AssertionError("reviewed subset scoring must not construct OafBackend")

monkeypatch.setattr("src.benchmark.oaf_corpus_run.create_backend", fail_backend)
outcome = score_oaf_reviewed_subset(request)
assert outcome.exit_code in {0, 1}
```

Hash parent-run reports before/after and require no change.

- [ ] **Step 3: Add fatal identity/membership tests**

Require exit 2 only for run/reference/timing/subset identity mismatch, subset member absent from parent population, noncanonical/unreadable run snapshot, or report-publication failure.

- [ ] **Step 4: Add item-level persisted prediction failure regression**

Delete or corrupt a prediction artifact referenced by a selected parent row that was previously `inferred`/`resumed`. Reconstruct and score the subset. Assert the selected item becomes HPA-325 `failed` with the existing `prediction_missing` or `prediction_artifact_invalid` reason and the subset outcome exits 1, not 2.

- [ ] **Step 5: Add diagnostics regression**

Wrap `score_cohort` and capture `diagnostics_for`:

```python
assert diagnostics_for == tuple(
    sorted(item.simfile_id for item in selected_items if item.status == "success")
)
```

For a successful fixture with scoreable events, require nonempty `event_diagnostics.jsonl`.

- [ ] **Step 6: Verify red**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py -k "score or rescore or diagnostic or prediction" -q
```

Expected: FAIL because subset scoring is absent.

- [ ] **Step 7: Implement persisted reconstruction + lineage checks**

Inside `score_oaf_reviewed_subset()` only:

```python
from src.benchmark.oaf_corpus_run import (
    build_oaf_cohort_from_snapshot,
    parse_oaf_corpus_run,
)
```

Parse run bytes canonically. Load HPA-324/HPA-323 and call `preflight_reference_mappings()`. Require the run's reference SHA, timing SHA, and timing version to match supplied manifests. Load HPA-327 and require its source-reference/timing identities to match the same manifests.

```python
parent_identity, parent_items = build_oaf_cohort_from_snapshot(
    snapshot,
    mappings=mappings,
    output_dir=request.run_path.parents[2],
)
```

Do not call `run_oaf_corpus()`.

- [ ] **Step 8: Filter exact membership and derive only a new cohort ID**

Require every subset ID in the parent population. Keep selected non-success items. Do not order by `candidate_rank`.

```python
subset_cohort_id = sha256(
    canonical_json_bytes(
        {
            "parent_run_id": parent_identity.cohort_id,
            "reviewed_subset_manifest_sha256": subset.manifest_sha256,
        }
    )
).hexdigest()
subset_identity = replace(parent_identity, cohort_id=subset_cohort_id)
```

- [ ] **Step 9: Reuse HPA-325 with successful selected diagnostics**

```python
diagnostics_for = tuple(
    sorted(item.simfile_id for item in selected_items if item.status == "success")
)
result = score_cohort(
    subset_identity,
    selected_items,
    diagnostics_for=diagnostics_for,
)
write_cohort_reports(result, request.output_dir)
```

Exit 0 when all selected items succeed; exit 1 when any selected item is failed/skipped/quarantined, including missing/invalid persisted prediction artifacts; exit 2 only for fatal run/lineage/membership/report errors.

- [ ] **Step 10: Run subset + unchanged broad acceptance**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py tests/benchmark/test_oaf_corpus_run_acceptance.py tests/benchmark/test_cohort_scoring_acceptance.py -q
```

Expected: PASS; broad reports stay unchanged and reviewed reports contain bounded diagnostics.

- [ ] **Step 11: Commit**

```bash
git add src/benchmark/reviewed_subset.py tests/benchmark/reviewed_subset_fixtures.py tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py
git commit -m "feat: score persisted OaF reviewed subset"
```

---

### Task 7: Wire three thin CLI commands and finish end-to-end verification

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Modify: `tests/benchmark/test_reviewed_subset_acceptance.py`

**Commands:**

```text
crux benchmark prepare-reviewed-subset
crux benchmark finalize-reviewed-subset
crux benchmark score-oaf-reviewed-subset
```

- [ ] **Step 1: Add red help/option tests**

Require:

```text
prepare-reviewed-subset: --manifest --timing-manifest --output-file [--prior-ledger]
finalize-reviewed-subset: --manifest --timing-manifest --review-file --output-dir [--prior-ledger]
score-oaf-reviewed-subset: --run --manifest --timing-manifest --subset-manifest --output-dir
```

Assert there is no seed/count/threshold/model/backend selector.

- [ ] **Step 2: Add callback forwarding + canonical JSON tests**

Monkeypatch each domain function and assert exact request construction. Omitted prior ledger becomes `None`.

Output fields:

```text
prepare: exit_code, candidate_count, carried_include_count, replacement_count, output_file
finalize: exit_code, included_count, excluded_count, manifest_path, review_ledger_path
score: exit_code, cohort_id, success_count, failed_count, skipped_count, quarantined_count, reports_path
```

- [ ] **Step 3: Verify red**

```bash
uv run pytest tests/test_cli_benchmark.py -k "reviewed_subset" -q
```

Expected: FAIL because the commands are absent.

- [ ] **Step 4: Implement Click callbacks only**

Use `click.Path(path_type=Path, exists=True, dir_okay=False)` for input files and `click.Path(path_type=Path)` for outputs. Call the domain function, render one object with the existing canonical encoder, then propagate the domain exit:

```python
click.echo(canonical_json_bytes(payload).decode("utf-8"))
if outcome.exit_code:
    raise click.exceptions.Exit(outcome.exit_code)
```

No HPA-327 domain behavior belongs in Click callbacks.

- [ ] **Step 5: Run the reusable synthetic chain end-to-end**

In `test_reviewed_subset_acceptance.py` execute:

```text
prepare -> fill manual fields -> finalize -> score persisted OaF subset
```

Then run the same through `CliRunner`. Compare manifest, canonical ledger, and report bytes where path-valued CLI output is excluded. Keep the test offline/backend-free.

- [ ] **Step 6: Run all HPA-327 + touched seams**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py tests/benchmark/test_reference_set_manifest.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py tests/benchmark/test_cohort_scoring_acceptance.py tests/benchmark/test_schema_goldens.py tests/test_cli_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 7: Run repository-wide gates**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check
```

Expected: all pass; do not weaken gates.

- [ ] **Step 8: Commit**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py tests/benchmark/test_reviewed_subset_acceptance.py
git commit -m "feat: add reviewed subset benchmark commands"
```

---

## Operational Completion Gate

The repository does not contain the real local HPA-323/HPA-324 artifacts or completed human review evidence. Do not fabricate this gate in tests.

After code lands with those real artifacts available:

1. Run `prepare-reviewed-subset` against the exact HPA-324 manifest and matching HPA-323 timing manifest before using OaF song-level scores for membership decisions.
2. Preserve the generated CSV and manually inspect every candidate's chart, full mix, BGM alignment, mapping, and musical fidelity.
3. Fill all 12 manual review fields.
4. If fewer than 20 are acceptable or pre-score diagnostic coverage is clearly inadequate, rerun prepare with `--prior-ledger`; preserve unchanged includes and review only deterministic unused replacements without consulting model scores.
5. Finalize with the same prior ledger used for that continuation pass. Preserve canonical `review-ledger.csv` plus `crux.reviewed-reference-subset/v1`.
6. Confirm the accepted manifest itself demonstrates materially different density/class-richness bands, timing-warning states, `real`/`full` selection, and manually recorded drum character.
7. Run `score-oaf-reviewed-subset` against the existing HPA-326 `run.json`; do not rerun inference.
8. Preserve subset reports and event diagnostics without modifying broad HPA-326 artifacts.
9. Mark HPA-327 Done only after the real 20–30-song audit evidence exists.

## Plan Self-Review Checklist

- Every durable schema has one loader and one golden; subset JSONL is never hand-parsed by consumers.
- Normal subset loading enforces 20–30 rows; the one-row golden is validated only by the schema-golden validator.
- Candidate preparation has no model/run/prediction/report/score parameter and no import-time OaF execution dependency.
- Both HPA-326 changes are characterization-tested no-behavior-change refactors before HPA-327 depends on them.
- Continuation prepare/finalize share the same prior ledger and the final manifest binds its hash.
- CSV generated cells never become authority; current evidence is re-derived from HPA-323/HPA-324.
- HPA-326 persisted prediction artifact failures preserve their existing item-level disposition.
- HPA-325 remains the sole scorer/report path, retains `simfile_id` order, and receives successful reviewed IDs via `diagnostics_for`.
- No task adds training, chart repair, UI, DB, generic runners, sampling configuration, or backward-compatibility machinery.
