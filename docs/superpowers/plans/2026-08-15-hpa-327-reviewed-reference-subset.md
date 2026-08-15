# HPA-327 Reviewed Reference Subset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze and manually audit a deterministic 20–30-song reference subset before score-informed selection, publish it as a canonical manifest, and rescore the existing persisted OaF cohort on that exact membership with HPA-325 diagnostics.

**Architecture:** Keep one model-independent HPA-327 module for selection, review CSV handling, subset publication, loading, and subset-score orchestration. First extract the two existing reusable seams from HPA-326 without changing behavior: reference/timing preflight moves to `reference_set_manifest.py`, and persisted cohort reconstruction becomes a public helper in `oaf_corpus_run.py`. New scoring still delegates to `score_cohort()` and `write_cohort_reports()`; candidate preparation never consumes run, prediction, or score inputs.

**Tech Stack:** Python 3.12, stdlib `csv`/`dataclasses`/`hashlib`, Click, pytest, existing Crux canonical JSON/JSONL helpers, HPA-323/HPA-324 reference manifests, HPA-325 scorer/report writer, HPA-326 persisted OaF run artifacts.

## Global Constraints

- `REVIEW_POLICY_VERSION = "hpa327-v1"`.
- `REVIEW_TARGET_COUNT = 30`, `REVIEW_MIN_COUNT = 20`, `REVIEW_MAX_COUNT = 30`.
- `REVIEW_SELECTION_SEED = "crux-hpa327-v1"`; no CLI seed/count overrides.
- Candidate preparation accepts only HPA-323/HPA-324 inputs plus an optional prior HPA-327 review ledger. It has no model/run/prediction/report/score input.
- Use the exact thirds formula `min(2, (i * 3) // n)` after sorting `(feature_value, simfile_id)`.
- Order strata and candidates by seeded SHA-256, never lexicographic stratum labels.
- Reconcile every eligible mapping with published HPA-324 `common_scored_event_count` before selection.
- `timing_warnings` comes from the HPA-323 timing row, not `ReferenceSetRowView`.
- Use `selects_real_or_full_chart`; do not call `real.dtx` / `full.dtx` nonstandard.
- CSV generated cells are display/evidence only. Finalization trusts current `simfile_id` membership plus the 12 manual fields and re-derives generated values.
- A continuation pass uses the same optional prior ledger in both prepare and finalize; unchanged valid includes carry forward, unchanged excludes remain consumed, and replacements come only from the unused deterministic candidate stream.
- Publish accepted rows as `crux.reviewed-reference-subset/v1` with six selection features plus both band labels, `review_ledger_sha256`, and optional `prior_review_ledger_sha256`.
- `load_reviewed_subset_manifest()` must use `read_canonical_manifest_core()` and the schema must be registered in the existing golden registry.
- HPA-325 owns scoring/report ordering. `candidate_rank` is provenance only; reports stay in `simfile_id` order.
- Reviewed-subset scoring passes only successful selected IDs through `diagnostics_for`.
- Do not instantiate `OafBackend` or rerun inference for subset scoring.
- Do not add a DB, reviewer UI, sampling DSL, experiment framework, generic backend runner, second scorer, automatic chart repair, training, or backward-compatibility layer.
- Real manual audit is operational acceptance, not an automated test. HPA-327 remains In Progress until that evidence exists.

---

### Task 1: Promote model-independent reference preflight out of the OaF runner

**Files:**
- Modify: `src/benchmark/reference_set_manifest.py`
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_reference_set_manifest.py`
- Verify unchanged: `tests/benchmark/test_oaf_corpus_run.py`
- Verify unchanged: `tests/benchmark/test_oaf_corpus_run_acceptance.py`

**Interfaces:**
- Consumes: `LoadedReferenceSetManifest`, `LoadedReferenceTimingManifest`, `read_native_reference_events()`, `map_reference_events()`.
- Produces:

```python
def preflight_reference_mappings(
    reference_manifest: LoadedReferenceSetManifest,
    timing_manifest: LoadedReferenceTimingManifest,
    *,
    timing_output_root: Path,
) -> dict[int, ReferenceMappingResult | None]:
    ...
```

- HPA-326 and HPA-327 both use this one implementation. Do not place it in `reference_set.py`; `reference_set_manifest.py` already imports `map_reference_events()` from that module.

- [ ] **Step 1: Add a failing public-seam test before moving code**

Add to `tests/benchmark/test_reference_set_manifest.py`:

```python
def test_reference_mapping_preflight_is_public_model_independent_contract() -> None:
    from src.benchmark.reference_set_manifest import preflight_reference_mappings

    assert preflight_reference_mappings.__module__ == "src.benchmark.reference_set_manifest"
```

This should fail before the extraction because the public function does not exist.

- [ ] **Step 2: Run the focused red test**

Run:

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py::test_reference_mapping_preflight_is_public_model_independent_contract -q
```

Expected: FAIL with import/name error for `preflight_reference_mappings`.

- [ ] **Step 3: Move the existing preflight implementation without semantic edits**

Move the body of `oaf_corpus_run.py::_preflight_reference_mappings()` into `reference_set_manifest.py` under the public name above. Preserve all current checks:

```python
if (
    reference_manifest.source_reference_timing_manifest_sha256
    != timing_manifest.manifest_sha256
    or reference_manifest.source_reference_timing_version != timing_manifest.corpus_version
):
    raise ValueError("reference and timing manifests have different lineage")
```

Keep the existing per-row identity checks for:

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

Keep the existing `None` mapping behavior for legitimate quarantined rows and fatal failure for broken eligible artifacts.

- [ ] **Step 4: Point HPA-326 at the promoted helper**

In `src/benchmark/oaf_corpus_run.py`, import:

```python
from src.benchmark.reference_set_manifest import (
    LoadedReferenceSetManifest,
    LoadedReferenceSetRow,
    load_reference_set_manifest,
    preflight_reference_mappings,
    read_native_reference_events,
)
```

Replace the private call with:

```python
mappings = preflight_reference_mappings(
    reference_manifest,
    timing_manifest,
    timing_output_root=request.timing_manifest_path.parent.parent,
)
```

Delete `_preflight_reference_mappings()` from `oaf_corpus_run.py`; do not leave an alias.

- [ ] **Step 5: Run focused and unchanged HPA-326 acceptance coverage**

Run:

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: PASS with no fixture or expected-output changes needed for HPA-326.

- [ ] **Step 6: Commit the no-behavior-change extraction**

```bash
git add src/benchmark/reference_set_manifest.py src/benchmark/oaf_corpus_run.py tests/benchmark/test_reference_set_manifest.py
git commit -m "refactor: share reference mapping preflight"
```

---

### Task 2: Extract persisted OaF cohort reconstruction with characterization coverage

**Files:**
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run_acceptance.py`
- Verify unchanged: `tests/benchmark/test_cohort_scoring_acceptance.py`

**Interfaces:**
- Consumes: validated HPA-326 snapshot, preflight mappings, immutable prediction artifacts, `_cohort_item_from_run_row()`.
- Produces:

```python
def build_oaf_cohort_from_snapshot(
    snapshot: Mapping[str, object],
    *,
    mappings: Mapping[int, ReferenceMappingResult | None],
    output_dir: Path,
) -> tuple[CohortIdentity, tuple[CohortItem, ...]]:
    ...
```

- Broad HPA-326 finalization keeps its current `output_dir or run_path.parents[2]` fallback before calling this helper. HPA-327 scoring passes `run_path.parents[2]` explicitly.

- [ ] **Step 1: Add a characterization test for report-byte equivalence**

Using the existing persisted-run setup in `tests/benchmark/test_oaf_corpus_run_acceptance.py`, add an assertion path that constructs the expected reports through the new seam:

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

Before extraction, the import of `build_oaf_cohort_from_snapshot` should fail.

- [ ] **Step 2: Run the characterization test red**

Run the specific acceptance test you extended:

```bash
uv run pytest tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: FAIL only because `build_oaf_cohort_from_snapshot` is absent.

- [ ] **Step 3: Implement the narrow reconstruction helper**

Place it next to `_cohort_item_from_run_row()`:

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

Do not move prediction parsing, failure mapping, or scoring into the new helper.

- [ ] **Step 4: Make broad finalization call the helper**

In `_finalize_scoring_and_outcome()` replace the inline identity/item reconstruction with:

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

Preserve every existing outcome/status/report behavior after that point.

- [ ] **Step 5: Run characterization plus existing scorer acceptance**

```bash
uv run pytest tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py tests/benchmark/test_cohort_scoring_acceptance.py -q
```

Expected: PASS; broad reports remain byte-identical.

- [ ] **Step 6: Commit the second no-behavior-change extraction**

```bash
git add src/benchmark/oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py
git commit -m "refactor: expose persisted OaF cohort reconstruction"
```

---

### Task 3: Establish HPA-327 schema rails and reusable synthetic reference fixture

**Files:**
- Create: `src/benchmark/reviewed_subset.py`
- Create: `tests/benchmark/test_reviewed_subset.py`
- Create: `tests/benchmark/reviewed_subset_fixtures.py`
- Create: `tests/benchmark/schema_goldens/crux.reviewed-reference-subset-v1.jsonl`
- Modify: `tests/benchmark/schema_goldens/manifest.json`
- Modify: `tests/benchmark/test_schema_goldens.py`

**Interfaces:**
- Produces constants:

```python
REVIEWED_REFERENCE_SUBSET_SCHEMA = "crux.reviewed-reference-subset/v1"
REVIEW_POLICY_VERSION = "hpa327-v1"
REVIEW_TARGET_COUNT = 30
REVIEW_MIN_COUNT = 20
REVIEW_MAX_COUNT = 30
REVIEW_SELECTION_SEED = "crux-hpa327-v1"
```

- Produces loader:

```python
def load_reviewed_subset_manifest(path: Path) -> LoadedReviewedSubsetManifest:
    ...
```

- Produces reusable test fixture interface:

```python
@dataclass(frozen=True)
class ReviewedSubsetReferenceFixture:
    reference_manifest_path: Path
    timing_manifest_path: Path
    timing_output_root: Path


def build_reviewed_subset_reference_fixture(
    tmp_path: Path,
    *,
    eligible_count: int = 36,
    reverse_rows: bool = False,
) -> ReviewedSubsetReferenceFixture:
    ...
```

Build that fixture by factoring the already-working canonical HPA-323/HPA-324 event/manifest construction from the existing OaF/reference-manifest tests into this shared test helper. Keep the produced manifest/event bytes equivalent; do not introduce a second fake schema.

- [ ] **Step 1: Register the new schema golden first**

Append this canonical registry row to `tests/benchmark/schema_goldens/manifest.json` in the existing schema order:

```json
{"golden_path":"tests/benchmark/schema_goldens/crux.reviewed-reference-subset-v1.jsonl","schema":"crux.reviewed-reference-subset/v1","validator_modules":["src.benchmark.reviewed_subset"]}
```

Update the exact expected schema list in `test_schema_goldens.py` to include `crux.reviewed-reference-subset/v1` after the benchmark-reference manifest and before the OaF smoke oracle.

- [ ] **Step 2: Add a one-row canonical accepted-subset golden**

The row must include exactly these fields:

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

Use canonical scalar examples: candidate rank `1`, simfile `42`, one common class/event, `density_band="medium"`, `class_richness_band="low"`, `has_timing_warning=false`, `selects_real_or_full_chart=true`, `musical_fidelity="usable_with_limits"`, `drum_character="acoustic"`, `reason_codes=["chart_simplification"]`, and valid lowercase 64-character SHA-256 strings. Set `prior_review_ledger_sha256` to `null` in the v1 golden.

- [ ] **Step 3: Add loader/golden tests before implementation**

In `tests/benchmark/test_reviewed_subset.py`:

```python
def test_reviewed_subset_golden_loads_canonically(tmp_path: Path) -> None:
    source = (
        Path(__file__).parent
        / "schema_goldens"
        / "crux.reviewed-reference-subset-v1.jsonl"
    )
    path = tmp_path / "subset.jsonl"
    path.write_bytes(source.read_bytes())

    loaded = load_reviewed_subset_manifest(path)

    assert loaded.manifest_sha256 == sha256(path.read_bytes()).hexdigest()
    assert len(loaded.rows) == 1
    assert loaded.rows[0].view.simfile_id == 42
    assert loaded.rows[0].view.candidate_rank == 1
```

Also add tests that duplicate `simfile_id`, duplicate `candidate_rank`, mixed source manifest/timing identity, mixed `review_ledger_sha256`, unknown enum values, and noncanonical JSONL are rejected.

- [ ] **Step 4: Run the schema/loader tests red**

```bash
uv run pytest tests/benchmark/test_schema_goldens.py tests/benchmark/test_reviewed_subset.py -q
```

Expected: FAIL because `src.benchmark.reviewed_subset` and its loader/validator do not exist.

- [ ] **Step 5: Implement the manifest types, exact-key validation, loader, and golden validator**

In `src/benchmark/reviewed_subset.py`, define closed types:

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

Define `ReviewedSubsetRowView`, `LoadedReviewedSubsetRow`, and `LoadedReviewedSubsetManifest`. The loaded manifest must expose at least:

```python
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

Implement the loader through:

```python
canonical = read_canonical_manifest_core(
    path,
    schema_version=REVIEWED_REFERENCE_SUBSET_SCHEMA,
    validate_rows=validate_rows,
)
```

Require one shared source-reference identity, timing identity, review-policy version, review-ledger hash, and prior-ledger hash across rows; unique candidate ranks and simfile IDs; accepted population 20–30 in normal loader mode. Keep the schema-golden validator's fixed one-row fixture rule separate from normal population validation, matching existing golden-validator patterns.

Expose:

```python
def validate_schema_golden(schema: str, content: bytes) -> None:
    ...
```

It must reject a schema argument other than `REVIEWED_REFERENCE_SUBSET_SCHEMA` and validate the canonical one-row golden against the same row-shape validator.

- [ ] **Step 6: Factor the reusable HPA-323/HPA-324 test fixture early**

Move the smallest existing canonical timing/reference/event builders needed by HPA-327 tests into `tests/benchmark/reviewed_subset_fixtures.py` under the interface above. Existing tests that owned those builders may import the shared helper; do not change their expected semantics. Support `eligible_count` by generating distinct simfile IDs and valid native event artifacts, and support `reverse_rows=True` by reversing input manifest row order before canonical publication.

- [ ] **Step 7: Run schema, fixture-owner, and reference tests**

```bash
uv run pytest tests/benchmark/test_schema_goldens.py tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reference_set_manifest.py tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit the schema rails and early fixture**

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
- Produces request/outcome:

```python
@dataclass(frozen=True)
class PrepareReviewedSubsetRequest:
    reference_manifest_path: Path
    timing_manifest_path: Path
    output_file: Path
    prior_ledger_path: Path | None = None


@dataclass(frozen=True)
class PrepareReviewedSubsetOutcome:
    exit_code: Literal[0, 2]
    output_file: Path | None
    candidate_count: int
    carried_include_count: int
    replacement_count: int
```

- Produces pure selection seams:

```python
def build_candidate_stream(
    reference_manifest: LoadedReferenceSetManifest,
    timing_manifest: LoadedReferenceTimingManifest,
    *,
    mappings: Mapping[int, ReferenceMappingResult | None],
) -> tuple[ReviewCandidate, ...]:
    ...


def prepare_reviewed_subset(
    request: PrepareReviewedSubsetRequest,
) -> PrepareReviewedSubsetOutcome:
    ...
```

- [ ] **Step 1: Write selection tests for the frozen population rules and anti-bias API**

Add tests using `build_reviewed_subset_reference_fixture()`:

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

Add separate tests for 20–29 selecting all and fewer than 20 returning exit 2 without producing a valid review file.

- [ ] **Step 2: Add deterministic feature/band/stratum tests**

Cover:

```python
assert candidate.common_event_count == loaded_reference.view.common_scored_event_count
assert candidate.density_band in {"low", "medium", "high"}
assert candidate.class_richness_band in {"low", "medium", "high"}
assert candidate.source_audio_cache_path == (
    f"sha256/{candidate.source_audio_content_hash[:2]}/{candidate.source_audio_content_hash}"
)
```

Create one fixture whose HPA-323 timing row contains a warning and assert `has_timing_warning is True`. Create `real.dtx`, `full.dtx`, and `mas.dtx` rows and assert only the first two set `selects_real_or_full_chart`.

Add a mismatch fixture where the HPA-324 row's `common_scored_event_count` differs from the reconstructed mapping and assert preparation exits 2.

- [ ] **Step 3: Add the lexical-bias regression**

Build a synthetic eligible population that populates more than 30 distinct strata/candidate positions. Assert the selected set is derived from seeded stratum hashes rather than lexical order by computing the expected first-round stratum order in the test:

```python
expected_order = sorted(
    nonempty_strata,
    key=lambda key: sha256(
        f"{REVIEW_SELECTION_SEED}:{canonical_stratum_key(key)}".encode()
    ).hexdigest(),
)
assert selected_first_round_strata == tuple(expected_order[: len(selected_first_round_strata)])
```

Also assert reversing manifest input rows produces byte-identical candidate `simfile_id` order.

- [ ] **Step 4: Run the selector tests red**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py -k "prepare or candidate or stratum or band" -q
```

Expected: FAIL because preparation/selection does not exist.

- [ ] **Step 5: Implement source-row hashing and feature extraction**

Use:

```python
def _source_row_sha256(source_row: Mapping[str, object]) -> str:
    payload = {
        key: value for key, value in source_row.items() if key != "corpus_version"
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()
```

Build a timing-row lookup from `LoadedReferenceTimingManifest.rows`, call `preflight_reference_mappings()` exactly once, require every eligible mapping to be non-`None`, and reconcile:

```python
if len(mapping.common_events) != loaded.view.common_scored_event_count:
    raise ValueError("reference common event count does not match HPA-324")
```

Compute span/density/class count only from `mapping.common_events`.

- [ ] **Step 6: Implement exact thirds and seeded round-robin**

Use one small helper:

```python
def _assign_bands(values: tuple[tuple[float, int], ...]) -> dict[int, Band]:
    ordered = sorted(values)
    count = len(ordered)
    labels: tuple[Band, Band, Band] = ("low", "medium", "high")
    return {
        simfile_id: labels[min(2, (index * 3) // count)]
        for index, (_, simfile_id) in enumerate(ordered)
    }
```

Canonical stratum key:

```python
def canonical_stratum_key(candidate: ReviewCandidate) -> str:
    return (
        f"{candidate.density_band}|{candidate.class_richness_band}|"
        f"{int(candidate.has_timing_warning)}|"
        f"{int(candidate.selects_real_or_full_chart)}"
    )
```

Order strata and members by:

```python
def _seeded_hash(value: str) -> str:
    return sha256(f"{REVIEW_SELECTION_SEED}:{value}".encode()).hexdigest()
```

Continue the stream beyond row 30 so continuation can draw deterministic replacements.

- [ ] **Step 7: Implement the exact review CSV boundary**

Define stable generated and manual column tuples. Generated columns are:

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

Manual columns are exactly:

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

Render floats with:

```python
canonical_json_bytes(quantize_six(value)).decode("ascii")
```

Use `csv.DictWriter(..., lineterminator="\n")`. Leave manual cells empty for new candidates.

- [ ] **Step 8: Add continuation tests before implementation**

Prepare an initial 30-row CSV, fill manual reviews so 24 unchanged rows are valid `include` and 6 are valid `exclude`, then call prepare again with `prior_ledger_path`.

Assert:

```python
assert outcome.carried_include_count == 24
assert outcome.replacement_count == 6
assert outcome.candidate_count == 30
assert not excluded_ids & current_ids
assert carried_ids <= current_ids
```

Mutate one source row before the continuation run and assert that row's prior review is not carried. Assert replacements are exactly the next unused IDs from `build_candidate_stream()`.

Malformed completed manual fields in the prior ledger must return exit 2 rather than silently dropping review evidence.

- [ ] **Step 9: Implement prior-ledger carry-forward**

Parse prior rows by canonical integer `simfile_id`. Read `source_row_sha256` only for the carry-forward guard. Validate completed manual review fields before carrying anything.

Rules:

```text
unchanged valid include -> carry manual fields
unchanged valid exclude -> consume and skip
changed source hash -> do not carry; row may reappear unreviewed
new/replacement row -> next unused deterministic stream candidate
```

Preserve carried included rows' relative order, append replacements in stream order, and assign fresh `candidate_rank = 1..N` for the new prepared CSV.

- [ ] **Step 10: Run all selection/continuation tests**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py -k "prepare or candidate or stratum or band or carry or prior" -q
```

Expected: PASS.

- [ ] **Step 11: Commit selection and preparation**

```bash
git add src/benchmark/reviewed_subset.py tests/benchmark/test_reviewed_subset.py tests/benchmark/reviewed_subset_fixtures.py
git commit -m "feat: prepare reviewed reference subset"
```

---

### Task 5: Validate manual reviews and publish the canonical subset + audit ledger

**Files:**
- Modify: `src/benchmark/reviewed_subset.py`
- Modify: `tests/benchmark/test_reviewed_subset.py`
- Create: `tests/benchmark/test_reviewed_subset_acceptance.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class FinalizeReviewedSubsetRequest:
    reference_manifest_path: Path
    timing_manifest_path: Path
    review_file: Path
    output_dir: Path
    prior_ledger_path: Path | None = None


@dataclass(frozen=True)
class FinalizeReviewedSubsetOutcome:
    exit_code: Literal[0, 2]
    manifest: PublishedManifest | None
    review_ledger_path: Path | None
    included_count: int
    excluded_count: int
```

```python
def finalize_reviewed_subset(
    request: FinalizeReviewedSubsetRequest,
) -> FinalizeReviewedSubsetOutcome:
    ...
```

- [ ] **Step 1: Write review-validation tests first**

Use a prepared CSV and fill manual columns. Add tests for:

```text
blank reviewer -> reject
invalid/non-UTC reviewed_at -> reject
invalid confirmation token -> reject
include + any false confirmation -> reject
include + not_representative -> reject
exclude + empty reason_codes -> reject
reason_codes containing other + empty notes -> reject
unknown reason -> reject
fewer than 20 includes -> reject
20–30 valid includes -> publishable
```

Use exact confirmation strings `true` / `false`, exact decision strings `include` / `exclude`, and semicolon-separated review reason tokens in the editing CSV.

- [ ] **Step 2: Test that generated spreadsheet cells are non-authoritative**

After preparation, deliberately rewrite generated cells such as density, hash display, booleans, and candidate rank while leaving `simfile_id` and manual review fields intact. Finalization must regenerate current evidence and still succeed when membership is current.

Then change a `simfile_id` to a stale/unknown member and assert finalization exits 2.

- [ ] **Step 3: Test continuation finalization requires the same prior ledger**

Prepare a continuation CSV with `prior_ledger_path`. Assert:

```python
without_prior = finalize_reviewed_subset(
    replace(finalize_request, prior_ledger_path=None)
)
assert without_prior.exit_code == 2

with_prior = finalize_reviewed_subset(finalize_request)
assert with_prior.exit_code == 0
```

This pins replacement-membership reproducibility rather than trusting editable generated cells.

- [ ] **Step 4: Run finalization tests red**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py -k "finalize or review" -q
```

Expected: FAIL because finalization is not implemented.

- [ ] **Step 5: Implement one closed manual-review parser**

Represent a completed review with a frozen dataclass and parse `reason_codes` into a sorted unique tuple. Validate all closed enums and RFC3339 UTC timestamps. Keep `known_limitations` / `notes` as strings. `other` requires nonempty notes.

Do not validate generated CSV display values as source identity.

- [ ] **Step 6: Reproduce current membership from authoritative inputs**

Finalization must rerun the same prepare selection logic in memory using the same optional prior ledger. Build `expected_by_id` from the reproduced slate, then require the submitted CSV to contain exactly one manual review for every expected `simfile_id` and no extra IDs.

Current source identity/features come from `expected_by_id`, not from submitted generated cells.

- [ ] **Step 7: Render and durably write the canonical complete audit ledger**

Re-render `review-ledger.csv` from fresh generated fields + validated manual fields using UTF-8, stable header order, `\n`, and `csv.DictWriter` quoting. Write it beneath `output_dir` with the existing shared durable byte-replacement helper.

Compute:

```python
review_ledger_sha256 = sha256(review_ledger_bytes).hexdigest()
prior_review_ledger_sha256 = (
    sha256(request.prior_ledger_path.read_bytes()).hexdigest()
    if request.prior_ledger_path is not None
    else None
)
```

- [ ] **Step 8: Publish accepted rows on existing manifest rails**

For each `include`, emit the exact reviewed-subset row contract from Task 3. Convert manual `reason_codes` to a JSON array. Carry selection evidence:

```text
common_event_count
reference_event_span_sec
common_event_density_per_sec
common_class_count
density_band
class_richness_band
has_timing_warning
selects_real_or_full_chart
```

Render/publish through:

```python
rendered = render_manifest(tuple(rows))
published = publish_manifest(request.output_dir, rendered)
publish_latest_manifest(request.output_dir, published, "complete", clock())
```

Bind `review_ledger_sha256` and `prior_review_ledger_sha256` into every accepted row.

- [ ] **Step 9: Add loader round-trip and ledger-hash acceptance assertions**

In `tests/benchmark/test_reviewed_subset_acceptance.py`, exercise prepare -> fill reviews -> finalize and assert:

```python
loaded = load_reviewed_subset_manifest(outcome.manifest.path)
assert 20 <= len(loaded.rows) <= 30
assert loaded.review_ledger_sha256 == sha256(
    outcome.review_ledger_path.read_bytes()
).hexdigest()
assert loaded.prior_review_ledger_sha256 is None
```

Add a continuation variant and assert its `prior_review_ledger_sha256` matches the supplied prior ledger bytes.

- [ ] **Step 10: Run finalization/schema acceptance**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py tests/benchmark/test_schema_goldens.py -q
```

Expected: PASS.

- [ ] **Step 11: Commit review finalization/publication**

```bash
git add src/benchmark/reviewed_subset.py tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py
git commit -m "feat: publish reviewed reference subset"
```

---

### Task 6: Rescore persisted OaF predictions on the reviewed subset with diagnostics

**Files:**
- Modify: `src/benchmark/reviewed_subset.py`
- Modify: `tests/benchmark/reviewed_subset_fixtures.py`
- Modify: `tests/benchmark/test_reviewed_subset.py`
- Modify: `tests/benchmark/test_reviewed_subset_acceptance.py`
- Verify unchanged: `tests/benchmark/test_oaf_corpus_run_acceptance.py`
- Verify unchanged: `tests/benchmark/test_cohort_scoring_acceptance.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class ScoreReviewedSubsetRequest:
    run_path: Path
    reference_manifest_path: Path
    timing_manifest_path: Path
    subset_manifest_path: Path
    output_dir: Path


@dataclass(frozen=True)
class ScoreReviewedSubsetOutcome:
    exit_code: Literal[0, 1, 2]
    cohort_id: str | None
    reports_path: Path | None
    success_count: int
    failed_count: int
    skipped_count: int
    quarantined_count: int
```

```python
def score_oaf_reviewed_subset(
    request: ScoreReviewedSubsetRequest,
) -> ScoreReviewedSubsetOutcome:
    ...
```

- To keep preparation model-independent, import `parse_oaf_corpus_run` and `build_oaf_cohort_from_snapshot` lazily inside `score_oaf_reviewed_subset()` instead of at `reviewed_subset.py` module import time.

- [ ] **Step 1: Extend the reusable synthetic fixture with a persisted OaF run**

Add a second fixture dataclass/interface in `tests/benchmark/reviewed_subset_fixtures.py` by factoring the smallest existing persisted-run construction from HPA-326 acceptance tests:

```python
@dataclass(frozen=True)
class ReviewedSubsetOafFixture:
    reference_manifest_path: Path
    timing_manifest_path: Path
    timing_output_root: Path
    run_path: Path
    oaf_output_dir: Path
```

The fixture must include at least successful items and one selectable non-success item, with valid immutable prediction artifacts for successes. It must not invoke Docker, TensorFlow, network, or a real backend.

- [ ] **Step 2: Write the no-inference subset-score test**

Finalize a valid subset, then monkeypatch backend creation to fail if touched and score it:

```python
def fail_backend(*args: object, **kwargs: object) -> object:
    raise AssertionError("reviewed subset scoring must not construct OafBackend")

monkeypatch.setattr("src.benchmark.oaf_corpus_run.create_backend", fail_backend)
outcome = score_oaf_reviewed_subset(request)
assert outcome.exit_code in {0, 1}
```

Also capture hashes of the parent run's existing report files before/after and assert they are unchanged.

- [ ] **Step 3: Add identity/membership failure tests**

Assert exit 2 for:

```text
run reference-manifest SHA mismatch
run timing-manifest SHA/version mismatch
subset source-reference identity mismatch
subset source-timing identity mismatch
subset simfile missing from parent run population
noncanonical run snapshot
unreadable prediction artifact needed by a selected success row
```

Do not silently drop any selected failed/skipped/quarantined parent item.

- [ ] **Step 4: Add the diagnostics hook regression**

Patch/wrap `score_cohort` in `reviewed_subset.py` and capture `diagnostics_for`. Assert it equals the sorted successful selected IDs only:

```python
assert diagnostics_for == tuple(
    sorted(item.simfile_id for item in selected_items if item.status == "success")
)
```

Then assert `event_diagnostics.jsonl` is nonempty for a fixture with at least one successful selected song containing scoreable events.

- [ ] **Step 5: Run rescore tests red**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py -k "score or rescore or diagnostic" -q
```

Expected: FAIL because subset scoring is absent.

- [ ] **Step 6: Implement exact persisted-run reconstruction and lineage checks**

Inside `score_oaf_reviewed_subset()`:

```python
from src.benchmark.oaf_corpus_run import (
    build_oaf_cohort_from_snapshot,
    parse_oaf_corpus_run,
)
```

Read `request.run_path`, parse canonically, load HPA-324/HPA-323, call `preflight_reference_mappings()`, and verify snapshot reference/timing hashes/versions against those loaded manifests.

Load the subset with `load_reviewed_subset_manifest()` and require its source manifest/timing identities to match the same loaded HPA-324/HPA-323 artifacts.

Reconstruct through:

```python
parent_identity, parent_items = build_oaf_cohort_from_snapshot(
    snapshot,
    mappings=mappings,
    output_dir=request.run_path.parents[2],
)
```

Do not call `run_oaf_corpus()`.

- [ ] **Step 7: Filter exact membership while preserving HPA-325 canonical ordering**

Build a set of subset IDs, require every ID in the parent population, and select the matching `CohortItem`s. Do not sort by `candidate_rank`; `score_cohort()` owns canonical item order.

Derive:

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

- [ ] **Step 8: Reuse HPA-325 scoring/reporting with successful-song diagnostics**

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

Set exit 0 when every selected item succeeds, exit 1 when item-level failed/skipped/quarantined states remain, and exit 2 for identity/artifact/report failure.

- [ ] **Step 9: Run subset + unchanged broad acceptance**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py tests/benchmark/test_oaf_corpus_run_acceptance.py tests/benchmark/test_cohort_scoring_acceptance.py -q
```

Expected: PASS. Broad reports remain unchanged; reviewed subset reports contain bounded diagnostics.

- [ ] **Step 10: Commit persisted subset rescoring**

```bash
git add src/benchmark/reviewed_subset.py tests/benchmark/reviewed_subset_fixtures.py tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py
git commit -m "feat: score persisted OaF reviewed subset"
```

---

### Task 7: Wire the three thin CLI commands and run end-to-end/repository verification

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Modify: `tests/benchmark/test_reviewed_subset_acceptance.py`

**Interfaces:**
- Adds commands:

```text
crux benchmark prepare-reviewed-subset
crux benchmark finalize-reviewed-subset
crux benchmark score-oaf-reviewed-subset
```

- CLI callbacks only construct request dataclasses, call domain functions, render concise canonical JSON, and exit with the domain outcome's 0/1/2 code.

- [ ] **Step 1: Write CLI help/option tests first**

Assert exact command availability and required option names:

```text
prepare-reviewed-subset:
  --manifest
  --timing-manifest
  --output-file
  --prior-ledger (optional)

finalize-reviewed-subset:
  --manifest
  --timing-manifest
  --review-file
  --output-dir
  --prior-ledger (optional)

score-oaf-reviewed-subset:
  --run
  --manifest
  --timing-manifest
  --subset-manifest
  --output-dir
```

Do not add `--seed`, `--target-count`, model IDs, score thresholds, or backend-selection flags.

- [ ] **Step 2: Add callback forwarding tests**

Monkeypatch each domain function and assert the Click callback builds exactly the request dataclass fields above. For prepare/finalize, verify omitted `--prior-ledger` becomes `None`.

Assert JSON output includes stable operator-useful facts only:

```text
prepare: exit_code, candidate_count, carried_include_count, replacement_count, output_file
finalize: exit_code, included_count, excluded_count, manifest_path, review_ledger_path
score: exit_code, cohort_id, success_count, failed_count, skipped_count, quarantined_count, reports_path
```

Render float-derived values, if any are later added, through `quantize_six()` before canonical JSON; v1 outcomes above need no floats.

- [ ] **Step 3: Run CLI tests red**

```bash
uv run pytest tests/test_cli_benchmark.py -k "reviewed_subset" -q
```

Expected: FAIL because commands are not registered.

- [ ] **Step 4: Implement the three Click callbacks**

Follow existing benchmark command option/path conventions (`click.Path(path_type=Path, ...)`, `click.IntRange` where already used elsewhere). Domain code stays in `src/benchmark/reviewed_subset.py`.

Each callback should emit one canonical JSON object and then:

```python
if outcome.exit_code:
    raise click.exceptions.Exit(outcome.exit_code)
```

Do not catch domain failures and convert them to success.

- [ ] **Step 5: Make the synthetic acceptance chain runnable end-to-end**

In `tests/benchmark/test_reviewed_subset_acceptance.py`, use the reusable fixture to execute domain-level:

```text
prepare -> fill manual fields -> finalize -> score persisted OaF subset
```

Then exercise the same flow through `CliRunner` and compare the resulting subset manifest/review-ledger/report bytes with the domain-level expectations where paths are normalized.

This test must remain offline and must not construct `OafBackend`.

- [ ] **Step 6: Run all HPA-327 and touched-seam tests**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py tests/benchmark/test_reference_set_manifest.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py tests/benchmark/test_cohort_scoring_acceptance.py tests/benchmark/test_schema_goldens.py tests/test_cli_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 7: Run repository-wide CI-equivalent verification**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check
```

Expected: all pass. Do not weaken coverage or lint gates to land HPA-327.

- [ ] **Step 8: Commit CLI and acceptance wiring**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py tests/benchmark/test_reviewed_subset_acceptance.py
git commit -m "feat: add reviewed subset benchmark commands"
```

---

## Operational Completion Gate (after code implementation)

The repository does not contain the user's real HPA-323/HPA-324 local artifacts or the completed human review evidence, so implementation must not fabricate this gate.

After the implementation lands in an environment with the real manifests/cache, complete HPA-327 in this order:

1. Run `prepare-reviewed-subset` against the exact HPA-324 manifest and its matching HPA-323 timing manifest **before using OaF song-level scores to make membership decisions**.
2. Preserve the initial review CSV and manually inspect every candidate's selected chart, matching full mix, BGM alignment, mapping, and musical fidelity.
3. Fill all 12 manual review fields.
4. If fewer than 20 are acceptable or pre-score coverage is clearly inadequate, rerun prepare with `--prior-ledger` to preserve unchanged includes and draw only deterministic unused replacements; continue review without consulting model scores.
5. Finalize with the same prior ledger used for that continuation pass. Preserve the canonical `review-ledger.csv` and `crux.reviewed-reference-subset/v1` manifest.
6. Confirm the published accepted subset itself demonstrates materially different density/class-richness bands, timing-warning states, `real`/`full` chart selection, and manually recorded drum character.
7. Run `score-oaf-reviewed-subset` against the existing HPA-326 `run.json`; do not rerun OaF inference.
8. Preserve reviewed-subset reports, including event diagnostics for successful reviewed songs, without modifying the broad HPA-326 run or reports.
9. Mark HPA-327 Done in Linear only after this real 20–30-song audit evidence exists.

## Plan Self-Review Checklist

Before implementation begins, verify these invariants remain true in this plan and the design spec:

- Every new durable schema has one loader and one golden; no call site hand-parses subset JSONL.
- Candidate preparation has no model/run/prediction/report/score parameter or import dependency on OaF execution code.
- The two HPA-326 changes are independently characterized no-behavior-change refactors before HPA-327 depends on them.
- Continuation selection is reproducible because prepare and finalize receive the same prior ledger and the final manifest binds its hash.
- CSV generated cells never become authority; source identities/features are re-derived from HPA-323/HPA-324.
- HPA-325 remains the only scorer/report path, preserves `simfile_id` ordering, and receives successful reviewed IDs through `diagnostics_for`.
- No task adds training, chart repair, UI, DB, generic runners, sampling configuration, or backward-compatibility machinery.
