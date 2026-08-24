# HPA-562 Paired Benchmark Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one deterministic HPA-562 bundle that composes the existing OaF/MuScriptor, OaF/separation, and OaF/IDM pairwise comparisons and exposes a clearly scoped model × input-view population matrix.

**Architecture:** Keep the three existing pairwise comparison drivers authoritative. Add one concrete `cross_comparison.py` coordinator that stages those outputs, validates shared reference/scoring identity plus frozen OaF model-lock and prediction-map identity, verifies a closed nested-artifact contract, and publishes only a top-level index/matrix. MuScriptor stays broad with `subset_manifest_path=None`; separation owns the supplied HPA-327 subset identity; IDM remains pilot-scoped by its validated HPA-396 run without adding another HPA-328 handoff input.

**Tech Stack:** Python 3.12, dataclasses, `pathlib`, existing strict/canonical JSON and SHA-256 helpers, Click (>=8.2; the CLI tests assert on `CliRunner`'s separately captured stderr), pytest, pytest-cov.

> Python-version note: this plan pins 3.12 to match `pyproject.toml` (`requires-python = "==3.12.*"`), the ruff/black/pylint `py312` targets, and CI's `python-version: '3.12'`. AGENTS.md's "Python 3.13" prose is stale and is intentionally not touched by this one-ticket PR.

**Spec:** `docs/superpowers/specs/2026-08-22-hpa-562-paired-comparisons-design.md`

## Global Constraints

- Keep all HPA-562 planning and implementation on `agent/hpa-562-paired-comparisons` and draft PR #29; do not open a second HPA-562 PR.
- Reuse `compare_oaf_muscriptor()`, `compare_oaf_separation()`, and `compare_oaf_idm()`; do not rebuild pair joins or rescore events.
- MuScriptor HPA-562 comparison is always broad full mix: construct `ComparisonRequest(..., subset_manifest_path=None)` even though the coordinator request carries the pilot subset for HPA-328.
- Same-input model comparisons keep exact input-audio hash matching in their existing drivers; cross-input OaF separation keeps HPA-328's source-hash + fixed-view contract.
- Full-corpus and pilot populations remain explicitly distinct; the headline matrix is not a leaderboard.
- Headline rows come from one closed `(scope, model, expected view, comparison id, models-key)` table; never select by a generic `"oaf"` lookup.
- Require OaF `model_lock_sha256` equality across MuScriptor, separation, and IDM summaries.
- Require nonempty equal OaF `prediction_map_version` across the five model entries checked by `_validate_oaf_identity()`: MuScriptor `oaf`, separation `full_mix`/`spleeter`/`htdemucs`, IDM `oaf`.
- Require separation HTDemucs and IDM's OaF peer to name the same HTDemucs stem input view.
- Do not publish a top-level reviewed-subset identity. Separation owns its HPA-327 subset identity; IDM is explicitly marked as not cross-verified against that subset at HPA-562 level.
- Nested artifact composition is a closed contract, not a glob. Missing or unexpected files fail closed.
- Persist only HPA-562-root-relative paths and use existing `read_regular_file_no_follow()` + `sha256_hex()` for hashes.
- `pairable_success_counts` has exactly four flat keys defined in Task 3 and is sourced from validated nested summaries, not heterogeneous driver return values.
- Do not add backward compatibility for old comparison summaries or historical comparison hashes.
- Do not add a generic comparison registry, experiment runner, leaderboard service, database, UI, significance layer, or automatic winner policy.

---

## File structure

```text
Create
  src/benchmark/cross_comparison.py
  tests/benchmark/test_cross_comparison.py

Modify
  src/benchmark/muscriptor_comparison.py
  src/benchmark/separation_comparison.py
  tests/benchmark/test_muscriptor_comparison.py
  tests/benchmark/test_separation_comparison.py
  src/cli/benchmark.py
  tests/test_cli_benchmark.py
```

`cross_comparison.py` owns only bundle orchestration, cross-summary validation, the closed matrix source table, closed artifact index, top-level rendering, and final publication. Existing pairwise modules continue to own pair construction and domain-specific integrity checks.

---

### Task 1: Complete the pairwise comparison identity HPA-562 needs

**Files:**
- Modify: `src/benchmark/muscriptor_comparison.py`
- Modify: `src/benchmark/separation_comparison.py`
- Test: `tests/benchmark/test_muscriptor_comparison.py`
- Test: `tests/benchmark/test_separation_comparison.py`

**Interfaces:**
- Consumes: `SCORING_VERSION`, `TAXONOMY_VERSION`, `DTX_LANE_MAP_VERSION`.
- Produces: MuScriptor and separation comparison `summary.json` identity mappings with the common key set HPA-562 validates.

- [ ] **Step 1: Add failing MuScriptor identity assertions**

In the existing successful `compare_oaf_muscriptor()` test, parse `summary.json` and add:

```python
assert summary["identity"]["taxonomy_version"] == TAXONOMY_VERSION
assert summary["identity"]["lane_map_version"] == DTX_LANE_MAP_VERSION
assert summary["identity"]["scoring_version"] == SCORING_VERSION
```

- [ ] **Step 2: Run the focused MuScriptor suite and observe RED**

```bash
uv run pytest tests/benchmark/test_muscriptor_comparison.py -q
```

Expected: taxonomy/lane assertions fail. `scoring_version` may already be present through the current default identity path; the RED gate is specifically the missing taxonomy/lane fields.

- [ ] **Step 3: Persist explicit MuScriptor comparison identity at the real call site**

`muscriptor_comparison.py` currently imports `_summary` as the private migration alias of `published_comparison.comparison_summary`. Do **not** rename or replace that seam in HPA-562.

Add:

```python
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, TAXONOMY_VERSION
```

Then change the final `_summary(...)` call inside `compare_oaf_muscriptor()` to pass its existing keyword-only `identity=` argument:

```python
summary = _summary(
    oaf,
    muscriptor,
    pairable_ids,
    exclusions,
    song_rows,
    class_rows,
    reference_manifest,
    timing_manifest,
    request.subset_manifest_path,
    subset_manifest,
    identity={
        "reference_manifest_sha256": reference_manifest.manifest_sha256,
        "reference_manifest_version": reference_manifest.corpus_version,
        "reference_timing_manifest_sha256": timing_manifest.manifest_sha256,
        "reference_timing_version": timing_manifest.corpus_version,
        "taxonomy_version": TAXONOMY_VERSION,
        "lane_map_version": DTX_LANE_MAP_VERSION,
        "input_view_id": oaf.identity.input_view_id,
        "scoring_version": SCORING_VERSION,
    },
)
```

Do not change pairing, model entries, output filenames, or schema.

- [ ] **Step 4: Add failing separation identity assertions**

Extend the successful separation comparison test with:

```python
assert summary["identity"]["taxonomy_version"] == TAXONOMY_VERSION
assert summary["identity"]["lane_map_version"] == DTX_LANE_MAP_VERSION
assert summary["identity"]["scoring_version"] == SCORING_VERSION
```

- [ ] **Step 5: Run the focused separation suite and observe RED**

```bash
uv run pytest tests/benchmark/test_separation_comparison.py -q
```

Expected: taxonomy/lane assertions fail. `separation_comparison._comparison_identity()` already emits `scoring_version`, so that assertion is characterization, not evidence of the new change.

- [ ] **Step 6: Extend `_comparison_identity()` only with taxonomy/lane**

Add existing constants to the returned mapping:

```python
return {
    "run_id": snapshot["run_id"],
    "parent_oaf_run_id": snapshot["parent_oaf_run_id"],
    "reference_manifest_sha256": getattr(reference_manifest, "manifest_sha256"),
    "reference_manifest_version": getattr(reference_manifest, "corpus_version"),
    "reference_timing_manifest_sha256": getattr(timing_manifest, "manifest_sha256"),
    "reference_timing_version": getattr(timing_manifest, "corpus_version"),
    "reviewed_subset_manifest_sha256": getattr(subset_manifest, "manifest_sha256"),
    "input_views": dict(_VIEW_IDS),
    "taxonomy_version": TAXONOMY_VERSION,
    "lane_map_version": DTX_LANE_MAP_VERSION,
    "scoring_version": SCORING_VERSION,
}
```

- [ ] **Step 7: Pin the intentional separation-summary hash change in documentation/test naming**

Add no compatibility code. Add or rename a focused test/comment so it is explicit that a newly generated separation `summary.json` now contains taxonomy/lane and therefore may not hash to the same bytes as an older HPA-328 handoff-recorded comparison summary.

The current HPA-562 contract is authoritative; old comparison-summary hashes are not a compatibility target.

- [ ] **Step 8: Run both focused suites GREEN**

```bash
uv run pytest \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_separation_comparison.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add \
  src/benchmark/muscriptor_comparison.py \
  src/benchmark/separation_comparison.py \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_separation_comparison.py
git commit -m "refactor: expose complete comparison identity"
```

---

### Task 2: Add the concrete coordinator, scope routing, and fail-closed identity validation

**Files:**
- Create: `src/benchmark/cross_comparison.py`
- Create: `tests/benchmark/test_cross_comparison.py`

**Interfaces:**
- Consumes:
  - `compare_oaf_muscriptor(ComparisonRequest) -> ComparisonOutcome`
  - `compare_oaf_separation(SeparationComparisonRequest) -> SeparationComparisonOutcome`
  - `compare_oaf_idm(IdmComparisonRequest) -> Path`
  - `read_regular_file_no_follow(path) -> bytes`
  - `strict_json_loads(..., require_canonical=True)`
  - `ComparisonIntegrityError`
- Produces:
  - `PAIRED_BENCHMARK_PUBLICATION_SCHEMA = "crux.paired-benchmark-publication/v1"`
  - `CrossComparisonRequest`
  - `CrossComparisonOutcome`
  - `publish_cross_comparisons()`

The public dataclasses are:

```python
@dataclass(frozen=True)
class CrossComparisonRequest:
    oaf_run_path: Path
    muscriptor_run_path: Path
    separation_run_path: Path
    idm_run_path: Path
    reference_manifest_path: Path
    timing_manifest_path: Path
    subset_manifest_path: Path
    output_dir: Path
    separation_cache_dir: Path | None = None


@dataclass(frozen=True)
class CrossComparisonOutcome:
    output_dir: Path
    headline_matrix_path: Path
    comparison_paths: dict[str, Path]
    pairable_success_counts: dict[str, int]
```

Both frozen dataclasses enforce these constraints at the runtime boundary in `__post_init__`: Path-only request fields, `separation_cache_dir` as `Path | None`, non-empty comparison paths keyed by non-empty strings, and counts that are non-negative `int`s with `bool` rejected. Step 1 pins each constraint with direct `TypeError`/`ValueError` tests.

- [ ] **Step 1: Write request/outcome validation tests**

Pin Path-only request fields, optional `separation_cache_dir`, non-empty comparison paths, and non-negative integer counts. Reject booleans as counts.

Example:

```python
def test_cross_comparison_request_requires_paths(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="oaf_run_path"):
        CrossComparisonRequest(
            oaf_run_path="run.json",  # type: ignore[arg-type]
            muscriptor_run_path=tmp_path / "muscriptor.json",
            separation_run_path=tmp_path / "separation.json",
            idm_run_path=tmp_path / "idm.json",
            reference_manifest_path=tmp_path / "reference.jsonl",
            timing_manifest_path=tmp_path / "timing.jsonl",
            subset_manifest_path=tmp_path / "subset.jsonl",
            output_dir=tmp_path / "out",
        )
```

- [ ] **Step 2: Run the new suite and observe import RED**

```bash
uv run pytest tests/benchmark/test_cross_comparison.py -q
```

Expected: import failure because `src.benchmark.cross_comparison` does not exist.

- [ ] **Step 3: Implement fixed comparison IDs and exact request routing**

Define:

```python
_COMPARISON_DIRS = {
    "oaf_muscriptor_full_mix": Path("comparisons/oaf-muscriptor"),
    "oaf_separation_pilot": Path("comparisons/oaf-separation"),
    "oaf_idm_htdemucs": Path("comparisons/oaf-idm"),
}
```

Inside the staging directory, construct exactly:

```python
muscriptor_request = ComparisonRequest(
    oaf_run_path=request.oaf_run_path,
    muscriptor_run_path=request.muscriptor_run_path,
    reference_manifest_path=request.reference_manifest_path,
    timing_manifest_path=request.timing_manifest_path,
    output_dir=stage / _COMPARISON_DIRS["oaf_muscriptor_full_mix"],
    subset_manifest_path=None,
)

separation_request = SeparationComparisonRequest(
    run_path=request.separation_run_path,
    reference_manifest_path=request.reference_manifest_path,
    timing_manifest_path=request.timing_manifest_path,
    subset_manifest_path=request.subset_manifest_path,
    output_dir=stage / _COMPARISON_DIRS["oaf_separation_pilot"],
    cache_dir=request.separation_cache_dir,
)

idm_request = IdmComparisonRequest(
    run_path=request.idm_run_path,
    output_dir=stage / _COMPARISON_DIRS["oaf_idm_htdemucs"],
)
```

Reject an existing final `output_dir`. Use one sibling `TemporaryDirectory`; only rename the complete staged bundle into `output_dir` after all checks/writes pass.

- [ ] **Step 4: Add delegation tests that pin scope routing**

Monkeypatch only the three driver call sites. Each fake records its request and writes a canonical minimal summary.

Assert:

```python
assert muscriptor_request.subset_manifest_path is None
assert separation_request.subset_manifest_path == request.subset_manifest_path
assert separation_request.cache_dir == request.separation_cache_dir
assert idm_request.run_path == request.idm_run_path
```

Also assert each driver is called exactly once.

- [ ] **Step 5: Implement strict canonical summary loading**

Use existing regular-file and canonical JSON helpers:

```python
def _read_summary(path: Path, *, expected_schema: str) -> Mapping[str, object]:
    try:
        content = read_regular_file_no_follow(path)
        value = strict_json_loads(content, require_canonical=True)
    except (OSError, StrictJsonError) as error:
        raise ComparisonIntegrityError(f"invalid comparison summary: {error}") from error
    if not isinstance(value, Mapping) or value.get("schema") != expected_schema:
        raise ComparisonIntegrityError("comparison summary schema mismatch")
    return value
```

Use `COMPARISON_SCHEMA`, `SEPARATION_COMPARISON_SCHEMA`, and `IDM_COMPARISON_SCHEMA`; do not accept arbitrary schemas.

- [ ] **Step 6: Implement shared reference/scoring identity validation**

Compare these exact fields across all three nested summary identities:

```python
_SHARED_FIELDS = (
    "reference_manifest_sha256",
    "reference_manifest_version",
    "reference_timing_manifest_sha256",
    "reference_timing_version",
    "taxonomy_version",
    "lane_map_version",
    "scoring_version",
)
```

For each field, require a non-empty value and exact equality. Raise `ComparisonIntegrityError` naming the field on missing/mismatch.

- [ ] **Step 7: Implement frozen OaF lock/map and HTDemucs-view checks**

Load exact model entries through a safe accessor so missing or malformed nested entries raise `ComparisonIntegrityError` instead of leaking `KeyError`:

```python
muscriptor_oaf = _model(muscriptor_summary, "oaf")
separation_full_mix = _model(separation_summary, "full_mix")
idm_oaf = _model(idm_summary, "oaf")
```

`_model(summary, key)` requires `summary["models"]` to be a mapping and the entry itself to be a mapping, raising `ComparisonIntegrityError("comparison summary models[...] is malformed")` otherwise.

Require all five OaF/separation model entries (`muscriptor_oaf`, `separation_full_mix`, separation `spleeter` and `htdemucs`, `idm_oaf`) to carry valid SHA-256 `model_lock_sha256` values, all equal. Require the same five entries to carry nonempty string `prediction_map_version` values, also equal; a missing or malformed value is a fatal error naming `prediction_map_version`. Then require:

```python
separation_view = separation_summary["models"]["htdemucs"]["input_view_id"]
idm_view = idm_oaf["input_view_id"]
```

with:

```python
if separation_view != HTDEMUCS_INPUT_VIEW_ID:
    raise ComparisonIntegrityError("HTDemucs input_view_id mismatch")
if idm_view != IDM_STEM_INPUT_VIEW_ID:
    raise ComparisonIntegrityError("HTDemucs input_view_id mismatch")
if separation_view != idm_view:
    raise ComparisonIntegrityError("HTDemucs input_view_id mismatch")
```

Do not add a new hash/input identity mechanism; these are cross-summary consistency checks only.

- [ ] **Step 8: Add mismatch tests for every cross-summary gate**

Parametrize shared fields:

```python
@pytest.mark.parametrize(
    "field",
    (
        "reference_manifest_sha256",
        "reference_manifest_version",
        "reference_timing_manifest_sha256",
        "reference_timing_version",
        "taxonomy_version",
        "lane_map_version",
        "scoring_version",
    ),
)
def test_cross_publication_rejects_shared_identity_mismatch(..., field: str) -> None:
    with pytest.raises(ComparisonIntegrityError, match=field):
        publish_cross_comparisons(request)
    assert not request.output_dir.exists()
```

Add separate tests for malformed/mismatched OaF `model_lock_sha256`, mismatched OaF `prediction_map_version`, mismatched HTDemucs `input_view_id`, malformed nested `models` mappings (including valid mappings whose required model keys are absent), and an existing final output directory. Every failure leaves no final HPA-562 bundle, so the CLI emits its canonical exit-2 integrity payload.

- [ ] **Step 9: Run the focused suite GREEN**

```bash
uv run pytest tests/benchmark/test_cross_comparison.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/benchmark/cross_comparison.py tests/benchmark/test_cross_comparison.py
git commit -m "feat: compose paired benchmark comparisons"
```

---

### Task 3: Publish the closed matrix, closed artifact index, scoped lineage notes, and four pair counts

**Files:**
- Modify: `src/benchmark/cross_comparison.py`
- Modify: `tests/benchmark/test_cross_comparison.py`

**Interfaces:**
- Consumes: validated nested summaries from Task 2.
- Produces:
  - `headline_matrix.csv`
  - `summary.json`
  - `summary.md`
  - closed nested artifact path/hash index
  - exact four-key `pairable_success_counts`

- [ ] **Step 1: Freeze the matrix source table**

Define one closed tuple:

```python
_HEADLINE_SOURCES = (
    (
        "broad_full_mix",
        "oaf",
        OAF_FULL_MIX_INPUT_VIEW_ID,
        "oaf_muscriptor_full_mix",
        "oaf",
    ),
    (
        "broad_full_mix",
        "muscriptor",
        MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID,
        "oaf_muscriptor_full_mix",
        "muscriptor",
    ),
    (
        "reviewed_pilot",
        "oaf",
        OAF_FULL_MIX_INPUT_VIEW_ID,
        "oaf_separation_pilot",
        "full_mix",
    ),
    (
        "reviewed_pilot",
        "oaf",
        SPLEETER_INPUT_VIEW_ID,
        "oaf_separation_pilot",
        "spleeter",
    ),
    (
        "reviewed_pilot",
        "oaf",
        HTDEMUCS_INPUT_VIEW_ID,
        "oaf_separation_pilot",
        "htdemucs",
    ),
    (
        "reviewed_pilot",
        "idm",
        IDM_STEM_INPUT_VIEW_ID,
        "oaf_idm_htdemucs",
        "idm",
    ),
)
```

Never source a headline row from IDM `models["oaf"]`.

- [ ] **Step 2: Write the sentinel population mapping test**

Give every possible source model entry a unique population, including an intentionally wrong sentinel:

```text
MuScriptor models["oaf"]        total_count = 101
MuScriptor models["muscriptor"] total_count = 102
separation models["full_mix"]   total_count = 201
separation models["spleeter"]   total_count = 202
separation models["htdemucs"]   total_count = 203
IDM models["idm"]               total_count = 301
IDM models["oaf"]               total_count = 999
```

Assert exact row order/counts:

```python
assert [(row["scope"], row["model"], int(row["total_count"])) for row in rows] == [
    ("broad_full_mix", "oaf", 101),
    ("broad_full_mix", "muscriptor", 102),
    ("reviewed_pilot", "oaf", 201),
    ("reviewed_pilot", "oaf", 202),
    ("reviewed_pilot", "oaf", 203),
    ("reviewed_pilot", "idm", 301),
]
```

- [ ] **Step 3: Freeze the closed nested-artifact contract**

Define:

```python
_EXPECTED_ARTIFACTS = {
    "oaf_muscriptor_full_mix": (
        "summary.json",
        "summary.md",
        "paired_per_song.csv",
        "paired_per_class.csv",
    ),
    "oaf_separation_pilot": (
        "summary.json",
        "summary.md",
        "spleeter/paired_per_song.csv",
        "spleeter/paired_per_class.csv",
        "htdemucs/paired_per_song.csv",
        "htdemucs/paired_per_class.csv",
    ),
    "oaf_idm_htdemucs": (
        "summary.json",
        "summary.md",
        "paired_per_song.csv",
        "paired_per_class.csv",
    ),
}
```

Do not glob by suffix to decide what gets indexed.

- [ ] **Step 4: Test missing and unexpected nested artifacts fail closed**

For each comparison fake, create exactly its expected files. Then add two tests:

```python
def test_cross_publication_rejects_missing_expected_artifact(...):
    (stage_source / "paired_per_class.csv").unlink()
    with pytest.raises(ComparisonIntegrityError, match="paired_per_class.csv"):
        publish_cross_comparisons(request)


def test_cross_publication_rejects_unexpected_artifact(...):
    (stage_source / "extra.csv").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ComparisonIntegrityError, match="unexpected comparison artifact"):
        publish_cross_comparisons(request)
```

Implementation compares the actual regular-file relative-path set against the expected set, then hashes each expected file using:

```python
sha256_hex(read_regular_file_no_follow(path))
```

- [ ] **Step 5: Define and test the four pairable-success count keys**

Use exactly:

```python
_PAIR_COUNT_SOURCES = {
    "oaf_muscriptor_full_mix": (
        "oaf_muscriptor_full_mix",
        ("pairing", "pairable_success_intersection"),
    ),
    "oaf_separation_pilot.spleeter": (
        "oaf_separation_pilot",
        ("pairing", "spleeter", "pairable_success_intersection"),
    ),
    "oaf_separation_pilot.htdemucs": (
        "oaf_separation_pilot",
        ("pairing", "htdemucs", "pairable_success_intersection"),
    ),
    "oaf_idm_htdemucs": (
        "oaf_idm_htdemucs",
        ("pairing", "pairable_success_intersection"),
    ),
}
```

The coordinator reads all counts from validated summaries after the drivers return. Reject missing, boolean, negative, or non-integer values.

Pin a test with distinct values:

```python
assert outcome.pairable_success_counts == {
    "oaf_muscriptor_full_mix": 11,
    "oaf_separation_pilot.spleeter": 12,
    "oaf_separation_pilot.htdemucs": 13,
    "oaf_idm_htdemucs": 14,
}
```

Do not read counts from `ComparisonOutcome`, `SeparationComparisonOutcome`, or the IDM driver's `Path` return.

- [ ] **Step 6: Implement matrix rendering with the existing CSV writer**

Use fixed fields:

```python
_HEADLINE_FIELDS = (
    "scope",
    "model",
    "input_view_id",
    "total_count",
    "eligible_count",
    "success_count",
    "failed_count",
    "skipped_count",
    "quarantined_count",
    "comparison_ids",
)
```

For each `_HEADLINE_SOURCES` tuple:

1. load only `summaries[comparison_id]["models"][model_key]`;
2. validate that entry's `input_view_id` equals `expected_view`;
3. validate its `population` has exactly the non-negative integer fields required by the matrix;
4. render with `published_comparison.write_csv()`.

- [ ] **Step 7: Scope reviewed-subset identity correctly**

Do **not** create `summary["reviewed_subset"]`.

Build comparison entries so separation alone carries its cross-verified subset identity:

```python
comparisons["oaf_separation_pilot"]["scope_identity"] = {
    "reviewed_subset_manifest_sha256": separation_summary["identity"][
        "reviewed_subset_manifest_sha256"
    ],
    "reviewed_subset_cross_verified": True,
}
```

For IDM, state the narrower claim:

```python
comparisons["oaf_idm_htdemucs"]["scope_identity"] = {
    "pilot_lineage": "validated_hpa396_run",
    "reviewed_subset_cross_verified": False,
}
```

Test:

```python
assert "reviewed_subset" not in summary
assert summary["comparisons"]["oaf_separation_pilot"]["scope_identity"][
    "reviewed_subset_cross_verified"
] is True
assert summary["comparisons"]["oaf_idm_htdemucs"]["scope_identity"][
    "reviewed_subset_cross_verified"
] is False
```

No extra HPA-328 handoff input is introduced.

- [ ] **Step 8: Implement top-level summary and Markdown**

`summary.json` has:

```text
schema
identity
  reference_manifest_sha256
  reference_manifest_version
  reference_timing_manifest_sha256
  reference_timing_version
  taxonomy_version
  lane_map_version
  scoring_version
  oaf_model_lock_sha256
pairable_success_counts
comparisons
  <comparison id>
    path
    artifacts
    scope_identity
headline_matrix
  path
  sha256
```

`pairable_success_counts` is the single persisted copy of the four-key map from Step 5. Comparison entries do not duplicate those counts.

Every artifact path is relative to the HPA-562 root. `artifacts` follows `_EXPECTED_ARTIFACTS` order and contains `{path, sha256}` objects.

`summary.md` renders the six matrix rows, four pair counts, and these two explicit cautions:

```text
Broad full-mix and reviewed-pilot rows have different populations and must not be ranked as one leaderboard.
The IDM pilot lineage is validated inside its HPA-396 run; HPA-562 does not cross-verify its reviewed-subset identity against the HPA-328 separation publication.
```

- [ ] **Step 9: Add root-independence tests**

Publish byte-identical fake evidence to two different non-existing roots and assert:

```python
for name in ("summary.json", "summary.md", "headline_matrix.csv"):
    assert (left / name).read_bytes() == (right / name).read_bytes()
```

Also assert every persisted path in `summary.json` is relative and contains no staging/root prefix.

- [ ] **Step 10: Add one real-driver summary-shape integration test**

Create one small internally consistent raw-evidence fixture in `tests/benchmark/test_cross_comparison.py` using the existing test fixture conventions/builders from:

```text
tests/benchmark/test_muscriptor_comparison.py
tests/benchmark/test_separation_comparison.py
tests/benchmark/test_idm_comparison.py
tests/benchmark/idm_pilot_fixtures.py
tests/benchmark/reviewed_subset_fixtures.py
```

The helper stays local to this test module; do not create production fixture infrastructure.

The test must invoke `publish_cross_comparisons(request)` **without monkeypatching**:

```text
compare_oaf_muscriptor
compare_oaf_separation
compare_oaf_idm
```

It may use ordinary test seams to create valid manifests/run snapshots/reports, but the three comparison drivers themselves execute for real.

The fixture must share:

```text
reference_manifest_sha256
reference_manifest_version
reference_timing_manifest_sha256
reference_timing_version
TAXONOMY_VERSION
DTX_LANE_MAP_VERSION
SCORING_VERSION
OaF model_lock_sha256
HTDemucs input-view identity between separation and IDM
```

Assert the publication succeeds and its three nested summaries use the actual driver schemas. If a real summary shape disagrees, the test must fail through `ComparisonIntegrityError` naming the concrete field rather than replacing it with a fake summary.

- [ ] **Step 11: Run the focused suite GREEN**

```bash
uv run pytest tests/benchmark/test_cross_comparison.py -q
```

Expected: PASS.

- [ ] **Step 12: Commit**

```bash
git add src/benchmark/cross_comparison.py tests/benchmark/test_cross_comparison.py
git commit -m "feat: publish benchmark comparison index"
```

---

### Task 4: Wire one thin CLI command with a frozen payload shape

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`

**Interfaces:**
- Consumes: `CrossComparisonRequest`, `CrossComparisonOutcome`, `publish_cross_comparisons()`.
- Produces: `crux benchmark publish-paired-comparisons` with exit `0` on complete publication and exit `2` on malformed/mismatched evidence/publication failure.

- [ ] **Step 1: Write the CLI success test with the exact four count keys**

Use `CliRunner` and monkeypatch only the coordinator boundary. The fake outcome contains:

```python
expected_counts = {
    "oaf_muscriptor_full_mix": 11,
    "oaf_separation_pilot.spleeter": 12,
    "oaf_separation_pilot.htdemucs": 13,
    "oaf_idm_htdemucs": 14,
}
```

Assert canonical stdout decodes to:

```python
{
    "comparison_paths": {
        "oaf_idm_htdemucs": str(output_dir / "comparisons/oaf-idm"),
        "oaf_muscriptor_full_mix": str(output_dir / "comparisons/oaf-muscriptor"),
        "oaf_separation_pilot": str(output_dir / "comparisons/oaf-separation"),
    },
    "exit_code": 0,
    "headline_matrix_path": str(output_dir / "headline_matrix.csv"),
    "output_dir": str(output_dir),
    "pairable_success_counts": expected_counts,
}
```

- [ ] **Step 2: Run the command-focused test and observe RED**

```bash
uv run pytest tests/test_cli_benchmark.py -q -k publish_paired_comparisons
```

Expected: command not registered.

- [ ] **Step 3: Add `publish-paired-comparisons` using the existing compare command style**

Wire these options:

```text
--oaf-run
--muscriptor-run
--separation-run
--idm-run
--manifest
--timing-manifest
--subset-manifest
--separation-cache-dir
--output-dir
```

Keep the `cross_comparison` import lazy inside the command body.

Construct `CrossComparisonRequest` directly. On success, write one canonical JSON object matching Step 1.

Catch:

```python
(ComparisonIntegrityError, OSError, TypeError, ValueError)
```

On failure:

1. write the concise exception string to stderr;
2. emit canonical stdout with:

```python
{
    "comparison_paths": {},
    "error": type(error).__name__,
    "exit_code": 2,
    "headline_matrix_path": None,
    "output_dir": None,
    "pairable_success_counts": {},
}
```

3. raise `click.exceptions.Exit(2)`.

- [ ] **Step 4: Add fatal CLI tests**

At minimum cover:

```text
ComparisonIntegrityError
ValueError from request validation
```

For integrity failure:

```python
assert result.exit_code == 2
assert "taxonomy_version mismatch" in result.stderr
payload = json.loads(result.stdout)
assert payload["exit_code"] == 2
assert payload["pairable_success_counts"] == {}
```

- [ ] **Step 5: Run CLI + coordinator tests GREEN**

```bash
uv run pytest \
  tests/benchmark/test_cross_comparison.py \
  tests/test_cli_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py
git commit -m "feat: add paired comparison publication CLI"
```

---

### Task 5: Prove acceptance and coverage, then publish real evidence when available

**Files:**
- Modify only files already listed above if a concrete failing test/coverage branch requires a correction.

**Interfaces:**
- Consumes: complete HPA-562 implementation from Tasks 1–4.
- Produces: pairwise + coordinator regression evidence, local coverage evidence matching the blocking Codecov target, CI-equivalent verification, and eventually one real production HPA-562 bundle.

- [ ] **Step 1: Run all pairwise/coordinator/CLI suites together**

```bash
uv run pytest \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_separation_comparison.py \
  tests/benchmark/test_idm_comparison.py \
  tests/benchmark/test_cross_comparison.py \
  tests/test_cli_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 2: Enforce the local 90% focused coverage proxy before CI**

`codecov.yml` has a blocking 90% patch target with zero threshold. Run a focused module gate:

`pytest.ini` sets a global `addopts = --cov=src`; override it so coverage scopes to the coordinator module instead of accumulating over the whole tree:

```bash
uv run pytest tests/benchmark/test_cross_comparison.py tests/benchmark/test_cross_comparison_coverage.py -q \
  -o addopts="--cov=src.benchmark.cross_comparison --cov-report=term-missing" \
  --cov-fail-under=90
```

Expected: PASS with `src/benchmark/cross_comparison.py` coverage >= 90%.

If this fails, add focused tests for the concrete uncovered validation/publication branches in `tests/benchmark/test_cross_comparison.py`. If the main test module becomes materially harder to navigate, split only those concrete branch tests into `tests/benchmark/test_cross_comparison_coverage.py`, following the existing `test_muscriptor_comparison_coverage.py` convention. Do not add production abstractions to satisfy coverage.

- [ ] **Step 3: Run the CI test command with coverage output**

Mirror `.github/workflows/ci.yml`:

```bash
uv run pytest -q \
  --cov=src \
  --cov-report=xml:coverage.xml \
  --cov-report=term-missing
```

Expected: PASS and `coverage.xml` exists.

- [ ] **Step 4: Run CI-equivalent static verification**

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check origin/main...HEAD
```

Expected: all succeed.

- [ ] **Step 5: Verify the branch still contains one-ticket scope**

```bash
git diff --name-only origin/main...HEAD
```

Expected changed production/test scope is limited to:

```text
src/benchmark/muscriptor_comparison.py
src/benchmark/separation_comparison.py
src/benchmark/cross_comparison.py
src/cli/benchmark.py
tests/benchmark/test_muscriptor_comparison.py
tests/benchmark/test_separation_comparison.py
tests/benchmark/test_cross_comparison.py
tests/benchmark/test_cross_comparison_coverage.py   # only if Step 2 required the split
tests/test_cli_benchmark.py
docs/superpowers/specs/2026-08-22-hpa-562-paired-comparisons-design.md
docs/superpowers/plans/2026-08-22-hpa-562-paired-comparisons.md
```

No inference/model/runtime implementation belongs in this PR.

- [ ] **Step 6: Execute the production publication only when real evidence paths exist**

```bash
uv run crux benchmark publish-paired-comparisons \
  --oaf-run "$OAF_RUN" \
  --muscriptor-run "$MUSCRIPTOR_RUN" \
  --separation-run "$SEPARATION_RUN" \
  --idm-run "$IDM_RUN" \
  --manifest "$REFERENCE_MANIFEST" \
  --timing-manifest "$TIMING_MANIFEST" \
  --subset-manifest "$REVIEWED_SUBSET_MANIFEST" \
  --separation-cache-dir "$SEPARATION_CACHE_DIR" \
  --output-dir "$HPA562_OUTPUT_DIR"
```

Expected when real HPA-627/HPA-328/HPA-396 evidence is available: exit `0` and one complete `crux.paired-benchmark-publication/v1` bundle.

If any production input is missing, do not substitute fixture evidence. Keep HPA-562 In Progress and record the exact operational block on the existing PR/Linear issue. The implementation PR itself is not blocked on Hugging Face access.

- [ ] **Step 7: Validate the production bundle before closing HPA-562**

Enforce the complete documented bundle contract with strict canonical JSON parsing, reusing the production validation helpers:

```bash
uv run python - <<'PY'
import csv
import os
from pathlib import Path

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import require_sha256, sha256_hex, strict_json_loads
from src.benchmark.cross_comparison import PAIRED_BENCHMARK_PUBLICATION_SCHEMA
from src.benchmark.idm_comparison import IDM_COMPARISON_SCHEMA
from src.benchmark.muscriptor_comparison import COMPARISON_SCHEMA
from src.benchmark.separation_comparison import SEPARATION_COMPARISON_SCHEMA

root = Path(os.environ["HPA562_OUTPUT_DIR"])
shared_fields = (
    "reference_manifest_sha256",
    "reference_manifest_version",
    "reference_timing_manifest_sha256",
    "reference_timing_version",
    "taxonomy_version",
    "lane_map_version",
    "scoring_version",
)
comparisons = {
    "oaf_muscriptor_full_mix": (
        "comparisons/oaf-muscriptor",
        COMPARISON_SCHEMA,
        ("summary.json", "summary.md", "paired_per_song.csv", "paired_per_class.csv"),
    ),
    "oaf_separation_pilot": (
        "comparisons/oaf-separation",
        SEPARATION_COMPARISON_SCHEMA,
        (
            "summary.json",
            "summary.md",
            "spleeter/paired_per_song.csv",
            "spleeter/paired_per_class.csv",
            "htdemucs/paired_per_song.csv",
            "htdemucs/paired_per_class.csv",
        ),
    ),
    "oaf_idm_htdemucs": (
        "comparisons/oaf-idm",
        IDM_COMPARISON_SCHEMA,
        ("summary.json", "summary.md", "paired_per_song.csv", "paired_per_class.csv"),
    ),
}


def load(path):
    return strict_json_loads(read_regular_file_no_follow(path), require_canonical=True)


def safe(relative):
    candidate = Path(relative)
    assert not candidate.is_absolute() and ".." not in candidate.parts, relative
    return root / candidate


def with_parents(relative):
    parts = Path(relative).parts
    return {"/".join(parts[: index + 1]) for index in range(len(parts))}


expected_entries = {
    entry for name in ("summary.json", "summary.md", "headline_matrix.csv")
    for entry in with_parents(name)
}
for comparison_dir, _schema, artifacts in comparisons.values():
    for artifact in artifacts:
        expected_entries |= with_parents(f"{comparison_dir}/{artifact}")

seen_files = set()
seen_dirs = set()
for current, dirnames, filenames in os.walk(root):
    base = Path(current)
    for name in dirnames:
        target = seen_dirs if not (base / name).is_symlink() else seen_files
        target.add((base / name).relative_to(root).as_posix())
    for name in filenames:
        seen_files.add((base / name).relative_to(root).as_posix())
assert seen_files | seen_dirs == expected_entries, (
    f"bundle entries diverge from contract: "
    f"missing={sorted(expected_entries - (seen_files | seen_dirs))} "
    f"unexpected={sorted((seen_files | seen_dirs) - expected_entries)}"
)

summary = load(safe("summary.json"))
assert summary["schema"] == PAIRED_BENCHMARK_PUBLICATION_SCHEMA, summary.get("schema")
assert "reviewed_subset" not in summary
identity = summary["identity"]
assert set(identity) == {*shared_fields, "oaf_model_lock_sha256"}
lock = require_sha256(identity["oaf_model_lock_sha256"], "identity.oaf_model_lock_sha256")
counts = summary["pairable_success_counts"]
assert set(counts) == {
    "oaf_muscriptor_full_mix",
    "oaf_separation_pilot.spleeter",
    "oaf_separation_pilot.htdemucs",
    "oaf_idm_htdemucs",
}
assert all(
    isinstance(value, int) and not isinstance(value, bool) and value >= 0
    for value in counts.values()
)
assert set(summary["comparisons"]) == set(comparisons)
assert summary["headline_matrix"]["path"] == "headline_matrix.csv"

for comparison_id, (comparison_dir, schema, artifacts) in comparisons.items():
    entry = summary["comparisons"][comparison_id]
    assert entry["path"] == comparison_dir
    nested = load(safe(comparison_dir) / "summary.json")
    assert nested["schema"] == schema
    nested_identity = nested["identity"]
    for field in shared_fields:
        assert nested_identity[field] == identity[field], field
    if comparison_id == "oaf_muscriptor_full_mix":
        assert nested["models"]["oaf"]["model_lock_sha256"] == lock
        oaf_prediction_map = nested["models"]["oaf"]["prediction_map_version"]
        assert isinstance(oaf_prediction_map, str) and bool(oaf_prediction_map)
    elif comparison_id == "oaf_separation_pilot":
        assert entry["scope_identity"]["reviewed_subset_cross_verified"] is True
        htdemucs_view = nested["models"]["htdemucs"]["input_view_id"]
        for view_key in ("full_mix", "spleeter", "htdemucs"):
            assert nested["models"][view_key]["model_lock_sha256"] == lock, view_key
            assert (
                nested["models"][view_key]["prediction_map_version"] == oaf_prediction_map
            ), view_key
    else:
        assert entry["scope_identity"] == {
            "pilot_lineage": "validated_hpa396_run",
            "reviewed_subset_cross_verified": False,
        }
        assert nested["models"]["oaf"]["model_lock_sha256"] == lock
        assert nested["models"]["oaf"]["input_view_id"] == htdemucs_view
        assert nested["models"]["oaf"]["prediction_map_version"] == oaf_prediction_map

    assert [item["path"] for item in entry["artifacts"]] == [
        f"{comparison_dir}/{relative}" for relative in artifacts
    ]
    for item in entry["artifacts"]:
        content = read_regular_file_no_follow(safe(item["path"]))
        assert item["sha256"] == sha256_hex(content), item["path"]

headline_bytes = read_regular_file_no_follow(
    safe(summary["headline_matrix"]["path"])
)
assert summary["headline_matrix"]["sha256"] == sha256_hex(headline_bytes), "headline hash"
with safe(summary["headline_matrix"]["path"]).open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 6, f"headline rows={len(rows)}"
print(root)
PY
```

Expected: prints the production bundle root and exits `0`.

- [ ] **Step 8: Commit only concrete acceptance corrections**

If Steps 1–7 require no code correction, do not create an empty commit. If a real defect is found, add its focused regression test and commit the minimal correction on the same PR.

---

## Self-review

- **Spec coverage:** all HPA-562 pair types remain owned by the existing three drivers; HPA-562 adds only cross-summary validation, closed artifact indexing, scoped population matrix, four pair counts, and top-level publication.
- **Reuse:** existing canonical JSON, safe read, SHA-256, CSV, staging, pairwise comparison, taxonomy, and scoring contracts are reused.
- **Scope routing:** MuScriptor always receives `subset_manifest_path=None`; separation receives the HPA-327 subset; IDM receives only its validated HPA-396 run.
- **Identity:** reference/timing/taxonomy/lane/scoring and OaF model-lock identity are cross-validated; same-input hashes remain inside pair drivers; HTDemucs view identity is cross-checked.
- **Pilot lineage:** no unverified bundle-level reviewed-subset field. Separation exposes its verified subset; IDM is explicitly marked as not cross-verified against separation at HPA-562 level.
- **Population labeling:** the matrix uses a closed source table and never uses IDM `models["oaf"]` as a headline source.
- **Artifact integrity:** every nested comparison has a fixed expected file set; missing/unexpected files fail closed; no glob-based contract.
- **Counts:** `pairable_success_counts` has four exact keys, is persisted once at top level, and comes from nested summaries independent of driver return-shape differences.
- **Determinism:** only relative paths are persisted; duplicate publications under different roots must have byte-identical top-level files.
- **Breaking-change policy:** regenerated separation-summary hashes may differ from historical HPA-328 handoff hashes after taxonomy/lane fields are added; no compatibility layer is added.
- **Coverage:** focused `cross_comparison.py` coverage must reach >=90% before the blocking Codecov patch gate.
- **Operational truthfulness:** fixture evidence proves code behavior; it does not complete production benchmarking.
- **PR shape:** planning and implementation remain one HPA-562 PR.
