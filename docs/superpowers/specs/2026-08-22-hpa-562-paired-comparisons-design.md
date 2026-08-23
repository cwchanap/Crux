# HPA-562 Paired Benchmark Publication Design

**Status:** Proposed
**Date:** 2026-08-22
**Linear:** HPA-562 — Publish paired cross-model and cross-input benchmark comparisons

## Decision summary

HPA-562 is a thin publication layer over comparison work that already exists.

Do not build another scorer, N-way comparison engine, experiment runner, leaderboard service, model registry, database, or report framework. HPA-395, HPA-328, and HPA-396 already own the three strict pairwise comparisons required by this ticket:

1. OaF full mix vs MuScriptor full mix;
2. OaF full mix vs OaF Spleeter/HTDemucs views;
3. OaF HTDemucs stem vs IDM on the exact same retained stem input.

HPA-562 invokes those existing comparison drivers from raw persisted run evidence, validates their published identities together, and publishes one deterministic top-level bundle containing:

- the three existing pairwise comparison trees;
- a concise model × input-view population matrix sourced from a closed comparison/model-key table;
- a top-level JSON/Markdown index with exact relative paths and SHA-256 hashes for a closed set of nested artifacts.

The draft planning PR is also the implementation PR for HPA-562. Production code and tests remain on this same branch/PR; no second HPA-562 PR is planned.

## Why this is the next actionable task

HPA-627 has higher nominal priority but is operationally blocked by gated Hugging Face access needed to freeze `runtime/muscriptor/model.json`.

HPA-562's formal prerequisite, HPA-325, is complete. The repository now also contains the HPA-395 MuScriptor comparator, HPA-328 separation comparator, and HPA-396 IDM comparator. HPA-329 is the final findings/decision ticket and should consume HPA-562 rather than recreate pair joins itself.

## Existing authorities

Reuse these modules without changing their ownership:

```text
src/benchmark/published_comparison.py
  pairable_success_ids()
  paired_song_rows()
  paired_class_rows()
  aggregate_delta_rows()
  population()
  comparison_summary()
  write_csv()
  write_comparison_artifacts()

src/benchmark/muscriptor_comparison.py
  compare_oaf_muscriptor()

src/benchmark/separation_comparison.py
  compare_oaf_separation()

src/benchmark/idm_comparison.py
  compare_oaf_idm()

src/benchmark/backend_identity.py
  sha256_hex()
  canonical_json_bytes()
  strict_json_loads()

src/benchmark/artifact_io.py
  read_regular_file_no_follow()

src/benchmark/separation_handoff.py
  closed comparison-artifact key convention

src/benchmark/reports.py
  HPA-325 published cohort reports
```

HPA-562 does not rebuild cohorts, reread raw prediction events for scoring, or recalculate pairwise deltas.

## Approaches considered

### A. Compose the existing strict comparison drivers — selected

Create one concrete `cross_comparison.py` coordinator. It stages the three current pairwise publications, validates their summaries as one evidence bundle, then writes the top-level matrix/index.

This is the smallest approach and keeps each existing ticket's integrity checks authoritative.

### B. Consume arbitrary pre-existing comparison directories — rejected

This would require a stable compatibility contract for older comparison summaries and would make it easier to mix stale outputs from different runs. Re-running the cheap report joins from immutable run/report evidence is simpler and safer.

### C. Replace the three drivers with a generic N-way comparison engine — rejected

The comparison topologies are materially different: same full mix across models, one model across different inputs, and two models on one retained stem pilot. Generalizing them now adds abstraction without removing work.

## Architecture

Add:

```text
src/benchmark/cross_comparison.py
```

with one public request/outcome pair and one public operation:

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


def publish_cross_comparisons(request: CrossComparisonRequest) -> CrossComparisonOutcome:
    ...
```

`publish_cross_comparisons()` uses a sibling staging directory and requires the final `output_dir` not to exist. It invokes:

```text
compare_oaf_muscriptor() -> stage/comparisons/oaf-muscriptor/
compare_oaf_separation()  -> stage/comparisons/oaf-separation/
compare_oaf_idm()         -> stage/comparisons/oaf-idm/
```

The request routing is deliberately asymmetric because the comparisons have different scopes:

- MuScriptor receives `subset_manifest_path=None` unconditionally. Its HPA-562 publication is the broad full-mix comparison, never the reviewed pilot.
- Separation receives the required HPA-327 subset manifest and optional cache root.
- IDM receives only its persisted HPA-396 run path plus output directory. HPA-396 itself validates the run and its retained-stem OaF peer.

Only after all three calls and cross-summary validation succeed does the coordinator write the top-level files and move the staged bundle into place. Any driver, identity, artifact, or rendering failure leaves no published HPA-562 bundle.

## Identity contract

The top-level publication validates these fields across all three nested summary `identity` objects:

```text
reference_manifest_sha256
reference_manifest_version
reference_timing_manifest_sha256
reference_timing_version
taxonomy_version
lane_map_version
scoring_version
```

Two small additions are needed before the coordinator can validate that contract from published evidence alone:

- MuScriptor comparison `summary.json` must persist `taxonomy_version` and `lane_map_version` through the existing `_summary(..., identity=...)` hook. `muscriptor_comparison.py` must import `TAXONOMY_VERSION` and `DTX_LANE_MAP_VERSION` from `src.benchmark.taxonomy`.
- Separation comparison `summary.json` must add `taxonomy_version` and `lane_map_version` to `_comparison_identity()`. It already persists `scoring_version`; that field is not new work.

IDM already publishes taxonomy, lane-map, and scoring identity and does not need an HPA-562 identity change.

No summary schema version bump or compatibility layer is needed. Current code is the only supported consumer.

### OaF model identity across all three comparisons

All three nested comparisons present an OaF population. HPA-562 must prove that they refer to the same frozen OaF checkpoint rather than silently publishing three different OaF identities under one label.

Require exact equality of:

```text
MuScriptor summary: models["oaf"]["model_lock_sha256"]
Separation summary: models["full_mix"]["model_lock_sha256"]
IDM summary: models["oaf"]["model_lock_sha256"]
```

A missing or unequal value is a fatal `ComparisonIntegrityError` naming `model_lock_sha256`.

Also require the two HTDemucs-stem identities to agree:

```text
Separation summary: models["htdemucs"]["input_view_id"]
IDM summary: models["oaf"]["input_view_id"]
```

Both must equal the existing HTDemucs stem input-view identity. This does not replace either pairwise driver's own input-hash checks; it only prevents the top-level bundle from presenting two differently named stem views as one.

### Reviewed-pilot lineage scope

HPA-562 must not make a bundle-level reviewed-subset claim that it has not cross-verified.

The separation driver directly validates its HPA-328 run against the supplied HPA-327 subset and publishes `reviewed_subset_manifest_sha256` in its comparison identity. The IDM comparison validates the HPA-396 run and its OaF/IDM identical-stem pairing, but its comparison summary does not republish the HPA-328 handoff or HPA-327 subset identity.

Therefore:

- **do not add another handoff-manifest input or a new lineage seam to HPA-562;**
- **do not publish a top-level `reviewed_subset` field;**
- keep the separation subset identity under `comparisons.oaf_separation_pilot.scope_identity`;
- record on `comparisons.oaf_idm_htdemucs.scope_identity` that the pilot lineage is carried by the validated HPA-396 run and that HPA-562 does not cross-verify its reviewed-subset identity against the separation publication;
- render the same qualification in `summary.md`.

The matrix can still label both as pilot-scoped results, but HPA-329 must not infer that the two nested publications share one cross-verified HPA-327 subset from a bundle-level field.

This is intentionally cheaper than adding the HPA-328 handoff as a fifth publication input and is sufficient for HPA-562's responsibility: preserve the pairwise comparisons and prevent unlabeled scope merging.

### Input-hash rule

The ticket's input-hash requirement is applied according to comparison topology:

- **same-input model comparisons** — OaF vs MuScriptor full mix and OaF vs IDM HTDemucs — require exact `input_audio_sha256` equality, as their current drivers already enforce;
- **cross-input ablation** — OaF full mix vs Spleeter/HTDemucs — cannot have equal input hashes by definition. It instead requires the same authoritative source hash plus fixed input-view identities and validated per-view input hashes, as HPA-328 already enforces.

HPA-562 must not weaken or duplicate those pair-level checks.

## Publication layout

```text
<output-dir>/
  summary.json
  summary.md
  headline_matrix.csv
  comparisons/
    oaf-muscriptor/
      summary.json
      summary.md
      paired_per_song.csv
      paired_per_class.csv
    oaf-separation/
      summary.json
      summary.md
      spleeter/
        paired_per_song.csv
        paired_per_class.csv
      htdemucs/
        paired_per_song.csv
        paired_per_class.csv
    oaf-idm/
      summary.json
      summary.md
      paired_per_song.csv
      paired_per_class.csv
```

The nested files remain owned by their existing writers. HPA-562 only validates and indexes them.

### Closed nested-artifact contract

Do not discover artifacts by globbing whatever happens to exist. Freeze the exact expected file set per comparison, following the existing `separation_handoff.py` convention:

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

For every nested comparison, the actual relative file set must equal the expected tuple's set. Missing or unexpected files fail closed with `ComparisonIntegrityError`. Hash each expected regular file using `read_regular_file_no_follow()` + `sha256_hex()` and persist only its HPA-562-root-relative path and hash.

This makes a missing pair CSV an integrity failure instead of silently producing a shorter index.

### Expected separation-summary hash divergence

Task 1 adds taxonomy/lane identity to newly generated HPA-328 comparison summaries. A separation `summary.json` regenerated by HPA-562 from an older separation run can therefore have different bytes/SHA-256 from the historical comparison summary whose hash may have been recorded in an HPA-328 handoff.

That divergence is expected. HPA-562 deliberately republishes the comparison from the persisted run under the current breaking-change contract; it does **not** claim byte identity with a historical handoff's comparison hash and does not add compatibility logic to preserve that hash.

## Headline matrix

`headline_matrix.csv` is a population/identity matrix, not a global accuracy leaderboard. Accuracy and deltas remain in the pairwise reports because broad-corpus and pilot-only populations are not directly rankable.

The matrix is not allowed to select a model by a shared name such as `"oaf"`. Its source is a closed table fixing the comparison summary and exact `models[...]` key:

| scope | model | input view | source summary/model key |
| --- | --- | --- | --- |
| `broad_full_mix` | OaF | full mix | MuScriptor `models["oaf"]` |
| `broad_full_mix` | MuScriptor | full mix | MuScriptor `models["muscriptor"]` |
| `reviewed_pilot` | OaF | full mix | separation `models["full_mix"]` |
| `reviewed_pilot` | OaF | Spleeter drums | separation `models["spleeter"]` |
| `reviewed_pilot` | OaF | HTDemucs drums | separation `models["htdemucs"]` |
| `reviewed_pilot` | IDM | HTDemucs drums | IDM `models["idm"]` |

Never use IDM `models["oaf"]` for a headline row: that entry is the retained HTDemucs-stem OaF peer inside the HPA-396 comparison, not the broad full-mix OaF population.

Fixed row order is the table order above.

Columns:

```text
scope
model
input_view_id
total_count
eligible_count
success_count
failed_count
skipped_count
quarantined_count
comparison_ids
```

Population fields come from the already-published `models[*].population` objects produced from the existing `published_comparison.population()` contract. The explicit `scope` column prevents full-corpus MuScriptor and pilot-only IDM from being presented as one unlabeled ranking.

Matrix tests give every possible source key a different population count, including a sentinel for IDM `models["oaf"]`, so an incorrect mapping fails deterministically.

## Pairable-success count contract

The three existing drivers expose pair counts in different return shapes. HPA-562 does not propagate those driver return types into its API. It reads the counts uniformly from the validated nested summaries after publication.

`CrossComparisonOutcome.pairable_success_counts`, top-level `summary.json["pairable_success_counts"]`, and the CLI all use exactly four flat keys:

```text
oaf_muscriptor_full_mix
oaf_separation_pilot.spleeter
oaf_separation_pilot.htdemucs
oaf_idm_htdemucs
```

Sources:

```text
oaf_muscriptor_full_mix
  -> muscriptor summary pairing.pairable_success_intersection

oaf_separation_pilot.spleeter
  -> separation summary pairing.spleeter.pairable_success_intersection

oaf_separation_pilot.htdemucs
  -> separation summary pairing.htdemucs.pairable_success_intersection

oaf_idm_htdemucs
  -> IDM summary pairing.pairable_success_intersection
```

Each value must be a non-negative integer. Missing, boolean, negative, or malformed counts fail closed. The coordinator does not depend on MuScriptor's scalar outcome count, separation's nested outcome dict, or IDM's bare `Path` return.

The four-key map is persisted once at top level. Individual comparison index entries do not duplicate it.

## Top-level summary

Use schema:

```text
crux.paired-benchmark-publication/v1
```

`summary.json` contains:

- shared reference/timing/taxonomy/lane/scoring identity;
- the validated frozen OaF `model_lock_sha256`;
- the four-key top-level `pairable_success_counts` map;
- one entry per pairwise comparison with relative path, closed artifact path/hash index, and scope information copied from the validated nested summary;
- separation-only reviewed-subset identity under `comparisons.oaf_separation_pilot.scope_identity`;
- an IDM scope note under `comparisons.oaf_idm_htdemucs.scope_identity` stating that HPA-562 does not cross-verify its reviewed subset against separation;
- `headline_matrix.csv` relative path and SHA-256.

There is no top-level `reviewed_subset` field.

Nested artifact hashes use `read_regular_file_no_follow()` and the existing `sha256_hex()` helper; do not add another hasher.

`summary.md` renders the same identity, six matrix rows, comparison paths, four explicit intersection counts, and both scope cautions:

```text
Broad full-mix and reviewed-pilot rows have different populations and must not be ranked as one leaderboard.
The IDM pilot lineage is validated inside its HPA-396 run; HPA-562 does not cross-verify its reviewed-subset identity against the HPA-328 separation publication.
```

All persisted paths are relative to the HPA-562 output root so identical evidence published in different directories produces identical top-level bytes.

## CLI

Add one thin command:

```text
crux benchmark publish-paired-comparisons
```

It accepts the four run snapshots, HPA-324 reference manifest, HPA-323 timing manifest, HPA-327 subset manifest, optional HPA-328 cache root, and one output directory.

The CLI requires the subset because the separation comparison requires it; the coordinator deliberately does not forward that subset to MuScriptor and does not add an HPA-328 handoff input for IDM.

Success prints one canonical JSON object with output paths and the four exact `pairable_success_counts` keys and exits `0`.

Malformed/mismatched evidence or publication failure prints a concise error to stderr, emits a canonical failure object, and exits `2`. There is no partial-success mode because this ticket's value is the complete comparison bundle.

## Testing

Focused tests must prove:

- MuScriptor and separation summaries expose taxonomy/lane identity; separation's pre-existing scoring identity remains unchanged;
- the coordinator calls the three existing comparison drivers rather than reimplementing joins;
- the coordinator passes `subset_manifest_path=None` to MuScriptor, the supplied subset to separation, and no subset argument to IDM;
- reference, timing, taxonomy, lane-map, scoring, or cross-comparison OaF model-lock mismatch fails closed before final publication;
- separation HTDemucs and IDM OaF input-view identities match the same frozen stem view;
- same-view input-hash enforcement remains owned by the existing MuScriptor/IDM drivers;
- there is no top-level reviewed-subset identity; separation owns its reviewed-subset scope identity and IDM is explicitly marked as not cross-verified at HPA-562 level;
- broad and reviewed-pilot rows remain distinct in the matrix;
- the six matrix rows have deterministic order and use the closed source-summary/model-key mapping;
- nested file sets must exactly match `_EXPECTED_ARTIFACTS`; missing or unexpected artifacts fail closed;
- top-level paths are relative and every indexed artifact hash matches regular-file bytes on disk;
- the four pairable-success count keys and summary sources are exact;
- publishing identical fixtures to two different roots produces byte-identical top-level files;
- one integration fixture executes `publish_cross_comparisons()` without monkeypatching the three comparison drivers, so the real MuScriptor, separation, and IDM `summary.json` shapes are validated together;
- focused HPA-562 coverage reaches at least 90% for `src/benchmark/cross_comparison.py` before CI;
- CLI success/fatal JSON and exit codes are stable.

The real-driver fixture reuses existing comparison fixture builders/evidence conventions and remains test-only. It is the authority that the three real summary shapes can satisfy HPA-562; production evidence is not required for this identity test.

If focused coverage is below 90%, add concrete tests for uncovered branches. If the main test module becomes materially harder to navigate, use a sibling `tests/benchmark/test_cross_comparison_coverage.py`, following the existing comparison/runner coverage-suite convention. Do not add production abstractions solely for coverage.

Existing pairwise comparator suites remain the acceptance authority for per-song/per-class pairing behavior.

## Operational completion gate

The code can be implemented and tested from synthetic fixtures before all production evidence is available, but HPA-562 must not be marked Done until the real bundle can be published.

Current known operational constraints are:

- HPA-627 must supply the canonical MuScriptor model lock required for reproducible production MuScriptor evidence;
- the real HPA-328 separation handoff/stem evidence must exist;
- the real HPA-396 IDM pilot evidence must exist.

If any of those are unavailable during implementation, the PR remains valid and HPA-562 remains In Progress with the missing evidence recorded; no synthetic result is presented as production benchmark evidence. The planning/implementation PR itself is not blocked on Hugging Face access.

## Non-goals

- new inference or rescoring;
- fine-tuning or model selection;
- production-model replacement;
- a generic comparison/experiment framework;
- a database, web UI, or dashboard;
- statistical significance/bootstrap analysis;
- automatic winner selection;
- merging full-corpus and pilot-only results into one ranking;
- a new HPA-328 handoff-lineage reader solely for HPA-562;
- changing HPA-329's final decision responsibility.
