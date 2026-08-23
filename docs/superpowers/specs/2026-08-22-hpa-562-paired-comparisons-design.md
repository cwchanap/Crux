# HPA-562 Paired Benchmark Publication Design

**Status:** Proposed
**Date:** 2026-08-22
**Linear:** HPA-562 — Publish paired cross-model and cross-input benchmark comparisons

## Decision summary

HPA-562 should be a thin publication layer over comparison work that already exists.

Do not build another scorer, N-way comparison engine, experiment runner, leaderboard service, model registry, database, or report framework. HPA-395, HPA-328, and HPA-396 already own the three strict pairwise comparisons required by this ticket:

1. OaF full mix vs MuScriptor full mix;
2. OaF full mix vs OaF Spleeter/HTDemucs views;
3. OaF HTDemucs stem vs IDM on the exact same retained stem input.

HPA-562 will invoke those existing comparison drivers against one supplied evidence set, validate their published identities together, and publish one deterministic top-level bundle containing:

- the three existing pairwise comparison trees;
- a concise model × input-view population matrix;
- a top-level JSON/Markdown index with exact relative paths and SHA-256 hashes for the nested comparison outputs.

The draft planning PR is also the implementation PR for HPA-562. Production code and tests will be added to this same branch/PR after the plan is approved; no second HPA-562 PR is planned.

## Why this is the next actionable task

HPA-627 has higher nominal priority but is operationally blocked by gated Hugging Face access needed to freeze `runtime/muscriptor/model.json`.

HPA-562's formal prerequisite, HPA-325, is complete. The repository now also contains the HPA-395 MuScriptor comparator, HPA-328 separation comparator, and HPA-396 IDM comparator. HPA-329 is the final findings/decision ticket and should consume HPA-562 rather than recreating pair joins itself.

## Existing authorities

Reuse these modules without changing their ownership:

```text
src/benchmark/published_comparison.py
  pairable_success_ids()
  paired_song_rows()
  paired_class_rows()
  aggregate_delta_rows()
  comparison_summary()
  write_comparison_artifacts()

src/benchmark/muscriptor_comparison.py
  compare_oaf_muscriptor()

src/benchmark/separation_comparison.py
  compare_oaf_separation()

src/benchmark/idm_comparison.py
  compare_oaf_idm()

src/benchmark/reports.py
  HPA-325 published cohort reports
```

HPA-562 does not rebuild cohorts, reread raw prediction events for scoring, or recalculate pairwise deltas.

## Approaches considered

### A. Compose the existing strict comparison drivers — selected

Create one concrete `cross_comparison.py` coordinator. It stages the three current pairwise publications, validates their summaries as one evidence bundle, then writes the top-level matrix/index.

This is the smallest approach and keeps each existing ticket's integrity checks authoritative.

### B. Consume arbitrary pre-existing comparison directories — rejected

This would require a new stable reader/compatibility contract for every comparison summary and would make it easier to accidentally combine stale outputs from different runs. Re-running the cheap report joins from immutable run/report evidence is simpler and safer.

### C. Replace the three drivers with a generic N-way comparison engine — rejected

The comparison topologies are materially different: same full mix across models, one model across different inputs, and two models on one retained stem pilot. Generalizing them now would add abstractions without reducing current work.

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

Only after all three calls and cross-summary validation succeed does the coordinator write the top-level files and move the staged bundle into place. A failed validation leaves no published HPA-562 bundle.

## Identity contract

The top-level publication validates these identities across the nested summaries wherever they are shared:

- reference manifest SHA-256 and version;
- reference timing manifest SHA-256 and version;
- canonical taxonomy version;
- DTX lane-map version;
- scoring version.

Two small additions are needed before the coordinator can validate that contract from published evidence alone:

- MuScriptor comparison `summary.json` must persist `taxonomy_version` and `lane_map_version` in its comparison identity;
- separation comparison `summary.json` must persist the same two fields in its comparison identity.

No summary schema version bump or compatibility layer is needed for this hobby project; the current repository is the only consumer.

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

The nested files remain owned by their existing writers. HPA-562 only indexes them.

## Headline matrix

`headline_matrix.csv` is intentionally a population/identity matrix, not a global accuracy leaderboard. Accuracy and deltas remain in the pairwise reports, because broad-corpus and pilot-only populations are not directly rankable.

Fixed row order:

1. `broad_full_mix` / OaF / full mix;
2. `broad_full_mix` / MuScriptor / full mix;
3. `reviewed_pilot` / OaF / full mix;
4. `reviewed_pilot` / OaF / Spleeter drums;
5. `reviewed_pilot` / OaF / HTDemucs drums;
6. `reviewed_pilot` / IDM / HTDemucs drums.

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

The explicit `scope` column prevents full-corpus MuScriptor and pilot-only IDM from being presented as one unlabeled ranking.

## Top-level summary

Use schema:

```text
crux.paired-benchmark-publication/v1
```

`summary.json` contains:

- shared reference/timing/taxonomy/lane/scoring identity;
- exact reviewed-subset identity for the pilot scope;
- one entry per pairwise comparison with relative path, summary SHA-256, pairable intersection count, and exclusion/population information copied from the validated nested summary;
- `headline_matrix.csv` relative path and SHA-256.

`summary.md` renders the same identity, the six matrix rows, the three comparison links/paths, intersection counts, and a prominent note that broad and pilot scopes must not be ranked together.

All persisted paths are relative to the HPA-562 output root so identical evidence published in different directories produces identical bytes.

## CLI

Add one thin command:

```text
crux benchmark publish-paired-comparisons
```

It accepts the four run snapshots, HPA-324 reference manifest, HPA-323 timing manifest, HPA-327 subset manifest, optional HPA-328 cache root, and one output directory.

Success prints one canonical JSON object with output paths and pairable counts and exits `0`.

Malformed/mismatched evidence or publication failure prints a concise error to stderr, emits a canonical failure object, and exits `2`. There is no partial-success mode because this ticket's value is the complete comparison bundle.

## Testing

Focused tests must prove:

- MuScriptor and separation summaries expose taxonomy/lane identity;
- the coordinator calls the three existing comparison drivers rather than reimplementing their joins;
- reference, timing, taxonomy, lane-map, or scoring mismatch fails closed before final publication;
- same-view input-hash enforcement remains owned by the existing MuScriptor/IDM drivers;
- broad and reviewed-pilot rows are always distinct in the matrix;
- the six matrix rows have deterministic order and exact population counts;
- top-level paths are relative and nested artifact hashes match bytes on disk;
- publishing identical fixtures to two different roots produces byte-identical top-level files;
- CLI success/fatal JSON and exit codes are stable.

Existing pairwise comparator suites remain the acceptance authority for per-song/per-class pairing behavior.

## Operational completion gate

The code can be implemented and tested from synthetic fixtures before all production evidence is available, but HPA-562 must not be marked Done until the real bundle can be published.

Current known operational constraints are:

- HPA-627 must supply the canonical MuScriptor model lock required for reproducible production MuScriptor evidence;
- the real HPA-328 separation handoff/stem evidence must exist;
- the real HPA-396 IDM pilot evidence must exist.

If any of those are unavailable during implementation, the PR remains valid and HPA-562 remains In Progress with the missing evidence recorded; no synthetic result is presented as production benchmark evidence.

## Non-goals

- new inference or rescoring;
- fine-tuning or model selection;
- production-model replacement;
- a generic comparison/experiment framework;
- a database, web UI, or dashboard;
- statistical significance/bootstrap analysis;
- automatic winner selection;
- merging full-corpus and pilot-only results into one ranking;
- changing HPA-329's final decision responsibility.
