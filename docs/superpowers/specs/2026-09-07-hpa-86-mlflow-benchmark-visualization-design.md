# HPA-86 MLflow Benchmark Visualization Design

## Summary

Add one optional, read-only projection from Crux's canonical published benchmark reports into a standard MLflow tracking server. DagsHub-hosted MLflow is the first operational target, but Crux contains no DagsHub SDK or provider-specific application API.

Crux remains the scientific source of truth. MLflow is disposable visualization/index state:

```text
model inference
  -> persisted Crux predictions
  -> Crux scoring
  -> canonical HPA-325 reports
  -> optional explicit MLflow publication
```

MLflow failure must never affect inference, scoring, report publication, or benchmark identity.

## Goals

1. Make existing OaF/separation results easy to compare interactively.
2. Preserve Crux model/reference/input/taxonomy/scoring identities as authoritative.
3. Keep MLflow entirely post-report and optional.
4. Use the standard MLflow client so the hosted provider remains replaceable.
5. Keep hosted storage and operational cost small enough for a hobby project.
6. Leave a clean path for future fine-tuning without redesigning today's benchmark pipeline.
7. Deliver design, implementation, tests, docs, CI coverage, and hosted smoke in one HPA-86 PR.

## Non-goals

HPA-86 does not add training/fine-tuning, hyperparameter search, Model Registry, MLflow dataset objects, autologging, CI publication of benchmark results, custom dashboards, audio/stem/checkpoint upload, R2 synchronization, a new scorer, a new report schema, or a generic experiment framework.

## Existing seams and reuse

Reuse the current benchmark contracts:

- `src/benchmark/reports.py`: add a public `read_cohort_report_identity(report_dir)` helper plus `load_published_cohort_reports(report_dir)`. The latter delegates full validation to `read_cohort_reports()`.
- `src/benchmark/separation_comparison.py`: make `_strict_report_identity()` delegate to the new report-identity helper, then preserve its existing snapshot expectation checks in `_validated_identity()`.
- `src/benchmark/artifact_io.py`: reuse `PublishedArtifact(path, sha256)` and `read_regular_file_no_follow()`.
- `src/benchmark/backend_identity.py`: reuse `canonical_json_bytes()` and `sha256_hex()` for the projection fingerprint.
- `src/benchmark/taxonomy.py`: reuse `CommonDrumClass` to close durable `class.<name>` metric keys.
- `src/cli/benchmark.py`: extend the existing Click group and canonical JSON publication-command convention.
- `.github/workflows/ci.yml`: follow the existing optional-dependency test pattern by installing `mlflow-skinny>=3.16,<4` in the hand-maintained CI test dependency list.

Do not add a projection-layer `COHORT_FAILURE_REASONS` validation. `read_cohort_reports()` already rejects unknown population reasons before a `PublishedCohortReports` reaches the exporter; duplicating that check would be unreachable through the supported loader.

No model runner/backend, separator execution, prediction, manifest, scorer, or R2/cache production path gains an MLflow dependency.

## Dependency and tracking-server boundary

Add only:

```toml
mlflow = ["mlflow-skinny>=3.16,<4"]
```

The base environment remains MLflow-free. MLflow is imported only on explicit publication.

The public publication command requires `MLFLOW_TRACKING_URI` and validates it as an explicit HTTP(S) tracking-server URI:

- scheme must be `http` or `https`;
- hostname must be present;
- username/password must not be embedded in the URI;
- query/fragment are rejected;
- `file:`, `sqlite:`, bare filesystem paths, and empty values are rejected.

This prevents accidental local tracking stores and inline credentials while still allowing explicit local/self-hosted HTTP servers. DagsHub's path-bearing HTTPS tracking URI remains valid, so the R2-specific `_normalized_https_origin()` cannot be reused directly.

Authentication remains MLflow-owned through normal environment variables such as `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`. Do not add credential CLI flags.

## Canonical report identity reuse

Add:

```python
read_cohort_report_identity(report_dir: Path) -> CohortIdentity
load_published_cohort_reports(report_dir: Path) -> PublishedCohortReports
```

`read_cohort_report_identity()` reads canonical `summary.json`, verifies report schema and the closed identity fields, and constructs `CohortIdentity`.

`load_published_cohort_reports()` calls that helper, then delegates all cross-file validation to:

```python
read_cohort_reports(report_dir, expected_identity=identity)
```

`separation_comparison._strict_report_identity()` delegates to the same helper and wraps `ReportIntegrityError` as `ComparisonIntegrityError`. Its caller `_validated_identity()` continues to own snapshot-specific expected-value checks. This removes the weaker duplicate without changing comparison semantics.

## Projection contract

Projection version is:

```text
crux.mlflow-projection/v1
```

Display scope is:

```text
broad | reviewed | pilot
```

The baseline benchmark views required for V1 are:

```text
30ms raw/aligned
50ms raw/aligned
100ms raw/aligned
```

These six views must be present, but they are not exclusive. If canonical HPA-325 reports later contain additional tolerances, project them too. `read_cohort_reports()` already guarantees raw/aligned completeness for every declared tolerance, and metric names are tolerance-qualified, so rejecting future extra views would only make disposable visualization state unnecessarily brittle.

Do not bind the required baseline set to `DEFAULT_TOLERANCES_MS`; the V1 minimum is explicit.

## Identity, scope, and run naming

Every run receives searchable `crux.*` tags for projection version/fingerprint, scope, cohort, backend, model/model lock, backend descriptor, input view, reference manifest/timing identity, taxonomy/lane-map identity, prediction-map identity, and scoring version.

Use a deterministic run name based on scope + model + input view + cohort. Do not duplicate identity into params in V1.

Scope is operator-supplied display metadata; it is not derivable from report bytes. Therefore scope is part of both the fingerprint and the idempotency search key. Correcting `--scope broad` to `--scope pilot` creates a distinct scoped projection instead of permanently conflicting with the original run.

## Scalar metrics

For every canonical aggregate tolerance/mode publish:

```text
event_micro.precision/recall/f1.<tolerance>ms.<mode>
song_macro.f1.<tolerance>ms.<mode>
class_macro.f1.<tolerance>ms.<mode>
song_f1.minimum/p10/p25/median/p75/p90/maximum.<tolerance>ms.<mode>
class.<common_class>.precision/recall/f1.<tolerance>ms.<mode>
class.<common_class>.reference_support.<tolerance>ms.<mode>
class.<common_class>.prediction_support.<tolerance>ms.<mode>
```

Before minting a `class.<name>` key, validate `PublishedAggregateClass.common_class` against `get_args(CommonDrumClass)`. The persisted report type stores this field as a bare string, so this is the one necessary taxonomy gate at the new durable-key boundary.

Publish population metrics:

```text
population.total
population.success
population.failed
population.skipped
population.quarantined
population.reason.<reason>
```

Population reasons need no second projection check because the supported report loader already closes them.

Undefined metrics are omitted. Preserve canonical `Decimal`/integer values in the pure projection and convert to finite floats only at the MLflow API boundary. Do not publish one scalar key per song; `per_song.csv` remains detailed evidence.

## Artifact policy

Reuse `PublishedArtifact` and publish only:

```text
summary.json
summary.md
items.csv
per_song.csv
per_class.csv
```

under `crux-reports/`.

Never publish diagnostics JSONL, audio, stems, predictions, checkpoints, runtime environments, or R2/cache data.

Each allowlisted file is read through `read_regular_file_no_follow()` and hashed from exact bytes with `sha256_hex()`.

## Projection fingerprint

Compute `crux.projection_sha256` locally before importing MLflow from canonical JSON containing:

- projection version;
- scope;
- deterministic run name;
- projected Crux identity tags excluding the fingerprint itself;
- projected scalar metric names and canonical source values;
- each allowlisted artifact basename and SHA-256.

The fingerprint is an integrity key for one `(cohort_id, scope, projection_version)` projection, not a replacement for Crux benchmark identity.

## Idempotent publication

Within one experiment, search active/non-deleted runs by all three tags:

```text
crux.cohort_id
crux.projection_version
crux.scope
```

Use MLflow's quoted tag-key syntax for dotted keys. Do not paginate during publication. `MlflowClient.search_runs()` returns a `PagedList`; if its continuation token is non-empty, fail closed as a saturated search rather than risking an unseen completed run.

For a complete result set:

1. Any matching run with a different fingerprint -> `run_conflict`.
2. Exactly one same-fingerprint `FINISHED` run -> no-op and return it.
3. More than one same-fingerprint `FINISHED` run -> `run_conflict`.
4. Same-fingerprint `RUNNING` or `SCHEDULED` -> `run_conflict` as publication in progress.
5. Only same-fingerprint `FAILED` and/or `KILLED` -> retry by creating a fresh run.
6. Unknown status -> `run_conflict`.
7. No match -> create a fresh run.

Do not mutate or delete prior runs automatically. For a genuine same-scope fingerprint conflict caused by bad historical MLflow state, the documented operator recovery is to delete the incorrect MLflow projection run and republish from canonical Crux reports.

## Bounded publication errors

Use a small MLflow-specific closed error code type rather than exposing raw provider exceptions. Keep codes tied to stable publication stages, not fragile vendor exception parsing:

```text
missing_optional_dependency
invalid_config
experiment_access_failed
run_search_failed
run_create_failed
metric_log_failed
artifact_upload_failed
run_finalize_failed
run_conflict
```

Each code maps to a fixed credential-free human message. Underlying MLflow/provider exceptions remain chained Python causes for debugging but are never interpolated into CLI output.

This intentionally does not guess `auth_failed` versus `network_failed` from vendor-specific exception strings/status objects; the failed stage plus chained cause is enough for this hobby-project CLI and is more maintainable.

## MLflow API boundary and tests

Use low-level `MlflowClient`, not global fluent run state. The side-effect path resolves/creates an experiment, searches matching runs, creates a run, logs one metric batch, uploads five artifacts, and terminates the run.

Behavior/status tests may use a tiny fake client for failure injection and statuses awkward to manufacture. In addition, CI must run a real-client no-network integration test using a temporary `file://` tracking store by constructing `MlflowClient` directly. This test exercises the actual search-filter parser, `Metric(...)`, `search_runs`, `create_run`, `log_batch`, `log_artifact`, and `set_terminated` without going through the public URI validator and without contacting DagsHub.

The real-client test uses `pytest.importorskip("mlflow")`; CI explicitly installs `mlflow-skinny>=3.16,<4`, so API/filter drift is exercised continuously after merge rather than only during hosted smoke.

A continuation-token saturation unit test may temporarily lower the module search limit or return a fake paged result with a token; do not construct 1000 fake runs.

## CLI

Add exactly:

```text
crux benchmark publish-mlflow-cohort \
  --reports PATH \
  --scope broad|reviewed|pilot \
  [--experiment crux-benchmark]
```

The command validates reports and builds the pure projection before any MLflow import/network work.

Success stdout is canonical JSON containing only non-secret result fields. Command-owned exit-2 failures follow sibling benchmark publication commands: bounded stderr plus canonical JSON stdout containing at least:

```text
exit_code: 2
status: "failed"
error_code: <closed MLflow error code or report/projection error class>
```

Never include credentials, tracking URI, raw provider exception text, or client objects.

## Hosted acceptance

Before HPA-86 is ready, publish five already-produced OaF projections to one `crux-benchmark` experiment:

- OaF full-mix broad;
- OaF full-mix reviewed;
- OaF full-mix pilot;
- OaF Spleeter pilot;
- OaF HTDemucs pilot.

Record non-secret run IDs/fingerprints, created/no-op outcomes, scope/model/input-view filtering, aggregate/per-class/population visibility, and confirmation that only the five small artifacts uploaded. Republish one cohort to prove no duplicate completed run is created.

If canonical workstation reports or hosted credentials are unavailable, keep the PR draft. Do not rerun inference/separation/R2/reference work merely to satisfy HPA-86.

## Risks and recovery

- **Search/filter semantics:** an invalid filter could defeat idempotency. The real `file://` client test executes the exact filter syntax in CI before hosted use.
- **Hosted artifact behavior:** DagsHub may differ from local FileStore in artifact routing or permissions. The hosted smoke explicitly confirms `log_artifact(..., artifact_path="crux-reports")` behavior.
- **Historical MLflow conflict:** MLflow is disposable state. A true same-scope fingerprint conflict is recovered by deleting the incorrect projection run and republishing from canonical Crux reports; never rewrite Crux evidence.
- **Provider outage/quota:** publication can fail independently and be retried later. It never blocks benchmark generation.

## Planned file surface

Create:

```text
src/benchmark/mlflow_export.py
tests/benchmark/test_mlflow_export.py
docs/benchmark/mlflow.md
```

Modify:

```text
src/benchmark/reports.py
src/benchmark/separation_comparison.py
src/cli/benchmark.py
tests/benchmark/test_reports.py
tests/benchmark/test_separation_comparison.py
tests/test_cli_benchmark.py
.github/workflows/ci.yml
pyproject.toml
uv.lock
README.md
```

## Delivery boundary

HPA-86 remains one ticket and one PR. The same branch carries design, plan, implementation, tests, docs, CI coverage, and hosted smoke evidence.
