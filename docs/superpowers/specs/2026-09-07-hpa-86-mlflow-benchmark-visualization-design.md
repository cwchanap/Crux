# HPA-86 MLflow Benchmark Visualization Design

## Summary

Add one optional, read-only projection from Crux's canonical published benchmark reports into a standard MLflow tracking server. DagsHub-hosted MLflow is the first operational target, but Crux contains no DagsHub-specific API or SDK dependency.

Crux remains the scientific source of truth. MLflow is only a disposable index, comparison, artifact-browsing, and visualization surface.

```text
model inference
    -> persisted Crux prediction evidence
    -> Crux scoring
    -> canonical HPA-325 reports
    -> optional explicit MLflow publication
```

A missing dependency, tracking-server outage, or authentication failure must never affect inference, scoring, canonical report publication, or benchmark identity.

## Goals

1. Make existing OaF/separation benchmark results easy to compare interactively.
2. Preserve Crux model/reference/input/taxonomy/scoring identities as the authoritative benchmark contract.
3. Keep MLflow completely outside inference/scoring/report publication.
4. Use only standard MLflow APIs and environment configuration so the hosted provider is replaceable.
5. Keep hosted storage small enough for a hobby project.
6. Establish a tracking surface that can later support fine-tuning without redesigning today's benchmark pipeline.
7. Deliver the design, implementation, tests, docs, and hosted smoke in one HPA-86 PR.

## Non-goals

HPA-86 does not add fine-tuning, training, hyperparameter search, MLflow Model Registry, MLflow dataset objects, autologging, CI publication, a DagsHub SDK, custom dashboards, audio/stem/checkpoint upload, R2 synchronization, a new scorer, a new report schema, or a generic experiment framework.

## Existing seams and reuse

Reuse the existing benchmark contracts rather than adding parallel abstractions:

- `src/benchmark/reports.py`: extend `read_cohort_reports()` / `_parse_summary_identity()` with a convenience self-identifying entry point. Do not reuse `separation_comparison._strict_report_identity`; that path intentionally verifies snapshot-expected identity and must remain separate.
- `src/benchmark/artifact_io.py`: reuse `PublishedArtifact(path, sha256)` and `read_regular_file_no_follow()` for the five projected report artifacts. Do not add a second artifact dataclass.
- `src/benchmark/backend_identity.py`: reuse `canonical_json_bytes()` and `sha256_hex()` for the projection fingerprint.
- `src/benchmark/taxonomy.py`: reuse the closed `CommonDrumClass` type for durable class metric keys.
- `src/benchmark/cohort_scoring.py`: reuse `COHORT_FAILURE_REASONS` for failure-reason metric keys.
- `src/cli/benchmark.py`: extend the existing Click group and its lazy-import convention.

No model-specific OaF, MuScriptor, IDM, separator, scorer, prediction, manifest, or R2 module changes are expected.

## Dependency and server boundary

Add only an optional extra:

```toml
mlflow = ["mlflow-skinny>=3.16,<4"]
```

The base environment must remain MLflow-free. MLflow is imported only when the explicit publication command runs.

Require a non-empty `MLFLOW_TRACKING_URI` so the command cannot silently create a local `./mlruns` store. Authentication remains MLflow-owned through standard environment variables such as `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`. Do not add credential CLI flags.

## Self-identifying published-report loader

Add:

```python
load_published_cohort_reports(report_dir: Path) -> PublishedCohortReports
```

The helper reads canonical `summary.json` only to reconstruct `CohortIdentity`, then delegates complete validation to the existing `read_cohort_reports(report_dir, expected_identity=...)`. It is not a second report reader and introduces no schema change.

## V1 projection contract

Freeze projection version:

```text
crux.mlflow-projection/v1
```

Scope is one of:

```text
broad | reviewed | pilot
```

A V1 cohort must contain exactly these six aggregate views:

```text
30ms raw
30ms aligned
50ms raw
50ms aligned
100ms raw
100ms aligned
```

This is a projection freeze, not a change to HPA-325. Do not bind the projection to `DEFAULT_TOLERANCES_MS`; future scorer defaults may change without changing V1.

## Identity and run naming

MLflow-generated IDs are transport metadata only. Every run receives searchable `crux.*` tags for projection version/fingerprint, scope, cohort, backend, model/model lock, backend descriptor, input view, reference manifest/timing identity, taxonomy/lane-map identity, prediction-map identity, and scoring version.

Use a deterministic readable run name based on scope + model + input view + cohort. Do not duplicate identity into params in V1. Training hyperparameters can use params when a future training integration exists.

## Scalar metrics

For every one of the six aggregate views publish:

```text
event_micro.precision.<tolerance>ms.<mode>
event_micro.recall.<tolerance>ms.<mode>
event_micro.f1.<tolerance>ms.<mode>
song_macro.f1.<tolerance>ms.<mode>
class_macro.f1.<tolerance>ms.<mode>

song_f1.minimum/p10/p25/median/p75/p90/maximum.<tolerance>ms.<mode>

class.<common_class>.precision.<tolerance>ms.<mode>
class.<common_class>.recall.<tolerance>ms.<mode>
class.<common_class>.f1.<tolerance>ms.<mode>
class.<common_class>.reference_support.<tolerance>ms.<mode>
class.<common_class>.prediction_support.<tolerance>ms.<mode>
```

Before creating any `class.<name>` key, validate `PublishedAggregateClass.common_class` against `get_args(CommonDrumClass)`. `PublishedAggregateClass` stores a bare string, so this projection boundary must fail closed on unknown/future/typo class names rather than minting a durable accidental metric key. This reuses the existing taxonomy; it does not create a second taxonomy.

Publish population metrics:

```text
population.total
population.success
population.failed
population.skipped
population.quarantined
population.reason.<reason>
```

Failure reasons must come from `COHORT_FAILURE_REASONS`.

Undefined metrics are omitted. Never publish NaN, Infinity, fabricated zeroes, or sentinel values. Preserve canonical `Decimal`/integer values in the pure projection and convert to finite floats only at the MLflow API boundary.

Do not publish one scalar metric per song; `per_song.csv` remains detailed evidence.

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

Never publish `event_diagnostics.jsonl`, audio, canonical audio, separator stems, prediction artifacts, checkpoints, runtime environments, or R2/cache data.

For each allowlisted basename, read through `read_regular_file_no_follow()`, hash the exact bytes with `sha256_hex()`, and represent it as the existing `PublishedArtifact`. The artifact basename is `artifact.path.name`.

## Projection fingerprint

`cohort_id` alone is insufficient for idempotency. Compute `crux.projection_sha256` locally, before importing MLflow, from canonical JSON containing:

- projection version;
- scope;
- deterministic run name;
- projected Crux identity tags excluding the fingerprint itself;
- scalar metric names plus canonical source values;
- each allowlisted artifact basename and SHA-256.

Any intentional mapping/metric/artifact change requires a new projection version rather than mutating completed V1 runs.

## Idempotent publication

Within the selected experiment, search non-deleted runs matching both `crux.cohort_id` and `crux.projection_version` with `max_results=1000`.

The publisher intentionally does not add pagination machinery. Because idempotency must fail closed, if the returned page length equals the 1000-result limit, treat the search as saturated and refuse to publish; an unseen completed run could otherwise cause a duplicate.

For an unsaturated result set:

1. Any matching run with a different fingerprint -> fail closed regardless of status.
2. Exactly one same-fingerprint `FINISHED` run -> no-op and return it.
3. More than one same-fingerprint `FINISHED` run -> fail closed.
4. Same-fingerprint `RUNNING` or `SCHEDULED` run -> fail closed as publication in progress.
5. Only same-fingerprint `FAILED` and/or `KILLED` runs -> retry by creating a fresh run.
6. Unknown run status -> fail closed.
7. No match -> create a fresh run.

`KILLED` is intentionally retryable like `FAILED`: interrupted/aborted publication is not a successful publication and must not permanently wedge the operator.

For a new run, log the complete projection and mark it `FINISHED` only after metrics and all five artifacts succeed. If any post-create operation fails, best-effort mark the new run `FAILED` and surface a generic publication error. Preserve provider/client exceptions only as Python causes; never interpolate raw provider errors into user-visible output.

Do not mutate/delete completed or failed runs automatically.

## MLflow API boundary

Use low-level `MlflowClient`, not global fluent run state. The path resolves/creates an experiment, searches runs, creates a run, sends one metric batch, uploads five artifacts, and terminates the run.

Automated behavior tests use a fake client. Separately, the optional-dependency verification must run with `--extra mlflow` without network access and instantiate `Metric("k", 1.0, 0, 0)` while asserting `MlflowClient` exposes:

```text
get_experiment_by_name
create_experiment
search_runs
create_run
log_batch
log_artifact
set_terminated
```

This catches client/API drift before the hosted smoke.

## CLI

Add exactly one command:

```text
crux benchmark publish-mlflow-cohort \
  --reports PATH \
  --scope broad|reviewed|pilot \
  [--experiment crux-benchmark]
```

The command validates reports and builds the projection before any MLflow import/network call.

Success stdout is canonical JSON with non-secret fields such as experiment ID/name, run ID, projection SHA, `created`, and status.

Exit-2 failures follow sibling benchmark publication commands: write a small canonical JSON error object to stdout and a bounded human-readable message to stderr. The JSON must remain parseable and must not contain credentials, tracking URI, raw provider exception text, or client objects.

## Testing

`tests/benchmark/test_reports.py` covers the convenience loader and proves it retains full cross-file validation.

`tests/benchmark/test_mlflow_export.py` covers exact six-view validation, CommonDrumClass validation, failure-reason validation, metric projection, undefined omission, `PublishedArtifact` allowlist/hashes, deterministic fingerprinting, no base MLflow import, idempotency status matrix including KILLED retry, saturated search fail-closed, failed-run cleanup, and generic transport errors.

`tests/test_cli_benchmark.py` covers created/no-op JSON success, JSON exit-2 failure, invalid scope, malformed reports before publisher invocation, missing dependency/configuration, publisher conflicts, and secret-redaction.

No automated test contacts DagsHub or another real server.

## Real hosted acceptance gate

Before HPA-86 is ready, use already-produced canonical OaF evidence to publish five runs to one `crux-benchmark` experiment:

- OaF full-mix broad;
- OaF full-mix reviewed;
- OaF full-mix pilot;
- OaF Spleeter pilot;
- OaF HTDemucs pilot.

The reviewed run is required so the `reviewed` scope is actually exercised rather than merely supported by the enum.

Record only non-secret experiment/run IDs, projection fingerprints, created/no-op outcomes, ability to filter by scope/model/input view, metric visibility, and confirmation that only the five small artifacts uploaded. Republish one cohort to prove it returns the same finished run with `created=false`.

If the already-produced workstation reports or hosted credentials are unavailable, keep the PR draft. Do not rerun OaF, separators, R2 sync, or reference construction under HPA-86 merely to satisfy this smoke gate.

MuScriptor/IDM publication is not required for HPA-86; future canonical cohorts use the same command.

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
src/cli/benchmark.py
tests/benchmark/test_reports.py
tests/test_cli_benchmark.py
pyproject.toml
uv.lock
README.md
```

No expected production changes to model backends/runners, separator execution, taxonomy definitions, scoring, predictions, manifests, or R2/cache code.

## Delivery boundary

HPA-86 remains one ticket and one PR. The same PR carries the approved design, implementation plan, implementation, tests, documentation, and hosted smoke evidence.
