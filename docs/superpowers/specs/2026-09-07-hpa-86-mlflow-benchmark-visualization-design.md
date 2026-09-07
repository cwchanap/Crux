# HPA-86 MLflow Benchmark Visualization Design

## Summary

Add one optional, read-only projection from Crux's canonical published benchmark reports into a standard MLflow tracking server. The first operational target is hosted MLflow for zero-maintenance hobby-project visualization (initially DagsHub), but Crux contains no DagsHub-specific API or SDK dependency.

Crux remains the scientific source of truth. MLflow is only an index, comparison, artifact-browsing, and visualization surface.

```text
model inference
    -> persisted Crux prediction evidence
    -> Crux scoring
    -> canonical HPA-325 reports
    -> optional explicit MLflow publication
```

A missing dependency, tracking-server outage, or authentication failure must never affect inference, scoring, canonical report publication, or benchmark identity.

## Problem

Crux already persists deterministic aggregate, per-song, per-class, distribution, population, and provenance evidence. HPA-562 adds paired comparison publications. The evidence is trustworthy but cumbersome to explore because the primary interfaces are JSON/CSV/JSONL/Markdown files.

Future fine-tuning will also need conventional experiment tracking. Adopting the standard MLflow tracking contract now gives the pretrained benchmark an interactive UI without replacing Crux's stronger domain-specific identity/evidence rules or committing the repository to a proprietary tracking model.

## Goals

1. Compare existing canonical cohort results in an MLflow UI.
2. Preserve Crux report/model-lock/reference/input-view/scoring identities as authoritative.
3. Keep MLflow optional and outside inference/scoring.
4. Use only the standard MLflow client and environment configuration so the hosted provider is replaceable.
5. Keep hosted storage small by publishing metadata, scalar metrics, and selected small reports only.
6. Establish a tracking contract future fine-tuning can reuse.
7. Deliver design, plan, implementation, tests, docs, and hosted smoke evidence in one HPA-86 PR.

## Non-goals

HPA-86 does not add:

- training or fine-tuning;
- hyperparameter search/Optuna;
- MLflow Model Registry or dataset objects;
- autologging;
- CI publication;
- DagsHub SDK calls;
- custom Plotly/Matplotlib dashboards;
- MLflow evaluation-framework integration;
- audio/stem/prediction/checkpoint/R2 synchronization;
- a replacement for HPA-325 or HPA-562 reports;
- a new scorer, report schema, runner framework, or Crux experiment database.

## Existing seams to reuse

`src/benchmark/reports.py` already owns the canonical persisted-report contract through `PublishedCohortReports`, its typed rows/aggregates, and `read_cohort_reports(...)`. MLflow must project from that reader rather than model-specific run internals.

`src/benchmark/backend_identity.py` already owns canonical JSON and SHA-256 helpers. Reuse those for the projection fingerprint.

`src/cli/benchmark.py` is the only new command surface.

No OaF, MuScriptor, IDM, separator, prediction, taxonomy, scoring, reference-manifest, or R2/cache module should need to change.

## Architecture

```text
                    canonical Crux evidence
         manifests / locks / predictions / reports
                              |
                         validated read
                              v
                    +-------------------+
                    | MLflow projection |
                    +---------+---------+
                              |
                    standard MLflow API
                              |
              +---------------+---------------+
              v                               v
       hosted MLflow UI                local/other MLflow
       (DagsHub first)                 server later
```

MLflow state is disposable. Deleting every MLflow run must not lose evidence needed to regenerate it.

### Optional dependency

Add:

```toml
[project.optional-dependencies]
mlflow = ["mlflow-skinny>=3.16,<4"]
```

`mlflow-skinny` is enough for a remote tracking client. Normal Crux imports must not require MLflow. Import the optional package only inside the explicit publication path and give missing-dependency guidance for `uv run --extra mlflow ...`.

Do not add the `dagshub` package.

### Server configuration

Use the standard MLflow environment contract:

```text
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
```

Do not accept credentials/tracking URI as Crux CLI options, persist them in benchmark artifacts, or echo secrets.

DagsHub is the initial hosted target, not an application dependency. A local server, MLflow Cloud, or another compatible MLflow host should work without code changes.

## Self-identifying published-report loader

Add one model-neutral helper in `reports.py`:

```python
load_published_cohort_reports(report_dir: Path) -> PublishedCohortReports
```

It must:

1. read canonical `summary.json` through the existing no-follow/strict-JSON path;
2. verify the existing single-cohort report schema;
3. construct the existing `CohortIdentity` from `summary.json`;
4. call `read_cohort_reports(report_dir, expected_identity=identity)`;
5. return its typed result.

This is a convenience entry point, not a second report reader and not a schema change.

## Projection module

Create `src/benchmark/mlflow_export.py`. Keep pure Crux-to-projection mapping separate from MLflow/network effects so projection correctness is testable without the optional dependency or server.

Conceptually:

```text
PublishedCohortReports + scope + report paths
             |
             v
      MlflowProjection
        run_name
        tags / params
        metrics
        artifact hashes
        projection_sha256
             |
             v
       MlflowClient publisher
```

The allowed scope is a closed value:

```text
broad | reviewed | pilot
```

Scope is presentation metadata only and cannot alter scores.

Freeze the first representation as:

```text
crux.mlflow-projection/v1
```

Changing metric names, identity mapping, artifact membership, or fingerprint semantics requires a new projection version; it does not change HPA-325 report schemas.

## Run metadata

MLflow's generated run ID is transport metadata only. Publish Crux-owned tags for at least:

```text
crux.projection_version
crux.projection_sha256
crux.scope
crux.cohort_id
crux.backend_id
crux.model_id
crux.model_lock_sha256
crux.backend_descriptor_sha256
crux.input_view_id
crux.reference_manifest_sha256
crux.reference_timing_version
crux.taxonomy_version
crux.lane_map_version
crux.prediction_map_version
crux.scoring_version
```

Do not use the reserved `mlflow.` tag prefix.

Use tags for identity/search/filter fields. Do not duplicate the same Crux identity into params unless a concrete MLflow UI need is demonstrated. Any v1 params should be limited to immutable projection configuration, and the fingerprint binds them.

Run name is deterministic/readable from model + input view + scope + cohort identity, never a timestamp.

## Scalar metric projection

MLflow scalar metrics are the v1 visualization surface.

For every canonical 30/50/100 ms × raw/aligned aggregate:

```text
event_micro.precision.<tolerance>ms.<mode>
event_micro.recall.<tolerance>ms.<mode>
event_micro.f1.<tolerance>ms.<mode>
song_macro.f1.<tolerance>ms.<mode>
class_macro.f1.<tolerance>ms.<mode>
```

Publish the same tolerance/mode's song-F1 distribution:

```text
song_f1.minimum.<tolerance>ms.<mode>
song_f1.p10.<tolerance>ms.<mode>
song_f1.p25.<tolerance>ms.<mode>
song_f1.median.<tolerance>ms.<mode>
song_f1.p75.<tolerance>ms.<mode>
song_f1.p90.<tolerance>ms.<mode>
song_f1.maximum.<tolerance>ms.<mode>
```

Publish aggregate common-class metrics:

```text
class.<common_class>.precision.<tolerance>ms.<mode>
class.<common_class>.recall.<tolerance>ms.<mode>
class.<common_class>.f1.<tolerance>ms.<mode>
class.<common_class>.reference_support.<tolerance>ms.<mode>
class.<common_class>.prediction_support.<tolerance>ms.<mode>
```

The common taxonomy is already closed to `kick`, `snare`, `hihat`, `crash`, `ride`, `tom`; do not add generic key sanitization or another taxonomy.

Publish population values:

```text
population.total
population.success
population.failed
population.skipped
population.quarantined
population.reason.<closed_failure_reason>
```

Undefined metrics are omitted. Never emit NaN, Infinity, fabricated zeroes, or sentinels.

Convert canonical six-decimal `Decimal` values to finite floats only at the MLflow API boundary. MLflow values are visualization copies, never benchmark evidence.

### Per-song detail

Do not create one MLflow metric key per song. `per_song.csv` remains the detail evidence; existing distribution percentiles provide a useful v1 overview. If actual use proves interactive song-level tables are necessary, scope that as a follow-up projection version rather than adding speculative complexity here.

## Artifact policy

Upload only these canonical report files under a stable MLflow artifact directory:

```text
summary.json
summary.md
items.csv
per_song.csv
per_class.csv
```

Never upload under HPA-86:

```text
event_diagnostics.jsonl
raw/canonical audio
separator stems
prediction artifacts
checkpoints
R2/cache objects
runtime environments
```

Verify every allowlisted artifact as a regular no-follow file and include its content SHA-256 in the projection fingerprint. The allowlist is closed for v1.

## Projection fingerprint

`cohort_id` alone is not enough for safe idempotency.

Compute `crux.projection_sha256` with existing canonical JSON/SHA helpers from exactly the projected meaning:

- projection version;
- scope;
- deterministic run name;
- projected Crux identity/tags;
- any v1 params;
- scalar metric keys plus canonical source tokens;
- each allowlisted relative artifact path and SHA-256.

The fingerprint is computed locally before importing/contacting MLflow.

## Idempotent publication and failed attempts

Within the requested experiment, search existing non-deleted runs matching:

```text
crux.cohort_id
crux.projection_version
```

Inspect both fingerprint and run status before writing.

Behavior:

1. **Any matching run has a different `crux.projection_sha256`**: fail closed. The same Crux cohort/projection version cannot mean two different projections, even if the older attempt failed.
2. **Exactly one matching `FINISHED` run has the same fingerprint**: no-op; return its run ID with `created=false`.
3. **More than one matching `FINISHED` run**: fail closed because completed-publication idempotency was violated.
4. **A matching `RUNNING`/scheduled run exists**: fail closed as publication-in-progress; do not create a concurrent duplicate.
5. **Only same-fingerprint failed/killed attempts exist**: retry is allowed by creating a fresh run. Failed attempts are not successful publications.
6. **No matching run exists**: create a new run.

For a new run, log the complete projection and mark it `FINISHED` only after tags/metrics/artifacts succeed. If any post-create operation fails, best-effort mark that new run `FAILED` and surface the original error. Crux artifacts remain untouched.

This rule fixes an important retry case: a partial failed run can never satisfy the successful no-op path.

Do not mutate a completed benchmark run in place or delete failed runs automatically. A deliberate mapping change uses a new projection version.

## MLflow client usage

Use `MlflowClient` rather than global fluent run state. The side-effect path explicitly:

- resolves/creates the experiment;
- searches matching runs;
- creates a run when needed;
- logs tags/any params/metrics;
- uploads the five allowlisted artifacts;
- terminates the run as `FINISHED` or best-effort `FAILED`.

Do not add a background worker, queue, retry engine, or local publication database. The operator reruns the explicit command; idempotency makes normal retries safe.

## CLI

Add one command:

```text
crux benchmark publish-mlflow-cohort \
  --reports PATH \
  --scope broad|reviewed|pilot \
  [--experiment crux-benchmark]
```

The command must:

1. fully validate canonical reports before MLflow import/network work;
2. build the deterministic projection;
3. invoke the publisher;
4. print canonical JSON with only non-secret result metadata;
5. exit 0 for newly-created and exact-no-op publications;
6. exit 2 for malformed reports, missing optional dependency, configuration/auth/network errors, publication-in-progress, or projection conflicts.

Do not add tracking URI, username, password, host, model, metric-selection, or artifact-selection flags.

## Hosted operation and cost boundary

DagsHub-hosted MLflow is the first smoke target. Documentation should show its normal MLflow environment-variable setup, but the code stays provider-neutral.

Do not encode vendor pricing/quota numbers in code. External plans change. Crux controls cost through the optional client dependency and closed small-artifact policy.

## Testing

### `tests/benchmark/test_reports.py`

Prove `load_published_cohort_reports()`:

- produces the same typed result as the explicit-identity reader;
- rejects malformed/non-canonical summary identity;
- still exercises full cross-file integrity validation.

### `tests/benchmark/test_mlflow_export.py`

Pure projection coverage:

- deterministic run name;
- identity/scope/projection tags;
- all 30/50/100 raw/aligned aggregate keys;
- all distribution keys;
- all six common classes when present;
- population/failure counts;
- undefined-value omission;
- artifact allowlist and hashes;
- stable fingerprint;
- scope/metric/artifact content changes fingerprint;
- pure projection construction does not import MLflow.

Publisher coverage with a fake client boundary:

- experiment resolution/creation;
- first publication;
- finished same-fingerprint no-op;
- failed same-fingerprint retry creates a new run;
- running attempt blocks concurrent publish;
- differing fingerprint conflict regardless of status;
- multiple finished-run conflict;
- artifact allowlist;
- post-create errors mark the new run failed;
- server/auth/network exceptions become one bounded publication error without changing Crux files.

### `tests/test_cli_benchmark.py`

Cover:

- created result/exit 0;
- no-op result/exit 0;
- malformed report exit 2;
- missing dependency guidance exit 2;
- publisher/config/auth/network/conflict exit 2;
- credentials never appear in output.

### Repository gates

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only src
uv run --extra mlflow pytest tests/benchmark/test_mlflow_export.py tests/test_cli_benchmark.py -q
```

No automated test contacts DagsHub or another real tracking server.

## Real hosted acceptance gate

Before HPA-86 is complete, run one operator-authenticated hosted smoke using environment-provided credentials. Initial evidence should use already-available OaF cohorts rather than waiting for the blocked MuScriptor model lock:

- OaF full-mix broad;
- OaF full-mix reviewed;
- OaF full-mix pilot;
- OaF Spleeter pilot;
- OaF HTDemucs pilot.

Record on HPA-86/PR:

- experiment name;
- non-secret MLflow run IDs;
- projection version/fingerprints;
- `created` versus no-op outcomes;
- ability to filter by model/input/scope;
- visibility of 30/50/100 raw/aligned and per-class metrics;
- confirmation that only the five allowlisted artifacts uploaded;
- a second publish proving one cohort creates no duplicate finished run.

If hosted credentials/service access are unavailable, deterministic implementation may be reviewed but the PR stays draft/not-ready until this real smoke is recorded. Do not substitute fixtures.

MuScriptor/IDM publication is not a prerequisite for HPA-86. Their eventual canonical cohorts use the same command without new code.

## Documentation

Implementation adds `docs/benchmark/mlflow.md` covering:

- projection-versus-source-of-truth boundary;
- optional dependency command;
- standard MLflow environment variables;
- DagsHub hosted example and provider-neutral compatibility note;
- CLI examples;
- metric naming;
- artifact allowlist;
- idempotency/retry behavior;
- secret/large-asset exclusions.

README receives only a link to the guide.

## Planned file surface

```text
Create:
  src/benchmark/mlflow_export.py
  tests/benchmark/test_mlflow_export.py
  docs/benchmark/mlflow.md

Modify:
  src/benchmark/reports.py
  src/cli/benchmark.py
  tests/benchmark/test_reports.py
  tests/test_cli_benchmark.py
  pyproject.toml
  uv.lock
  README.md

Planning:
  docs/superpowers/specs/2026-09-07-hpa-86-mlflow-benchmark-visualization-design.md
  docs/superpowers/plans/2026-09-07-hpa-86-mlflow-benchmark-visualization.md
```

No expected changes to model backends/runners, separator execution, taxonomy, scoring, predictions, manifests, or R2/cache code.

## Delivery boundary

HPA-86 is one ticket and one PR. The same branch carries the approved design, implementation plan, implementation, tests, documentation, and hosted smoke evidence. Do not create a second implementation PR.
