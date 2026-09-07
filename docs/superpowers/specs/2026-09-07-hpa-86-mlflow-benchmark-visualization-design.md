# HPA-86 MLflow Benchmark Visualization Design

## Summary

Add one optional, read-only projection from Crux's canonical published benchmark reports into a standard MLflow tracking server. The first operational target is hosted MLflow for zero-maintenance hobby-project visualization (initially DagsHub), but the repository must contain no DagsHub-specific API or SDK dependency. Any MLflow-compatible server can be selected through the standard MLflow environment configuration.

Crux remains the scientific source of truth. MLflow is an index, comparison, and visualization surface only.

The integration is deliberately post-publication:

```text
model inference
    -> persisted Crux prediction evidence
    -> Crux scoring
    -> canonical HPA-325 reports
    -> optional explicit MLflow publication
```

A tracking-server outage, authentication failure, or missing MLflow dependency must never affect inference, scoring, canonical report publication, or benchmark identity.

## Problem

Crux now publishes rich deterministic benchmark evidence, but it is cumbersome to explore visually. HPA-325 already persists aggregate, per-song, per-class, distribution, population, and provenance data. HPA-562 adds paired comparison publications. The data is trustworthy and reproducible, but the primary interfaces are JSON/CSV/JSONL/Markdown files rather than an interactive experiment UI.

Future fine-tuning will also need a conventional experiment-tracking surface for run comparison. Introducing the standard MLflow tracking contract now gives the current pretrained benchmark a usable UI without committing Crux to a proprietary experiment model or requiring a later tracking migration.

The design must avoid replacing the benchmark pipeline with MLflow. Crux has stronger domain-specific identity and evidence rules than a generic experiment tracker, and those contracts remain load-bearing.

## Goals

1. Make existing canonical cohort results easy to compare in an MLflow UI.
2. Preserve Crux's existing immutable report, model-lock, reference, input-view, and scoring identities.
3. Keep MLflow entirely optional and outside the inference/scoring critical path.
4. Keep the integration host-neutral by using only the standard MLflow Python client and environment configuration.
5. Keep hosted storage small enough for hobby-project use by publishing metadata, scalar metrics, and selected small report artifacts only.
6. Establish a tracking contract that future fine-tuning work can reuse without changing today's benchmark architecture.
7. Deliver the complete integration as one HPA-86 PR.

## Non-goals

HPA-86 does not add:

- model fine-tuning or training;
- hyperparameter search or Optuna;
- MLflow Model Registry;
- MLflow dataset objects;
- MLflow autologging;
- CI-driven publication;
- DagsHub SDK calls or DagsHub-specific repository APIs;
- custom Plotly/Matplotlib dashboards;
- MLflow evaluation-framework integration;
- audio, separator-stem, prediction, checkpoint, or R2 synchronization;
- a replacement for HPA-325 or HPA-562 reports;
- a new scorer, report schema, runner framework, or experiment database inside Crux.

These can be separately scoped later if real usage justifies them.

## Existing seams to reuse

The implementation must project from existing published evidence rather than parse model-specific run internals.

`src/benchmark/reports.py` already owns the canonical HPA-325 persisted-report contract:

- `PublishedCohortReports`
- `PublishedAggregate`
- `PublishedSongRow`
- `PublishedClassRow`
- `PublishedItemRow`
- `read_cohort_reports(...)`

That typed reader is the correct source for MLflow metrics.

`src/benchmark/backend_identity.py` already owns canonical JSON encoding and SHA-256 helpers. HPA-86 reuses those helpers for a deterministic projection fingerprint rather than introducing a second serialization/hash convention.

The existing benchmark CLI group in `src/cli/benchmark.py` is the only new command surface.

No model-specific OaF, MuScriptor, IDM, separator, prediction, taxonomy, or scoring module should need to change.

## Architecture

### Canonical versus projected state

```text
                         canonical
                    +----------------+
                    | Crux manifests |
                    | model locks    |
                    | predictions    |
                    | HPA-325 reports|
                    +--------+-------+
                             |
                      validated read
                             |
                             v
                    +----------------+
                    | MLflow         |
                    | projection     |
                    +--------+-------+
                             |
                   standard MLflow API
                             |
               +-------------+-------------+
               |                           |
               v                           v
        hosted MLflow UI             local/other MLflow
        (DagsHub first)              server later
```

MLflow data is disposable and reproducible. Deleting every MLflow run must not lose any benchmark evidence required to regenerate it.

### Dependency boundary

Use an optional extra:

```toml
[project.optional-dependencies]
mlflow = ["mlflow-skinny>=3.16,<4"]
```

`mlflow-skinny` is sufficient because Crux only needs the remote tracking client. The base install does not gain MLflow server, SQL backend, UI, or data-science dependencies.

The repository must not import `mlflow` from normal benchmark module-import paths. The optional dependency is imported only when the explicit MLflow publication path executes. Missing dependency errors must tell the operator to run the command with `--extra mlflow`.

No `dagshub` Python package is added.

### Server configuration

Use the standard MLflow environment contract only:

```text
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
```

DagsHub is an operational default, not an application dependency. A local MLflow server or another compatible hosted MLflow service works with the same Crux command.

Secrets are never accepted as CLI arguments, persisted in Crux reports, committed to configuration, or echoed in canonical CLI output.

## Self-identifying published-report loader

The current `read_cohort_reports(report_dir, expected_identity=...)` API intentionally verifies reports against a caller-supplied `CohortIdentity`. That is ideal for benchmark pipelines that already know the expected lineage, but a generic visualization exporter should not know which runner produced the report.

Add one model-neutral helper in `reports.py`:

```python
load_published_cohort_reports(report_dir: Path) -> PublishedCohortReports
```

Its behavior is narrowly defined:

1. read canonical `summary.json` through the existing no-follow/strict-JSON path;
2. validate that the summary uses the existing single-cohort report schema;
3. construct `CohortIdentity` from the existing summary identity object;
4. delegate all full report validation to `read_cohort_reports(report_dir, expected_identity=identity)`;
5. return the resulting `PublishedCohortReports`.

This is not a second report reader. It is a convenience entry point into the existing reader and introduces no schema change.

## Projection model

Add a focused `src/benchmark/mlflow_export.py` module. It owns both the pure Crux-to-MLflow projection and the small side-effecting publisher.

The module should keep pure mapping separate from network operations so almost all behavior can be tested without MLflow or a tracking server.

Conceptually:

```text
PublishedCohortReports + scope
        |
        v
MlflowProjection
  - run name
  - identity tags/params
  - scalar metrics
  - artifact allowlist + hashes
  - projection fingerprint
        |
        v
publish via MlflowClient
```

### Scope

The CLI requires one closed scope value:

```text
broad
reviewed
pilot
```

Scope is presentation metadata. It must not alter or reinterpret any benchmark score.

### Projection version

Freeze the initial mapping as:

```text
crux.mlflow-projection/v1
```

Changing metric names, artifact membership, fingerprint semantics, or identity projection in a way that changes the published representation requires a new projection version. The underlying HPA-325 report schema remains unchanged.

## Run identity and metadata

MLflow's generated run ID is transport metadata only. Crux identity remains authoritative.

Every published run receives Crux-owned tags/params for at least:

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

Do not use the reserved `mlflow.` prefix for Crux-owned tags.

The run name should be deterministic and readable, derived from model/input/scope/cohort identity rather than timestamps. The exact format is part of the v1 projection and must have a focused test.

Do not duplicate every identity value into both a parameter and a tag without a UI need. Use one stable ownership rule in the implementation plan: tags for identity/search/filter fields; params only for immutable projection configuration that benefits from the parameter column. The projection fingerprint binds both.

## Scalar metric projection

MLflow scalar metrics are the primary visualization surface.

For each canonical aggregate at 30, 50, and 100 ms in both `raw` and `aligned` modes, publish:

```text
event_micro.precision.<tolerance>ms.<mode>
event_micro.recall.<tolerance>ms.<mode>
event_micro.f1.<tolerance>ms.<mode>
song_macro.f1.<tolerance>ms.<mode>
class_macro.f1.<tolerance>ms.<mode>
```

Publish song-F1 distribution values for the same tolerance/mode:

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

The common taxonomy is already closed to safe stable IDs (`kick`, `snare`, `hihat`, `crash`, `ride`, `tom`); HPA-86 does not add generic key sanitization or a second taxonomy.

Publish population counts once per run:

```text
population.total
population.success
population.failed
population.skipped
population.quarantined
```

Also publish the existing closed failure-reason counts as:

```text
population.reason.<reason>
```

`None`/undefined metrics are omitted. Do not emit NaN, Infinity, fabricated zeroes, or sentinel values.

Convert Crux's canonical six-decimal `Decimal` values to finite MLflow floating-point values only at the MLflow API boundary. The canonical source value remains the report token; MLflow rounding is not benchmark evidence.

### Per-song data

Do not publish one MLflow scalar metric per song. Hundreds of song-specific metric keys would make the experiment view harder to use and would couple the tracker schema to corpus membership.

`per_song.csv` remains the detailed song-level evidence artifact. The v1 projection exposes its distribution through the existing aggregate percentiles. If real MLflow usage proves that interactive song-level tables are necessary, that is a follow-up projection version rather than speculative HPA-86 scope.

## Artifact policy

Publish only these existing canonical report files under a stable MLflow artifact subdirectory:

```text
summary.json
summary.md
items.csv
per_song.csv
per_class.csv
```

Do not publish:

```text
event_diagnostics.jsonl
raw audio
canonicalized audio
separator stems
prediction artifacts
model checkpoints
R2 cache/object data
runtime environments
```

The publisher verifies every allowlisted file as a regular no-follow file before upload and includes each content SHA-256 in the projection fingerprint.

The allowlist is closed. Adding an artifact changes the projection contract and requires deliberate review.

## Deterministic projection fingerprint

Idempotency must not rely on `cohort_id` alone.

Compute `crux.projection_sha256` from a canonical JSON payload containing exactly the v1 projected meaning:

- projection version;
- scope;
- deterministic run name;
- projected Crux identity metadata;
- projected params/tags excluding server-generated values;
- all scalar metric keys and canonical source tokens;
- each allowlisted artifact relative path and SHA-256.

Use the repository's existing canonical JSON and SHA-256 helpers.

The fingerprint is local and deterministic. It must be computable before importing/contacting MLflow so projection correctness can be tested offline.

## Idempotent publication

Within the selected MLflow experiment, search active runs by both:

```text
crux.cohort_id
crux.projection_version
```

Behavior:

1. **No matching run**: create one run, log the complete projection, mark it finished, return its MLflow run ID and `created=true`.
2. **Exactly one matching run with the same `crux.projection_sha256`**: perform no writes and return the existing run ID with `created=false`.
3. **Exactly one matching run with a different projection fingerprint**: fail closed with a projection-conflict error; do not mutate or delete the existing run.
4. **Multiple matching runs**: fail closed because idempotency has already been violated; do not guess a winner.

Do not update existing benchmark runs in place. A future intentional mapping change uses a new projection version.

If publication fails after creating a new MLflow run, attempt to mark that run failed and surface the original publication error. Crux canonical artifacts remain untouched.

## MLflow client usage

Prefer `MlflowClient` over global fluent run state for the publication path. The exporter should explicitly:

- resolve/create the requested experiment;
- search matching runs;
- create a run when needed;
- log tags/params/metrics;
- upload the allowlisted artifacts;
- terminate the run as finished or failed.

This keeps the side-effect boundary obvious and easier to fake in unit tests.

No background worker, retry engine, queue, or local publication ledger is introduced. An operator can rerun the explicit command; idempotency makes the retry safe.

## CLI

Add one command to the existing benchmark group:

```text
crux benchmark publish-mlflow-cohort \
  --reports PATH \
  --scope broad|reviewed|pilot \
  [--experiment crux-benchmark]
```

The command:

1. loads and fully validates the canonical reports before MLflow import/network work;
2. constructs the deterministic projection;
3. invokes the publisher;
4. prints one canonical JSON result containing only non-secret publication metadata;
5. exits 0 for both newly created and exact no-op publications;
6. exits 2 for malformed reports, missing optional dependency, configuration/auth/network errors, or projection conflicts.

Do not add tracking URI, username, password, DagsHub repository, model name, metric selection, or artifact selection flags.

## Hosted operation

DagsHub-hosted MLflow is the initial zero-maintenance target because it accepts the normal MLflow tracking protocol. HPA-86 documentation should show the standard environment-variable setup and keep all host-specific instructions outside application configuration.

The code must also work unchanged against a local/self-hosted MLflow server or another fully compatible hosted service. Host choice is therefore reversible without a Crux code migration.

Do not encode current vendor pricing or quota numbers in repository code. External service plans can change. The closed small-artifact policy is the cost-control mechanism Crux owns.

## Testing strategy

### Report loader tests

Extend `tests/benchmark/test_reports.py` to prove the self-identifying helper:

- accepts a canonical published report and returns the same typed content as the explicit-identity reader;
- rejects malformed/non-canonical summary identity;
- delegates full cross-file integrity validation rather than weakening it.

### Pure projection tests

Add `tests/benchmark/test_mlflow_export.py` covering at least:

- deterministic run name;
- exact identity/scope/projection tags;
- all 30/50/100 ms raw/aligned event-micro, song-macro, and class-macro metric keys;
- all distribution keys;
- all six common-class aggregate mappings when present;
- population and closed failure-reason counts;
- undefined metric omission;
- closed artifact allowlist;
- artifact SHA-256 participation in the fingerprint;
- stable projection fingerprint for the same evidence;
- changed scope/metric/artifact content changes the fingerprint;
- no import of MLflow for pure projection construction.

### Publisher tests

Use a small fake client boundary; never contact a real server from automated tests.

Cover:

- experiment resolution/creation;
- first publication;
- exact duplicate no-op;
- same cohort/projection-version fingerprint conflict;
- multiple-match conflict;
- artifact upload allowlist;
- run marked failed on post-create publish error;
- server/auth/network exceptions translated to one bounded publication error without changing Crux files.

### CLI tests

Extend `tests/test_cli_benchmark.py` for:

- successful create output/exit 0;
- duplicate no-op output/exit 0;
- malformed canonical report exit 2;
- missing `mlflow-skinny` guidance exit 2;
- publisher/config/auth/network failure exit 2;
- no credential values in output.

### Repository gates

Final deterministic gates remain:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only src
```

The optional MLflow-focused suites also run with:

```bash
uv run --extra mlflow pytest tests/benchmark/test_mlflow_export.py tests/test_cli_benchmark.py -q
```

## Real acceptance gate

Automated tests do not prove hosted compatibility. Before HPA-86 is considered complete, perform one operator-authenticated publication to a real hosted MLflow server using environment-provided credentials.

Initial real evidence should publish the already-available OaF cohorts that do not depend on the still-blocked MuScriptor model lock:

- OaF full-mix broad;
- OaF full-mix reviewed;
- OaF full-mix separation-pilot view;
- OaF Spleeter separation-pilot view;
- OaF HTDemucs separation-pilot view.

Record on HPA-86/its PR:

- experiment name;
- non-secret MLflow run IDs;
- projection version/fingerprints;
- whether each publish created or no-op'd;
- confirmation that model/input/scope filters work;
- confirmation that 30/50/100 ms raw/aligned metrics and per-class metrics are visible;
- confirmation that only the five allowlisted report artifacts were uploaded;
- a second publication of one cohort proving no duplicate run is created.

If hosted credentials or the service are unavailable, deterministic implementation may be reviewed, but the PR remains draft/not ready until the real smoke is recorded. Do not substitute fixtures for this acceptance gate.

MuScriptor and IDM publication are not prerequisites for HPA-86 because their production evidence has separate upstream operational dependencies. Once those cohorts exist, the same command publishes them without new code.

## Documentation

Add a small `docs/benchmark/mlflow.md` during implementation with:

- why MLflow is a projection rather than source of truth;
- optional dependency command;
- standard environment configuration;
- DagsHub hosted setup example;
- local/other-server compatibility note;
- CLI examples for broad/reviewed/pilot cohorts;
- metric naming contract;
- artifact allowlist;
- idempotency behavior;
- reminder that secrets and large benchmark assets remain outside MLflow.

README should only link to that guide; do not duplicate the full setup.

## Planned file surface

Expected implementation surface for the single PR:

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

No changes are expected in OaF/MuScriptor/IDM backends, corpus runners, separation execution, taxonomy, scoring, prediction artifacts, reference manifests, or R2/cache code.

## Delivery boundary

HPA-86 is one ticket and one PR. The draft begins with this design and its implementation plan; implementation, tests, documentation, and hosted smoke evidence land on the same branch. Do not create a separate implementation PR.
