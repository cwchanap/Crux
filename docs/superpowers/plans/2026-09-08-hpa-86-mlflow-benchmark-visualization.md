# HPA-86 MLflow Benchmark Visualization Implementation Plan

> **Execution:** implement task-by-task on draft PR #34 / `agent/hpa-86-mlflow-benchmark-visualization`. Use TDD for production behavior. One HPA-86 ticket, one PR.

**Goal:** Add one optional, host-neutral MLflow projection over canonical Crux cohort reports so existing benchmark results are browsable now and the same tracking contract can support later fine-tuning.

**Architecture:** canonical HPA-325 reports remain authoritative. `reports.py` gains one self-identifying convenience loader; `mlflow_export.py` builds a deterministic pure projection and then optionally publishes it through a lazily imported `MlflowClient`. No runner/scorer/separator path knows about MLflow.

**Spec:** `docs/superpowers/specs/2026-09-07-hpa-86-mlflow-benchmark-visualization-design.md`

## Global constraints

- Base is `main` at `31b7784ddb951ec2f780d3b00554ec948415f526`; re-check assumptions if main moves before implementation.
- MLflow remains absent from the base dependency set; add only `mlflow-skinny>=3.16,<4` under optional extra `mlflow`.
- No DagsHub SDK or provider-specific API.
- Require explicit `MLFLOW_TRACKING_URI`; do not add credential URI/user/password CLI flags.
- V1 projection is exactly `crux.mlflow-projection/v1`.
- V1 scope is exactly `broad | reviewed | pilot`.
- V1 aggregate grid is exactly 30/50/100 ms × raw/aligned; do not bind this freeze to `DEFAULT_TOLERANCES_MS`.
- V1 artifact allowlist is exactly `summary.json`, `summary.md`, `items.csv`, `per_song.csv`, `per_class.csv` under `crux-reports/`.
- Reuse `PublishedArtifact`; do not add `ProjectedArtifact`.
- Validate durable class metric keys against `get_args(CommonDrumClass)`.
- Validate failure-reason metric keys against `COHORT_FAILURE_REASONS`.
- Undefined metrics are omitted; conversion to float happens only at the MLflow API boundary.
- No per-song scalar metrics.
- Idempotency is scoped by `crux.cohort_id + crux.projection_version` and a deterministic projection SHA.
- `FAILED` and `KILLED` same-fingerprint runs are retryable; `RUNNING`/`SCHEDULED`, unknown status, mismatching fingerprint, multiple finished runs, or saturated run search fail closed.
- User-visible provider/client failures are generic; raw exceptions remain chained causes only.
- Success and exit-2 failure stdout are canonical JSON, matching sibling publication-command conventions.
- No Model Registry, datasets, autologging, training, Optuna, CI publication, dashboard framework, R2 sync, or model-runner changes.
- Automated tests never contact DagsHub or another real MLflow server.

## File surface

Create:

```text
src/benchmark/mlflow_export.py
tests/benchmark/test_mlflow_export.py
docs/benchmark/mlflow.md
```

Modify:

```text
src/benchmark/reports.py
tests/benchmark/test_reports.py
src/cli/benchmark.py
tests/test_cli_benchmark.py
pyproject.toml
uv.lock
README.md
```

Do not change `cohort_scoring.py`, model runners/backends, separation execution, prediction artifacts, taxonomy definitions, manifests, or R2/cache production code.

---

## Task 1 — Self-identifying canonical cohort-report loader

**Files:** `src/benchmark/reports.py`, `tests/benchmark/test_reports.py`

### 1.1 Write red loader-equivalence test

Add `load_published_cohort_reports` to the existing report imports and test:

```python
def test_load_published_cohort_reports_reads_identity_from_summary(tmp_path: Path) -> None:
    write_cohort_reports(_result(), tmp_path)

    explicit = read_cohort_reports(tmp_path, expected_identity=_identity())
    discovered = load_published_cohort_reports(tmp_path)

    assert discovered == explicit
```

Run:

```bash
uv run pytest tests/benchmark/test_reports.py::test_load_published_cohort_reports_reads_identity_from_summary -q
```

Expected: fails because the helper does not exist.

### 1.2 Extract identity construction only

Refactor existing `_parse_summary_identity()` so construction is reusable:

```python
def _cohort_identity_from_summary(value: object) -> CohortIdentity:
    payload = _require_keys(value, _IDENTITY_FIELDS, "summary identity")
    for field in _IDENTITY_FIELDS:
        _json_text(payload[field], f"identity.{field}")
    try:
        return CohortIdentity(**payload)  # type: ignore[arg-type]
    except (TypeError, ValueError, StrictJsonError) as error:
        _report_error(f"summary identity is malformed: {error}")


def _parse_summary_identity(value: object, expected: CohortIdentity) -> CohortIdentity:
    identity = _cohort_identity_from_summary(value)
    if identity != expected:
        _report_error("summary identity mismatch")
    return identity
```

Do not fold in `separation_comparison._strict_report_identity`; that path intentionally verifies an externally expected snapshot identity.

### 1.3 Implement convenience loader by delegation

```python
def load_published_cohort_reports(report_dir: Path) -> PublishedCohortReports:
    if not isinstance(report_dir, Path):
        raise TypeError("report_dir must be a Path")

    summary = _read_report_json(report_dir / "summary.json")
    _require_keys(summary, _SUMMARY_FIELDS, "summary")
    if summary["schema"] != REPORT_SCHEMA:
        _report_error("summary schema is invalid")
    identity = _cohort_identity_from_summary(summary["identity"])
    return read_cohort_reports(report_dir, expected_identity=identity)
```

Reading `summary.json` twice is acceptable; full cross-file validation remains owned by `read_cohort_reports()`.

### 1.4 Add integrity regressions

Test malformed summary identity and one existing cross-file corruption case. Both must raise `ReportIntegrityError` through the new entry point.

### 1.5 Verify and commit

```bash
uv run pytest tests/benchmark/test_reports.py -q
uv run ruff check src/benchmark/reports.py tests/benchmark/test_reports.py
uv run ruff format --check src/benchmark/reports.py tests/benchmark/test_reports.py
git add src/benchmark/reports.py tests/benchmark/test_reports.py
git commit -m "feat: load canonical cohort reports by persisted identity"
```

---

## Task 2 — Pure deterministic MLflow projection

**Files:** create `src/benchmark/mlflow_export.py`, `tests/benchmark/test_mlflow_export.py`

This task must not import `mlflow`.

### 2.1 Write red projection fixture/tests

Build one `PublishedCohortReports` fixture containing exactly:

```python
EXPECTED_VIEWS = (
    (30, "raw"),
    (30, "aligned"),
    (50, "raw"),
    (50, "aligned"),
    (100, "raw"),
    (100, "aligned"),
)
```

At 50 ms aligned include aggregate class rows for all six `CommonDrumClass` values.

Write deterministic bytes for the five allowlisted files.

First red test pins deterministic run name and identity tags:

```python
def test_projection_keeps_crux_identity_as_tags(tmp_path: Path) -> None:
    reports = _published_reports()
    _write_allowed_artifacts(tmp_path)

    projection = build_mlflow_projection(reports, report_dir=tmp_path, scope="broad")

    assert projection.run_name == (
        f"broad:{reports.identity.model_id}:{reports.identity.input_view_id}:"
        f"{reports.identity.cohort_id}"
    )
    assert projection.tags["crux.cohort_id"] == reports.identity.cohort_id
    assert projection.tags["crux.projection_version"] == PROJECTION_VERSION
    assert projection.tags["crux.projection_sha256"] == projection.projection_sha256
```

### 2.2 Add pure projection contract

Use:

```python
from src.benchmark.artifact_io import PublishedArtifact, read_regular_file_no_follow
from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex
from src.benchmark.cohort_scoring import COHORT_FAILURE_REASONS
from src.benchmark.taxonomy import CommonDrumClass
```

Define only:

```text
MlflowScope
MlflowMetricValue
PROJECTION_VERSION
REPORT_ARTIFACT_PATH
REPORT_ARTIFACT_NAMES
EXPECTED_AGGREGATE_VIEWS
MlflowProjectionError
MlflowProjection
build_mlflow_projection(...)
```

`MlflowProjection.artifacts` is `tuple[PublishedArtifact, ...]`.

### 2.3 Test exact six-view grid and metric mapping

Tests must assert all six aggregate views expose:

```text
event_micro.precision/recall/f1
song_macro.f1
class_macro.f1
song_f1.minimum/p10/p25/median/p75/p90/maximum
```

and common-class aggregate precision/recall/f1/support metrics plus population/failure-reason metrics.

Parameterize missing, extra, and duplicate aggregate views; all fail with `MlflowProjectionError`.

### 2.4 Test closed class and failure-reason keys

Because persisted `PublishedAggregateClass.common_class` is a bare `str`, add a regression that replaces one class with `"ride_typo"` and requires `MlflowProjectionError` before a metric key is created.

Implementation validates:

```python
common_classes = frozenset(get_args(CommonDrumClass))
```

Do the equivalent closed check for population reason names using `COHORT_FAILURE_REASONS` before creating `population.reason.*` keys.

### 2.5 Test undefined-value omission

Set a macro/distribution metric to `None` and assert the MLflow metric key is absent, not zero/NaN.

### 2.6 Test/reuse artifact hashing

For each allowlisted basename:

```python
content = read_regular_file_no_follow(report_dir / name)
artifact = PublishedArtifact(path=report_dir / name, sha256=sha256_hex(content))
```

Reject missing/symlink/unreadable allowed files as `MlflowProjectionError`.

Assert extra files such as `event_diagnostics.jsonl` and `audio.wav` never appear in `projection.artifacts`.

### 2.7 Fingerprint tests

Fingerprint canonical JSON contains projection version, scope, run name, base identity tags, canonical metric values, and `{path: artifact.path.name, sha256}` for each allowed artifact. Exclude `crux.projection_sha256` itself.

Prove same evidence gives the same SHA and changing scope, one metric, or one allowlisted file changes the SHA.

### 2.8 Protect optional-dependency boundary

Use a subprocess test that imports `src.benchmark.mlflow_export` and asserts `mlflow` is not added to `sys.modules`.

### 2.9 Verify and commit

```bash
uv run pytest tests/benchmark/test_mlflow_export.py -q
uv run ruff check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
uv run ruff format --check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
git add src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
git commit -m "feat: project canonical benchmark reports for MLflow"
```

---

## Task 3 — Optional `mlflow-skinny` publisher and idempotency

**Files:** `pyproject.toml`, `uv.lock`, `src/benchmark/mlflow_export.py`, `tests/benchmark/test_mlflow_export.py`

### 3.1 Add optional dependency only

```toml
mlflow = ["mlflow-skinny>=3.16,<4"]
```

Run `uv lock` and inspect `pyproject.toml`/`uv.lock`; unrelated existing-package version churn is not accepted.

### 3.2 Write fake-client tests before publisher code

Create a minimal `_FakeMlflowClient` exposing only:

```text
get_experiment_by_name
create_experiment
search_runs
create_run
log_batch
log_artifact
set_terminated
```

and fake run objects with `.info.run_id`, `.info.status`, `.data.tags`.

Add `MlflowPublicationError`, `MlflowProjectionConflictError`, and `MlflowPublishOutcome` only after the tests are red.

### 3.3 Pin first publish and exact no-op

First publication creates one run, logs one metric batch, uploads exactly five `PublishedArtifact.path` files to `crux-reports`, and finishes the run.

A same-fingerprint single `FINISHED` run returns `created=False` and performs no write.

### 3.4 Pin retry/status matrix

Tests must cover:

```text
only FAILED same SHA                  -> create retry
only KILLED same SHA                  -> create retry
FAILED + KILLED same SHA              -> create retry
one FINISHED same SHA + failed/killed -> no-op to FINISHED
RUNNING same SHA                      -> conflict
SCHEDULED same SHA                    -> conflict
unknown status                        -> conflict
any different SHA                     -> conflict regardless of status
multiple FINISHED                     -> conflict
```

`KILLED` is deliberately retryable like `FAILED` because it represents an interrupted/aborted attempt, not a successful publication.

### 3.5 Pin saturated search fail-closed

Publisher uses:

```python
runs = list(
    client.search_runs(
        [experiment_id],
        filter_string=_matching_run_filter(projection),
        max_results=1000,
    )
)
```

Add a fake result with exactly 1000 matching runs and require `MlflowProjectionConflictError("MLflow run search is saturated")` before any create call. Do not add pagination machinery in HPA-86.

### 3.6 Implement low-level client path

Resolve/create experiment, classify existing runs, convert canonical values to finite floats, construct MLflow `Metric` entities, call `log_batch` once, upload each of five artifacts, then `set_terminated(..., "FINISHED")`.

If a post-create call raises, best-effort set the created run to `FAILED`, then raise generic `MlflowPublicationError("MLflow publication failed")` from the original exception. Never interpolate raw provider exception text.

### 3.7 Lazy real binding and explicit URI

`publish_mlflow_projection()` first requires non-empty `MLFLOW_TRACKING_URI`, then lazily imports:

```python
from mlflow import MlflowClient
from mlflow.entities import Metric
```

Missing dependency produces bounded guidance mentioning the optional `mlflow` extra. Do not inspect username/password values.

### 3.8 Add no-network real-client API smoke

Do not wait for DagsHub smoke to discover signature drift. Run under the optional extra with no server calls:

```bash
uv run --extra mlflow python - <<'PY'
from mlflow import MlflowClient
from mlflow.entities import Metric

Metric("k", 1.0, 0, 0)
required = (
    "get_experiment_by_name",
    "create_experiment",
    "search_runs",
    "create_run",
    "log_batch",
    "log_artifact",
    "set_terminated",
)
for name in required:
    assert callable(getattr(MlflowClient, name, None)), name
print("mlflow-client-api-ok")
PY
```

Also encode this as an automated no-network optional-extra test if the repository's test command can select it without polluting the base environment.

### 3.9 Verify and commit

```bash
uv run pytest tests/benchmark/test_mlflow_export.py -q
uv run python -c 'import importlib.util; assert importlib.util.find_spec("mlflow") is None'
uv run --extra mlflow python - <<'PY'
from mlflow import MlflowClient
from mlflow.entities import Metric
Metric("k", 1.0, 0, 0)
for name in ("get_experiment_by_name", "create_experiment", "search_runs", "create_run", "log_batch", "log_artifact", "set_terminated"):
    assert callable(getattr(MlflowClient, name, None)), name
PY
uv lock --check
uv run ruff check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
uv run ruff format --check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
git add pyproject.toml uv.lock src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
git commit -m "feat: publish benchmark projections through MLflow"
```

---

## Task 4 — One explicit CLI with parseable failure output

**Files:** `src/cli/benchmark.py`, `tests/test_cli_benchmark.py`

### 4.1 Write success/no-op tests

For `publish-mlflow-cohort`, monkeypatch loader/projection/publisher and require canonical JSON stdout containing:

```text
created
experiment_id
experiment_name
exit_code = 0
projection_sha256
run_id
status = published | already_published
```

No-op uses `created=false` and the existing run ID.

### 4.2 Write failure-ordering and JSON-error tests

Cover invalid `--scope`, malformed reports, projection failure, missing URI/dependency, publication conflict, and generic transport failure.

Malformed reports must prove the publisher was never invoked.

For command-owned exit-2 domain failures, require stderr contains a bounded human-readable message and stdout remains canonical JSON containing:

```text
error = domain exception type
exit_code = 2
experiment_name
projection_sha256 = null or known local projection SHA
run_id = null
status = failed
```

Never include `MLFLOW_TRACKING_URI`, username/password, raw provider exception text, or client repr.

Click's own argument-validation failures may remain Click-formatted because domain execution has not started.

### 4.3 Implement thin command

Command shape:

```text
crux benchmark publish-mlflow-cohort \
  --reports PATH \
  --scope broad|reviewed|pilot \
  [--experiment crux-benchmark]
```

Use `click.Choice` for scope and lazy imports inside the command. Sequence is strictly:

```text
load/validate canonical reports
-> build pure projection
-> publish MLflow
```

On caught domain failure, write the canonical error object to stdout, bounded message to stderr, then exit 2. Follow sibling `compare-oaf-muscriptor` / `publish-paired-comparisons` style.

### 4.4 Verify and commit

```bash
uv run pytest tests/test_cli_benchmark.py tests/benchmark/test_mlflow_export.py tests/benchmark/test_reports.py -q
uv run ruff check src/cli/benchmark.py tests/test_cli_benchmark.py
uv run ruff format --check src/cli/benchmark.py tests/test_cli_benchmark.py
uv run pylint --errors-only src/cli/benchmark.py src/benchmark/mlflow_export.py src/benchmark/reports.py
git add src/cli/benchmark.py tests/test_cli_benchmark.py
git commit -m "feat: expose explicit MLflow benchmark publication command"
```

---

## Task 5 — Docs, five-run hosted smoke, final verification

**Files:** create `docs/benchmark/mlflow.md`, modify `README.md`; record non-secret evidence in PR #34 and HPA-86.

### 5.1 Document provider-neutral hosted use

Guide covers canonical-authority boundary, optional `mlflow` extra, standard MLflow environment variables, metric/tag names, five-file allowlist, large-artifact exclusions, idempotency/retry behavior, and provider-neutral configuration. Do not encode vendor pricing/quota numbers.

README gets only a short pointer to `docs/benchmark/mlflow.md`.

### 5.2 Pre-host deterministic verification

```bash
uv lock --check
uv run pytest tests/benchmark/test_reports.py tests/benchmark/test_mlflow_export.py tests/test_cli_benchmark.py -q
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only src
```

### 5.3 Resolve five already-produced report directories

Known broad and separation evidence:

```text
artifacts/benchmark/oaf-corpus/runs/oaf-149faa97328e20eb/reports
artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-76e8dc7e249deb90/views/full_mix/reports
artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-76e8dc7e249deb90/views/spleeter/reports
artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-76e8dc7e249deb90/views/htdemucs/reports
```

Resolve the fifth path from the already-produced HPA-327 reviewed-subset `ScoreReviewedSubsetOutcome.reports_path` on the workstation. Verify its `summary.json` identifies the reviewed OaF cohort. Do not regenerate the reviewed subset or rerun OaF merely to satisfy HPA-86.

If any required existing report directory is unavailable, leave PR #34 draft at the hosted-smoke gate and restore/copy the canonical report directory instead of running inference/reference/separation work.

### 5.4 Publish exactly five initial OaF projections

Use one `crux-benchmark` experiment:

```text
OaF full-mix broad       --scope broad
OaF full-mix reviewed    --scope reviewed
OaF full-mix pilot       --scope pilot
OaF Spleeter pilot       --scope pilot
OaF HTDemucs pilot       --scope pilot
```

The reviewed run is mandatory so the `reviewed` scope is exercised in real hosted evidence.

Record only non-secret experiment/run IDs, projection SHAs, `created`, and status.

### 5.5 Prove idempotency and UI usefulness

Republish broad OaF exactly. Require exit 0, `created=false`, `already_published`, same run ID, same projection SHA.

In hosted UI verify five runs appear, filters work for scope/model/input view, all six aggregate views are visible, headline/per-class/population metrics can be compared, exactly five `crux-reports/*` artifacts are attached, and large excluded artifacts are absent.

### 5.6 Final repo verification

```bash
uv lock --check
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only src
uv run --extra mlflow python - <<'PY'
from mlflow import MlflowClient
from mlflow.entities import Metric
Metric("k", 1.0, 0, 0)
for name in ("get_experiment_by_name", "create_experiment", "search_runs", "create_run", "log_batch", "log_artifact", "set_terminated"):
    assert callable(getattr(MlflowClient, name, None)), name
PY
git diff --check origin/main...HEAD
```

Commit docs:

```bash
git add docs/benchmark/mlflow.md README.md
git commit -m "docs: explain Crux MLflow benchmark visualization"
```

Record final verification + hosted evidence in PR #34/HPA-86, then and only then mark the PR ready and HPA-86 eligible for Done.

## Plan self-review

- Reuses `PublishedArtifact`, canonical JSON/SHA helpers, `CommonDrumClass`, and `COHORT_FAILURE_REASONS`; no duplicate artifact/taxonomy/failure abstractions.
- `KILLED` and `FAILED` are retryable; only successful `FINISHED` publication can satisfy idempotency.
- Exactly-full `search_runs(max_results=1000)` fails closed; no pagination framework added.
- Pure projection validates class names before minting durable MLflow keys.
- Optional-extra verification exercises the real `Metric` constructor and required `MlflowClient` methods without network access.
- CLI success and domain failures both keep stdout machine-readable canonical JSON.
- Hosted smoke exercises all three scope values with five existing OaF projections.
- MLflow stays optional and post-publication; no benchmark source-of-truth changes.
- One ticket / one PR remains intact.