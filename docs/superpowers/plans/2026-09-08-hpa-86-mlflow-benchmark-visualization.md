# HPA-86 MLflow Benchmark Visualization Implementation Plan

> **Execution:** implement task-by-task on draft PR #34 / `agent/hpa-86-mlflow-benchmark-visualization`. One HPA-86 ticket, one PR. Use TDD for production behavior.

**Goal:** Add an optional MLflow visualization/index projection over canonical Crux benchmark reports, with DagsHub as the first hosted target and no MLflow dependency in inference/scoring/report generation.

**Spec:** `docs/superpowers/specs/2026-09-07-hpa-86-mlflow-benchmark-visualization-design.md`

## Global constraints

- Planning base is `main` at `31b7784ddb951ec2f780d3b00554ec948415f526`; re-check if main moves before implementation.
- Crux HPA-325 reports remain authoritative; MLflow is disposable projection state.
- Add only optional `mlflow-skinny>=3.16,<4`; no DagsHub SDK.
- Public publication accepts only explicit HTTP(S) tracking-server URIs with no inline credentials/query/fragment. Reject empty/file/sqlite/path tracking URIs.
- Scope is `broad | reviewed | pilot` and participates in both fingerprint and run search.
- V1 requires 30/50/100 ms raw+aligned views but also projects additional canonical tolerances if they exist.
- Reuse `PublishedArtifact`; no second artifact dataclass.
- Validate durable class metric keys against `CommonDrumClass`.
- Do not duplicate failure-reason validation already enforced by `read_cohort_reports()`.
- `FAILED`/`KILLED` matching attempts retry; `FINISHED` may satisfy no-op; nonterminal/unknown/conflicting state fails closed.
- Do not paginate matching-run search; fail closed if MLflow returns a continuation token.
- Use bounded MLflow error codes/messages; raw provider exceptions remain chained causes only.
- Success and command-owned exit-2 paths keep stdout canonical JSON.
- No registry, datasets, autologging, fine-tuning, training orchestration, CI benchmark publication, custom dashboard framework, or model-runner changes.
- Automated tests never contact DagsHub.

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

---

## Task 1 — One canonical report-identity reader

**Files:** `src/benchmark/reports.py`, `src/benchmark/separation_comparison.py`, `tests/benchmark/test_reports.py`, `tests/benchmark/test_separation_comparison.py`

### 1.1 Write red tests

Add report tests proving:

```python
identity = read_cohort_report_identity(report_dir)
reports = load_published_cohort_reports(report_dir)

assert identity == _identity()
assert reports == read_cohort_reports(report_dir, expected_identity=_identity())
```

Add a separation-comparison regression proving its existing expected-snapshot identity rejection still works after delegation.

### 1.2 Add the reusable public helper

In `reports.py`, extract the current summary identity construction into:

```python
def read_cohort_report_identity(report_dir: Path) -> CohortIdentity:
    if not isinstance(report_dir, Path):
        raise TypeError("report_dir must be a Path")
    summary = _read_report_json(report_dir / "summary.json")
    _require_keys(summary, _SUMMARY_FIELDS, "summary")
    if summary["schema"] != REPORT_SCHEMA:
        _report_error("summary schema is invalid")
    return _cohort_identity_from_summary(summary["identity"])
```

Keep `_parse_summary_identity(value, expected)` for the existing full reader; it delegates construction to `_cohort_identity_from_summary()` and still enforces equality.

Add:

```python
def load_published_cohort_reports(report_dir: Path) -> PublishedCohortReports:
    identity = read_cohort_report_identity(report_dir)
    return read_cohort_reports(report_dir, expected_identity=identity)
```

Reading `summary.json` twice is acceptable; the existing reader remains the only complete cross-file validator.

### 1.3 Collapse the laxer separation duplicate

Replace `separation_comparison._strict_report_identity()` internals with delegation:

```python
def _strict_report_identity(report_dir: Path) -> CohortIdentity:
    try:
        return read_cohort_report_identity(report_dir)
    except (ReportIntegrityError, OSError, TypeError, ValueError) as error:
        raise ComparisonIntegrityError("cannot read HPA-325 identity") from error
```

Do not move snapshot expectations. `_validated_identity()` continues checking expected cohort/backend/model/input/reference/scoring values exactly as today.

### 1.4 Verify

```bash
uv run pytest tests/benchmark/test_reports.py tests/benchmark/test_separation_comparison.py -q
uv run ruff check src/benchmark/reports.py src/benchmark/separation_comparison.py tests/benchmark/test_reports.py tests/benchmark/test_separation_comparison.py
uv run ruff format --check src/benchmark/reports.py src/benchmark/separation_comparison.py tests/benchmark/test_reports.py tests/benchmark/test_separation_comparison.py
```

Commit:

```bash
git add src/benchmark/reports.py src/benchmark/separation_comparison.py tests/benchmark/test_reports.py tests/benchmark/test_separation_comparison.py
git commit -m "refactor: reuse canonical cohort report identity reader"
```

---

## Task 2 — Pure deterministic MLflow projection

**Files:** create `src/benchmark/mlflow_export.py`, `tests/benchmark/test_mlflow_export.py`

This task must not import `mlflow`.

### 2.1 Build a red six-baseline-view fixture

Create `PublishedCohortReports` fixture rows containing at least:

```python
REQUIRED_BASELINE_VIEWS = frozenset(
    {
        (30, "raw"),
        (30, "aligned"),
        (50, "raw"),
        (50, "aligned"),
        (100, "raw"),
        (100, "aligned"),
    }
)
```

At 50 ms aligned include all six common classes. Write deterministic bytes for the five allowlisted files.

### 2.2 Add minimal projection types

Use:

```python
from src.benchmark.artifact_io import PublishedArtifact, read_regular_file_no_follow
from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex
from src.benchmark.taxonomy import CommonDrumClass
```

Define only:

```text
MlflowScope
MlflowMetricValue
PROJECTION_VERSION
REPORT_ARTIFACT_PATH
REPORT_ARTIFACT_NAMES
REQUIRED_BASELINE_VIEWS
MlflowProjectionError
MlflowProjection
build_mlflow_projection(...)
```

`MlflowProjection.artifacts` is `tuple[PublishedArtifact, ...]`.

### 2.3 Require baseline views, allow extras

`read_cohort_reports()` already guarantees raw/aligned completeness for every declared tolerance. The exporter only checks:

```python
actual = {(row.tolerance_ms, row.mode) for row in reports.aggregates}
if not REQUIRED_BASELINE_VIEWS <= actual:
    raise MlflowProjectionError("required benchmark views are missing")
```

Do not reject additional tolerances. Add one test where 200 ms raw/aligned is present and assert both 200 ms metric families are projected.

### 2.4 Map metrics and close class names

Project every aggregate row:

```text
event_micro.precision/recall/f1.<tol>ms.<mode>
song_macro.f1.<tol>ms.<mode>
class_macro.f1.<tol>ms.<mode>
song_f1.minimum/p10/p25/median/p75/p90/maximum.<tol>ms.<mode>
class.<common>.precision/recall/f1.<tol>ms.<mode>
class.<common>.reference_support/prediction_support.<tol>ms.<mode>
```

Before creating class keys:

```python
COMMON_CLASSES = frozenset(get_args(CommonDrumClass))
```

Reject an unknown `common_class` with `MlflowProjectionError`.

Population metrics:

```text
population.total/success/failed/skipped/quarantined
population.reason.<reason>
```

Do not add a second `COHORT_FAILURE_REASONS` check. The loader already rejects unknown reasons.

Omit `None`; never synthesize zero/NaN.

### 2.5 Reuse artifact hashing

For each closed filename:

```python
path = report_dir / name
content = read_regular_file_no_follow(path)
PublishedArtifact(path=path, sha256=sha256_hex(content))
```

Wrap file errors as `MlflowProjectionError`. Extra files never enter the projection.

### 2.6 Fingerprint

Hash canonical JSON containing projection version, scope, run name, base identity tags, canonical metric values, and each allowlisted `{path: artifact.path.name, sha256}`. Add `crux.projection_sha256` only after computing the hash.

Tests prove same input is stable and scope/metric/artifact changes alter the SHA.

### 2.7 Verify base optionality

Use a subprocess test proving importing `src.benchmark.mlflow_export` does not import `mlflow`.

Run:

```bash
uv run pytest tests/benchmark/test_mlflow_export.py -q
uv run ruff check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
uv run ruff format --check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
```

Commit:

```bash
git add src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
git commit -m "feat: project canonical benchmark reports for MLflow"
```

---

## Task 3 — Optional real MLflow publisher, idempotency, and CI drift guard

**Files:** `pyproject.toml`, `uv.lock`, `src/benchmark/mlflow_export.py`, `tests/benchmark/test_mlflow_export.py`, `.github/workflows/ci.yml`

### 3.1 Add optional dependency and CI package

In `pyproject.toml`:

```toml
mlflow = ["mlflow-skinny>=3.16,<4"]
```

In the CI minimal dependency list add:

```text
"mlflow-skinny>=3.16,<4"
```

Run `uv lock` and reject unrelated lock churn.

### 3.2 Add bounded publication errors

Define:

```python
MlflowErrorCode = Literal[
    "missing_optional_dependency",
    "invalid_config",
    "experiment_access_failed",
    "run_search_failed",
    "run_create_failed",
    "metric_log_failed",
    "artifact_upload_failed",
    "run_finalize_failed",
    "run_conflict",
]
```

`MlflowPublicationError` carries `.code` and a fixed bounded message. Do not inspect vendor strings to guess auth-vs-network categories.

### 3.3 Validate public tracking URI

Add a small local validator using `urllib.parse.urlsplit`; do not reuse R2's origin-only helper because MLflow/DagsHub URIs may contain a path.

Accept explicit `http`/`https` URI + hostname. Reject inline username/password, query, fragment, `file:`, `sqlite:`, bare path, or empty input with `invalid_config`.

The file-store integration test below constructs `MlflowClient` directly and therefore does not weaken this public CLI contract.

### 3.4 Search identity includes scope

Build the run filter from:

```text
crux.cohort_id
crux.projection_version
crux.scope
```

Use valid MLflow quoted-key syntax for dotted tag names. Search with a module constant, e.g. `MATCHING_RUN_LIMIT = 1000`.

Keep the returned `PagedList`; do not immediately cast to `list`. If `runs.token` is non-empty, raise `run_conflict` before any create call. No pagination loop.

### 3.5 Unit-test status/error semantics

Use a minimal fake only where useful for failure injection/awkward statuses:

```text
no matches                              -> create
FAILED same fingerprint                 -> retry
KILLED same fingerprint                 -> retry
FAILED + KILLED same fingerprint        -> retry
one FINISHED same fingerprint           -> no-op
FINISHED + failed/killed same fingerprint -> no-op to FINISHED
RUNNING / SCHEDULED                     -> run_conflict
unknown status                          -> run_conflict
any different fingerprint in same search key -> run_conflict
multiple FINISHED                       -> run_conflict
continuation token present              -> run_conflict
```

Do not construct 1000 runs to test saturation. Return a fake paged result with a non-empty token or temporarily lower `MATCHING_RUN_LIMIT` in a focused test.

### 3.6 Implement side effects with phase-specific errors

Use low-level `MlflowClient`:

1. resolve/create experiment;
2. search/classify matching runs;
3. create run;
4. convert canonical numeric values to finite float and `log_batch()` once;
5. `log_artifact()` exactly five times to `crux-reports`;
6. `set_terminated(..., "FINISHED")`.

Wrap each stage into its bounded code. If a post-create stage fails, best-effort terminate the new run as `FAILED`; preserve the original exception only as the cause.

### 3.7 Real-client no-network integration test

Inside `tests/benchmark/test_mlflow_export.py` add a test that locally does:

```python
mlflow = pytest.importorskip("mlflow")
from mlflow import MlflowClient
from mlflow.entities import Metric

Metric("k", 1.0, 0, 0)
```

Assert the required method surface exists.

Then create a temporary `file://` tracking URI with `Path(tmp_path / "mlruns").resolve().as_uri()`, construct `MlflowClient(tracking_uri=uri)` directly, and call the internal publisher seam with a real projection. Verify:

- first publication succeeds;
- the exact dotted-tag filter finds it;
- metric logging works;
- all five artifacts exist in the run artifact tree;
- run status is `FINISHED`;
- second publication returns the same run ID with `created=False`.

This is the pre-host end-to-end test for filter syntax and real client signatures. It does not contact a server.

Current MLflow documentation still supports local `file:` tracking stores, and `mlflow-skinny` excludes SQL/server/UI dependencies rather than the file store; the CI test is the final executable proof for the locked package.

### 3.8 Verify

```bash
uv lock --check
uv run pytest tests/benchmark/test_mlflow_export.py -q
uv run python -c 'import importlib.util; assert importlib.util.find_spec("mlflow") is None'
uv run --extra mlflow pytest tests/benchmark/test_mlflow_export.py -q
uv run ruff check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py .github/workflows/ci.yml
uv run ruff format --check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
```

Commit:

```bash
git add pyproject.toml uv.lock src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py .github/workflows/ci.yml
git commit -m "feat: publish benchmark projections through MLflow"
```

---

## Task 4 — One explicit CLI with parseable errors

**Files:** `src/cli/benchmark.py`, `tests/test_cli_benchmark.py`

### 4.1 Success/no-op tests

For:

```text
crux benchmark publish-mlflow-cohort --reports PATH --scope broad|reviewed|pilot [--experiment NAME]
```

mock loader/projection/publisher and require exit 0 canonical JSON containing:

```text
created
experiment_id
experiment_name
projection_sha256
run_id
status = published | already_published
```

### 4.2 Failure tests

Pin:

- invalid scope -> Click exit 2 before domain work;
- malformed report -> publisher not called;
- projection error -> canonical failure JSON;
- missing dependency/config -> bounded MLflow error code;
- run conflict -> `error_code="run_conflict"`;
- secret-looking environment values/raw provider cause never appear.

For command-owned domain failures, stdout JSON contains:

```json
{"error_code":"...","exit_code":2,"status":"failed"}
```

and stderr carries only the fixed safe message.

Click's own option-validation error may remain Click-formatted; do not wrap parser errors into a second parser.

### 4.3 Implement thin command

Lazy-import reports/exporter inside the command. Order is:

```text
load/validate canonical report
-> build pure projection
-> publish to MLflow
```

No MLflow work occurs if report/projection validation fails.

### 4.4 Verify

```bash
uv run pytest tests/test_cli_benchmark.py tests/benchmark/test_mlflow_export.py tests/benchmark/test_reports.py tests/benchmark/test_separation_comparison.py -q
uv run ruff check src/cli/benchmark.py tests/test_cli_benchmark.py
uv run ruff format --check src/cli/benchmark.py tests/test_cli_benchmark.py
uv run pylint --errors-only src/cli/benchmark.py src/benchmark/mlflow_export.py src/benchmark/reports.py src/benchmark/separation_comparison.py
```

Commit:

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py
git commit -m "feat: expose MLflow benchmark publication command"
```

---

## Task 5 — Docs, hosted smoke, and final verification

**Files:** create `docs/benchmark/mlflow.md`, modify `README.md`, update PR #34/HPA-86 discussion.

### 5.1 Document operation and recovery

Document:

- Crux reports are canonical; MLflow can be deleted/rebuilt.
- install/use `--extra mlflow`;
- `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` setup;
- URI must be an explicit HTTP(S) server URL and must not embed credentials;
- five uploaded files and explicit no-upload list;
- metric/tag naming and UI filters;
- `FAILED`/`KILLED` retry semantics;
- scope is part of projection identity/search;
- true same-scope fingerprint conflict recovery: delete the incorrect MLflow run in the tracking UI, then republish from canonical Crux reports;
- changing provider requires environment changes only.

Do not encode vendor quota/pricing numbers.

README gets one short link only.

### 5.2 Gate smoke on existing evidence

Require already-produced report directories for:

```text
OaF full-mix broad
OaF full-mix reviewed
OaF full-mix pilot
OaF Spleeter pilot
OaF HTDemucs pilot
```

If an expected local report directory is missing, keep PR draft and restore/copy existing canonical evidence. Do not rerun inference, separation, R2 sync, or reference generation under HPA-86.

### 5.3 Hosted smoke

Publish all five to `crux-benchmark`. Record only non-secret experiment/run IDs, fingerprints, scope, created/status.

Verify in hosted UI:

- broad/reviewed/pilot filters all have evidence;
- full-mix/Spleeter/HTDemucs can be compared;
- 30/50/100 raw/aligned metrics visible;
- per-class/population metrics visible;
- exactly five small report artifacts attached;
- no diagnostics/audio/stems/predictions/checkpoints uploaded.

Repeat one exact publication and require same run ID + `created=false`.

### 5.4 Final gates

```bash
uv lock --check
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only src
uv run --extra mlflow pytest tests/benchmark/test_mlflow_export.py -q
git diff --check origin/main...HEAD
```

Commit docs, record hosted evidence in PR/Linear, then mark PR ready only if hosted smoke passed.

---

## Plan self-review

- One new production module only: `mlflow_export.py`.
- No second scorer/report parser/artifact type/failure-reason taxonomy.
- `separation_comparison` reuses the new canonical identity reader instead of leaving a weaker duplicate.
- Baseline 30/50/100 views remain guaranteed without bricking future extra tolerances.
- Scope correction is recoverable because scope participates in search identity.
- Exact filter syntax and real `MlflowClient` operations run offline in CI before hosted smoke.
- Search saturation checks the real pagination token, not a synthetic 1000-row fixture.
- URI validation prevents local file/sqlite fallbacks and embedded credentials without rejecting DagsHub's path-bearing URL.
- Closed error codes improve CLI diagnosis without vendor-specific exception parsing.
- Hosted smoke is confirmation, not first discovery of client/filter behavior.
- Remaining risks are provider artifact routing/outage/quota and genuinely conflicting disposable MLflow state; recovery is documented.
