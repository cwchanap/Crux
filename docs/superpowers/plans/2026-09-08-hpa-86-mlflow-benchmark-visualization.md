# HPA-86 MLflow Benchmark Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one optional, host-neutral MLflow projection over canonical Crux benchmark cohort reports so existing pretrained-model results are browsable in MLflow/DagsHub today and the same tracking contract can support future fine-tuning work later.

**Architecture:** Crux remains the only scientific source of truth. `reports.py` gains one self-identifying convenience loader; a new `mlflow_export.py` converts validated `PublishedCohortReports` into deterministic tags, metrics, artifact hashes, and a projection fingerprint, then publishes that projection through a lazily imported `MlflowClient`. The CLI runs this exporter only after canonical report publication, so MLflow configuration, auth, and network failures cannot affect inference, scoring, or report generation.

**Tech Stack:** Python 3.12, Click, existing Crux canonical JSON/SHA helpers, `mlflow-skinny>=3.16,<4` as an optional extra, MLflow 3.x `MlflowClient`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-07-hpa-86-mlflow-benchmark-visualization-design.md`

## Global Constraints

- One HPA-86 ticket and one PR only; continue on draft PR #34 / branch `agent/hpa-86-mlflow-benchmark-visualization`.
- Planning base is `main` at `31b7784ddb951ec2f780d3b00554ec948415f526`; if `main` moves before execution, rebase first and re-check the file/contract assumptions in this plan.
- Canonical HPA-325 reports remain authoritative; MLflow state is disposable and reproducible.
- The MLflow path is explicit and post-publication only; no model runner, scorer, separator, manifest builder, or prediction path may call MLflow.
- The base dependency set must not include MLflow; add only the optional `mlflow` extra using `mlflow-skinny>=3.16,<4`.
- Do not add the DagsHub Python SDK. Configure hosted operation only through standard MLflow environment variables.
- Require an explicit non-empty `MLFLOW_TRACKING_URI` for publication so MLflow cannot silently fall back to a local `./mlruns` file store. Username/password remain optional because a local/self-hosted server may not require auth.
- Do not accept tracking URI, username, password, DagsHub repo, model name, metric-selection, or artifact-selection CLI flags.
- V1 projection version is exactly `crux.mlflow-projection/v1`.
- V1 scope is exactly `broad | reviewed | pilot`; scope is display metadata only.
- A V1 cohort must contain exactly these aggregate views: `(30, raw)`, `(30, aligned)`, `(50, raw)`, `(50, aligned)`, `(100, raw)`, `(100, aligned)`. Missing, duplicate, or additional tolerance/mode views fail closed.
- V1 artifact allowlist is exactly `summary.json`, `summary.md`, `items.csv`, `per_song.csv`, and `per_class.csv`, logged below MLflow artifact path `crux-reports`.
- Never upload `event_diagnostics.jsonl`, audio, separator stems, predictions, model checkpoints, runtime environments, or R2/cache data.
- Do not emit per-song scalar metrics; detailed song rows remain in `per_song.csv`.
- Undefined metrics are omitted; never publish NaN, Infinity, fabricated zeroes, or sentinel values.
- Fingerprint payload excludes the self-referential `crux.projection_sha256` tag; compute the SHA first, then add that tag to the final run tags.
- Idempotency is scoped by `crux.cohort_id + crux.projection_version`. A matching `FINISHED` run with the same fingerprint is a no-op. Same-fingerprint `FAILED` attempts are retryable. Any mismatching fingerprint, `RUNNING`, `SCHEDULED`, unexpected execution status, or multiple `FINISHED` matches fails closed.
- User-visible MLflow transport errors must be generic and must not interpolate raw provider/client exception text; preserve underlying exceptions only as Python causes.
- No Model Registry, MLflow datasets, autologging, Optuna, training/fine-tuning, CI publishing, custom dashboard generation, or backward-compatibility layer in HPA-86.
- Automated tests must never contact DagsHub or any real MLflow server.
- Repository quality gates remain `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check src tests`, and `uv run pylint --errors-only src`.

---

## File Structure

### Create

- `src/benchmark/mlflow_export.py` — pure projection contract plus the small lazy-import MLflow publication boundary.
- `tests/benchmark/test_mlflow_export.py` — projection, idempotency, optional-dependency, and fake-client coverage.
- `docs/benchmark/mlflow.md` — operator setup, DagsHub-hosted usage, source-of-truth boundary, artifact policy, and UI inspection workflow.

### Modify

- `src/benchmark/reports.py` — add one self-identifying canonical report loader and refactor the existing identity parser only enough to reuse it.
- `tests/benchmark/test_reports.py` — prove the convenience loader preserves the existing full report-integrity contract.
- `src/cli/benchmark.py` — add `publish-mlflow-cohort` with lazy imports and canonical JSON success output.
- `tests/test_cli_benchmark.py` — command wiring, exit-code, and no-secret-output tests.
- `pyproject.toml` — add the optional `mlflow` dependency extra only.
- `uv.lock` — lock the new optional dependency without unrelated upgrades.
- `README.md` — add a short pointer to MLflow visualization documentation; do not rewrite the broader README under this ticket.

### Explicitly unchanged

- `src/benchmark/cohort_scoring.py`
- `src/benchmark/oaf_corpus_run.py`
- `src/benchmark/muscriptor_corpus_run.py`
- `src/benchmark/idm_pilot_run.py`
- `src/benchmark/separation_pilot.py`
- `src/benchmark/cross_comparison.py`
- model backends, prediction artifacts, taxonomy/mapping, R2/cache, and runtime model locks.

---

### Task 1: Add a self-identifying canonical cohort-report loader

**Files:**
- Modify: `src/benchmark/reports.py`
- Modify: `tests/benchmark/test_reports.py`

**Interfaces:**
- Consumes: existing `_read_report_json()`, `_require_keys()`, `_json_text()`, `CohortIdentity`, `read_cohort_reports()` and `REPORT_SCHEMA`.
- Produces: `load_published_cohort_reports(report_dir: Path) -> PublishedCohortReports`.
- Existing public interface `read_cohort_reports(report_dir, expected_identity=identity)` remains unchanged.

- [ ] **Step 1: Write the red self-identifying loader test**

Add `load_published_cohort_reports` to the existing `src.benchmark.reports` import list in `tests/benchmark/test_reports.py`, then add:

```python
def test_load_published_cohort_reports_reads_identity_from_canonical_summary(
    tmp_path: Path,
) -> None:
    write_cohort_reports(_result(), tmp_path)

    explicit = read_cohort_reports(tmp_path, expected_identity=_identity())
    discovered = load_published_cohort_reports(tmp_path)

    assert discovered == explicit
    assert discovered.identity == _identity()
```

- [ ] **Step 2: Run the focused test red**

```bash
uv run pytest tests/benchmark/test_reports.py::test_load_published_cohort_reports_reads_identity_from_canonical_summary -q
```

Expected: import failure for `load_published_cohort_reports`.

- [ ] **Step 3: Extract one reusable summary-identity constructor**

Refactor the construction part of `_parse_summary_identity()` without changing its equality behavior:

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

- [ ] **Step 4: Implement the convenience loader by delegating full validation to the existing reader**

```python
def load_published_cohort_reports(report_dir: Path) -> PublishedCohortReports:
    """Load one canonical cohort report using the identity persisted in summary.json."""
    if not isinstance(report_dir, Path):
        raise TypeError("report_dir must be a Path")

    summary = _read_report_json(report_dir / "summary.json")
    _require_keys(summary, _SUMMARY_FIELDS, "summary")
    if summary["schema"] != REPORT_SCHEMA:
        _report_error("summary schema is invalid")
    identity = _cohort_identity_from_summary(summary["identity"])
    return read_cohort_reports(report_dir, expected_identity=identity)
```

Reading `summary.json` twice is intentional: the first pass discovers identity; the existing reader remains the sole owner of complete cross-file validation.

- [ ] **Step 5: Add malformed identity and cross-file-integrity regressions**

```python
def test_load_published_cohort_reports_rejects_malformed_summary_identity(
    tmp_path: Path,
) -> None:
    artifacts = write_cohort_reports(_result(), tmp_path)
    summary = strict_json_loads(
        artifacts.summary_json.read_bytes(),
        require_canonical=True,
    )
    assert isinstance(summary, dict)
    assert isinstance(summary["identity"], dict)
    summary["identity"]["model_lock_sha256"] = "not-a-sha"
    artifacts.summary_json.write_bytes(canonical_json_bytes(summary))

    with pytest.raises(ReportIntegrityError, match="summary identity is malformed"):
        load_published_cohort_reports(tmp_path)


def test_load_published_cohort_reports_keeps_cross_file_validation(
    tmp_path: Path,
) -> None:
    artifacts = write_cohort_reports(_result(), tmp_path)
    summary = strict_json_loads(
        artifacts.summary_json.read_bytes(),
        require_canonical=True,
    )
    assert isinstance(summary, dict)
    assert isinstance(summary["population"], dict)
    summary["population"]["reason_counts"] = {"prediction_missing": 99}
    artifacts.summary_json.write_bytes(canonical_json_bytes(summary))

    with pytest.raises(ReportIntegrityError, match="reason_counts does not match"):
        load_published_cohort_reports(tmp_path)
```

- [ ] **Step 6: Run Task 1 verification**

```bash
uv run pytest tests/benchmark/test_reports.py -q
uv run ruff check src/benchmark/reports.py tests/benchmark/test_reports.py
uv run ruff format --check src/benchmark/reports.py tests/benchmark/test_reports.py
```

- [ ] **Step 7: Commit Task 1**

```bash
git add src/benchmark/reports.py tests/benchmark/test_reports.py
git commit -m "feat: load canonical cohort reports by persisted identity"
```

---

### Task 2: Build the deterministic Crux-to-MLflow projection without importing MLflow

**Files:**
- Create: `src/benchmark/mlflow_export.py`
- Create: `tests/benchmark/test_mlflow_export.py`

**Interfaces:**
- Consumes: `PublishedCohortReports`, `read_regular_file_no_follow()`, `canonical_json_bytes()`, `sha256_hex()`.
- Produces:
  - `MlflowScope = Literal["broad", "reviewed", "pilot"]`
  - `PROJECTION_VERSION = "crux.mlflow-projection/v1"`
  - `REPORT_ARTIFACT_PATH = "crux-reports"`
  - `REPORT_ARTIFACT_NAMES = ("summary.json", "summary.md", "items.csv", "per_song.csv", "per_class.csv")`
  - `MlflowProjectionError`
  - `ProjectedArtifact`
  - `MlflowProjection`
  - `build_mlflow_projection(reports, *, report_dir, scope) -> MlflowProjection`
- This task imports no `mlflow` package symbol.

- [ ] **Step 1: Create a deterministic six-view published-report fixture**

In `tests/benchmark/test_mlflow_export.py`, construct `PublishedCohortReports` directly from the existing published dataclasses. Use one helper that returns aggregates for this exact sequence:

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

For the 50 ms aligned aggregate, include six `PublishedAggregateClass` rows named `kick`, `snare`, `hihat`, `crash`, `ride`, and `tom`. Make `ride` use `f1=Decimal("0.700000")`, `reference_support=12`, and `prediction_support=10` so the class mapping is pinned by tests.

Add `_write_allowed_artifacts(tmp_path)` that writes deterministic bytes for exactly the five allowlisted filenames.

- [ ] **Step 2: Write and run the red identity projection test**

```python
def test_build_mlflow_projection_keeps_crux_identity_as_tags(tmp_path: Path) -> None:
    reports = _published_reports()
    _write_allowed_artifacts(tmp_path)

    projection = build_mlflow_projection(reports, report_dir=tmp_path, scope="broad")

    assert projection.run_name == (
        f"broad:{reports.identity.model_id}:{reports.identity.input_view_id}:"
        f"{reports.identity.cohort_id}"
    )
    assert projection.tags["crux.projection_version"] == PROJECTION_VERSION
    assert projection.tags["crux.scope"] == "broad"
    assert projection.tags["crux.cohort_id"] == reports.identity.cohort_id
    assert projection.tags["crux.model_id"] == reports.identity.model_id
    assert projection.tags["crux.input_view_id"] == reports.identity.input_view_id
    assert projection.tags["crux.projection_sha256"] == projection.projection_sha256
```

Run:

```bash
uv run pytest tests/benchmark/test_mlflow_export.py::test_build_mlflow_projection_keeps_crux_identity_as_tags -q
```

Expected: module/function missing.

- [ ] **Step 3: Add the pure projection types and constants**

```python
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal, TypeAlias

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex
from src.benchmark.reports import PublishedCohortReports

MlflowScope = Literal["broad", "reviewed", "pilot"]
MlflowMetricValue: TypeAlias = int | Decimal

PROJECTION_VERSION = "crux.mlflow-projection/v1"
REPORT_ARTIFACT_PATH = "crux-reports"
REPORT_ARTIFACT_NAMES = (
    "summary.json",
    "summary.md",
    "items.csv",
    "per_song.csv",
    "per_class.csv",
)
EXPECTED_AGGREGATE_VIEWS = frozenset(
    {
        (30, "raw"),
        (30, "aligned"),
        (50, "raw"),
        (50, "aligned"),
        (100, "raw"),
        (100, "aligned"),
    }
)


class MlflowProjectionError(ValueError):
    """Raised when canonical Crux reports cannot form the frozen V1 projection."""


@dataclass(frozen=True)
class ProjectedArtifact:
    name: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class MlflowProjection:
    run_name: str
    tags: dict[str, str]
    metrics: dict[str, MlflowMetricValue]
    artifacts: tuple[ProjectedArtifact, ...]
    projection_sha256: str
```

V1 has no MLflow params. Identity and search fields are tags; measured values are metrics.

- [ ] **Step 4: Pin complete aggregate/population metric mapping**

Add:

```python
def test_projection_publishes_all_six_aggregate_views_and_population(
    tmp_path: Path,
) -> None:
    reports = _published_reports()
    _write_allowed_artifacts(tmp_path)
    projection = build_mlflow_projection(reports, report_dir=tmp_path, scope="reviewed")

    for tolerance_ms, mode in EXPECTED_VIEWS:
        suffix = f"{tolerance_ms}ms.{mode}"
        assert f"event_micro.precision.{suffix}" in projection.metrics
        assert f"event_micro.recall.{suffix}" in projection.metrics
        assert f"event_micro.f1.{suffix}" in projection.metrics
        assert f"song_macro.f1.{suffix}" in projection.metrics
        assert f"class_macro.f1.{suffix}" in projection.metrics
        assert f"song_f1.median.{suffix}" in projection.metrics

    assert projection.metrics["class.ride.f1.50ms.aligned"] == Decimal("0.700000")
    assert projection.metrics["class.ride.reference_support.50ms.aligned"] == 12
    assert projection.metrics["class.ride.prediction_support.50ms.aligned"] == 10
    assert projection.metrics["population.total"] == reports.population.total_count
    assert projection.metrics["population.success"] == reports.population.success_count
    assert projection.metrics["population.reason.prediction_missing"] == 1
```

- [ ] **Step 5: Pin exact six-view completeness and undefined-value behavior**

```python
@pytest.mark.parametrize(
    ("aggregate_rows", "message"),
    [
        (EXPECTED_VIEWS[:-1], "aggregate views do not match"),
        (EXPECTED_VIEWS + ((200, "raw"),), "aggregate views do not match"),
        (EXPECTED_VIEWS + ((50, "aligned"),), "aggregate views contain duplicates"),
    ],
)
def test_projection_rejects_missing_extra_or_duplicate_aggregate_views(
    tmp_path: Path,
    aggregate_rows: tuple[tuple[int, str], ...],
    message: str,
) -> None:
    reports = _published_reports(aggregate_views=aggregate_rows)
    _write_allowed_artifacts(tmp_path)

    with pytest.raises(MlflowProjectionError, match=message):
        build_mlflow_projection(reports, report_dir=tmp_path, scope="broad")


def test_projection_omits_undefined_metrics(tmp_path: Path) -> None:
    reports = _published_reports(song_macro_none_at=(30, "raw"))
    _write_allowed_artifacts(tmp_path)

    projection = build_mlflow_projection(reports, report_dir=tmp_path, scope="broad")

    assert "song_macro.f1.30ms.raw" not in projection.metrics
```

The fixture helpers accept the exact optional arguments shown above; do not create separate one-off fixture families.

- [ ] **Step 6: Implement metric mapping and exact aggregate validation**

Implement a small helper:

```python
def _put_metric(
    metrics: dict[str, MlflowMetricValue],
    key: str,
    value: MlflowMetricValue | None,
) -> None:
    if value is not None:
        metrics[key] = value
```

Before mapping, derive `actual_views = tuple((row.tolerance_ms, row.mode) for row in reports.aggregates)`. Reject duplicate tuples first, then require `set(actual_views) == EXPECTED_AGGREGATE_VIEWS`.

Map each aggregate into:

```text
event_micro.precision/recall/f1.<tolerance>ms.<mode>
song_macro.f1.<tolerance>ms.<mode>
class_macro.f1.<tolerance>ms.<mode>
song_f1.minimum/p10/p25/median/p75/p90/maximum.<tolerance>ms.<mode>
class.<common_class>.precision/recall/f1/reference_support/prediction_support.<tolerance>ms.<mode>
```

Map population once:

```text
population.total
population.success
population.failed
population.skipped
population.quarantined
population.reason.<reason>
```

Keep canonical `Decimal` and integer values in the projection.

- [ ] **Step 7: Pin and implement the artifact allowlist plus deterministic fingerprint**

Add tests:

```python
def test_projection_hashes_only_closed_small_artifact_allowlist(tmp_path: Path) -> None:
    reports = _published_reports()
    _write_allowed_artifacts(tmp_path)
    (tmp_path / "event_diagnostics.jsonl").write_bytes(b"diagnostics")
    (tmp_path / "audio.wav").write_bytes(b"audio")

    projection = build_mlflow_projection(reports, report_dir=tmp_path, scope="pilot")

    assert [artifact.name for artifact in projection.artifacts] == list(REPORT_ARTIFACT_NAMES)


def test_projection_fingerprint_changes_with_scope_metric_and_allowed_artifact(
    tmp_path: Path,
) -> None:
    reports = _published_reports()
    _write_allowed_artifacts(tmp_path)
    broad = build_mlflow_projection(reports, report_dir=tmp_path, scope="broad")
    broad_repeat = build_mlflow_projection(reports, report_dir=tmp_path, scope="broad")
    reviewed = build_mlflow_projection(reports, report_dir=tmp_path, scope="reviewed")
    changed_metric = build_mlflow_projection(
        _published_reports(event_micro_f1_50_aligned=Decimal("0.123456")),
        report_dir=tmp_path,
        scope="broad",
    )

    assert broad.projection_sha256 == broad_repeat.projection_sha256
    assert broad.projection_sha256 != reviewed.projection_sha256
    assert broad.projection_sha256 != changed_metric.projection_sha256

    (tmp_path / "summary.md").write_bytes(b"changed summary bytes")
    changed_artifact = build_mlflow_projection(reports, report_dir=tmp_path, scope="broad")
    assert broad.projection_sha256 != changed_artifact.projection_sha256
```

Read every allowlisted file with `read_regular_file_no_follow()` and wrap missing/symlink/unreadable failures as `MlflowProjectionError` before any MLflow import.

Use this fingerprint shape:

```python
base_tags = {
    "crux.projection_version": PROJECTION_VERSION,
    "crux.scope": scope,
    "crux.cohort_id": identity.cohort_id,
    "crux.backend_id": identity.backend_id,
    "crux.model_id": identity.model_id,
    "crux.model_lock_sha256": identity.model_lock_sha256,
    "crux.backend_descriptor_sha256": identity.backend_descriptor_sha256,
    "crux.input_view_id": identity.input_view_id,
    "crux.reference_manifest_sha256": identity.reference_manifest_sha256,
    "crux.reference_timing_version": identity.reference_timing_version,
    "crux.taxonomy_version": identity.taxonomy_version,
    "crux.lane_map_version": identity.lane_map_version,
    "crux.prediction_map_version": identity.prediction_map_version,
    "crux.scoring_version": identity.scoring_version,
}

fingerprint_payload = {
    "projection_version": PROJECTION_VERSION,
    "run_name": run_name,
    "tags": dict(sorted(base_tags.items())),
    "metrics": dict(sorted(metrics.items())),
    "artifacts": [
        {"path": artifact.name, "sha256": artifact.sha256}
        for artifact in artifacts
    ],
}
projection_sha256 = sha256_hex(canonical_json_bytes(fingerprint_payload))
tags = {**base_tags, "crux.projection_sha256": projection_sha256}
```

- [ ] **Step 8: Protect the optional dependency boundary**

```python
def test_projection_module_does_not_import_optional_mlflow_dependency() -> None:
    script = (
        "import sys; "
        "import src.benchmark.mlflow_export; "
        "assert 'mlflow' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
```

- [ ] **Step 9: Run Task 2 verification**

```bash
uv run pytest tests/benchmark/test_mlflow_export.py -q
uv run ruff check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
uv run ruff format --check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
```

- [ ] **Step 10: Commit Task 2**

```bash
git add src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
git commit -m "feat: project canonical benchmark reports for MLflow"
```

---

### Task 3: Add the optional MLflow dependency and idempotent publisher

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `src/benchmark/mlflow_export.py`
- Modify: `tests/benchmark/test_mlflow_export.py`

**Interfaces:**
- Consumes: Task 2 `MlflowProjection`.
- Produces:
  - `MlflowPublicationError`
  - `MlflowProjectionConflictError`
  - `MlflowPublishOutcome`
  - `publish_mlflow_projection(projection, *, experiment_name) -> MlflowPublishOutcome`
- Internal test seam: `_publish_mlflow_projection_with_client(projection, experiment_name, client, metric_type)`.

- [ ] **Step 1: Add the optional dependency only and inspect lock churn**

Add:

```toml
mlflow = ["mlflow-skinny>=3.16,<4"]
```

under `[project.optional-dependencies]`, then run:

```bash
uv lock
git diff -- pyproject.toml uv.lock
```

Stop and investigate if existing package versions change for reasons unrelated to introducing the optional extra. Do not add `mlflow`, `dagshub`, pandas, SQLAlchemy, Flask, or server/UI dependencies to base/dev dependencies.

- [ ] **Step 2: Add publisher outcome and errors**

```python
class MlflowPublicationError(RuntimeError):
    """Raised when the optional MLflow transport cannot publish a projection."""


class MlflowProjectionConflictError(MlflowPublicationError):
    """Raised when existing MLflow state violates Crux projection identity."""


@dataclass(frozen=True)
class MlflowPublishOutcome:
    experiment_id: str
    run_id: str
    created: bool
```

- [ ] **Step 3: Create the fake MLflow client used by publisher tests**

In `tests/benchmark/test_mlflow_export.py`, define small test-only dataclasses `_FakeExperiment`, `_FakeRunInfo`, `_FakeRunData`, `_FakeRun`, and `_FakeMetric`, plus `_FakeMlflowClient` implementing only:

```text
get_experiment_by_name
create_experiment
search_runs
create_run
log_batch
log_artifact
set_terminated
```

The fake records create count, created tags, logged metric keys, uploaded artifact `(Path, artifact_path)` pairs, and terminations.

Add helper:

```python
def _fake_run(
    run_id: str,
    status: str,
    projection_sha256: str,
) -> _FakeRun:
    return _FakeRun(
        info=_FakeRunInfo(run_id=run_id, status=status),
        data=_FakeRunData(
            tags={"crux.projection_sha256": projection_sha256},
        ),
    )
```

- [ ] **Step 4: Pin first-publication and exact-no-op behavior**

```python
def test_publish_creates_finishes_and_logs_one_projection(tmp_path: Path) -> None:
    projection = _projection(tmp_path)
    client = _FakeMlflowClient()

    outcome = _publish_mlflow_projection_with_client(
        projection,
        "crux-benchmark",
        client,
        _FakeMetric,
    )

    assert outcome.created is True
    assert client.created_run_tags == projection.tags
    assert client.logged_metric_keys == set(projection.metrics)
    assert client.logged_artifacts == [
        (artifact.path, REPORT_ARTIFACT_PATH)
        for artifact in projection.artifacts
    ]
    assert client.terminated[-1] == (outcome.run_id, "FINISHED")


def test_publish_exact_finished_projection_is_noop(tmp_path: Path) -> None:
    projection = _projection(tmp_path)
    client = _FakeMlflowClient(
        search_results=[
            _fake_run("existing", "FINISHED", projection.projection_sha256),
        ]
    )

    outcome = _publish_mlflow_projection_with_client(
        projection,
        "crux-benchmark",
        client,
        _FakeMetric,
    )

    assert outcome == MlflowPublishOutcome(
        experiment_id=client.experiment_id,
        run_id="existing",
        created=False,
    )
    assert client.create_run_calls == 0
    assert client.logged_metric_keys == set()
    assert client.logged_artifacts == []
```

- [ ] **Step 5: Pin retry and fail-closed status semantics with concrete tests**

```python
def test_publish_retries_only_failed_runs_with_same_fingerprint(tmp_path: Path) -> None:
    projection = _projection(tmp_path)
    client = _FakeMlflowClient(
        search_results=[
            _fake_run("failed", "FAILED", projection.projection_sha256),
        ]
    )

    outcome = _publish_mlflow_projection_with_client(
        projection,
        "crux-benchmark",
        client,
        _FakeMetric,
    )

    assert outcome.created is True
    assert outcome.run_id != "failed"


def test_finished_same_fingerprint_wins_over_failed_attempt(tmp_path: Path) -> None:
    projection = _projection(tmp_path)
    client = _FakeMlflowClient(
        search_results=[
            _fake_run("failed", "FAILED", projection.projection_sha256),
            _fake_run("finished", "FINISHED", projection.projection_sha256),
        ]
    )

    outcome = _publish_mlflow_projection_with_client(
        projection,
        "crux-benchmark",
        client,
        _FakeMetric,
    )

    assert outcome.run_id == "finished"
    assert outcome.created is False


@pytest.mark.parametrize(
    ("runs", "message"),
    [
        (
            [_fake_run("running", "RUNNING", "a" * 64)],
            "existing MLflow run is not terminal",
        ),
        (
            [_fake_run("scheduled", "SCHEDULED", "a" * 64)],
            "existing MLflow run is not terminal",
        ),
        (
            [_fake_run("killed", "KILLED", "a" * 64)],
            "unexpected MLflow run status",
        ),
        (
            [_fake_run("failed-other", "FAILED", "b" * 64)],
            "projection fingerprint conflict",
        ),
        (
            [
                _fake_run("finished-a", "FINISHED", "a" * 64),
                _fake_run("finished-b", "FINISHED", "a" * 64),
            ],
            "multiple finished MLflow runs",
        ),
    ],
)
def test_existing_mlflow_state_conflicts(
    tmp_path: Path,
    runs: list[_FakeRun],
    message: str,
) -> None:
    projection = replace(_projection(tmp_path), projection_sha256="a" * 64)
    projection.tags["crux.projection_sha256"] = "a" * 64
    client = _FakeMlflowClient(search_results=runs)

    with pytest.raises(MlflowProjectionConflictError, match=message):
        _publish_mlflow_projection_with_client(
            projection,
            "crux-benchmark",
            client,
            _FakeMetric,
        )
```

If `MlflowProjection` tags are immutable in the implementation, make `_projection(tmp_path, projection_sha256="a" * 64)` the fixture API instead of mutating the returned tag dictionary. Keep exactly one fixture pattern in the final test file.

- [ ] **Step 6: Implement experiment resolution, filtered search, and status classification**

Use standard low-level client calls:

```python
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = client.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

runs = list(
    client.search_runs(
        [experiment_id],
        filter_string=_matching_run_filter(projection),
        max_results=1000,
    )
)
```

Build `_matching_run_filter()` with backticked dotted tag keys and JSON-quoted string values:

```python
def _matching_run_filter(projection: MlflowProjection) -> str:
    cohort_id = json.dumps(projection.tags["crux.cohort_id"])
    version = json.dumps(projection.tags["crux.projection_version"])
    return (
        f"tags.`crux.cohort_id` = {cohort_id} AND "
        f"tags.`crux.projection_version` = {version}"
    )
```

Classify results locally using the exact semantics from Step 5. Never update or delete an existing benchmark run.

- [ ] **Step 7: Batch metrics once, upload exactly five artifacts, terminate explicitly**

At the transport boundary only:

```python
timestamp_ms = int(time.time() * 1000)
metric_entities = []
for key, value in sorted(projection.metrics.items()):
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise MlflowPublicationError("projected metric is not finite")
    metric_entities.append(metric_type(key, numeric_value, timestamp_ms, 0))
```

Then:

```python
run = client.create_run(
    experiment_id,
    tags=projection.tags,
    run_name=projection.run_name,
)
run_id = run.info.run_id
client.log_batch(run_id, metrics=metric_entities)
for artifact in projection.artifacts:
    client.log_artifact(
        run_id,
        str(artifact.path),
        artifact_path=REPORT_ARTIFACT_PATH,
    )
client.set_terminated(run_id, status="FINISHED")
```

- [ ] **Step 8: Pin failure cleanup without exposing raw provider exceptions**

Create a fake-client mode where `log_artifact()` raises `RuntimeError("provider-secret-looking-detail")`, then assert:

```python
def test_post_create_transport_failure_marks_run_failed_and_redacts_provider_error(
    tmp_path: Path,
) -> None:
    projection = _projection(tmp_path)
    client = _FakeMlflowClient(fail_on_artifact=True)

    with pytest.raises(MlflowPublicationError, match="MLflow publication failed") as raised:
        _publish_mlflow_projection_with_client(
            projection,
            "crux-benchmark",
            client,
            _FakeMetric,
        )

    assert "provider-secret-looking-detail" not in str(raised.value)
    assert client.terminated[-1][1] == "FAILED"
```

Implementation shape:

```python
created_run_id: str | None = None
try:
    # resolve, search, create, log, finish
except MlflowProjectionConflictError:
    raise
except Exception as error:
    if created_run_id is not None:
        try:
            client.set_terminated(created_run_id, status="FAILED")
        except Exception:
            pass
    raise MlflowPublicationError("MLflow publication failed") from error
```

Do not catch `BaseException`. If marking failed also fails, preserve the original exception as the cause; a later retry will see the leftover nonterminal run and fail closed for manual inspection.

- [ ] **Step 9: Add lazy real-client bindings and explicit tracking-URI requirement**

```python
def _import_mlflow_bindings():
    from mlflow import MlflowClient
    from mlflow.entities import Metric

    return MlflowClient, Metric


def publish_mlflow_projection(
    projection: MlflowProjection,
    *,
    experiment_name: str,
) -> MlflowPublishOutcome:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        raise MlflowPublicationError("MLFLOW_TRACKING_URI is required")
    try:
        client_type, metric_type = _import_mlflow_bindings()
    except (ImportError, ModuleNotFoundError) as error:
        raise MlflowPublicationError(
            "MLflow client is unavailable; run with the optional mlflow extra"
        ) from error
    try:
        client = client_type(tracking_uri=tracking_uri)
    except Exception as error:
        raise MlflowPublicationError("cannot create MLflow client") from error
    return _publish_mlflow_projection_with_client(
        projection,
        experiment_name,
        client,
        metric_type,
    )
```

Add:

```python
def test_public_publish_requires_explicit_tracking_uri(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(MlflowPublicationError, match="MLFLOW_TRACKING_URI is required"):
        publish_mlflow_projection(
            _projection(tmp_path),
            experiment_name="crux-benchmark",
        )


def test_public_publish_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    def raise_missing_mlflow():
        raise ModuleNotFoundError("mlflow")

    monkeypatch.setattr(export_module, "_import_mlflow_bindings", raise_missing_mlflow)

    with pytest.raises(MlflowPublicationError, match="optional mlflow extra"):
        publish_mlflow_projection(
            _projection(tmp_path),
            experiment_name="crux-benchmark",
        )
```

Do not read or validate `MLFLOW_TRACKING_USERNAME` or `MLFLOW_TRACKING_PASSWORD`; MLflow owns those standard auth environment variables.

- [ ] **Step 10: Verify base and optional dependency modes**

```bash
uv run pytest tests/benchmark/test_mlflow_export.py -q
uv run python -c 'import importlib.util; assert importlib.util.find_spec("mlflow") is None'
uv run --extra mlflow python - <<'PY'
from mlflow import MlflowClient
from mlflow.entities import Metric
print(MlflowClient.__name__, Metric.__name__)
PY
uv lock --check
uv run ruff check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
uv run ruff format --check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
```

If an existing base dependency unexpectedly supplies MLflow, investigate dependency drift rather than deleting the base-dependency assertion.

- [ ] **Step 11: Commit Task 3**

```bash
git add pyproject.toml uv.lock src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
git commit -m "feat: publish benchmark projections through MLflow"
```

---

### Task 4: Add the single explicit benchmark CLI command

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`

**Interfaces:**
- Consumes:
  - Task 1 `load_published_cohort_reports()`
  - Task 2 `build_mlflow_projection()`
  - Task 3 `publish_mlflow_projection()`
- Produces: `crux benchmark publish-mlflow-cohort --reports PATH --scope broad|reviewed|pilot [--experiment NAME]`.
- Exit 0: newly published or exact idempotent no-op.
- Exit 2: malformed canonical evidence, projection failure, missing optional dependency/configuration, auth/network failure, or idempotency conflict.

- [ ] **Step 1: Write command success and exact-no-op tests**

Use a real temporary report directory only for Click path validation, and monkeypatch the three domain functions.

```python
@pytest.mark.parametrize(
    ("created", "expected_status"),
    [
        (True, "published"),
        (False, "already_published"),
    ],
)
def test_publish_mlflow_cohort_emits_canonical_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    created: bool,
    expected_status: str,
) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    monkeypatch.setattr(
        "src.benchmark.reports.load_published_cohort_reports",
        lambda path: _mlflow_reports,
    )
    monkeypatch.setattr(
        "src.benchmark.mlflow_export.build_mlflow_projection",
        lambda reports, *, report_dir, scope: _mlflow_projection,
    )
    monkeypatch.setattr(
        "src.benchmark.mlflow_export.publish_mlflow_projection",
        lambda projection, *, experiment_name: MlflowPublishOutcome(
            experiment_id="7",
            run_id="run-123",
            created=created,
        ),
    )

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "publish-mlflow-cohort",
            "--reports",
            str(report_dir),
            "--scope",
            "broad",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "created": created,
        "experiment_id": "7",
        "experiment_name": "crux-benchmark",
        "projection_sha256": _mlflow_projection.projection_sha256,
        "run_id": "run-123",
        "status": expected_status,
    }
```

Define `_mlflow_reports` and `_mlflow_projection` as compact module-level fixtures/objects near other CLI test fixtures; use the production dataclasses rather than generic mocks.

- [ ] **Step 2: Write concrete failure-ordering and safe-error tests**

```python
def test_publish_mlflow_cohort_rejects_invalid_scope(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "publish-mlflow-cohort",
            "--reports",
            str(report_dir),
            "--scope",
            "invalid",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--scope'" in result.output


def test_publish_mlflow_cohort_invalid_reports_never_call_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    publisher_called = False

    def reject_reports(path: Path):
        raise ReportIntegrityError("invalid canonical report")

    def record_publish(projection, *, experiment_name):
        nonlocal publisher_called
        publisher_called = True
        raise AssertionError("publisher must not run")

    monkeypatch.setattr(
        "src.benchmark.reports.load_published_cohort_reports",
        reject_reports,
    )
    monkeypatch.setattr(
        "src.benchmark.mlflow_export.publish_mlflow_projection",
        record_publish,
    )

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "publish-mlflow-cohort",
            "--reports",
            str(report_dir),
            "--scope",
            "broad",
        ],
    )

    assert result.exit_code == 2
    assert publisher_called is False


def test_publish_mlflow_cohort_transport_error_does_not_echo_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    secret = "dagshub-secret-token-value"
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", secret)
    monkeypatch.setattr(
        "src.benchmark.reports.load_published_cohort_reports",
        lambda path: _mlflow_reports,
    )
    monkeypatch.setattr(
        "src.benchmark.mlflow_export.build_mlflow_projection",
        lambda reports, *, report_dir, scope: _mlflow_projection,
    )

    def reject_publish(projection, *, experiment_name):
        raise MlflowPublicationError("MLflow publication failed")

    monkeypatch.setattr(
        "src.benchmark.mlflow_export.publish_mlflow_projection",
        reject_publish,
    )

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "publish-mlflow-cohort",
            "--reports",
            str(report_dir),
            "--scope",
            "broad",
        ],
    )

    assert result.exit_code == 2
    assert "MLflow publication failed" in result.output
    assert secret not in result.output
```

- [ ] **Step 3: Implement one thin command with lazy domain imports**

```python
@benchmark.command("publish-mlflow-cohort")
@click.option(
    "--reports",
    "report_dir",
    type=click.Path(exists=True, path_type=Path, file_okay=False),
    required=True,
)
@click.option(
    "--scope",
    type=click.Choice(("broad", "reviewed", "pilot"), case_sensitive=True),
    required=True,
)
@click.option("--experiment", default="crux-benchmark", show_default=True)
@click.pass_context
def publish_mlflow_cohort_command(
    ctx: click.Context,
    report_dir: Path,
    scope: str,
    experiment: str,
) -> None:
    from typing import cast

    from src.benchmark.mlflow_export import (
        MlflowProjectionError,
        MlflowPublicationError,
        MlflowScope,
        build_mlflow_projection,
        publish_mlflow_projection,
    )
    from src.benchmark.reports import (
        ReportIntegrityError,
        load_published_cohort_reports,
    )

    try:
        reports = load_published_cohort_reports(report_dir)
        projection = build_mlflow_projection(
            reports,
            report_dir=report_dir,
            scope=cast(MlflowScope, scope),
        )
        outcome = publish_mlflow_projection(
            projection,
            experiment_name=experiment,
        )
    except (
        OSError,
        ReportIntegrityError,
        MlflowProjectionError,
        MlflowPublicationError,
        TypeError,
        ValueError,
    ) as error:
        click.echo(str(error), err=True)
        ctx.exit(2)

    payload = {
        "created": outcome.created,
        "experiment_id": outcome.experiment_id,
        "experiment_name": experiment,
        "projection_sha256": projection.projection_sha256,
        "run_id": outcome.run_id,
        "status": "published" if outcome.created else "already_published",
    }
    stdout = click.get_binary_stream("stdout")
    stdout.write(canonical_json_bytes(payload, trailing_newline=True))
    stdout.flush()
```

The CLI may echo domain errors because the MLflow transport layer has already reduced provider/client exceptions to generic safe messages.

- [ ] **Step 4: Run Task 4 verification**

```bash
uv run pytest tests/test_cli_benchmark.py tests/benchmark/test_mlflow_export.py tests/benchmark/test_reports.py -q
uv run ruff check src/cli/benchmark.py tests/test_cli_benchmark.py
uv run ruff format --check src/cli/benchmark.py tests/test_cli_benchmark.py
uv run pylint --errors-only src/cli/benchmark.py src/benchmark/mlflow_export.py src/benchmark/reports.py
```

- [ ] **Step 5: Commit Task 4**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py
git commit -m "feat: expose explicit MLflow benchmark publication command"
```

---

### Task 5: Document hosted usage, publish real existing OaF evidence, and close verification

**Files:**
- Create: `docs/benchmark/mlflow.md`
- Modify: `README.md`
- Update discussion: draft PR #34 and Linear HPA-86 comments; no generated smoke-evidence file is required in git.

**Interfaces:**
- Consumes: Task 4 command plus existing production HPA-326/HPA-328/HPA-305 report directories.
- Produces: operator documentation and recorded hosted smoke evidence demonstrating UI usefulness and idempotency.

- [ ] **Step 1: Write concise operator documentation**

`docs/benchmark/mlflow.md` must contain these sections and commands:

```markdown
# MLflow benchmark visualization

Crux benchmark reports are canonical. MLflow is an optional visualization/index projection and can be deleted/recreated without losing benchmark evidence.

## Install and configure

Run the publication command with the optional client dependency:

`uv run --extra mlflow crux benchmark publish-mlflow-cohort ...`

Configure the standard MLflow environment variables supplied by the hosted provider:

```bash
export MLFLOW_TRACKING_URI="<copy the Crux MLflow tracking URI from the provider>"
export MLFLOW_TRACKING_USERNAME="<provider username>"
export MLFLOW_TRACKING_PASSWORD="${DAGSHUB_TOKEN}"
```

The password/token is external secret state and must never be committed.

## Publish one cohort

```bash
uv run --extra mlflow crux benchmark publish-mlflow-cohort \
  --reports artifacts/benchmark/oaf-corpus/runs/oaf-149faa97328e20eb/reports \
  --scope broad
```
```

The document must also explain:

- logged tags and 30/50/100 raw/aligned aggregate/distribution/class/population metrics;
- the five allowed small report artifacts;
- the explicit no-upload list for diagnostics/audio/stems/predictions/checkpoints/R2;
- exact republish returns `created=false`;
- conflicting same-cohort projection fails closed;
- UI filtering by `crux.scope`, `crux.model_id`, `crux.input_view_id`;
- compare `event_micro.f1.50ms.aligned`, `class.ride.f1.50ms.aligned`, and `population.success`;
- download `per_song.csv` for detailed song analysis;
- another MLflow-compatible server works by changing environment configuration only.

Do not encode vendor pricing/quota numbers.

- [ ] **Step 2: Add only a short README pointer**

```markdown
### Benchmark visualization

Canonical benchmark reports can optionally be projected to a standard MLflow tracking server for interactive comparison. See [`docs/benchmark/mlflow.md`](docs/benchmark/mlflow.md). MLflow is not required for inference, scoring, or report publication.
```

Do not rewrite the broader README under HPA-86.

- [ ] **Step 3: Run deterministic pre-host verification**

```bash
uv lock --check
uv run pytest tests/benchmark/test_reports.py tests/benchmark/test_mlflow_export.py tests/test_cli_benchmark.py -q
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only src
```

- [ ] **Step 4: Gate hosted smoke on already-produced real reports**

Check:

```bash
test -d artifacts/benchmark/oaf-corpus/runs/oaf-149faa97328e20eb/reports

test -d artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-76e8dc7e249deb90/views/full_mix/reports

test -d artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-76e8dc7e249deb90/views/spleeter/reports

test -d artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-76e8dc7e249deb90/views/htdemucs/reports
```

If these workstation artifacts are absent, HPA-86 remains draft at the hosted-smoke gate. Do not rerun OaF, separators, R2 sync, or reference construction under HPA-86. Restore/copy already-produced canonical report directories, then rerun this smoke step.

- [ ] **Step 5: Publish four existing OaF cohorts to the hosted `crux-benchmark` experiment**

With hosted credentials supplied only through environment variables:

```bash
uv run --extra mlflow crux benchmark publish-mlflow-cohort \
  --reports artifacts/benchmark/oaf-corpus/runs/oaf-149faa97328e20eb/reports \
  --scope broad \
  --experiment crux-benchmark

uv run --extra mlflow crux benchmark publish-mlflow-cohort \
  --reports artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-76e8dc7e249deb90/views/full_mix/reports \
  --scope pilot \
  --experiment crux-benchmark

uv run --extra mlflow crux benchmark publish-mlflow-cohort \
  --reports artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-76e8dc7e249deb90/views/spleeter/reports \
  --scope pilot \
  --experiment crux-benchmark

uv run --extra mlflow crux benchmark publish-mlflow-cohort \
  --reports artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-76e8dc7e249deb90/views/htdemucs/reports \
  --scope pilot \
  --experiment crux-benchmark
```

Record only non-secret canonical JSON output fields in PR #34 / HPA-86: experiment ID/name, run ID, projection SHA, `created`, and status.

- [ ] **Step 6: Prove real idempotency**

Repeat the broad OaF publication exactly. Require:

- exit 0;
- `created=false`;
- `status="already_published"`;
- same MLflow run ID as the first broad publication;
- same `projection_sha256`.

- [ ] **Step 7: Inspect the hosted UI against the product goal**

Verify and record:

- four OaF projections are visible without duplicate broad publication;
- filters work for `crux.scope` and `crux.input_view_id`;
- 30/50/100 ms raw/aligned metrics are visible;
- `event_micro.f1.50ms.aligned` can be compared across full mix/Spleeter/HTDemucs views;
- `class.ride.f1.50ms.aligned` is visible where that class metric exists;
- population counts are visible;
- exactly the five `crux-reports/*` artifacts are attached;
- `event_diagnostics.jsonl`, audio, stems, predictions, and checkpoints are absent.

- [ ] **Step 8: Run final repository verification on the final implementation tree**

```bash
uv lock --check
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only src
uv run --extra mlflow python - <<'PY'
from mlflow import MlflowClient
from mlflow.entities import Metric
print(MlflowClient.__name__, Metric.__name__)
PY
git diff --check origin/main...HEAD
```

- [ ] **Step 9: Commit documentation**

```bash
git add docs/benchmark/mlflow.md README.md
git commit -m "docs: explain Crux MLflow benchmark visualization"
```

- [ ] **Step 10: Record completion evidence before making the PR ready**

PR #34 and HPA-86 must record:

- final commit SHA;
- focused and full verification results;
- locked `mlflow-skinny` version from `uv.lock`;
- hosted experiment/run IDs for the four real OaF projections;
- broad republish idempotency result;
- confirmation that only five small artifacts were uploaded;
- confirmation that canonical Crux reports remain the source of truth;
- confirmation that no fine-tuning, registry, autologging, or CI publication was introduced.

Only after hosted evidence is recorded should PR #34 be marked ready and HPA-86 be eligible for Done.

---

## Plan Self-Review

- **Spec coverage:** Tasks 1-5 cover the report loader, pure projection, all metric families, identity tags, closed artifact allowlist, fingerprint, optional dependency, idempotent publisher, CLI, hosted docs, and real smoke.
- **No second truth:** no task writes back into canonical reports or replaces Crux identities with MLflow IDs.
- **No hidden dependency:** Tasks 1-2 and the normal base test suite run without MLflow installed.
- **Exact report shape:** V1 requires all six 30/50/100 × raw/aligned aggregates and rejects missing/extra/duplicate views.
- **Fingerprint is non-circular:** `crux.projection_sha256` is added only after hashing the base payload.
- **Retry semantics:** `FINISHED` same fingerprint no-ops; `FAILED` same fingerprint retries; nonterminal/unexpected/mismatching/multiple-finished state fails closed.
- **Provider errors are redacted:** user-visible transport errors are generic; raw exceptions survive only as causes for debugging.
- **Cost boundary:** only five existing small files can be uploaded.
- **Single PR:** all changes remain on PR #34.
- **No placeholders:** operational credentials/report availability are explicit gates; code/test steps are concrete.
