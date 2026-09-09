# HPA-86 MLflow Benchmark Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one optional, host-neutral MLflow projection over canonical Crux benchmark cohort reports so existing pretrained-model results are browsable in MLflow/DagsHub today and the same tracking contract can support future fine-tuning work later.

**Architecture:** Crux remains the only scientific source of truth. `reports.py` gains one self-identifying convenience loader; a new `mlflow_export.py` converts already-validated `PublishedCohortReports` into deterministic tags/metrics/small-artifact metadata and publishes that projection through a lazily imported `MlflowClient`. The CLI runs this exporter only after canonical report publication, so MLflow configuration, auth, and network failures cannot affect inference, scoring, or report generation.

**Tech Stack:** Python 3.12, Click, existing Crux canonical JSON/SHA helpers, `mlflow-skinny>=3.16,<4` as an optional extra, MLflow 3.x `MlflowClient`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-07-hpa-86-mlflow-benchmark-visualization-design.md`

## Global Constraints

- One HPA-86 ticket and one PR only; continue on draft PR #34 / branch `agent/hpa-86-mlflow-benchmark-visualization`.
- Planning base is `main` at `31b7784ddb951ec2f780d3b00554ec948415f526`; if `main` moves before execution, rebase first and re-check the file/contract assumptions in this plan.
- Canonical HPA-325 reports remain authoritative; MLflow state is disposable and reproducible.
- The MLflow path is explicit and post-publication only; no model runner, scorer, separator, manifest builder, or prediction path may call MLflow.
- The base dependency set must not include MLflow; add only the optional `mlflow` extra using `mlflow-skinny>=3.16,<4`.
- Do not add the DagsHub Python SDK. Configure hosted operation only through standard MLflow environment variables.
- Require an explicit non-empty `MLFLOW_TRACKING_URI` for the publication command so MLflow cannot silently fall back to a local `./mlruns` file store. Username/password remain optional because a local/self-hosted server may not require auth.
- Do not accept tracking URI, username, password, DagsHub repo, model name, metric-selection, or artifact-selection CLI flags.
- V1 projection version is exactly `crux.mlflow-projection/v1`.
- V1 scope is exactly `broad | reviewed | pilot`; it is display metadata only and must not change scores.
- V1 artifact allowlist is exactly `summary.json`, `summary.md`, `items.csv`, `per_song.csv`, and `per_class.csv`, logged below MLflow artifact path `crux-reports`.
- Never upload `event_diagnostics.jsonl`, audio, separator stems, predictions, model checkpoints, runtime environments, or R2/cache data.
- Do not emit per-song scalar metrics; detailed song rows remain in `per_song.csv`.
- Undefined metrics are omitted; never publish NaN, Infinity, fabricated zeroes, or sentinel values.
- Fingerprint payload excludes the self-referential `crux.projection_sha256` tag; compute the SHA first, then add that tag to the final run tags.
- Idempotency is scoped by `crux.cohort_id + crux.projection_version`. A matching `FINISHED` run with the same fingerprint is a no-op. Same-fingerprint `FAILED` attempts are retryable. Any mismatching fingerprint, `RUNNING`/`SCHEDULED`/unexpected execution status, or multiple `FINISHED` matches fails closed.
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
- `README.md` — add a short pointer to the MLflow visualization documentation; do not rewrite the existing application README under this ticket.
- `docs/superpowers/specs/2026-09-07-hpa-86-mlflow-benchmark-visualization-design.md` — only if implementation uncovers a genuine contract clarification; do not expand scope.

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
- Existing public interface `read_cohort_reports(report_dir, expected_identity=...)` remains unchanged.

- [ ] **Step 1: Write a red test proving self-identifying load equals explicit-identity load**

Add the new import and test next to the current `test_read_cohort_reports_parses_all_published_rows_and_summary_aggregates` coverage:

```python
from src.benchmark.reports import (
    # existing imports...
    load_published_cohort_reports,
)


def test_load_published_cohort_reports_reads_identity_from_canonical_summary(
    tmp_path: Path,
) -> None:
    write_cohort_reports(_result(), tmp_path)

    explicit = read_cohort_reports(tmp_path, expected_identity=_identity())
    discovered = load_published_cohort_reports(tmp_path)

    assert discovered == explicit
    assert discovered.identity == _identity()
```

- [ ] **Step 2: Run the focused test and confirm it fails only because the helper is absent**

Run:

```bash
uv run pytest tests/benchmark/test_reports.py::test_load_published_cohort_reports_reads_identity_from_canonical_summary -q
```

Expected: import/attribute failure for `load_published_cohort_reports`.

- [ ] **Step 3: Refactor summary identity construction into one reusable private helper**

Replace the construction part of `_parse_summary_identity()` with one helper; keep its existing equality check exactly where it is:

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

Do not change `_IDENTITY_FIELDS`, `CohortIdentity`, report schemas, or error text beyond what the extraction requires.

- [ ] **Step 4: Implement the convenience loader by delegating full validation back to the existing reader**

Add near `read_cohort_reports()`:

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

It is acceptable that `summary.json` is read twice: the new entry point discovers identity, then the existing canonical reader independently performs the complete cross-file validation. Do not optimize this into a second read path.

- [ ] **Step 5: Add malformed-identity and delegated-cross-file-integrity regressions**

Use the existing canonical report fixture, mutate only the relevant evidence, and prove both failures remain `ReportIntegrityError`:

```python
def test_load_published_cohort_reports_rejects_malformed_summary_identity(tmp_path: Path) -> None:
    artifacts = write_cohort_reports(_result(), tmp_path)
    summary = strict_json_loads(artifacts.summary_json.read_bytes(), require_canonical=True)
    summary["identity"]["model_lock_sha256"] = "not-a-sha"
    artifacts.summary_json.write_bytes(canonical_json_bytes(summary))

    with pytest.raises(ReportIntegrityError, match="summary identity is malformed"):
        load_published_cohort_reports(tmp_path)


def test_load_published_cohort_reports_keeps_cross_file_validation(tmp_path: Path) -> None:
    artifacts = write_cohort_reports(_result(), tmp_path)
    rows = artifacts.items_csv.read_text(encoding="utf-8")
    artifacts.items_csv.write_text(rows.replace("prediction_missing", "backend_unavailable", 1), encoding="utf-8")

    with pytest.raises(ReportIntegrityError):
        load_published_cohort_reports(tmp_path)
```

If the exact existing report invariant fails earlier/later than this sample mutation, reuse an already-tested corruption from `test_reports.py`; the acceptance is that the convenience loader cannot bypass a check enforced by `read_cohort_reports()`.

- [ ] **Step 6: Run the full report test module**

```bash
uv run pytest tests/benchmark/test_reports.py -q
uv run ruff check src/benchmark/reports.py tests/benchmark/test_reports.py
uv run ruff format --check src/benchmark/reports.py tests/benchmark/test_reports.py
```

Expected: all pass.

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
- This task must not import any `mlflow` package symbol.

- [ ] **Step 1: Write the projection fixture and red test for exact run identity/tags**

Build a small `PublishedCohortReports` fixture directly from the existing published dataclasses. It should contain six aggregate rows: 30/50/100 ms × raw/aligned, and at least one aggregate with all six common classes (`kick`, `snare`, `hihat`, `crash`, `ride`, `tom`). Create the five allowlisted files with deterministic bytes under `tmp_path`.

The first test pins the V1 readable run-name format and all identity tags:

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

- [ ] **Step 2: Run it red**

```bash
uv run pytest tests/benchmark/test_mlflow_export.py::test_build_mlflow_projection_keeps_crux_identity_as_tags -q
```

Expected: module/function missing.

- [ ] **Step 3: Add the projection types/constants with only Crux imports**

Start `src/benchmark/mlflow_export.py` with the minimal contract:

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

Do not add params in V1. Identity/filter configuration belongs in tags; measured values belong in metrics. Future training runs can introduce hyperparameter params under a separately scoped training integration.

- [ ] **Step 4: Add exact aggregate/population metric mapping tests before implementation**

Pin all metric families, not just F1:

```python
def test_projection_publishes_all_six_aggregate_views_and_population(tmp_path: Path) -> None:
    reports = _published_reports()
    _write_allowed_artifacts(tmp_path)
    projection = build_mlflow_projection(reports, report_dir=tmp_path, scope="reviewed")

    assert projection.metrics["event_micro.precision.30ms.raw"] == Decimal("0.800000")
    assert projection.metrics["event_micro.recall.50ms.aligned"] == Decimal("0.900000")
    assert projection.metrics["event_micro.f1.100ms.aligned"] == Decimal("0.850000")
    assert projection.metrics["song_macro.f1.50ms.aligned"] == Decimal("0.820000")
    assert projection.metrics["class_macro.f1.50ms.aligned"] == Decimal("0.810000")
    assert projection.metrics["song_f1.p90.50ms.aligned"] == Decimal("0.930000")
    assert projection.metrics["class.ride.f1.50ms.aligned"] == Decimal("0.700000")
    assert projection.metrics["class.ride.reference_support.50ms.aligned"] == 12
    assert projection.metrics["class.ride.prediction_support.50ms.aligned"] == 10
    assert projection.metrics["population.total"] == reports.population.total_count
    assert projection.metrics["population.success"] == reports.population.success_count
    assert projection.metrics["population.reason.prediction_missing"] == 1
```

Also assert the key set contains every tolerance/mode pair for event micro, macro, and distribution families.

- [ ] **Step 5: Implement the pure metric mapping with omission for undefined values**

Use one tiny helper to avoid six families of repetitive `None` checks:

```python
def _put_metric(
    metrics: dict[str, MlflowMetricValue],
    key: str,
    value: MlflowMetricValue | None,
) -> None:
    if value is not None:
        metrics[key] = value
```

For each `PublishedAggregate`, use its own `tolerance_ms` and `mode`. V1 accepts only tolerance values `{30, 50, 100}` and modes `{raw, aligned}`; reject another tolerance rather than silently creating an unversioned metric namespace.

Map:

```text
event_micro.precision/recall/f1
song_macro.f1
class_macro.f1
song_f1.minimum/p10/p25/median/p75/p90/maximum
class.<common>.precision/recall/f1/reference_support/prediction_support
population.total/success/failed/skipped/quarantined
population.reason.<closed reason>
```

Use the canonical `Decimal`/integer values from `PublishedCohortReports`; do not convert to float in this task.

- [ ] **Step 6: Add closed artifact-policy and fingerprint tests**

```python
def test_projection_hashes_only_the_closed_small_artifact_allowlist(tmp_path: Path) -> None:
    reports = _published_reports()
    _write_allowed_artifacts(tmp_path)
    (tmp_path / "event_diagnostics.jsonl").write_bytes(b"large diagnostics")
    (tmp_path / "audio.wav").write_bytes(b"audio")

    projection = build_mlflow_projection(reports, report_dir=tmp_path, scope="pilot")

    assert [item.name for item in projection.artifacts] == list(REPORT_ARTIFACT_NAMES)
    assert all(item.name not in {"event_diagnostics.jsonl", "audio.wav"} for item in projection.artifacts)


def test_projection_fingerprint_changes_with_scope_metric_or_allowed_artifact(tmp_path: Path) -> None:
    reports = _published_reports()
    _write_allowed_artifacts(tmp_path)
    first = build_mlflow_projection(reports, report_dir=tmp_path, scope="broad")
    same = build_mlflow_projection(reports, report_dir=tmp_path, scope="broad")
    reviewed = build_mlflow_projection(reports, report_dir=tmp_path, scope="reviewed")

    assert first.projection_sha256 == same.projection_sha256
    assert first.projection_sha256 != reviewed.projection_sha256

    (tmp_path / "summary.md").write_bytes(b"changed canonical report bytes")
    changed_artifact = build_mlflow_projection(reports, report_dir=tmp_path, scope="broad")
    assert first.projection_sha256 != changed_artifact.projection_sha256
```

Add one `dataclasses.replace()` metric mutation and assert the fingerprint changes there too.

- [ ] **Step 7: Implement closed artifact hashing and canonical fingerprint construction**

Read every allowlisted file with `read_regular_file_no_follow()` and hash those exact bytes. Missing, symlinked, or unreadable allowlisted files must surface as `MlflowProjectionError` before any MLflow import/network call.

Build the fingerprint payload before adding the fingerprint tag itself:

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
        {"path": item.name, "sha256": item.sha256}
        for item in artifacts
    ],
}
projection_sha256 = sha256_hex(canonical_json_bytes(fingerprint_payload))
tags = {**base_tags, "crux.projection_sha256": projection_sha256}
```

The `crux.projection_sha256` tag is deliberately excluded from `fingerprint_payload` to avoid self-reference.

- [ ] **Step 8: Add the no-MLflow-import regression**

```python
def test_projection_module_does_not_import_optional_mlflow_dependency(monkeypatch) -> None:
    import importlib
    import sys
    import src.benchmark.mlflow_export as export_module

    sys.modules.pop("mlflow", None)
    importlib.reload(export_module)
    assert "mlflow" not in sys.modules
```

If another unrelated installed package has already imported MLflow in the test process, isolate this check in a subprocess using the base environment instead; do not weaken the architectural assertion.

- [ ] **Step 9: Run pure projection coverage and formatting**

```bash
uv run pytest tests/benchmark/test_mlflow_export.py -q
uv run ruff check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
uv run ruff format --check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
```

Expected: all pass without installing the `mlflow` extra.

- [ ] **Step 10: Commit Task 2**

```bash
git add src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
git commit -m "feat: project canonical benchmark reports for MLflow"
```

---

### Task 3: Add the optional MLflow client dependency and idempotent publisher

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
- Internal test seam: `_publish_mlflow_projection_with_client(projection, experiment_name, client, metric_type)`; the public function lazily loads real MLflow bindings and delegates to it.

- [ ] **Step 1: Add the optional dependency only**

In `pyproject.toml`:

```toml
[project.optional-dependencies]
# existing extras...
mlflow = ["mlflow-skinny>=3.16,<4"]
```

Do not add `mlflow`, `dagshub`, pandas, SQLAlchemy, Flask, or any tracking package to base/dev dependencies.

Regenerate the lock:

```bash
uv lock
```

Inspect:

```bash
git diff -- pyproject.toml uv.lock
```

Expected: the project metadata gains the optional extra and `uv.lock` adds the `mlflow-skinny` dependency graph needed by that extra. Stop and investigate unrelated existing-package version churn before committing.

- [ ] **Step 2: Write fake-client tests for first publication and exact no-op**

Create a small fake that mirrors only the `MlflowClient` calls HPA-86 uses. Fake run objects need `.info.run_id`, `.info.status`, and `.data.tags`; fake experiment objects need `.experiment_id`.

Pin the happy path:

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
    assert client.created_run_tags["crux.projection_sha256"] == projection.projection_sha256
    assert client.logged_metric_keys == set(projection.metrics)
    assert client.logged_artifacts == [
        (item.path, REPORT_ARTIFACT_PATH) for item in projection.artifacts
    ]
    assert client.terminated[-1] == (outcome.run_id, "FINISHED")


def test_publish_exact_finished_projection_is_noop(tmp_path: Path) -> None:
    projection = _projection(tmp_path)
    existing = _fake_run(
        run_id="existing",
        status="FINISHED",
        projection_sha256=projection.projection_sha256,
    )
    client = _FakeMlflowClient(search_results=[existing])

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

- [ ] **Step 3: Write all fail-closed/retry idempotency tests before implementation**

Cover the exact status matrix from the spec:

```python
def test_failed_same_fingerprint_is_retryable(...): ...
def test_finished_plus_failed_same_fingerprint_noops_to_finished(...): ...
def test_different_fingerprint_conflicts_even_when_old_run_failed(...): ...
def test_running_same_fingerprint_conflicts(...): ...
def test_multiple_finished_matches_conflict(...): ...
def test_unexpected_killed_status_conflicts(...): ...
```

Required behavior:

- no matches -> create;
- only `FAILED` same fingerprint -> create retry;
- one `FINISHED` same fingerprint, optionally plus `FAILED` same-fingerprint attempts -> no-op to the one finished run;
- any mismatching projection SHA -> `MlflowProjectionConflictError`;
- `RUNNING`, `SCHEDULED`, or any unowned status such as `KILLED` -> conflict;
- more than one `FINISHED` -> conflict.

- [ ] **Step 4: Add the publisher outcome/errors and deterministic search-filter helper**

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

Build the MLflow DSL filter with backticked dot-containing tag keys and JSON-quoted values rather than ad-hoc escaping:

```python
def _matching_run_filter(projection: MlflowProjection) -> str:
    return (
        f"tags.`crux.cohort_id` = {json.dumps(projection.tags['crux.cohort_id'])} AND "
        f"tags.`crux.projection_version` = "
        f"{json.dumps(projection.tags['crux.projection_version'])}"
    )
```

- [ ] **Step 5: Implement experiment resolution and existing-run classification against the fake client**

Use:

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

Inspect `run.info.status` and `run.data.tags["crux.projection_sha256"]` locally according to Step 3. Do not update/delete an existing benchmark run.

- [ ] **Step 6: Batch metrics once, upload exactly five artifacts, and terminate explicitly**

Convert canonical Crux metric values to floats only here:

```python
timestamp_ms = int(time.time() * 1000)
metric_entities = [
    metric_type(key, float(value), timestamp_ms, 0)
    for key, value in sorted(projection.metrics.items())
]
```

Before creating the run, assert the converted floats are finite; raise `MlflowPublicationError` rather than sending invalid transport values.

Create/log/finish:

```python
run = client.create_run(
    experiment_id,
    tags=projection.tags,
    run_name=projection.run_name,
)
run_id = run.info.run_id
client.log_batch(run_id, metrics=metric_entities)
for artifact in projection.artifacts:
    client.log_artifact(run_id, str(artifact.path), artifact_path=REPORT_ARTIFACT_PATH)
client.set_terminated(run_id, status="FINISHED")
```

Five artifact calls are intentional and simpler than staging/copying another directory.

- [ ] **Step 7: Preserve the original publication failure while best-effort marking a newly created run failed**

Write a fake-client regression where artifact logging raises after `create_run()`:

```python
def test_post_create_transport_failure_marks_new_run_failed_and_preserves_error(...):
    ...
    with pytest.raises(MlflowPublicationError, match="artifact upload failed"):
        _publish_mlflow_projection_with_client(...)
    assert client.terminated[-1][1] == "FAILED"
```

Implementation shape:

```python
created_run_id: str | None = None
try:
    # resolve/search/create/log/finish
except MlflowProjectionConflictError:
    raise
except Exception as error:
    if created_run_id is not None:
        try:
            client.set_terminated(created_run_id, status="FAILED")
        except Exception:
            pass
    raise MlflowPublicationError(str(error)) from error
```

Do not catch `BaseException`. If marking `FAILED` itself fails, preserve the original transport error; a later retry will see the leftover running run and fail closed for manual inspection rather than guessing.

- [ ] **Step 8: Add lazy real-MLflow binding and explicit tracking-URI configuration tests**

The base environment must remain importable without MLflow. Add:

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
            "MLflow client is unavailable; run with `uv run --extra mlflow ...`"
        ) from error
    try:
        client = client_type(tracking_uri=tracking_uri)
    except Exception as error:
        raise MlflowPublicationError(f"cannot create MLflow client: {error}") from error
    return _publish_mlflow_projection_with_client(
        projection,
        experiment_name,
        client,
        metric_type,
    )
```

Tests:

```python
def test_public_publish_requires_explicit_tracking_uri(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(MlflowPublicationError, match="MLFLOW_TRACKING_URI is required"):
        publish_mlflow_projection(_projection(tmp_path), experiment_name="crux-benchmark")


def test_public_publish_reports_missing_optional_dependency(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    monkeypatch.setattr(export_module, "_import_mlflow_bindings", _raise_module_not_found)
    with pytest.raises(MlflowPublicationError, match="--extra mlflow"):
        publish_mlflow_projection(_projection(tmp_path), experiment_name="crux-benchmark")
```

Do not read or validate `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`; MLflow owns those standard auth environment variables and local servers may not need them.

- [ ] **Step 9: Run base tests without the optional extra**

```bash
uv run pytest tests/benchmark/test_mlflow_export.py -q
uv run python -c 'import importlib.util; assert importlib.util.find_spec("mlflow") is None'
```

Expected: tests pass and the base project environment still has no MLflow package.

If another existing base dependency unexpectedly provides MLflow, investigate dependency drift rather than deleting this guard silently.

- [ ] **Step 10: Verify the optional extra imports the real current client**

```bash
uv run --extra mlflow python - <<'PY'
from mlflow import MlflowClient
from mlflow.entities import Metric
print(MlflowClient.__name__, Metric.__name__)
PY
```

Expected output includes `MlflowClient Metric`. Do not contact a server in this verification.

- [ ] **Step 11: Run focused lint/format and lock verification**

```bash
uv lock --check
uv run ruff check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
uv run ruff format --check src/benchmark/mlflow_export.py tests/benchmark/test_mlflow_export.py
```

- [ ] **Step 12: Commit Task 3**

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
  - Task 3 `publish_mlflow_projection()` and publication/projection errors.
- Produces: `crux benchmark publish-mlflow-cohort --reports PATH --scope broad|reviewed|pilot [--experiment NAME]`.
- Exit 0: newly published or exact idempotent no-op.
- Exit 2: malformed canonical evidence, projection failure, missing optional dependency/configuration, auth/network failure, or idempotency conflict.

- [ ] **Step 1: Write CLI wiring tests with all domain functions monkeypatched**

Use an existing directory only to satisfy Click path validation; do not need real MLflow/report bytes in this wiring test.

```python
def test_publish_mlflow_cohort_builds_projection_and_emits_canonical_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    captured: list[object] = []

    monkeypatch.setattr(reports_module, "load_published_cohort_reports", lambda path: _reports)
    monkeypatch.setattr(
        mlflow_export_module,
        "build_mlflow_projection",
        lambda reports, *, report_dir, scope: _projection,
    )
    monkeypatch.setattr(
        mlflow_export_module,
        "publish_mlflow_projection",
        lambda projection, *, experiment_name: MlflowPublishOutcome(
            experiment_id="7",
            run_id="run-123",
            created=True,
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
        "created": True,
        "experiment_id": "7",
        "experiment_name": "crux-benchmark",
        "projection_sha256": _projection.projection_sha256,
        "run_id": "run-123",
        "status": "published",
    }
```

Add a second case with `created=False` and require status `already_published` with the same run ID.

- [ ] **Step 2: Write red failure-path tests pinning exit 2**

Cover:

```python
def test_publish_mlflow_cohort_rejects_invalid_scope_before_domain_work(...): ...
def test_publish_mlflow_cohort_report_integrity_failure_exits_two(...): ...
def test_publish_mlflow_cohort_mlflow_failure_exits_two_without_success_json(...): ...
```

In the MLflow failure case, include a fake exception message containing a server URL/token-looking value and assert the command does not print environment credentials. The implementation should only print the exception's safe message produced by `MlflowPublicationError`; never dump environment variables or the MLflow client object.

- [ ] **Step 3: Implement one thin command with lazy domain imports**

Add near the other publication commands:

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
    from src.benchmark.mlflow_export import (
        MlflowProjectionError,
        MlflowPublicationError,
        build_mlflow_projection,
        publish_mlflow_projection,
    )
    from src.benchmark.reports import ReportIntegrityError, load_published_cohort_reports

    try:
        reports = load_published_cohort_reports(report_dir)
        projection = build_mlflow_projection(
            reports,
            report_dir=report_dir,
            scope=scope,  # Click has already closed this value.
        )
        outcome = publish_mlflow_projection(
            projection,
            experiment_name=experiment,
        )
    except (OSError, ReportIntegrityError, MlflowProjectionError, MlflowPublicationError, TypeError, ValueError) as error:
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

If static typing rejects raw `scope: str`, add one small `cast(MlflowScope, scope)` after Click validation; do not add a second runtime scope parser.

- [ ] **Step 4: Prove report/projection work happens before MLflow import/network work**

Add a CLI regression where `load_published_cohort_reports()` raises and a monkeypatched publisher records whether it was called:

```python
assert publisher_called is False
```

This protects the architectural boundary that malformed Crux evidence fails before any MLflow side effect.

- [ ] **Step 5: Run the focused CLI and exporter suites**

```bash
uv run pytest tests/test_cli_benchmark.py tests/benchmark/test_mlflow_export.py tests/benchmark/test_reports.py -q
uv run ruff check src/cli/benchmark.py tests/test_cli_benchmark.py
uv run ruff format --check src/cli/benchmark.py tests/test_cli_benchmark.py
uv run pylint --errors-only src/cli/benchmark.py src/benchmark/mlflow_export.py src/benchmark/reports.py
```

Expected: all pass without the optional MLflow extra.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py
git commit -m "feat: expose explicit MLflow benchmark publication command"
```

---

### Task 5: Document hosted usage, publish real existing OaF evidence, and close verification

**Files:**
- Create: `docs/benchmark/mlflow.md`
- Modify: `README.md`
- Update tracking discussion only: draft PR #34 and Linear HPA-86 comments; no generated smoke-evidence file is required in git.

**Interfaces:**
- Consumes: Task 4 command plus existing production HPA-326/HPA-328/HPA-305 report directories.
- Produces: operator documentation and recorded hosted smoke evidence demonstrating real UI usefulness/idempotency.

- [ ] **Step 1: Write concise operator documentation before the real smoke**

`docs/benchmark/mlflow.md` must contain these sections:

1. **Purpose and authority** — MLflow is a visualization/index projection; canonical Crux reports remain authoritative.
2. **Install** — `uv run --extra mlflow ...`; base install remains unchanged.
3. **Tracking configuration** — copy the MLflow URI/username/token from the hosted provider and set:

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/cwchanap/Crux.mlflow"
export MLFLOW_TRACKING_USERNAME="cwchanap"
export MLFLOW_TRACKING_PASSWORD="${DAGSHUB_TOKEN}"
```

Document that the actual hosted URI must be copied from the DagsHub Crux repository if its casing/path differs; the token is external secret state and must never be committed.

4. **Publish one cohort**:

```bash
uv run --extra mlflow crux benchmark publish-mlflow-cohort \
  --reports artifacts/benchmark/oaf-corpus/runs/oaf-149faa97328e20eb/reports \
  --scope broad
```

5. **What is logged** — tags, 30/50/100 raw/aligned aggregate/distribution/class/population metrics, five small report files.
6. **What is never logged** — diagnostics JSONL, audio/stems/predictions/checkpoints/R2.
7. **Idempotency** — exact republish returns `created=false`; conflicting same-cohort projection fails rather than mutating.
8. **UI workflow** — filter by `crux.scope`, `crux.model_id`, `crux.input_view_id`; compare `event_micro.f1.50ms.aligned`, `class.ride.f1.50ms.aligned`, and `population.success`; download `per_song.csv` for detailed song analysis.
9. **Other MLflow servers** — set another `MLFLOW_TRACKING_URI`; no Crux code changes.

Do not encode DagsHub pricing/quota numbers, since external plans can change.

- [ ] **Step 2: Add only a short README pointer**

Under the existing Development/benchmark guidance, add a concise paragraph/link such as:

```markdown
### Benchmark visualization

Canonical benchmark reports can optionally be projected to a standard MLflow tracking server for interactive comparison. See [`docs/benchmark/mlflow.md`](docs/benchmark/mlflow.md). MLflow is not required for inference, scoring, or report publication.
```

Do not use HPA-86 to rewrite the stale broader product description.

- [ ] **Step 3: Run documentation and dependency contract checks**

```bash
uv lock --check
uv run pytest tests/benchmark/test_reports.py tests/benchmark/test_mlflow_export.py tests/test_cli_benchmark.py -q
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only src
```

Expected: all green.

- [ ] **Step 4: Gate the hosted smoke on existing real report availability; do not rerun upstream benchmarks under HPA-86**

Check the recorded production directories:

```bash
test -d artifacts/benchmark/oaf-corpus/runs/oaf-149faa97328e20eb/reports

test -d artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-76e8dc7e249deb90/views/full_mix/reports

test -d artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-76e8dc7e249deb90/views/spleeter/reports

test -d artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-76e8dc7e249deb90/views/htdemucs/reports
```

If these workstation artifacts are no longer present, HPA-86 remains draft at the hosted-smoke gate. Do **not** rerun OaF/separators or rebuild the R2/reference chain as part of this ticket; restore/copy an already-produced canonical report directory or rerun the explicit HPA-86 smoke when that evidence is available. Automated implementation/testing can still be complete, but acceptance cannot claim hosted real-evidence verification until this gate passes.

- [ ] **Step 5: Configure hosted MLflow outside the repository and publish four existing OaF cohorts**

With the operator's DagsHub token supplied only through the environment, publish:

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

Record the canonical JSON output for each in the PR/Linear discussion, especially run ID, experiment ID, projection SHA, and `created` state. Do not paste secrets or shell environment dumps.

- [ ] **Step 6: Prove real idempotency by immediately republishing the broad OaF cohort**

Repeat the first command exactly.

Require:

- exit 0;
- `created=false`;
- `status="already_published"`;
- same MLflow run ID as the first broad publication;
- same `projection_sha256`.

- [ ] **Step 7: Inspect the hosted UI against the actual product goal**

In the `crux-benchmark` experiment, verify and record in PR #34 / HPA-86:

- four OaF runs are visible without duplicate broad publication;
- runs can be filtered by `crux.scope` and `crux.input_view_id`;
- 30/50/100 ms raw/aligned metrics are available;
- `event_micro.f1.50ms.aligned` can compare broad/full-mix and pilot input views;
- `class.ride.f1.50ms.aligned` (or another present class metric) can be compared between full mix/Spleeter/HTDemucs;
- population counts are visible;
- exactly the five `crux-reports/*` small artifacts are attached;
- `event_diagnostics.jsonl`, audio, stems, predictions, and checkpoints are absent.

This is the real acceptance check that MLflow materially improves benchmark visibility; do not substitute fixture screenshots for it.

- [ ] **Step 8: Run final repository verification on the final committed implementation tree**

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

Expected: full pytest, Ruff, formatter, Pylint error checks, optional client import, and diff hygiene all pass.

- [ ] **Step 9: Commit documentation after the hosted smoke text is accurate**

```bash
git add docs/benchmark/mlflow.md README.md
git commit -m "docs: explain Crux MLflow benchmark visualization"
```

- [ ] **Step 10: Update PR #34 and HPA-86 with completion evidence**

The PR body/comment and Linear comment must state:

- exact final commit SHA;
- focused and full verification results;
- locked `mlflow-skinny` version from `uv.lock`;
- hosted MLflow experiment/run IDs for the four real OaF projections;
- broad republish idempotency result;
- confirmation that only five small artifacts were uploaded;
- confirmation that the canonical Crux report directories remain the source of truth;
- no fine-tuning/model-registry/autologging/CI work was introduced.

Only after this evidence is recorded should the draft PR be marked ready and HPA-86 be eligible for Done.

---

## Plan Self-Review Checklist

Before implementation begins, verify these points against the spec:

- **Spec coverage:** report loader, pure projection, identity tags, all metric families, artifact allowlist, deterministic fingerprint, lazy optional dependency, idempotent publication, CLI, hosted docs, real smoke, and no-go boundaries each map to a task above.
- **No second truth:** no task writes back into canonical benchmark reports or replaces their identities with MLflow run IDs.
- **No hidden MLflow import:** Tasks 1-2 and the base test suite remain runnable with no `mlflow` package installed.
- **Fingerprint is non-circular:** payload excludes `crux.projection_sha256`; final tag is added only after hashing.
- **Retry semantics are unambiguous:** `FINISHED` same fingerprint no-ops; `FAILED` same fingerprint retries; running/scheduled/unexpected/mismatching/multiple-finished state fails closed.
- **Cost boundary is structural:** only five existing small report files can be uploaded.
- **Single PR:** all commits remain on PR #34; no implementation PR is opened.
- **No placeholders:** external credentials/report availability are explicit operational gates, not unspecified implementation work.
