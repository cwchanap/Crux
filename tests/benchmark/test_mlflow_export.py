"""Tests for the MLflow projection of cohort reports and its publisher."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import types
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from src.benchmark.artifact_io import PublishedArtifact
from src.benchmark.backend_identity import (
    OAF_BACKEND_ID,
    canonical_json_bytes,
    sha256_hex,
)
from src.benchmark.cohort_scoring import CohortIdentity, PopulationSummary
from src.benchmark.mlflow_export import (
    MATCHING_RUN_LIMIT,
    PROJECTION_VERSION,
    REPORT_ARTIFACT_NAMES,
    REPORT_ARTIFACT_PATH,
    REQUIRED_BASELINE_VIEWS,
    MlflowProjection,
    MlflowProjectionError,
    MlflowPublication,
    MlflowPublicationError,
    build_matching_runs_filter,
    build_mlflow_projection,
    mlflow_insecure_tls_configured,
    mlflow_redirects_disabled,
    publish_mlflow_projection,
    staged_report_snapshot,
    validate_tracking_uri,
)
from src.benchmark.reports import (
    PublishedAggregate,
    PublishedAggregateClass,
    PublishedCohortReports,
    PublishedF1Distribution,
    PublishedMetric,
)
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, TAXONOMY_VERSION

SIX_COMMON_CLASSES = ("kick", "snare", "hihat", "crash", "ride", "tom")
EXTRA_ARTIFACT_NAME = "event_diagnostics.jsonl"
ARTIFACT_BYTES = {
    "summary.json": b'{"schema":"crux.single-cohort-report/v1"}\n',
    "summary.md": b"# Single-Cohort Drum Benchmark\n",
    "items.csv": b"cohort_id,simfile_id\n",
    "per_song.csv": b"cohort_id,simfile_id\n",
    "per_class.csv": b"cohort_id,simfile_id\n",
    EXTRA_ARTIFACT_NAME: b"{}\n",
}


def _identity() -> CohortIdentity:
    return CohortIdentity(
        cohort_id="oaf-full-mix-v1",
        reference_manifest_sha256="a" * 64,
        reference_timing_version="sha256:" + "b" * 64,
        taxonomy_version=TAXONOMY_VERSION,
        lane_map_version=DTX_LANE_MAP_VERSION,
        backend_id=OAF_BACKEND_ID,
        model_id="magenta-egmd-ckpt-569400-v1",
        model_lock_sha256="c" * 64,
        backend_descriptor_sha256="d" * 64,
        prediction_map_version="crux.prediction-map/oaf-egmd-8hit-v1",
        input_view_id="full-mix-v1",
    )


def _metric(precision: str = "0.95") -> PublishedMetric:
    return PublishedMetric(
        true_positives=10,
        false_positives=2,
        false_negatives=1,
        precision=Decimal(precision),
        recall=Decimal("0.90"),
        f1=Decimal("0.925"),
    )


def _class_row(common_class: str, precision: str = "0.95") -> PublishedAggregateClass:
    return PublishedAggregateClass(
        common_class=common_class,
        metric=_metric(precision),
        reference_support=12,
        prediction_support=10,
    )


def _aggregate(tolerance_ms: int, mode: str, *, precision: str = "0.95") -> PublishedAggregate:
    # The 30ms raw view carries undefined optional metrics so the tests can pin
    # that ``None`` values are omitted instead of becoming zeros.
    sparse = (tolerance_ms, mode) == (30, "raw")
    per_class = (
        [_class_row(common, precision) for common in SIX_COMMON_CLASSES]
        if (tolerance_ms, mode) == (50, "aligned")
        else [_class_row("kick", precision)]
    )
    return PublishedAggregate(
        tolerance_ms=tolerance_ms,
        mode=mode,  # type: ignore[arg-type]
        event_micro=PublishedMetric(
            true_positives=10,
            false_positives=2,
            false_negatives=1,
            precision=Decimal(precision),
            recall=Decimal("0.90"),
            f1=None if sparse else Decimal("0.925"),
        ),
        song_macro_f1=None if sparse else Decimal("0.88"),
        class_macro_f1=Decimal("0.86"),
        song_f1_distribution=PublishedF1Distribution(
            minimum=Decimal("0.5"),
            p10=None if sparse else Decimal("0.6"),
            p25=Decimal("0.7"),
            median=Decimal("0.8"),
            p75=Decimal("0.9"),
            p90=Decimal("0.95"),
            maximum=Decimal("1.0"),
        ),
        per_class=tuple(per_class),
        successful_song_count=3,
    )


def _reports(extra_tolerance_ms: int | None = None, *, precision: str = "0.95"):
    tolerances_ms = [30, 50, 100]
    if extra_tolerance_ms is not None:
        tolerances_ms.append(extra_tolerance_ms)
    return PublishedCohortReports(
        identity=_identity(),
        population=PopulationSummary(
            total_count=5,
            success_count=3,
            failed_count=1,
            skipped_count=1,
            quarantined_count=0,
            reason_counts=(("prediction_missing", 1),),
        ),
        aggregates=tuple(
            _aggregate(tolerance_ms, mode, precision=precision)
            for tolerance_ms in tolerances_ms
            for mode in ("raw", "aligned")
        ),
        items=(),
        songs=(),
        classes=(),
    )


def _report_dir(tmp_path: Path, *, write_extra: bool = True) -> Path:
    for name, content in ARTIFACT_BYTES.items():
        if name == EXTRA_ARTIFACT_NAME and not write_extra:
            continue
        (tmp_path / name).write_bytes(content)
    return tmp_path


def test_required_baseline_views_are_the_six_v1_views() -> None:
    assert REQUIRED_BASELINE_VIEWS == frozenset(
        {
            (30, "raw"),
            (30, "aligned"),
            (50, "raw"),
            (50, "aligned"),
            (100, "raw"),
            (100, "aligned"),
        }
    )


def test_projection_covers_every_metric_family(tmp_path: Path) -> None:
    projection = build_mlflow_projection(_reports(), _report_dir(tmp_path), scope="broad")
    metrics = projection.metrics
    for name, value in (
        ("event_micro.precision.50ms.aligned", "0.95"),
        ("event_micro.recall.50ms.aligned", "0.90"),
        ("event_micro.f1.50ms.aligned", "0.925"),
        ("song_macro.f1.50ms.aligned", "0.88"),
        ("class_macro.f1.50ms.aligned", "0.86"),
        ("song_f1.minimum.50ms.aligned", "0.5"),
        ("song_f1.p10.50ms.aligned", "0.6"),
        ("song_f1.p25.50ms.aligned", "0.7"),
        ("song_f1.median.50ms.aligned", "0.8"),
        ("song_f1.p75.50ms.aligned", "0.9"),
        ("song_f1.p90.50ms.aligned", "0.95"),
        ("song_f1.maximum.50ms.aligned", "1.0"),
    ):
        assert metrics[name] == Decimal(value), name
    for common in SIX_COMMON_CLASSES:
        assert metrics[f"class.{common}.precision.50ms.aligned"] == Decimal("0.95")
        assert metrics[f"class.{common}.recall.50ms.aligned"] == Decimal("0.90")
        assert metrics[f"class.{common}.f1.50ms.aligned"] == Decimal("0.925")
        assert metrics[f"class.{common}.reference_support.50ms.aligned"] == 12
        assert metrics[f"class.{common}.prediction_support.50ms.aligned"] == 10
    assert metrics["population.total"] == 5
    assert metrics["population.success"] == 3
    assert metrics["population.failed"] == 1
    assert metrics["population.skipped"] == 1
    assert metrics["population.quarantined"] == 0
    assert metrics["population.reason.prediction_missing"] == 1


def test_projection_omits_undefined_metrics(tmp_path: Path) -> None:
    projection = build_mlflow_projection(_reports(), _report_dir(tmp_path), scope="broad")
    metrics = projection.metrics
    assert "event_micro.f1.30ms.raw" not in metrics
    assert "song_macro.f1.30ms.raw" not in metrics
    assert "song_f1.p10.30ms.raw" not in metrics
    assert metrics["event_micro.precision.30ms.raw"] == Decimal("0.95")


def test_projection_projects_extra_tolerance_views(tmp_path: Path) -> None:
    projection = build_mlflow_projection(
        _reports(extra_tolerance_ms=200), _report_dir(tmp_path), scope="broad"
    )
    assert projection.metrics["event_micro.precision.200ms.raw"] == Decimal("0.95")
    assert projection.metrics["event_micro.precision.200ms.aligned"] == Decimal("0.95")


def test_projection_requires_all_six_baseline_views(tmp_path: Path) -> None:
    reports = _reports()
    aggregates = tuple(
        row for row in reports.aggregates if (row.tolerance_ms, row.mode) != (100, "aligned")
    )
    with pytest.raises(MlflowProjectionError, match="required benchmark views are missing"):
        build_mlflow_projection(
            replace(reports, aggregates=aggregates), _report_dir(tmp_path), scope="broad"
        )


def test_projection_rejects_unknown_common_class(tmp_path: Path) -> None:
    reports = _reports()
    extra = PublishedAggregateClass("cowbell", _metric(), 1, 1)
    aggregates = tuple(
        replace(row, per_class=row.per_class + (extra,))
        if (row.tolerance_ms, row.mode) == (50, "aligned")
        else row
        for row in reports.aggregates
    )
    with pytest.raises(MlflowProjectionError, match="cowbell"):
        build_mlflow_projection(
            replace(reports, aggregates=aggregates), _report_dir(tmp_path), scope="broad"
        )


def test_projection_artifacts_cover_only_allowlisted_files(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    projection = build_mlflow_projection(_reports(), report_dir, scope="broad")
    assert [artifact.path.name for artifact in projection.artifacts] == list(REPORT_ARTIFACT_NAMES)
    for artifact in projection.artifacts:
        assert isinstance(artifact, PublishedArtifact)
        assert artifact.path == report_dir / artifact.path.name
        assert artifact.sha256 == sha256_hex(ARTIFACT_BYTES[artifact.path.name])
    assert EXTRA_ARTIFACT_NAME not in {artifact.path.name for artifact in projection.artifacts}


def test_projection_wraps_unreadable_artifact_error(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    (report_dir / "summary.md").unlink()
    with pytest.raises(MlflowProjectionError, match="summary.md"):
        build_mlflow_projection(_reports(), report_dir, scope="broad")


def test_staged_report_snapshot_copies_only_the_report_artifacts(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    with staged_report_snapshot(report_dir) as snapshot_dir:
        assert snapshot_dir != report_dir
        assert sorted(path.name for path in snapshot_dir.iterdir()) == sorted(REPORT_ARTIFACT_NAMES)
        for name in REPORT_ARTIFACT_NAMES:
            assert (snapshot_dir / name).read_bytes() == ARTIFACT_BYTES[name]
    assert not snapshot_dir.exists()


def test_staged_report_snapshot_isolates_from_report_dir_writes(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    with staged_report_snapshot(report_dir) as snapshot_dir:
        for name in REPORT_ARTIFACT_NAMES:
            (report_dir / name).write_bytes(b"regenerated\n")
        projection = build_mlflow_projection(_reports(), snapshot_dir, scope="broad")
    digests = {artifact.path.name: artifact.sha256 for artifact in projection.artifacts}
    assert digests == {name: sha256_hex(ARTIFACT_BYTES[name]) for name in REPORT_ARTIFACT_NAMES}


def test_staged_report_snapshot_missing_artifact_is_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError), staged_report_snapshot(tmp_path):
        pass


def test_staged_report_snapshot_rejects_non_path() -> None:
    with pytest.raises(TypeError), staged_report_snapshot("reports"):  # type: ignore[arg-type]
        pass


def test_projection_is_stable_for_same_input(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    first = build_mlflow_projection(_reports(), report_dir, scope="broad")
    second = build_mlflow_projection(_reports(), report_dir, scope="broad")
    assert first == second


def test_fingerprint_changes_with_scope(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    broad = build_mlflow_projection(_reports(), report_dir, scope="broad")
    reviewed = build_mlflow_projection(_reports(), report_dir, scope="reviewed")
    assert broad.tags["crux.projection_sha256"] != reviewed.tags["crux.projection_sha256"]


def test_fingerprint_changes_with_metric_value(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    baseline = build_mlflow_projection(_reports(), report_dir, scope="broad")
    changed = build_mlflow_projection(_reports(precision="0.80"), report_dir, scope="broad")
    assert baseline.tags["crux.projection_sha256"] != changed.tags["crux.projection_sha256"]


def test_fingerprint_changes_with_artifact_bytes(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    baseline = build_mlflow_projection(_reports(), report_dir, scope="broad")
    (report_dir / "summary.md").write_bytes(b"# Changed\n")
    changed = build_mlflow_projection(_reports(), report_dir, scope="broad")
    assert baseline.tags["crux.projection_sha256"] != changed.tags["crux.projection_sha256"]


def test_projection_sha256_hashes_base_payload(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    projection = build_mlflow_projection(_reports(), report_dir, scope="broad")
    base_tags = {
        name: value for name, value in projection.tags.items() if name != "crux.projection_sha256"
    }
    payload = {
        "projection_version": PROJECTION_VERSION,
        "scope": "broad",
        "run_name": projection.run_name,
        "tags": base_tags,
        "metrics": dict(projection.metrics),
        "artifacts": [
            {"path": artifact.path.name, "sha256": artifact.sha256}
            for artifact in projection.artifacts
        ],
    }
    assert projection.tags["crux.projection_sha256"] == sha256_hex(canonical_json_bytes(payload))


def test_projection_tags_carry_scope_and_identity(tmp_path: Path) -> None:
    projection = build_mlflow_projection(_reports(), _report_dir(tmp_path), scope="pilot")
    identity = _identity()
    assert projection.tags["crux.projection_version"] == PROJECTION_VERSION
    assert projection.tags["crux.scope"] == "pilot"
    assert projection.tags["crux.cohort_id"] == identity.cohort_id
    assert projection.tags["crux.backend_id"] == identity.backend_id
    assert projection.tags["crux.model_id"] == identity.model_id
    assert projection.tags["crux.model_lock_sha256"] == identity.model_lock_sha256
    assert projection.tags["crux.backend_descriptor_sha256"] == identity.backend_descriptor_sha256
    assert projection.tags["crux.input_view_id"] == identity.input_view_id
    assert projection.tags["crux.reference_manifest_sha256"] == identity.reference_manifest_sha256
    assert projection.tags["crux.reference_timing_version"] == identity.reference_timing_version
    assert projection.tags["crux.taxonomy_version"] == identity.taxonomy_version
    assert projection.tags["crux.lane_map_version"] == identity.lane_map_version
    assert projection.tags["crux.prediction_map_version"] == identity.prediction_map_version
    assert projection.tags["crux.scoring_version"] == identity.scoring_version


def test_run_name_is_deterministic_from_scope_model_view_and_cohort(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    identity = _identity()
    projection = build_mlflow_projection(_reports(), report_dir, scope="broad")
    again = build_mlflow_projection(_reports(), report_dir, scope="broad")
    assert projection.run_name == again.run_name
    assert projection.run_name == (
        f"broad-{identity.model_id}-{identity.input_view_id}-{identity.cohort_id}"
    )


def test_import_does_not_import_mlflow() -> None:
    code = "\n".join(
        [
            "import sys",
            "import src.benchmark.mlflow_export",
            "loaded = [n for n in sys.modules if n == 'mlflow' or n.startswith('mlflow.')]",
            "assert not loaded, loaded",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr


# --- Publication (Task 3): publisher seam, idempotency, bounded errors ---

_OTHER_FINGERPRINT = "e" * 64


@pytest.fixture
def mlflow_entities_stub(monkeypatch: pytest.MonkeyPatch) -> type:
    """Stub mlflow.entities so create-path unit tests run without mlflow."""

    class Metric:  # pylint: disable=too-few-public-methods
        def __init__(self, name: str, value: float, timestamp: int, step: int) -> None:
            self.name = name
            self.value = value
            self.timestamp = timestamp
            self.step = step

    entities = types.ModuleType("mlflow.entities")
    entities.Metric = Metric
    package = types.ModuleType("mlflow")
    package.entities = entities
    monkeypatch.setitem(sys.modules, "mlflow", package)
    monkeypatch.setitem(sys.modules, "mlflow.entities", entities)
    return Metric


class _FakeRunInfo:
    def __init__(
        self, run_id: str, status: str, artifact_uri: str = "https://artifacts.example.com"
    ) -> None:
        self.run_id = run_id
        self.status = status
        self.artifact_uri = artifact_uri


class _FakeRun:
    def __init__(
        self,
        run_id: str,
        status: str,
        tags: dict[str, str],
        artifact_uri: str = "https://artifacts.example.com",
    ) -> None:
        self.info = _FakeRunInfo(run_id, status, artifact_uri)
        self.data = SimpleNamespace(tags=tags)


class _FakePage:
    """Stands in for mlflow's PagedList: iterable plus a continuation token."""

    def __init__(self, runs: list[_FakeRun], token: str | None = None) -> None:
        self._runs = runs
        self.token = token

    def __iter__(self):
        return iter(self._runs)

    def __len__(self) -> int:
        return len(self._runs)


class _FakeClient:
    """Records publisher calls; injects one-shot failures per stage."""

    def __init__(
        self,
        page: _FakePage,
        *,
        has_experiment: bool = True,
        experiment_lifecycle_stage: str = "active",
        experiment_error: Exception | None = None,
        search_error: Exception | None = None,
        create_error: Exception | None = None,
        log_batch_error: Exception | None = None,
        artifact_error: Exception | None = None,
        list_artifact_error: Exception | None = None,
        finalize_error: Exception | None = None,
        artifact_uri: str = "https://artifacts.example.com",
        listed_artifacts: list | None = None,
    ) -> None:
        self.page = page
        self.has_experiment = has_experiment
        self.experiment_lifecycle_stage = experiment_lifecycle_stage
        self.experiment_error = experiment_error
        self.search_error = search_error
        self.create_error = create_error
        self.log_batch_error = log_batch_error
        self.artifact_error = artifact_error
        self.list_artifact_error = list_artifact_error
        self.finalize_error = finalize_error
        self.artifact_uri = artifact_uri
        self.listed_artifacts = listed_artifacts
        self.events: list[tuple] = []
        self.created_experiment_names: list[str] = []
        self.searches: list[dict] = []
        self.created_runs: list[dict] = []
        self.logged_batches: list[dict] = []
        self.logged_artifacts: list[tuple[str, str, str | None]] = []
        self.logged_artifact_bytes: list[bytes] = []
        self.terminated: list[tuple[str, str | None]] = []
        self.terminated_suppression: list[str | None] = []
        self.redirects_env: list[str | None] = []
        self.multipart_env: list[str | None] = []

    def _fire(self, key: str) -> None:
        error = getattr(self, key)
        if error is not None:
            setattr(self, key, None)  # one-shot: the best-effort retry succeeds
            raise error

    def get_experiment_by_name(self, name: str):
        self.events.append(("get_experiment_by_name", name))
        self.redirects_env.append(os.environ.get("MLFLOW_ALLOW_HTTP_REDIRECTS"))
        self._fire("experiment_error")
        if not self.has_experiment:
            return None
        return SimpleNamespace(
            experiment_id="exp-1", lifecycle_stage=self.experiment_lifecycle_stage
        )

    def create_experiment(self, name: str) -> str:
        self.events.append(("create_experiment", name))
        self._fire("experiment_error")
        self.created_experiment_names.append(name)
        return "exp-created"

    def search_runs(self, experiment_ids, filter_string: str = "", max_results=None):
        self.events.append(("search_runs", tuple(experiment_ids)))
        self._fire("search_error")
        self.searches.append(
            {
                "experiment_ids": list(experiment_ids),
                "filter_string": filter_string,
                "max_results": max_results,
            }
        )
        return self.page

    def create_run(self, experiment_id: str, tags=None, run_name=None):
        self.events.append(("create_run", experiment_id))
        self._fire("create_error")
        stored_tags = dict(tags or {})
        self.created_runs.append(
            {
                "experiment_id": experiment_id,
                "tags": stored_tags,
                "run_name": run_name,
            }
        )
        return _FakeRun("run-new", "RUNNING", stored_tags, artifact_uri=self.artifact_uri)

    def log_batch(self, run_id: str, metrics=(), params=(), tags=(), synchronous=None):
        self.events.append(("log_batch", run_id))
        self._fire("log_batch_error")
        self.logged_batches.append(
            {"run_id": run_id, "metrics": list(metrics), "synchronous": synchronous}
        )

    def log_artifact(self, run_id: str, local_path, artifact_path=None):
        self.events.append(("log_artifact", run_id))
        self.redirects_env.append(os.environ.get("MLFLOW_ALLOW_HTTP_REDIRECTS"))
        self.multipart_env.append(os.environ.get("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"))
        self._fire("artifact_error")
        self.logged_artifacts.append((run_id, str(local_path), artifact_path))
        self.logged_artifact_bytes.append(Path(local_path).read_bytes())

    def list_artifacts(self, run_id: str, path=None):
        self.events.append(("list_artifacts", run_id))
        self.redirects_env.append(os.environ.get("MLFLOW_ALLOW_HTTP_REDIRECTS"))
        self.multipart_env.append(os.environ.get("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"))
        self._fire("list_artifact_error")
        if self.listed_artifacts is not None:
            return self.listed_artifacts
        return [
            SimpleNamespace(
                path=f"{artifact_path}/{Path(local_path).name}",
                file_size=len(content),
            )
            for (_, local_path, artifact_path), content in zip(
                self.logged_artifacts, self.logged_artifact_bytes
            )
        ]

    def set_terminated(self, run_id: str, status=None):
        self.events.append(("set_terminated", run_id, status))
        self.terminated_suppression.append(os.environ.get("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"))
        self.redirects_env.append(os.environ.get("MLFLOW_ALLOW_HTTP_REDIRECTS"))
        self._fire("finalize_error")
        self.terminated.append((run_id, status))


def _publisher_projection(tmp_path: Path) -> MlflowProjection:
    return build_mlflow_projection(_reports(), _report_dir(tmp_path), scope="broad")


def _matching_run(
    fingerprint: str,
    status: str,
    *,
    run_id: str = "run-existing",
    artifact_uri: str = "https://artifacts.example.com",
) -> _FakeRun:
    return _FakeRun(run_id, status, {"crux.projection_sha256": fingerprint}, artifact_uri)


def _publish(projection: MlflowProjection, client: _FakeClient) -> MlflowPublication:
    return publish_mlflow_projection(projection, client=client, experiment_name="crux-benchmarks")


@pytest.mark.parametrize(
    "raw_uri",
    [
        "",
        "file:///tmp/mlruns",
        "sqlite:///mlflow.db",
        "ftp://host.example/store",
        "mlruns",
        "/tmp/mlruns",
        "http://",
        "http://localhost:5000",
        "http://host.example",
        "https://",
        "https://user@host.example",
        "https://user:pass@host.example",
        "https://host.example/?x=1",
        "https://host.example/#frag",
    ],
)
def test_validate_tracking_uri_rejects_invalid(raw_uri: str) -> None:
    with pytest.raises(MlflowPublicationError) as raised:
        validate_tracking_uri(raw_uri)
    assert raised.value.code == "invalid_config"
    assert str(raised.value) == MlflowPublicationError("invalid_config").message


@pytest.mark.parametrize(
    "uri",
    [
        "https://localhost:5000",
        "https://dagshub.com/owner/repo.mlflow",
        "https://dagshub.com/owner/repo.mlflow/",
    ],
)
def test_validate_tracking_uri_accepts_https_uris(uri: str) -> None:
    assert validate_tracking_uri(uri) == uri


def test_validate_tracking_uri_rejects_http_despite_matching_netrc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # requests would attach ~/.netrc Basic credentials to a plaintext request;
    # the URI must be refused rather than probing every credential source.
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".netrc").write_text("machine mlflow.example.com login operator password hunter2\n")
    with pytest.raises(MlflowPublicationError) as raised:
        validate_tracking_uri("http://mlflow.example.com")
    assert raised.value.code == "invalid_config"


def test_validate_tracking_uri_rejects_insecure_tls() -> None:
    with pytest.raises(MlflowPublicationError) as raised:
        validate_tracking_uri("https://mlflow.example.com", insecure_tls=True)
    assert raised.value.code == "invalid_config"
    assert validate_tracking_uri("https://mlflow.example.com") == "https://mlflow.example.com"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("false", False),
        ("0", False),
        ("", False),
        ("yes", False),
    ],
)
def test_mlflow_insecure_tls_configured_parses_boolean_env(
    value: str, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_INSECURE_TLS", value)
    assert mlflow_insecure_tls_configured() is expected
    monkeypatch.delenv("MLFLOW_TRACKING_INSECURE_TLS", raising=False)
    assert mlflow_insecure_tls_configured() is False


def test_mlflow_redirects_disabled_sets_and_restores_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_ALLOW_HTTP_REDIRECTS", raising=False)
    with mlflow_redirects_disabled():
        assert os.environ["MLFLOW_ALLOW_HTTP_REDIRECTS"] == "false"
    assert "MLFLOW_ALLOW_HTTP_REDIRECTS" not in os.environ

    monkeypatch.setenv("MLFLOW_ALLOW_HTTP_REDIRECTS", "true")
    with mlflow_redirects_disabled():
        assert os.environ["MLFLOW_ALLOW_HTTP_REDIRECTS"] == "false"
    assert os.environ["MLFLOW_ALLOW_HTTP_REDIRECTS"] == "true"


def test_publish_runs_client_calls_with_redirects_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An https->http (or cross-origin) redirect would let requests re-resolve
    # ~/.netrc credentials for the target; every mlflow call must run with
    # redirects off, not just the finalize path.
    monkeypatch.delenv("MLFLOW_ALLOW_HTTP_REDIRECTS", raising=False)
    projection = _publisher_projection(tmp_path)
    client = _FakeClient(_FakePage([]))

    publication = _publish(projection, client)

    assert publication.status == "published"
    assert client.redirects_env
    assert all(value == "false" for value in client.redirects_env)
    assert "MLFLOW_ALLOW_HTTP_REDIRECTS" not in os.environ


def test_projection_rejects_quote_in_search_identity(tmp_path: Path) -> None:
    identity = replace(_identity(), cohort_id="quoted'cohort")
    with pytest.raises(MlflowProjectionError, match="unrepresentable quote"):
        build_mlflow_projection(
            replace(_reports(), identity=identity), _report_dir(tmp_path), scope="broad"
        )


def test_matching_runs_filter_does_not_escape_backslashes() -> None:
    filter_string = build_matching_runs_filter(
        {
            "crux.cohort_id": "raw\\cohort",
            "crux.projection_version": PROJECTION_VERSION,
            "crux.scope": "broad",
        }
    )
    assert "tags.\"crux.cohort_id\" = 'raw\\cohort'" in filter_string
    assert "raw\\\\cohort" not in filter_string


def test_matching_runs_filter_quotes_dotted_tag_keys(tmp_path: Path) -> None:
    projection = _publisher_projection(tmp_path)
    assert build_matching_runs_filter(projection.tags) == (
        "tags.\"crux.cohort_id\" = 'oaf-full-mix-v1'"
        " and tags.\"crux.projection_version\" = 'crux.mlflow-projection/v1'"
        " and tags.\"crux.scope\" = 'broad'"
    )


def test_publish_with_no_matches_creates_new_run(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    projection = _publisher_projection(tmp_path)
    client = _FakeClient(_FakePage([]))
    result = _publish(projection, client)
    assert result.created is True
    assert result.status == "published"
    assert result.run_id == "run-new"
    assert result.experiment_id == "exp-1"
    assert result.experiment_name == "crux-benchmarks"
    assert result.projection_sha256 == projection.tags["crux.projection_sha256"]


@pytest.mark.parametrize("status", ["FAILED", "KILLED"])
def test_publish_retries_after_failed_or_killed(
    tmp_path: Path, mlflow_entities_stub: type, status: str
) -> None:
    fingerprint = _publisher_projection(tmp_path).tags["crux.projection_sha256"]
    client = _FakeClient(_FakePage([_matching_run(fingerprint, status)]))
    result = _publish(_publisher_projection(tmp_path), client)
    assert result.created is True
    assert result.status == "published"
    assert result.run_id == "run-new"


def test_publish_retries_after_failed_and_killed(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    fingerprint = _publisher_projection(tmp_path).tags["crux.projection_sha256"]
    client = _FakeClient(
        _FakePage(
            [
                _matching_run(fingerprint, "FAILED", run_id="run-a"),
                _matching_run(fingerprint, "KILLED", run_id="run-b"),
            ]
        )
    )
    result = _publish(_publisher_projection(tmp_path), client)
    assert result.created is True
    assert result.status == "published"


def test_publish_with_one_finished_run_is_noop(tmp_path: Path, mlflow_entities_stub: type) -> None:
    fingerprint = _publisher_projection(tmp_path).tags["crux.projection_sha256"]
    client = _FakeClient(_FakePage([_matching_run(fingerprint, "FINISHED", run_id="run-done")]))
    result = _publish(_publisher_projection(tmp_path), client)
    assert result.created is False
    assert result.status == "already_published"
    assert result.run_id == "run-done"
    assert result.projection_sha256 == fingerprint
    assert client.created_runs == []


def test_publish_finished_plus_failed_is_noop_to_finished(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    fingerprint = _publisher_projection(tmp_path).tags["crux.projection_sha256"]
    client = _FakeClient(
        _FakePage(
            [
                _matching_run(fingerprint, "FAILED", run_id="run-a"),
                _matching_run(fingerprint, "FINISHED", run_id="run-b"),
            ]
        )
    )
    result = _publish(_publisher_projection(tmp_path), client)
    assert result.created is False
    assert result.status == "already_published"
    assert result.run_id == "run-b"
    assert client.created_runs == []


@pytest.mark.parametrize("status", ["RUNNING", "SCHEDULED"])
def test_publish_resumes_orphaned_live_run(
    tmp_path: Path, mlflow_entities_stub: type, status: str
) -> None:
    fingerprint = _publisher_projection(tmp_path).tags["crux.projection_sha256"]
    client = _FakeClient(_FakePage([_matching_run(fingerprint, status, run_id="run-orphan")]))
    result = _publish(_publisher_projection(tmp_path), client)
    assert result.created is False
    assert result.status == "published"
    assert result.run_id == "run-orphan"
    assert client.created_runs == []
    assert client.terminated == [("run-orphan", "FINISHED")]


def test_publish_finished_wins_over_live_match(tmp_path: Path, mlflow_entities_stub: type) -> None:
    fingerprint = _publisher_projection(tmp_path).tags["crux.projection_sha256"]
    client = _FakeClient(
        _FakePage(
            [
                _matching_run(fingerprint, "RUNNING", run_id="run-live"),
                _matching_run(fingerprint, "FINISHED", run_id="run-done"),
            ]
        )
    )
    result = _publish(_publisher_projection(tmp_path), client)
    assert result.created is False
    assert result.status == "already_published"
    assert result.run_id == "run-done"


def test_publish_multiple_live_runs_conflict(tmp_path: Path, mlflow_entities_stub: type) -> None:
    fingerprint = _publisher_projection(tmp_path).tags["crux.projection_sha256"]
    client = _FakeClient(
        _FakePage(
            [
                _matching_run(fingerprint, "RUNNING", run_id="run-a"),
                _matching_run(fingerprint, "SCHEDULED", run_id="run-b"),
            ]
        )
    )
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "run_conflict"


def test_publish_unknown_status_conflicts(tmp_path: Path, mlflow_entities_stub: type) -> None:
    fingerprint = _publisher_projection(tmp_path).tags["crux.projection_sha256"]
    client = _FakeClient(_FakePage([_matching_run(fingerprint, "UNKNOWN")]))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "run_conflict"
    assert client.created_runs == []


def test_publish_retries_after_failed_run_with_other_fingerprint(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    client = _FakeClient(_FakePage([_matching_run(_OTHER_FINGERPRINT, "FAILED")]))
    result = _publish(_publisher_projection(tmp_path), client)
    assert result.created is True
    assert result.status == "published"
    assert result.run_id == "run-new"


def test_publish_different_fingerprint_conflicts(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    client = _FakeClient(_FakePage([_matching_run(_OTHER_FINGERPRINT, "FINISHED")]))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "run_conflict"
    assert client.created_runs == []


def test_publish_multiple_finished_conflicts(tmp_path: Path, mlflow_entities_stub: type) -> None:
    fingerprint = _publisher_projection(tmp_path).tags["crux.projection_sha256"]
    client = _FakeClient(
        _FakePage(
            [
                _matching_run(fingerprint, "FINISHED", run_id="run-a"),
                _matching_run(fingerprint, "FINISHED", run_id="run-b"),
            ]
        )
    )
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "run_conflict"
    assert client.created_runs == []


def test_publish_continuation_token_conflicts_without_pagination(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    fingerprint = _publisher_projection(tmp_path).tags["crux.projection_sha256"]
    client = _FakeClient(_FakePage([_matching_run(fingerprint, "FAILED")], token="token-1"))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "run_conflict"
    assert client.created_runs == []
    assert len(client.searches) == 1


def test_publish_creates_missing_experiment(tmp_path: Path, mlflow_entities_stub: type) -> None:
    projection = _publisher_projection(tmp_path)
    client = _FakeClient(_FakePage([]), has_experiment=False)
    result = _publish(projection, client)
    assert client.created_experiment_names == ["crux-benchmarks"]
    assert result.experiment_id == "exp-created"
    assert client.searches[0]["experiment_ids"] == ["exp-created"]


def test_publish_rejects_deleted_experiment(tmp_path: Path, mlflow_entities_stub: type) -> None:
    client = _FakeClient(_FakePage([]), experiment_lifecycle_stage="deleted")
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "experiment_access_failed"
    assert client.created_experiment_names == []
    assert client.created_runs == []


def test_publish_success_records_client_surface_in_order(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    projection = _publisher_projection(tmp_path)
    client = _FakeClient(_FakePage([]))
    result = _publish(projection, client)
    assert [event[0] for event in client.events] == [
        "get_experiment_by_name",
        "search_runs",
        "create_run",
        "log_batch",
        "log_artifact",
        "log_artifact",
        "log_artifact",
        "log_artifact",
        "log_artifact",
        "list_artifacts",
        "set_terminated",
    ]
    assert client.searches[0]["filter_string"] == build_matching_runs_filter(projection.tags)
    assert client.searches[0]["max_results"] == MATCHING_RUN_LIMIT
    created = client.created_runs[0]
    assert created["experiment_id"] == "exp-1"
    assert created["tags"] == dict(projection.tags)
    assert created["run_name"] == projection.run_name
    entries = client.logged_batches[0]["metrics"]
    assert {entry.name for entry in entries} == set(projection.metrics)
    for entry in entries:
        assert isinstance(entry.value, float)
        assert math.isfinite(entry.value)
        assert entry.value == float(projection.metrics[entry.name])
    assert [
        (run_id, Path(local_path).name, artifact_path)
        for run_id, local_path, artifact_path in client.logged_artifacts
    ] == [
        ("run-new", artifact.path.name, REPORT_ARTIFACT_PATH) for artifact in projection.artifacts
    ]
    assert client.logged_artifact_bytes == [
        ARTIFACT_BYTES[artifact.path.name] for artifact in projection.artifacts
    ]
    assert client.terminated == [("run-new", "FINISHED")]
    assert result.created is True
    assert result.status == "published"


def test_publish_artifact_changed_after_projection_fails_and_terminates(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    projection = _publisher_projection(tmp_path)
    (tmp_path / "summary.md").write_bytes(b"# changed after projection\n")
    client = _FakeClient(_FakePage([]))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(projection, client)
    assert raised.value.code == "artifact_upload_failed"
    assert client.logged_artifacts == []
    assert client.terminated == [("run-new", "FAILED")]


def test_publication_error_pickles_with_its_code() -> None:
    import pickle

    error = MlflowPublicationError("run_conflict")
    restored = pickle.loads(pickle.dumps(error))
    assert restored.code == "run_conflict"
    assert str(restored) == error.message


def test_publish_wraps_search_failure_with_bounded_message(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    projection = _publisher_projection(tmp_path)
    client = _FakeClient(_FakePage([]), search_error=RuntimeError("vendor secret 42"))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(projection, client)
    assert raised.value.code == "run_search_failed"
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value) == MlflowPublicationError("run_search_failed").message
    assert "vendor secret 42" not in str(raised.value)


def test_publish_experiment_failure_bounds_code(tmp_path: Path, mlflow_entities_stub: type) -> None:
    client = _FakeClient(_FakePage([]), experiment_error=OSError("boom"))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "experiment_access_failed"
    assert isinstance(raised.value.__cause__, OSError)


def test_publish_create_failure_bounds_code_and_skips_termination(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    client = _FakeClient(_FakePage([]), create_error=OSError("boom"))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "run_create_failed"
    assert client.terminated == []


def test_publish_metric_failure_terminates_new_run_as_failed(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    client = _FakeClient(_FakePage([]), log_batch_error=OSError("boom"))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "metric_log_failed"
    assert client.terminated == [("run-new", "FAILED")]
    assert isinstance(raised.value.__cause__, OSError)


def test_publish_metric_stage_import_failure_bounds_code_and_terminates(
    tmp_path: Path, mlflow_entities_stub: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Half-broken install: mlflow imports but mlflow.entities does not.
    monkeypatch.setitem(sys.modules, "mlflow.entities", None)
    client = _FakeClient(_FakePage([]))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "metric_log_failed"
    assert str(raised.value) == MlflowPublicationError("metric_log_failed").message
    assert isinstance(raised.value.__cause__, ImportError)
    assert client.logged_batches == []
    assert client.terminated == [("run-new", "FAILED")]


def test_publish_metric_stage_metric_construction_failure_bounds_and_terminates(
    tmp_path: Path, mlflow_entities_stub: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_metric(*_args: object) -> None:
        raise RuntimeError("vendor metric construction boom")

    monkeypatch.setattr(sys.modules["mlflow.entities"], "Metric", broken_metric)
    client = _FakeClient(_FakePage([]))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "metric_log_failed"
    assert str(raised.value) == MlflowPublicationError("metric_log_failed").message
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "vendor metric construction boom" not in str(raised.value)
    assert client.logged_batches == []
    assert client.terminated == [("run-new", "FAILED")]


def test_publish_artifact_failure_terminates_new_run_as_failed(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    client = _FakeClient(_FakePage([]), artifact_error=OSError("boom"))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "artifact_upload_failed"
    assert client.terminated == [("run-new", "FAILED")]
    assert isinstance(raised.value.__cause__, OSError)


@pytest.mark.parametrize(
    "artifact_uri",
    [
        # Plaintext http puts file bytes and resolvable credentials on the wire.
        "http://artifacts.example.com",
        "http://mlflow.example.com",
        # Embedded credentials and query/fragment are rejected like tracking URIs.
        "https://user@artifacts.example.com",
        "https://user:pass@artifacts.example.com",
        "https://artifacts.example.com/?x=1",
        "https://artifacts.example.com/#frag",
        # A host-bearing mlflow-artifacts URI repoints uploads at another origin.
        "mlflow-artifacts://artifacts.example.com/0/run/artifacts",
        # Other remote schemes are out of scope for the verified https policy.
        "s3://bucket/prefix",
        "ftp://host.example/store",
        "runs:/other-run/artifacts",
        "models:/model/1",
        # Malformed URIs fail parsing outright.
        "https://[bad",
        # An empty destination resolves to a local root; it is not a valid one.
        "",
    ],
)
def test_publish_rejects_unsafe_artifact_destinations(
    tmp_path: Path, mlflow_entities_stub: type, artifact_uri: str
) -> None:
    client = _FakeClient(_FakePage([]), artifact_uri=artifact_uri)
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "unsafe_artifact_uri"
    # The run stays RUNNING so a retry resumes it after the server artifact
    # destination is fixed; nothing is logged or terminated.
    assert client.logged_artifacts == []
    assert client.logged_batches == []
    assert client.terminated == []


def test_publish_rejects_unsafe_artifact_uri_on_resumed_run(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    projection = _publisher_projection(tmp_path)
    fingerprint = projection.tags["crux.projection_sha256"]
    leftover = _matching_run(fingerprint, "RUNNING", artifact_uri="http://artifacts.example.com")
    client = _FakeClient(_FakePage([leftover]))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(projection, client)
    assert raised.value.code == "unsafe_artifact_uri"
    assert client.created_runs == []
    assert client.logged_artifacts == []


@pytest.mark.parametrize(
    "artifact_uri",
    [
        "https://artifacts.example.com",
        "https://artifacts.example.com:8443/base/path",
        "mlflow-artifacts:/0/run-new/artifacts",
        "file:///tmp/mlruns/0/run-new/artifacts",
        "mlruns/0/run-new/artifacts",
    ],
)
def test_publish_accepts_safe_artifact_destinations(
    tmp_path: Path, mlflow_entities_stub: type, artifact_uri: str
) -> None:
    client = _FakeClient(_FakePage([]), artifact_uri=artifact_uri)
    assert _publish(_publisher_projection(tmp_path), client).status == "published"


def test_publish_disables_proxy_multipart_uploads(
    tmp_path: Path, mlflow_entities_stub: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Presigned multipart part uploads use raw requests.put calls that bypass
    # both the artifact-URI check and MLFLOW_ALLOW_HTTP_REDIRECTS; the env var
    # must be forced off for artifact calls and restored afterward.
    monkeypatch.delenv("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", raising=False)
    client = _FakeClient(_FakePage([]))
    _publish(_publisher_projection(tmp_path), client)
    assert client.multipart_env
    assert all(value == "false" for value in client.multipart_env)
    assert "MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD" not in os.environ


def test_publish_restores_proxy_multipart_env(
    tmp_path: Path, mlflow_entities_stub: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "true")
    client = _FakeClient(_FakePage([]))
    _publish(_publisher_projection(tmp_path), client)
    assert all(value == "false" for value in client.multipart_env)
    assert os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] == "true"


def test_publish_fails_when_uploaded_artifact_is_not_listed(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    # An unfollowed 3xx passes mlflow's raise_for_status as "uploaded" while
    # nothing was stored; the post-upload listing must catch the gap.
    client = _FakeClient(
        _FakePage([]),
        listed_artifacts=[
            SimpleNamespace(path=f"{REPORT_ARTIFACT_PATH}/summary.json", file_size=None)
        ],
    )
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "artifact_upload_failed"
    assert client.terminated == [("run-new", "FAILED")]


def test_publish_fails_when_listed_artifact_size_mismatches(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    client = _FakeClient(
        _FakePage([]),
        listed_artifacts=[
            SimpleNamespace(path=f"{REPORT_ARTIFACT_PATH}/{name}", file_size=0)
            for name in REPORT_ARTIFACT_NAMES
        ],
    )
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "artifact_upload_failed"
    assert client.terminated == [("run-new", "FAILED")]


def test_publish_list_artifacts_failure_bounds_code(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    client = _FakeClient(_FakePage([]), list_artifact_error=OSError("boom"))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "artifact_upload_failed"
    assert client.terminated == [("run-new", "FAILED")]
    assert isinstance(raised.value.__cause__, OSError)


def test_publish_accepts_listing_without_reported_sizes(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    client = _FakeClient(
        _FakePage([]),
        listed_artifacts=[
            SimpleNamespace(path=f"{REPORT_ARTIFACT_PATH}/{name}", file_size=None)
            for name in REPORT_ARTIFACT_NAMES
        ],
    )
    assert _publish(_publisher_projection(tmp_path), client).status == "published"


def test_publish_finalize_failure_leaves_run_status_unchanged(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    # set_terminated can commit FINISHED before its response is lost, and
    # mlflow accepts a later FAILED on that completed run; marking failed
    # here would hide the run from the idempotent match and force a
    # duplicate on the next publish.
    client = _FakeClient(_FakePage([]), finalize_error=OSError("boom"))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "run_finalize_failed"
    assert client.terminated == []
    assert [event for event in client.events if event[0] == "set_terminated"] == [
        ("set_terminated", "run-new", "FINISHED")
    ]
    assert isinstance(raised.value.__cause__, OSError)


def test_publish_finalize_failure_retry_resumes_the_same_run(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    projection = _publisher_projection(tmp_path)
    interrupted = _FakeClient(_FakePage([]), finalize_error=OSError("boom"))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(projection, interrupted)
    assert raised.value.code == "run_finalize_failed"

    fingerprint = projection.tags["crux.projection_sha256"]
    leftover = _matching_run(fingerprint, "RUNNING", run_id="run-new")
    recovered = _FakeClient(_FakePage([leftover]))
    result = _publish(projection, recovered)
    assert result.status == "published"
    assert result.created is False
    assert result.run_id == "run-new"
    assert recovered.created_runs == []
    assert recovered.terminated == [("run-new", "FINISHED")]


def test_publish_requests_synchronous_metric_logging(
    tmp_path: Path, mlflow_entities_stub: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    # MLFLOW_ENABLE_ASYNC_LOGGING must not leave metrics pending past finalize.
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "true")
    client = _FakeClient(_FakePage([]))
    _publish(_publisher_projection(tmp_path), client)
    assert client.logged_batches[0]["synchronous"] is True


def test_publish_chunks_metrics_at_mlflow_batch_limit(
    tmp_path: Path, mlflow_entities_stub: type
) -> None:
    # mlflow's RestStore rejects log_batch calls over 1,000 metrics; reports
    # with extra canonical tolerances cross that cap and must still publish.
    projection = _publisher_projection(tmp_path)
    metrics = dict(projection.metrics)
    while len(metrics) <= 2 * 1000:
        metrics[f"synthetic.metric_{len(metrics)}"] = Decimal("0.5")
    projection = replace(projection, metrics=MappingProxyType(metrics))
    client = _FakeClient(_FakePage([]))
    _publish(projection, client)
    assert len(client.logged_batches) > 1
    logged_names = []
    for batch in client.logged_batches:
        assert 0 < len(batch["metrics"]) <= 1000
        assert batch["synchronous"] is True
        logged_names.extend(entry.name for entry in batch["metrics"])
    assert logged_names == list(projection.metrics)


def test_finalize_run_suppresses_mlflow_url_output(
    tmp_path: Path, mlflow_entities_stub: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", raising=False)
    client = _FakeClient(_FakePage([]))
    _publish(_publisher_projection(tmp_path), client)
    assert client.terminated_suppression == ["true"]
    assert os.environ.get("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT") is None


def test_failed_run_cleanup_suppresses_mlflow_url_output(
    tmp_path: Path, mlflow_entities_stub: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "false")
    client = _FakeClient(_FakePage([]), log_batch_error=OSError("boom"))
    with pytest.raises(MlflowPublicationError) as raised:
        _publish(_publisher_projection(tmp_path), client)
    assert raised.value.code == "metric_log_failed"
    assert client.terminated_suppression == ["true"]
    assert os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] == "false"


def test_real_client_file_store_publication_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow = pytest.importorskip("mlflow")
    from mlflow import MlflowClient
    from mlflow.entities import Metric

    assert mlflow is not None
    Metric("k", 1.0, 0, 0)
    for method in (
        "get_experiment_by_name",
        "create_experiment",
        "search_runs",
        "create_run",
        "log_batch",
        "log_artifact",
        "set_terminated",
    ):
        assert callable(getattr(MlflowClient, method, None)), method

    report_dir = tmp_path / "report"
    report_dir.mkdir()
    _report_dir(report_dir, write_extra=False)
    projection = build_mlflow_projection(_reports(), report_dir, scope="broad")
    # mlflow-skinny 3.16 gates the maintenance-mode file store behind this opt-in.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    client = MlflowClient(tracking_uri=Path(tmp_path / "mlruns").resolve().as_uri())
    experiment_name = "crux-benchmarks"

    first = publish_mlflow_projection(projection, client=client, experiment_name=experiment_name)
    assert first.created is True
    assert first.status == "published"
    assert first.experiment_name == experiment_name
    assert first.projection_sha256 == projection.tags["crux.projection_sha256"]

    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None
    assert first.experiment_id == experiment.experiment_id
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=build_matching_runs_filter(projection.tags),
        max_results=MATCHING_RUN_LIMIT,
    )
    assert len(runs) == 1
    run = runs[0]
    assert run.info.run_id == first.run_id
    assert run.info.status == "FINISHED"
    assert run.data.tags["crux.projection_sha256"] == projection.tags["crux.projection_sha256"]

    logged = client.get_run(first.run_id).data.metrics
    assert logged["population.total"] == 5.0
    assert len(logged) == len(projection.metrics)

    artifacts = client.list_artifacts(first.run_id, REPORT_ARTIFACT_PATH)
    assert sorted(item.path for item in artifacts) == sorted(
        f"{REPORT_ARTIFACT_PATH}/{name}" for name in REPORT_ARTIFACT_NAMES
    )

    second = publish_mlflow_projection(projection, client=client, experiment_name=experiment_name)
    assert second.created is False
    assert second.status == "already_published"
    assert second.run_id == first.run_id
