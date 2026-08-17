from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.benchmark.backend_identity import BackendDescriptor
from src.benchmark.backends.base import CanonicalAudio, NativePrediction
from src.benchmark.cohort_scoring import score_cohort
from src.benchmark.muscriptor_corpus_run import (
    build_muscriptor_cohort_from_snapshot,
    parse_muscriptor_corpus_run,
    run_muscriptor_corpus,
)
from src.benchmark.reports import write_cohort_reports
from tests.benchmark.muscriptor_run_fixtures import (
    _install_seams,
    _manifests,
    _mapping,
    _prediction,
    _request,
)


def test_persisted_muscriptor_run_acceptance_reconstructs_hpa325_reports(
    tmp_path: Path, monkeypatch
) -> None:
    _, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)
    reference_manifest, _ = _manifests()

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return _prediction(audio, descriptor)

        def close(self) -> None:
            return None

    outcome = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert outcome.exit_code == 0
    assert outcome.run_id is not None
    assert outcome.run_path is not None
    assert outcome.reports_path is not None
    assert outcome.reports_path.is_dir()
    assert (outcome.reports_path / "summary.json").exists()
    assert (outcome.reports_path / "per_song.csv").exists()
    assert (outcome.reports_path / "per_class.csv").exists()

    snapshot = parse_muscriptor_corpus_run(outcome.run_path.read_bytes())
    mappings = {
        row.view.simfile_id: _mapping(row.view.simfile_id) for row in reference_manifest.rows
    }
    identity, items = build_muscriptor_cohort_from_snapshot(
        snapshot,
        mappings=mappings,
        output_dir=tmp_path / "output",
    )
    expected = score_cohort(identity, items, diagnostics_for=())
    expected_reports = tmp_path / "expected-reports"
    write_cohort_reports(expected, expected_reports)
    for name in (
        "summary.json",
        "items.csv",
        "per_song.csv",
        "per_class.csv",
        "event_diagnostics.jsonl",
        "summary.md",
    ):
        assert (outcome.reports_path / name).read_bytes() == (expected_reports / name).read_bytes()
