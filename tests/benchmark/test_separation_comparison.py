"""Focused HPA-328 published separation-comparison coverage."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from src.benchmark.published_comparison import (
    PublishedRunEvidence,
    PublishedRunItem,
    pairable_success_ids,
)
from src.benchmark.separation_comparison import aggregate_paired_event_micro


def _song(
    simfile_id: str,
    *,
    tp: int,
    fp: int,
    fn: int,
    f1: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        simfile_id=simfile_id,
        tolerance_ms=50,
        mode="raw",
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=Decimal("0.5"),
        recall=Decimal("0.5"),
        f1=Decimal(f1),
    )


def test_paired_event_micro_sums_published_counts_and_uses_authoritative_minutes() -> None:
    left = SimpleNamespace(
        songs=(_song("1", tp=1, fp=1, fn=3, f1="0.2"), _song("2", tp=3, fp=0, fn=1, f1="0.75")),
    )
    right = SimpleNamespace(
        songs=(_song("1", tp=2, fp=0, fn=2, f1="0.5"), _song("2", tp=4, fp=1, fn=0, f1="0.8")),
    )

    rows = aggregate_paired_event_micro(
        left,
        right,
        {"1", "2"},
        {"1": Decimal("60"), "2": Decimal("120")},
        left_label="full_mix",
        right_label="spleeter",
    )

    row = rows[0]
    assert row["full_mix"]["tp"] == 4
    assert row["full_mix"]["fp"] == 1
    assert row["full_mix"]["fn"] == 4
    assert row["full_mix"]["f1"] == Decimal("0.615385")
    assert row["spleeter"]["tp"] == 6
    assert row["spleeter"]["fp"] == 1
    assert row["spleeter"]["fn"] == 2
    assert row["spleeter"]["f1"] == Decimal("0.8")
    assert row["spleeter"]["fp_per_minute"] == Decimal("0.333333")
    assert row["spleeter"]["fn_per_minute"] == Decimal("0.666667")


def test_separation_pairing_allows_distinct_derived_input_hashes() -> None:
    left = PublishedRunEvidence(
        identity=SimpleNamespace(),
        items={"1": PublishedRunItem("1", "success", "a" * 64, "c" * 64)},
        reports=SimpleNamespace(),  # type: ignore[arg-type]
    )
    right = PublishedRunEvidence(
        identity=SimpleNamespace(),
        items={"1": PublishedRunItem("1", "success", "a" * 64, "d" * 64)},
        reports=SimpleNamespace(),  # type: ignore[arg-type]
    )

    pairable, exclusions = pairable_success_ids(
        left,
        right,
        None,
        require_identical_input_hash=False,
        left_label="full_mix",
        right_label="spleeter",
    )

    assert pairable == {"1"}
    assert exclusions["source_audio_mismatch"] == 0


def test_pilot_comparison_call_uses_run_scoped_output_after_scoring(
    tmp_path,
    monkeypatch,
) -> None:
    from tests.benchmark.reviewed_subset_fixtures import build_reviewed_subset_oaf_fixture
    from tests.benchmark.test_separation_pilot import (
        _install_fixture_locks,
        _request,
        _subset_path,
        _task6_seams,
    )

    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    _install_fixture_locks(monkeypatch)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    comparison_requests = []

    def fake_compare(comparison_request):
        comparison_requests.append(comparison_request)
        return None

    import src.benchmark.separation_pilot as pilot

    monkeypatch.setattr(pilot, "compare_oaf_separation", fake_compare)
    monkeypatch.setattr(pilot, "_comparison_reports_ready", lambda _run_dir: True)

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    outcome = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    assert outcome.exit_code == 0
    assert len(comparison_requests) == 1
    comparison_request = comparison_requests[0]
    assert comparison_request.run_path == outcome.run_path
    assert comparison_request.output_dir == outcome.run_path.parent / "comparison"
    assert comparison_request.subset_manifest_path == subset


def test_comparison_publishes_paired_csvs_summary_and_native_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    from tests.benchmark.reviewed_subset_fixtures import build_reviewed_subset_oaf_fixture
    from tests.benchmark.test_separation_pilot import (
        _install_fixture_locks,
        _request,
        _subset_path,
        _task6_seams,
    )

    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20, failed_count=0)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    _install_fixture_locks(monkeypatch)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)

    import src.benchmark.separation_pilot as pilot
    from src.benchmark.reviewed_subset import score_oaf_reviewed_subset

    monkeypatch.setattr(pilot, "score_oaf_reviewed_subset", score_oaf_reviewed_subset)

    from dataclasses import replace

    base_factory = calls["factory"][0]

    def aligned_backend_factory(**kwargs):
        backend = base_factory(**kwargs)

        class AlignedBackend:
            def descriptor(self):
                return backend.descriptor()

            def transcribe(self, audio):
                prediction = backend.transcribe(audio)
                return replace(
                    prediction,
                    events=tuple(replace(event, time_sec=0.5) for event in prediction.events),
                )

            def close(self):
                return backend.close()

        return AlignedBackend()

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    outcome = run_oaf_separation_pilot(
        request,
        backend_factory=aligned_backend_factory,
    )

    assert outcome.exit_code == 0
    assert outcome.run_path is not None
    comparison_dir = outcome.run_path.parent / "comparison"
    expected_files = {
        "spleeter/paired_per_song.csv",
        "spleeter/paired_per_class.csv",
        "htdemucs/paired_per_song.csv",
        "htdemucs/paired_per_class.csv",
        "summary.json",
        "summary.md",
    }
    actual_files = {
        path.relative_to(comparison_dir).as_posix()
        for path in comparison_dir.rglob("*")
        if path.is_file()
    }
    assert actual_files == expected_files

    import json

    summary = json.loads((comparison_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "crux.oaf-separation-comparison/v1"
    for view_name in ("spleeter", "htdemucs"):
        assert summary["pairing"][view_name]["pairable_success_intersection"] == 20
        assert summary["models"][view_name]["failure_code_histogram"] == {}
        resources = summary["models"][view_name]["resources"]
        assert resources["retained_stem_bytes"] > 0
        assert resources["retained_prediction_bytes"] > 0
        assert resources["retained_report_bytes"] > 0
        assert summary["comparisons"][view_name]["event_micro"]
    assert "cost" not in (comparison_dir / "summary.json").read_text(encoding="utf-8")
