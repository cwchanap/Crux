"""Fixture contract for the HPA-324 reference-lane audit."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.benchmark.backend_identity import strict_json_loads
from src.benchmark.corpus_manifest import render_manifest
from src.benchmark.reference_timing import NativeReferenceEvent, render_reference_events


def _event(
    *,
    simfile_id: int,
    lane_id: str,
    source_order: int,
    measure: int,
    audio_time_sec: float,
) -> NativeReferenceEvent:
    return NativeReferenceEvent(
        simfile_id=simfile_id,
        selected_chart_key=f"{simfile_id}/real.dtx",
        selected_chart_content_hash="5dbd7639514e76fbd11cc8d6c518adae8b168ea944461c555236dadb1a3a4809",
        source_audio_key=f"{simfile_id}/bgm.wav",
        source_audio_content_hash="c" * 64,
        source_order=source_order,
        measure=measure,
        position=0.0,
        lane_id=lane_id,
        note_id=f"{source_order:02X}",
        chart_time_sec=audio_time_sec,
        audio_time_sec=audio_time_sec,
    )


def _ready_row(simfile_id: int, event_path: str) -> dict[str, object]:
    golden = (
        Path(__file__).parents[2]
        / "benchmark/schema_goldens/crux.reference-timing-manifest-v1.jsonl"
    )
    row = strict_json_loads(golden.read_bytes().splitlines()[0], require_canonical=True)
    assert isinstance(row, dict)
    row = dict(row)
    row["simfile_id"] = simfile_id
    row["object_prefix"] = f"{simfile_id}/"
    objects = row["objects"]
    assert isinstance(objects, list) and len(objects) == 1
    object_row = dict(objects[0])
    object_row["key"] = f"{simfile_id}/real.dtx"
    row["objects"] = [object_row]
    row["selected_chart_key"] = f"{simfile_id}/real.dtx"
    row["source_audio_key"] = f"{simfile_id}/bgm.wav"
    row["reference_events_cache_path"] = event_path
    return row


def _quarantined_row() -> dict[str, object]:
    golden = (
        Path(__file__).parents[2]
        / "benchmark/schema_goldens/crux.reference-timing-manifest-v1.jsonl"
    )
    row = strict_json_loads(golden.read_bytes().splitlines()[1], require_canonical=True)
    assert isinstance(row, dict)
    return dict(row)


def _write_fixture(tmp_path: Path) -> Path:
    events_by_simfile = {
        42: (
            _event(simfile_id=42, lane_id="11", source_order=0, measure=0, audio_time_sec=1.0),
            _event(simfile_id=42, lane_id="14", source_order=1, measure=1, audio_time_sec=2.0),
            _event(simfile_id=42, lane_id="15", source_order=2, measure=1, audio_time_sec=2.0),
        ),
        44: (
            _event(simfile_id=44, lane_id="13", source_order=0, measure=0, audio_time_sec=1.0),
            _event(simfile_id=44, lane_id="2A", source_order=1, measure=1, audio_time_sec=2.0),
        ),
    }
    output_dir = tmp_path / "reference-timing"
    events_dir = output_dir / "events"
    events_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for simfile_id, events in events_by_simfile.items():
        content = render_reference_events(events)
        digest = hashlib.sha256(content).hexdigest()
        (events_dir / f"{digest}.jsonl").write_bytes(content)
        rows.append(_ready_row(simfile_id, f"events/{digest}.jsonl"))
    rows.append(_quarantined_row())
    rendered = render_manifest(
        tuple({key: value for key, value in row.items() if key != "corpus_version"} for row in rows)
    )
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir()
    manifest_path = manifests_dir / f"{rendered.manifest_sha256}.jsonl"
    manifest_path.write_bytes(rendered.content)
    return manifest_path


def test_audit_reports_unknown_lanes_collisions_and_prospective_status(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)

    from tools.hpa324.analyze_reference_lanes import run_reference_lane_audit

    report = run_reference_lane_audit(manifest_path)

    assert (
        report["source_reference_timing_manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert report["source_reference_timing_version"].startswith("sha256:")
    assert report["row_count"] == 3
    assert report["ready_row_count"] == 2
    assert report["lane_event_counts"] == {"11": 1, "13": 1, "14": 1, "15": 1, "2A": 1}
    assert report["unmapped_lane_event_counts"] == {"2A": 1}
    assert report["unmapped_lane_simfile_counts"] == {"2A": 1}
    assert report["common_collision_count"] == 1
    assert report["common_collision_simfile_count"] == 1
    assert report["ignored_non_drum_lanes"] == ["54", "C2"]
    assert report["prospective_eligible_row_count"] == 1
    assert report["prospective_quarantined_row_count"] == 1
    assert report["ready_row_count"] == (
        report["prospective_eligible_row_count"] + report["prospective_quarantined_row_count"]
    )


def test_audit_rejects_content_addressed_event_hash_drift(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    row = strict_json_loads(manifest_path.read_bytes().splitlines()[0], require_canonical=True)
    assert isinstance(row, dict)
    event_path = row["reference_events_cache_path"]
    assert isinstance(event_path, str)
    artifact = manifest_path.parent.parent / event_path
    artifact.write_bytes(artifact.read_bytes() + b" ")

    from tools.hpa324.analyze_reference_lanes import run_reference_lane_audit

    with pytest.raises(ValueError, match="content hash"):
        run_reference_lane_audit(manifest_path)
