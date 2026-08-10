"""Reproducible HPA-324 audit of persisted HPA-323 reference lanes."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.reference_set import map_reference_events
from src.benchmark.reference_timing import read_reference_events
from src.benchmark.reference_timing_manifest import load_reference_timing_manifest
from src.benchmark.taxonomy import IGNORED_NON_DRUM_LANES


def _timing_artifact_root(manifest_path: Path) -> Path:
    """Return the HPA-323 output root containing ``manifests/`` and ``events/``."""
    manifest_parent = manifest_path.parent.resolve()
    return manifest_parent.parent if manifest_parent.name == "manifests" else manifest_parent


def _read_event_artifact(manifest_path: Path, relative_path: str) -> tuple[Any, ...]:
    """Read one manifest-referenced, content-addressed native event artifact."""
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or relative.parts[:1] != ("events",):
        raise ValueError("reference event artifact path escapes the timing artifact root")

    root = _timing_artifact_root(manifest_path)
    artifact = (root / Path(*relative.parts)).resolve()
    try:
        artifact.relative_to(root)
    except ValueError:
        raise ValueError("reference event artifact path escapes the timing artifact root") from None

    try:
        content = artifact.read_bytes()
    except OSError:
        raise ValueError(f"reference event artifact is unavailable: {relative_path}") from None

    expected_sha256 = relative.name.removesuffix(".jsonl")
    actual_sha256 = sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"reference event artifact content hash does not match filename: {relative_path}"
        )
    return read_reference_events(content)


def run_reference_lane_audit(manifest_path: Path) -> dict[str, Any]:
    """Aggregate deterministic lane and prospective eligibility diagnostics."""
    manifest_path = Path(manifest_path)
    loaded = load_reference_timing_manifest(manifest_path)

    lane_event_counts: Counter[str] = Counter()
    unmapped_lane_event_counts: Counter[str] = Counter()
    unmapped_lane_simfile_counts: Counter[str] = Counter()
    common_collision_count = 0
    common_collision_simfile_count = 0
    ready_row_count = 0
    prospective_eligible_row_count = 0
    prospective_quarantined_row_count = 0

    for loaded_row in loaded.rows:
        view = loaded_row.view
        if view.timing_status != "ready":
            continue

        ready_row_count += 1
        if view.reference_events_cache_path is None:
            raise ValueError(f"ready row {view.simfile_id} has no reference event artifact")
        events = _read_event_artifact(manifest_path, view.reference_events_cache_path)
        if any(event.simfile_id != view.simfile_id for event in events):
            raise ValueError(
                f"reference event artifact identity does not match simfile {view.simfile_id}"
            )

        for event in events:
            lane_event_counts[event.lane_id] += 1

        mapped = map_reference_events(events)
        for lane_id, count in mapped.diagnostics.unmapped.items():
            unmapped_lane_event_counts[lane_id] += count
            unmapped_lane_simfile_counts[lane_id] += 1

        collision_count = mapped.diagnostics.duplicate_common_event_count
        common_collision_count += collision_count
        if collision_count:
            common_collision_simfile_count += 1

        if mapped.diagnostics.unmapped or not mapped.mapped_events:
            prospective_quarantined_row_count += 1
        else:
            prospective_eligible_row_count += 1

    return {
        "source_reference_timing_manifest_sha256": loaded.manifest_sha256,
        "source_reference_timing_version": loaded.corpus_version,
        "row_count": len(loaded.rows),
        "ready_row_count": ready_row_count,
        "lane_event_counts": dict(sorted(lane_event_counts.items())),
        "unmapped_lane_event_counts": dict(sorted(unmapped_lane_event_counts.items())),
        "unmapped_lane_simfile_counts": dict(sorted(unmapped_lane_simfile_counts.items())),
        "common_collision_count": common_collision_count,
        "common_collision_simfile_count": common_collision_simfile_count,
        "ignored_non_drum_lanes": sorted(IGNORED_NON_DRUM_LANES),
        "prospective_eligible_row_count": prospective_eligible_row_count,
        "prospective_quarantined_row_count": prospective_quarantined_row_count,
    }


def render_report(report: dict[str, Any]) -> bytes:
    """Render the audit report as canonical JSON with one final newline."""
    return canonical_json_bytes(report, trailing_newline=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Immutable HPA-323 reference-timing manifest (canonical JSONL).",
    )
    args = parser.parse_args(argv)
    report = run_reference_lane_audit(args.manifest)
    print(render_report(report).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
