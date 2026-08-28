"""Reusable synthetic HPA-323/HPA-324 reference fixture for reviewed-subset tests.

Factors the smallest already-working canonical timing/reference/event builders
from the HPA-323/HPA-324/HPA-326 tests: one ready timing row is cloned from the
``crux.reference-timing-manifest/v1`` schema golden per distinct simfile ID, its
event artifact is re-keyed and written under the timing output root, and the
HPA-324 reference manifest is derived through the real ``run_reference_set``
pipeline so every row is eligible.  Optional per-row recipes vary timing
warnings, selected-chart basename, and reference events so tests can build
populations with distinct density/richness strata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from src.benchmark.corpus_manifest import render_manifest
from src.benchmark.reference_set_manifest import (
    ReferenceSetRequest,
    load_reference_set_manifest,
    run_reference_set,
)
from src.benchmark.reference_timing import NativeReferenceEvent, render_reference_events
from src.benchmark.reference_timing_manifest import load_reference_timing_manifest

_MODEL_LOCK_PATH = Path(__file__).resolve().parents[2] / "runtime" / "oaf_tf1" / "model.json"

_TIMING_GOLDEN = Path(__file__).parent / "schema_goldens/crux.reference-timing-manifest-v1.jsonl"
_FIRST_SIMFILE_ID = 100


@dataclass(frozen=True)
class ReviewedSubsetReferenceFixture:
    reference_manifest_path: Path
    timing_manifest_path: Path
    timing_output_root: Path


@dataclass(frozen=True)
class ReviewedSubsetOafFixture:
    """One persisted OaF corpus run over the synthetic reviewed-subset population.

    Mirrors the smallest persisted-run/prediction setup from the HPA-326
    acceptance suite: the run snapshot lives at ``run_path`` under
    ``oaf_output_dir`` and every successful row carries a real prediction
    artifact referenced by a relative ``prediction_path``.
    """

    reference_manifest_path: Path
    timing_manifest_path: Path
    timing_output_root: Path
    run_path: Path
    oaf_output_dir: Path


@dataclass(frozen=True)
class ReviewedSubsetRowSpec:
    """One per-simfile recipe for the synthetic HPA-323/HPA-324 population.

    ``events`` holds ``(lane_id, audio_time_sec)`` pairs rendered as native
    reference events in order; ``chart_basename`` is the selected chart's file
    basename; ``timing_warnings`` is published verbatim on the timing row.
    """

    timing_warnings: tuple[str, ...] = ()
    chart_basename: str = "real.dtx"
    events: tuple[tuple[str, float], ...] = (("13", 0.5),)


def build_reviewed_subset_reference_fixture(
    tmp_path: Path,
    *,
    eligible_count: int = 36,
    reverse_rows: bool = False,
    row_specs: tuple[ReviewedSubsetRowSpec, ...] | None = None,
) -> ReviewedSubsetReferenceFixture:
    """Build an all-eligible HPA-323 timing + HPA-324 reference population.

    ``eligible_count`` controls the population size: one ready timing row per
    distinct simfile ID, each backed by a valid events artifact.  ``row_specs``
    overrides the per-row recipe (one entry per simfile ID, in ID order).  With
    ``reverse_rows=True`` the timing rows are published in reverse input order
    (and the derived reference manifest inherits that order) so consumers can
    prove candidate selection is order-independent.
    """
    ready_row = json.loads(_TIMING_GOLDEN.read_text(encoding="utf-8").splitlines()[0])
    if row_specs is None:
        row_specs = tuple(ReviewedSubsetRowSpec() for _ in range(eligible_count))
    if len(row_specs) != eligible_count:
        raise ValueError("row_specs must match eligible_count")
    timing_rows: list[dict[str, object]] = []
    event_artifacts: list[tuple[Path, bytes]] = []
    for offset in range(eligible_count):
        spec = row_specs[offset]
        simfile_id = _FIRST_SIMFILE_ID + offset
        selected_chart_key = f"{simfile_id}/{spec.chart_basename}"
        events = tuple(
            NativeReferenceEvent(
                simfile_id=simfile_id,
                selected_chart_key=selected_chart_key,
                selected_chart_content_hash=ready_row["selected_chart_content_hash"],
                source_audio_key=f"{simfile_id}/bgm.wav",
                source_audio_content_hash=ready_row["source_audio_content_hash"],
                source_order=index,
                measure=1,
                position=0.0,
                lane_id=lane_id,
                note_id=f"{index + 1:02d}",
                chart_time_sec=audio_time_sec + 0.5,
                audio_time_sec=audio_time_sec,
            )
            for index, (lane_id, audio_time_sec) in enumerate(spec.events)
        )
        event_content = render_reference_events(events)
        event_hash = sha256(event_content).hexdigest()
        row = dict(ready_row)
        row.pop("corpus_version", None)
        row["simfile_id"] = simfile_id
        row["object_prefix"] = f"{simfile_id}/"
        source_audio_key = f"{simfile_id}/bgm.wav"
        source_audio_hash = str(ready_row["source_audio_content_hash"])
        objects = ready_row["objects"]
        assert isinstance(objects, list) and objects
        remote_template = objects[0]
        assert isinstance(remote_template, dict)

        chart_remote = {**remote_template, "key": selected_chart_key}
        audio_remote = {
            **remote_template,
            "key": source_audio_key,
            "content_type": "audio/wav",
            "etag": f'"audio-{simfile_id}"',
            "size": 88244,
            "sha256": source_audio_hash,
            "cache_path": f"sha256/{source_audio_hash[:2]}/{source_audio_hash}",
        }

        row["objects"] = [chart_remote, audio_remote]
        row["selected_chart_key"] = selected_chart_key
        row["source_audio_key"] = source_audio_key
        row["reference_events_cache_path"] = f"events/{event_hash}.jsonl"
        row["timing_warnings"] = list(spec.timing_warnings)
        timing_rows.append(row)
        event_artifacts.append((Path("events") / f"{event_hash}.jsonl", event_content))

    source_rows = tuple(reversed(timing_rows) if reverse_rows else timing_rows)
    rendered = render_manifest(source_rows)
    timing_root = tmp_path / "timing"
    timing_path = timing_root / "manifests" / f"{rendered.manifest_sha256}.jsonl"
    timing_path.parent.mkdir(parents=True)
    timing_path.write_bytes(rendered.content)
    for relative, content in event_artifacts:
        artifact_path = timing_root / relative
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(content)

    outcome = run_reference_set(ReferenceSetRequest(timing_path, tmp_path / "reference-set"))
    if outcome.manifest is None or outcome.exit_code != 0:
        raise ValueError("reviewed subset fixture must produce an all-eligible reference manifest")
    return ReviewedSubsetReferenceFixture(
        reference_manifest_path=outcome.manifest.path,
        timing_manifest_path=timing_path,
        timing_output_root=timing_root,
    )


def build_reviewed_subset_oaf_fixture(
    tmp_path: Path,
    *,
    eligible_count: int = 36,
    failed_count: int = 1,
) -> ReviewedSubsetOafFixture:
    """Build a persisted OaF corpus run over the synthetic population.

    Every eligible row except the trailing ``failed_count`` is persisted as an
    ``inferred`` row with a real, source-keyed prediction artifact; the
    trailing rows are persisted as ``failed`` (``inference_failed``) so tests
    can select a non-success item into a reviewed subset.  No Docker,
    TensorFlow, network, or backend is invoked.
    """
    # Heavy OaF imports stay inside the builder so suites importing the
    # fixture module never pull librosa/TF model code at collection time.
    from runtime.oaf_tf1.model import load_model_config
    from src.benchmark.backends import CanonicalAudio, NativeEvent, NativePrediction
    from src.benchmark.mapping import map_oaf_prediction
    from src.benchmark.oaf_corpus_run import (
        OAF_ADAPTER_REVISION,
        OAF_CANONICALIZATION_REVISION,
        OAF_CORPUS_RUN_SCHEMA,
        OAF_FULL_MIX_INPUT_VIEW_ID,
        _expected_oaf_descriptor,
        build_inference_config,
        build_run_id,
        compute_model_lock_sha256,
        inference_config_sha256,
        prediction_path,
        write_oaf_corpus_run,
    )
    from src.benchmark.prediction_artifact import publish_prediction_artifact

    reference_fixture = build_reviewed_subset_reference_fixture(
        tmp_path, eligible_count=eligible_count
    )
    reference = load_reference_set_manifest(reference_fixture.reference_manifest_path)
    timing = load_reference_timing_manifest(reference_fixture.timing_manifest_path)
    config = load_model_config(_MODEL_LOCK_PATH)
    descriptor = _expected_oaf_descriptor(config)
    model_lock_sha = compute_model_lock_sha256(_MODEL_LOCK_PATH)
    inference_payload = build_inference_config(config, descriptor, model_lock_sha)
    config_sha = inference_config_sha256(inference_payload)
    run_id = build_run_id(
        reference.manifest_sha256,
        timing.manifest_sha256,
        descriptor.sha256,
        model_lock_sha,
        config.checkpoint.archive_sha256,
        config_sha,
    )
    oaf_output_dir = tmp_path / "oaf-output"
    run_dir = oaf_output_dir / "runs" / run_id
    run_path = run_dir / "run.json"
    run_dir.mkdir(parents=True)
    input_audio_sha256 = sha256(b"canonical-oaf-full-mix").hexdigest()

    rows: list[dict[str, object]] = []
    for offset, loaded in enumerate(reference.rows):
        simfile_id = loaded.view.simfile_id
        row: dict[str, object] = {
            "simfile_id": simfile_id,
            "eligibility_status": loaded.view.eligibility_status,
            "eligibility_reason_codes": list(loaded.view.eligibility_reason_codes),
            "eligibility_warnings": list(loaded.view.eligibility_warnings),
            "selected_chart_key": loaded.source_row["selected_chart_key"],
            "selected_chart_content_hash": loaded.source_row["selected_chart_content_hash"],
            "source_audio_key": loaded.source_row["source_audio_key"],
            "source_audio_content_hash": loaded.source_row["source_audio_content_hash"],
        }
        if offset >= eligible_count - failed_count:
            row["execution_disposition"] = "failed"
            row["runner_failure_code"] = "inference_failed"
            row["failure_detail"] = "synthetic persisted inference failure"
            rows.append(row)
            continue
        audio = CanonicalAudio(
            path=oaf_output_dir / "inputs" / str(simfile_id) / "full-mix.wav",
            source_audio_id=f"{simfile_id}/bgm.wav",
            source_audio_sha256=str(loaded.source_row["source_audio_content_hash"]),
            input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
            input_audio_sha256=input_audio_sha256,
            byte_length=88244,
            sample_rate=44100,
            channel_count=1,
            sample_width_bytes=2,
            audio_frame_count=44100,
        )
        native = NativePrediction(
            audio=audio,
            descriptor=descriptor,
            events=(
                NativeEvent(
                    time_sec=0.5,
                    native_class_id="midi_36",
                    model_output_bin=15,
                    native_midi_note=36,
                    native_metadata={"upstream_8hit_group_id": "kick"},
                    confidence=0.9,
                    velocity_midi=100,
                ),
            ),
        )
        mapped, _ = map_oaf_prediction(native)
        target = prediction_path(
            oaf_output_dir,
            simfile_id=simfile_id,
            source_audio_sha256=audio.source_audio_sha256,
            backend_descriptor_sha256=descriptor.sha256,
            inference_config_sha256=config_sha,
        )
        published = publish_prediction_artifact(target, mapped)
        row["execution_disposition"] = "inferred"
        row["prediction_path"] = target.relative_to(oaf_output_dir).as_posix()
        row["prediction_artifact_sha256"] = published.sha256
        row["source_audio_id"] = audio.source_audio_id
        row["source_audio_sha256"] = audio.source_audio_sha256
        row["input_view_id"] = OAF_FULL_MIX_INPUT_VIEW_ID
        row["input_audio_sha256"] = input_audio_sha256
        rows.append(row)

    counts = {"success_count": 0, "failed_count": 0, "skipped_count": 0, "quarantined_count": 0}
    for row in rows:
        disposition = row["execution_disposition"]
        if disposition in {"inferred", "resumed"}:
            counts["success_count"] += 1
        elif disposition == "failed":
            counts["failed_count"] += 1
        elif disposition == "skipped":
            counts["skipped_count"] += 1
        elif disposition == "quarantined":
            counts["quarantined_count"] += 1
    snapshot: dict[str, object] = {
        "schema": OAF_CORPUS_RUN_SCHEMA,
        "run_id": run_id,
        "reference_manifest_sha256": reference.manifest_sha256,
        "reference_manifest_version": reference.corpus_version,
        "reference_timing_manifest_sha256": timing.manifest_sha256,
        "reference_timing_version": timing.corpus_version,
        "backend_descriptor_sha256": descriptor.sha256,
        "backend_descriptor": dict(descriptor.payload),
        "model_id": config.model_id,
        "model_lock_sha256": model_lock_sha,
        "checkpoint_archive_sha256": config.checkpoint.archive_sha256,
        "adapter_revision": OAF_ADAPTER_REVISION,
        "inference_config": dict(inference_payload),
        "inference_config_sha256": config_sha,
        "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
        "canonicalization_revision": OAF_CANONICALIZATION_REVISION,
        "include_simfile_ids": [],
        "exclude_simfile_ids": [],
        "items": rows,
        "overall_status": "partial" if failed_count else "complete",
    }
    snapshot.update(counts)
    write_oaf_corpus_run(run_path, snapshot)
    return ReviewedSubsetOafFixture(
        reference_manifest_path=reference_fixture.reference_manifest_path,
        timing_manifest_path=reference_fixture.timing_manifest_path,
        timing_output_root=reference_fixture.timing_output_root,
        run_path=run_path,
        oaf_output_dir=oaf_output_dir,
    )
