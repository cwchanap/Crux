from __future__ import annotations

import csv
import shutil
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from experiments.oaf_tf215 import selected_compare
from experiments.oaf_tf215 import selected_inputs as selected
from runtime.oaf_tf1.model import load_model_config
from src.benchmark.backend_identity import (
    MUSCRIPTOR_DESCRIPTOR_IDENTITIES,
    MUSCRIPTOR_DESCRIPTOR_KEYS,
    MUSCRIPTOR_DESCRIPTOR_SCHEMA,
    OAF_DESCRIPTOR_KEYS,
    OAF_DESCRIPTOR_SCHEMA,
    build_descriptor,
    canonical_json_bytes,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backends.base import NativeEvent, NativePrediction
from src.benchmark.corpus_cache import ResolvedSourceAudio
from src.benchmark.input_view import materialize_full_mix_audio
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.oaf_corpus_run import (
    OAF_CORPUS_RUN_SCHEMA,
    OAF_FULL_MIX_INPUT_VIEW_ID,
    OAF_PREDICTION_MAP_ID,
    _expected_oaf_descriptor,
    _model_lock_path,
    build_inference_config,
    build_run_id,
    compute_model_lock_sha256,
    inference_config_sha256,
    render_oaf_corpus_run,
)
from src.benchmark.prediction_artifact import (
    MappedPrediction,
    read_prediction_artifact,
    render_prediction_artifact,
)
from src.benchmark.reference_set_manifest import (
    LoadedReferenceSetManifest,
    LoadedReferenceSetRow,
    ReferenceSetRowView,
)
from src.benchmark.reference_timing import NativeReferenceEvent, render_reference_events
from src.benchmark.reference_timing_manifest import (
    LoadedReferenceTimingManifest,
    LoadedReferenceTimingRow,
    ReferenceTimingRowView,
)
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, TAXONOMY_VERSION

SONGS = (389, 391, 27, 241, 207)
RANKS = (0, 45, 90, 135, 181)
PANEL_IDS = (
    131,
    384,
    329,
    209,
    112,
    68,
    56,
    201,
    298,
    364,
    194,
    216,
    138,
    187,
    225,
    294,
    154,
    262,
    148,
    200,
)
PANEL_PROTOCOL_DIGEST = "9ecdd8e12a0033203ddbdc6b974d201436426a00fac80398c1853cdd47bf622b"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def archived_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> dict[str, Path]:
    """Use real run, prediction and WAV bytes with synthetic manifest/cache access."""
    archive, reports = tmp_path / "archive", tmp_path / "reports"
    archive.mkdir()
    reports.mkdir()
    timing = tmp_path / "timing" / "manifests" / "timing.jsonl"
    timing.parent.mkdir(parents=True)
    reference = tmp_path / "reference.jsonl"
    for path in (timing, reference):
        path.touch()

    config = load_model_config()
    descriptor = _expected_oaf_descriptor(config)
    lock_sha = compute_model_lock_sha256(_model_lock_path())
    inference = build_inference_config(config, descriptor, lock_sha)
    inference_sha = inference_config_sha256(inference)
    reference_sha, timing_sha = "a" * 64, "b" * 64
    reference_version, timing_version = "sha256:" + "c" * 64, "sha256:" + "d" * 64
    run_id = build_run_id(
        reference_sha,
        timing_sha,
        descriptor.sha256,
        lock_sha,
        config.checkpoint.archive_sha256,
        inference_sha,
    )
    monkeypatch.setattr(selected, "ARCHIVED_RUN_ID", run_id)
    monkeypatch.setattr(selected, "REFERENCE_MANIFEST_SHA256", reference_sha)
    monkeypatch.setattr(selected, "REFERENCE_TIMING_VERSION", timing_version)
    success_ids = list(range(1000, 1182))
    for rank, song in zip(RANKS, SONGS, strict=True):
        success_ids[rank] = song
    extra_ids = getattr(request, "param", ())
    success_ids[1 : len(extra_ids) + 1] = extra_ids
    bound_ids = (*SONGS, *extra_ids)
    quarantined_ids = list(range(2000, 2138))
    report_identity = {
        "cohort_id": run_id,
        "backend_id": config.backend_id,
        "backend_descriptor_sha256": descriptor.sha256,
        "model_id": config.model_id,
        "model_lock_sha256": lock_sha,
        "reference_manifest_sha256": reference_sha,
        "reference_timing_version": timing_version,
        "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
        "prediction_map_version": OAF_PREDICTION_MAP_ID,
        "lane_map_version": DTX_LANE_MAP_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "scoring_version": "crux.single-cohort-scoring/v1",
    }
    population = {
        "success_count": 182,
        "quarantined_count": 138,
        "failed_count": 0,
        "skipped_count": 0,
    }
    (reports / "summary.json").write_bytes(
        canonical_json_bytes(
            {"identity": report_identity, "population": {"total_count": 320, **population}}
        )
    )
    _write_csv(
        reports / "items.csv",
        [
            {
                "simfile_id": song,
                "cohort_id": run_id,
                "status": "success" if song in success_ids else "quarantined",
                "prediction_native_event_count": 1,
                "prediction_native_class_counts": "midi_38=1",
            }
            for song in (*success_ids, *quarantined_ids)
        ],
    )
    score_fields = {
        key: report_identity[key]
        for key in (
            "cohort_id",
            "model_id",
            "model_lock_sha256",
            "prediction_map_version",
            "input_view_id",
            "scoring_version",
        )
    }
    _write_csv(
        reports / "per_song.csv",
        [
            {
                **score_fields,
                "simfile_id": song,
                "mode": "raw",
                "tolerance_ms": 50,
                "f1": f"0.{rank:03d}",
            }
            for rank, song in enumerate(success_ids)
        ],
    )
    monkeypatch.setattr(
        selected,
        "REPORT_SHA256",
        {path.name: sha256_hex(path.read_bytes()) for path in reports.iterdir()},
    )

    source = Path("tests/fixtures/oaf_tf1_smoke/canonical.wav").resolve()
    source_sha = sha256_hex(source.read_bytes())
    sources = {
        song: ResolvedSourceAudio(source, f"{song}/mix.wav", source_sha, 1.0) for song in bound_ids
    }
    bindings: dict[int, dict[str, object]] = {}
    for song, resolved in sources.items():
        audio = materialize_full_mix_audio(
            resolved,
            tmp_path / "expected" / f"{song}.wav",
            input_root=tmp_path / "expected",
            input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
            max_input_audio_frames=config.max_input_audio_frames,
        )
        events = (
            NativeEvent(0.1, "midi_38", 17, 38, {"upstream_8hit_group_id": "snare"}, 0.9, 100),
        )
        prediction, _ = map_oaf_prediction(NativePrediction(audio, descriptor, events))
        artifact = render_prediction_artifact(prediction)
        path = archive / "predictions" / f"{song}.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(artifact)
        bindings[song] = {
            "source_audio_id": audio.source_audio_id,
            "source_audio_sha256": audio.source_audio_sha256,
            "input_audio_sha256": audio.input_audio_sha256,
            "prediction_path": f"predictions/{song}.jsonl",
            "prediction_artifact_sha256": sha256_hex(artifact),
        }
    baseline = archive / "run.json"
    baseline.write_bytes(
        render_oaf_corpus_run(
            {
                "schema": OAF_CORPUS_RUN_SCHEMA,
                "run_id": run_id,
                "overall_status": "complete",
                "backend_descriptor": dict(descriptor.payload),
                "backend_descriptor_sha256": descriptor.sha256,
                "model_id": config.model_id,
                "model_lock_sha256": lock_sha,
                "checkpoint_archive_sha256": config.checkpoint.archive_sha256,
                "reference_manifest_sha256": reference_sha,
                "reference_manifest_version": reference_version,
                "reference_timing_manifest_sha256": timing_sha,
                "reference_timing_version": timing_version,
                "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
                "inference_config": inference,
                "inference_config_sha256": inference_sha,
                "adapter_revision": inference["adapter_revision"],
                "canonicalization_revision": inference["canonicalization_revision"],
                **population,
                "items": [
                    {
                        "simfile_id": song,
                        "execution_disposition": "inferred",
                        **bindings.get(song, {}),
                    }
                    for song in success_ids
                ]
                + [
                    {"simfile_id": song, "execution_disposition": "quarantined"}
                    for song in quarantined_ids
                ],
            }
        )
    )
    manifest_rows, timing_rows = [], []
    for song in bound_ids:
        event = NativeReferenceEvent(
            simfile_id=song,
            selected_chart_key=f"{song}/chart.dtx",
            selected_chart_content_hash="e" * 64,
            source_audio_key=f"{song}/mix.wav",
            source_audio_content_hash=source_sha,
            source_order=0,
            measure=1,
            position=0.0,
            lane_id="11",
            note_id="01",
            chart_time_sec=0.0,
            audio_time_sec=0.0,
        )
        event_content = render_reference_events((event,))
        event_relative_path = f"events/{sha256_hex(event_content)}.jsonl"
        event_path = timing.parent.parent / event_relative_path
        event_path.parent.mkdir(exist_ok=True)
        event_path.write_bytes(event_content)
        source_row = {
            "selected_chart_key": event.selected_chart_key,
            "selected_chart_content_hash": event.selected_chart_content_hash,
            "source_audio_key": event.source_audio_key,
            "source_audio_content_hash": source_sha,
            "reference_events_cache_path": event_relative_path,
        }
        manifest_rows.append(
            LoadedReferenceSetRow(
                source_row, ReferenceSetRowView(song, "eligible", (), (), 1, 1, 0, 0, 0)
            )
        )
        timing_rows.append(
            LoadedReferenceTimingRow(
                source_row,
                ReferenceTimingRowView(
                    song,
                    timing_version,
                    "ready",
                    (),
                    (),
                    event_relative_path,
                    event.source_audio_key,
                    source_sha,
                ),
            )
        )
    monkeypatch.setattr(
        selected,
        "load_reference_set_manifest",
        lambda _: LoadedReferenceSetManifest(
            reference_sha, reference_version, timing_sha, timing_version, tuple(manifest_rows)
        ),
    )
    monkeypatch.setattr(
        selected,
        "load_reference_timing_manifest",
        lambda _: LoadedReferenceTimingManifest(timing_sha, timing_version, tuple(timing_rows)),
    )
    monkeypatch.setattr(
        selected,
        "resolve_source_audio",
        lambda row, *_args, **_kwargs: sources[int(row["source_audio_key"].split("/")[0])],
    )
    return {
        "baseline_evidence_dir": reports,
        "baseline_run": baseline,
        "reference_manifest": reference,
        "timing_manifest": timing,
        "cache_dir": tmp_path / "cache",
        "input_root": tmp_path / "inputs",
    }


def test_rank_scores_breaks_decimal_ties_by_integer_id() -> None:
    rows = [
        (10, Decimal("0.10")),
        (5, Decimal("0.001")),
        (2, Decimal("0.100")),
        (8, Decimal("0.01")),
        (4, Decimal("1")),
    ]
    assert selected.rank_scores(rows) == [
        (5, rows[1][1]),
        (8, rows[3][1]),
        (2, rows[2][1]),
        (10, rows[0][1]),
        (4, rows[4][1]),
    ]
    with pytest.raises(selected.SelectionError, match="duplicate"):
        selected.rank_scores(rows + [rows[0]])


def test_prepares_selection_and_verifies_it_after_root_relocation(
    archived_bundle, tmp_path
) -> None:
    selection = selected.prepare_selection(**archived_bundle)
    selection_path = tmp_path / "selection.json"
    selection_path.write_bytes(canonical_json_bytes(selection))
    digest = selected.selection_sha256(selection)
    baseline = archived_bundle["baseline_run"]
    archive = shutil.move(baseline.parent, tmp_path / "moved-archive")
    audio_root = shutil.move(archived_bundle["input_root"], tmp_path / "moved-inputs")

    restored, songs = selected.verify_selection(
        selection_path=selection_path,
        expected_selection_sha256=digest,
        baseline_run=archive / baseline.name,
        input_root=audio_root,
    )
    assert restored == selection
    assert [row["simfile_id"] for row in selection["items"]] == list(SONGS)
    assert [audio.path.name for audio, _ in songs] == [f"{song}.wav" for song in SONGS]
    assert [artifact.event_count for _, artifact in songs] == [1] * 5


def test_prepare_blocks_missing_body_and_rejects_report_changes(archived_bundle) -> None:
    body = archived_bundle["baseline_run"].parent / "predictions" / "389.jsonl"
    body.unlink()
    with pytest.raises(selected.SelectionBlocked, match="389.jsonl"):
        selected.prepare_selection(**archived_bundle)
    report = archived_bundle["baseline_evidence_dir"] / "items.csv"
    report.write_bytes(report.read_bytes() + b"\n")
    with pytest.raises(selected.SelectionError, match="digest") as error:
        selected.prepare_selection(**archived_bundle)
    assert not isinstance(error.value, selected.SelectionBlocked)


def test_preparation_distinguishes_corrupt_native_reference_from_missing_file(
    archived_bundle,
) -> None:
    event_root = archived_bundle["timing_manifest"].parent.parent / "events"
    path = next(event_root.glob("*.jsonl"))
    path.write_bytes(b"corrupt native reference")
    with pytest.raises(selected.SelectionError, match="reference preflight") as error:
        selected.prepare_selection(**archived_bundle)
    assert not isinstance(error.value, selected.SelectionBlocked)
    path.unlink()
    with pytest.raises(selected.SelectionBlocked, match="required input unavailable"):
        selected.prepare_selection(**archived_bundle)


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": "wrong-cohort"},
        {"model_lock_sha256": "d" * 64},
        {"backend_descriptor_sha256": "d" * 64},
        {"inference_config_sha256": "d" * 64},
        {"reference_timing_manifest_sha256": "d" * 64},
    ],
)
def test_prepare_checks_archived_run_identity(archived_bundle, change) -> None:
    baseline = archived_bundle["baseline_run"]
    run = strict_json_loads(baseline.read_bytes())
    run.update(change)
    baseline.write_bytes(render_oaf_corpus_run(run))
    with pytest.raises(selected.SelectionError) as error:
        selected.prepare_selection(**archived_bundle)
    assert not isinstance(error.value, selected.SelectionBlocked)


@pytest.mark.parametrize(
    ("change", "diagnostic"),
    [("other-backend", "descriptor"), ("other-upstream", "descriptor"), ("event-count", "counts")],
)
def test_preparation_rejects_a_valid_prediction_with_wrong_lineage(
    archived_bundle, change, diagnostic
) -> None:
    baseline = archived_bundle["baseline_run"]
    run = strict_json_loads(baseline.read_bytes())
    row = next(row for row in run["items"] if row["simfile_id"] == 389)
    path = baseline.parent / row["prediction_path"]
    prediction = read_prediction_artifact(path.read_bytes()).prediction
    if change == "other-backend":
        payload = MUSCRIPTOR_DESCRIPTOR_IDENTITIES | {
            "model_id": "muscriptor-small-aaaaaaaaaaaa-bbbbbbbbbbbb",
            "upstream_source_commit": "c" * 40,
        }
        descriptor = build_descriptor(
            payload, MUSCRIPTOR_DESCRIPTOR_KEYS, MUSCRIPTOR_DESCRIPTOR_SCHEMA
        )
        prediction = MappedPrediction(prediction.audio, descriptor, ())
    if change == "other-upstream":
        payload = dict(prediction.descriptor.payload) | {"upstream_source_commit": "c" * 40}
        descriptor = build_descriptor(payload, OAF_DESCRIPTOR_KEYS, OAF_DESCRIPTOR_SCHEMA)
        prediction = replace(prediction, descriptor=descriptor)
    if change == "event-count":
        (event,) = prediction.events
        extra = replace(event, native=replace(event.native, time_sec=0.2))
        prediction = replace(prediction, events=(event, extra))
    content = render_prediction_artifact(prediction)
    path.write_bytes(content)
    row["prediction_artifact_sha256"] = sha256_hex(content)
    baseline.write_bytes(render_oaf_corpus_run(run))

    with pytest.raises(selected.SelectionError, match=diagnostic) as error:
        selected.prepare_selection(**archived_bundle)
    assert not isinstance(error.value, selected.SelectionBlocked)
    assert not archived_bundle["input_root"].exists()


def test_preparation_does_not_publish_a_partial_selection_after_source_blocker(
    archived_bundle, monkeypatch
) -> None:
    original = selected.resolve_source_audio

    def missing_final_source(row, *args, **kwargs):
        if row["source_audio_key"] == "207/mix.wav":
            raise ValueError("verified source audio unavailable")
        return original(row, *args, **kwargs)

    monkeypatch.setattr(selected, "resolve_source_audio", missing_final_source)
    with pytest.raises(selected.SelectionBlocked, match="required input unavailable"):
        selected.prepare_selection(**archived_bundle)
    assert not archived_bundle["input_root"].exists()


def test_verify_rejects_changed_seal_wav_and_symlink(archived_bundle, tmp_path) -> None:
    selection = selected.prepare_selection(**archived_bundle)
    path = tmp_path / "selection.json"
    path.write_bytes(canonical_json_bytes(selection))
    digest = selected.selection_sha256(selection)
    arguments = {
        "selection_path": path,
        "expected_selection_sha256": digest,
        "baseline_run": archived_bundle["baseline_run"],
        "input_root": archived_bundle["input_root"],
    }
    with pytest.raises(selected.SelectionError, match="selection digest"):
        selected.verify_selection(**(arguments | {"expected_selection_sha256": "d" * 64}))
    wav = arguments["input_root"] / "389.wav"
    good = wav.read_bytes()
    wav.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
    with pytest.raises(selected.SelectionError, match="canonical WAV"):
        selected.verify_selection(**arguments)
    wav.write_bytes(good)
    predictions = arguments["baseline_run"].parent / "predictions"
    target = predictions.with_name("outside-predictions")
    predictions.rename(target)
    predictions.symlink_to(target, target_is_directory=True)
    with pytest.raises(selected.SelectionError, match="symlink"):
        selected.verify_selection(**arguments)


def test_prepare_rejects_paths_that_escape_the_baseline_root(archived_bundle, tmp_path) -> None:
    baseline = archived_bundle["baseline_run"]
    run = strict_json_loads(baseline.read_bytes())
    row = next(row for row in run["items"] if row["simfile_id"] == 389)
    body = baseline.parent / row["prediction_path"]
    outside = tmp_path / "outside.jsonl"
    shutil.copyfile(body, outside)
    row["prediction_path"] = f"../{outside.name}"
    baseline.write_bytes(render_oaf_corpus_run(run))
    with pytest.raises(selected.SelectionError, match="unsafe relative"):
        selected.prepare_selection(**archived_bundle)


@pytest.fixture
def fresh_bundle(archived_bundle, monkeypatch) -> dict[str, Path]:
    """Use the production corpus output layout with a synthetic fresh scoped run."""
    arguments = dict(archived_bundle)
    del arguments["baseline_evidence_dir"]
    archive_snapshot = arguments["baseline_run"]
    run = strict_json_loads(archive_snapshot.read_bytes())
    run_id = build_run_id(
        run["reference_manifest_sha256"],
        run["reference_timing_manifest_sha256"],
        run["backend_descriptor_sha256"],
        run["model_lock_sha256"],
        run["checkpoint_archive_sha256"],
        run["inference_config_sha256"],
        SONGS,
    )
    for name, value in {
        "FRESH_RUN_ID": run_id,
        "FRESH_REFERENCE_MANIFEST_SHA256": run["reference_manifest_sha256"],
        "FRESH_REFERENCE_MANIFEST_VERSION": run["reference_manifest_version"],
        "FRESH_TIMING_MANIFEST_SHA256": run["reference_timing_manifest_sha256"],
        "FRESH_TIMING_MANIFEST_VERSION": run["reference_timing_version"],
    }.items():
        monkeypatch.setattr(selected, name, value, raising=False)
    run.update(
        run_id=run_id,
        include_simfile_ids=list(SONGS),
        success_count=5,
        quarantined_count=0,
        items=[row for row in run["items"] if row["simfile_id"] in SONGS],
    )
    row = next(row for row in run["items"] if row["simfile_id"] == 389)
    path = archive_snapshot.parent / row["prediction_path"]
    prediction = read_prediction_artifact(path.read_bytes()).prediction
    (event,) = prediction.events
    second = replace(event, native=replace(event.native, time_sec=0.2))
    artifact = render_prediction_artifact(replace(prediction, events=(event, second)))
    path.write_bytes(artifact)
    row["prediction_artifact_sha256"] = sha256_hex(artifact)

    baseline = archive_snapshot.parent / "runs" / run_id / "run.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(render_oaf_corpus_run(run))
    arguments["baseline_run"] = baseline
    return arguments


def test_prepare_fresh_cli_seals_production_layout_with_measured_counts(
    fresh_bundle, tmp_path, capsys
) -> None:
    output = tmp_path / "fresh-selection.json"
    arguments = [
        argument
        for key, value in (fresh_bundle | {"output": output}).items()
        for argument in (f"--{key.replace('_', '-')}", str(value))
    ]
    assert selected_compare.main(["prepare-fresh", *arguments]) == 0
    digest = capsys.readouterr().out.strip()
    selection, bindings = selected.verify_selection(
        selection_path=output,
        expected_selection_sha256=digest,
        baseline_run=fresh_bundle["baseline_run"],
        input_root=fresh_bundle["input_root"],
    )
    assert selection["schema"] == "crux.oaf-tf215-fresh-selection/v1"
    assert selection["selection_provenance"]["run_id"] == selected.ARCHIVED_RUN_ID
    assert selection["baseline_identity"]["run_id"] == fresh_bundle["baseline_run"].parent.name
    assert [row["simfile_id"] for row in selection["items"]] == list(SONGS)
    assert [row["rank"] for row in selection["items"]] == list(RANKS)
    assert "archived_f1" not in selection["items"][0]
    assert selection["items"][0]["prediction_native_class_counts"] == {"midi_38": 2}
    assert [artifact.event_count for _, artifact in bindings] == [2, 1, 1, 1, 1]


@pytest.mark.parametrize(
    ("change", "diagnostic"),
    [
        ({"include_simfile_ids": [27, 207, 241, 389]}, "include_simfile_ids"),
        ({"reference_timing_manifest_sha256": "d" * 64}, "reference_timing_manifest_sha256"),
        ({"overall_status": "partial"}, "overall_status"),
    ],
)
def test_prepare_fresh_rejects_wrong_scope_or_lineage(fresh_bundle, change, diagnostic) -> None:
    baseline = fresh_bundle["baseline_run"]
    run = strict_json_loads(baseline.read_bytes())
    baseline.write_bytes(render_oaf_corpus_run(run | change))
    with pytest.raises(selected.SelectionError, match=diagnostic):
        selected.prepare_fresh_selection(**fresh_bundle)
    assert not fresh_bundle["input_root"].exists()


def test_prepare_fresh_rejects_a_different_successful_song_in_same_scope(fresh_bundle) -> None:
    baseline = fresh_bundle["baseline_run"]
    run = strict_json_loads(baseline.read_bytes())
    row = next(row for row in run["items"] if row["simfile_id"] == 389)
    row["simfile_id"] = 999
    baseline.write_bytes(render_oaf_corpus_run(run))
    with pytest.raises(selected.SelectionError, match="selected membership"):
        selected.prepare_fresh_selection(**fresh_bundle)
    assert not fresh_bundle["input_root"].exists()


def test_verify_fresh_rebinds_prediction_bytes(fresh_bundle, tmp_path) -> None:
    selection = selected.prepare_fresh_selection(**fresh_bundle)
    output = tmp_path / "fresh-selection.json"
    output.write_bytes(canonical_json_bytes(selection))
    baseline = fresh_bundle["baseline_run"]
    prediction = baseline.parents[2] / selection["items"][0]["prediction_path"]
    prediction.write_bytes(prediction.read_bytes() + b"\n")
    with pytest.raises(selected.SelectionError, match="prediction"):
        selected.verify_selection(
            selection_path=output,
            expected_selection_sha256=selected.selection_sha256(selection),
            baseline_run=baseline,
            input_root=fresh_bundle["input_root"],
        )


@pytest.fixture
def panel_bundle(archived_bundle, monkeypatch):
    arguments = dict(archived_bundle)
    del arguments["baseline_evidence_dir"]
    snapshot = arguments["baseline_run"]
    run = strict_json_loads(snapshot.read_bytes())
    run_id = build_run_id(
        run["reference_manifest_sha256"],
        run["reference_timing_manifest_sha256"],
        run["backend_descriptor_sha256"],
        run["model_lock_sha256"],
        run["checkpoint_archive_sha256"],
        run["inference_config_sha256"],
        PANEL_IDS,
    )
    for name, value in {
        "PANEL_RUN_ID": run_id,
        "FRESH_REFERENCE_MANIFEST_SHA256": run["reference_manifest_sha256"],
        "FRESH_REFERENCE_MANIFEST_VERSION": run["reference_manifest_version"],
        "FRESH_TIMING_MANIFEST_SHA256": run["reference_timing_manifest_sha256"],
        "FRESH_TIMING_MANIFEST_VERSION": run["reference_timing_version"],
    }.items():
        monkeypatch.setattr(selected, name, value, raising=False)
    run.update(
        run_id=run_id,
        include_simfile_ids=sorted(PANEL_IDS),
        success_count=len(PANEL_IDS),
        quarantined_count=0,
        items=[row for row in run["items"] if row["simfile_id"] in PANEL_IDS],
    )
    baseline = snapshot.parent / "runs" / run_id / "run.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(render_oaf_corpus_run(run))
    arguments["baseline_run"] = baseline
    return arguments


@pytest.mark.parametrize("archived_bundle", [PANEL_IDS], indirect=True)
def test_panel_seal_binds_all_members_and_requires_opt_in(panel_bundle, tmp_path) -> None:
    selection = selected.prepare_panel_selection(**panel_bundle)
    assert selection["schema"] == "crux.oaf-tf215-panel-selection/v1"
    assert selection["panel_protocol_sha256"] == PANEL_PROTOCOL_DIGEST
    assert selection["baseline_identity"]["include_simfile_ids"] == sorted(PANEL_IDS)
    assert [row["simfile_id"] for row in selection["items"]] == list(PANEL_IDS)
    assert [row["panel_ordinal"] for row in selection["items"]] == list(range(1, 21))
    assert all("rank" not in row for row in selection["items"])
    path = tmp_path / "panel-selection.json"
    path.write_bytes(canonical_json_bytes(selection))
    arguments = {
        "selection_path": path,
        "expected_selection_sha256": selected.selection_sha256(selection),
        "baseline_run": panel_bundle["baseline_run"],
        "input_root": panel_bundle["input_root"],
    }
    verified, bindings = selected.verify_selection(**arguments, panel=True)
    assert verified == selection
    assert [audio.path.name for audio, _ in bindings] == [f"{song}.wav" for song in PANEL_IDS]
    with pytest.raises(selected.SelectionError):
        selected.verify_selection(**arguments)


@pytest.mark.parametrize("archived_bundle", [PANEL_IDS], indirect=True)
def test_panel_seal_rejects_changed_protocol(panel_bundle, tmp_path, monkeypatch) -> None:
    protocol = tmp_path / "PANEL.md"
    protocol.write_text("Changed panel protocol\n", encoding="utf-8")
    monkeypatch.setattr(selected, "PANEL_PROTOCOL_PATH", protocol, raising=False)
    with pytest.raises(selected.SelectionError, match="panel protocol"):
        selected.prepare_panel_selection(**panel_bundle)
    assert not panel_bundle["input_root"].exists()
