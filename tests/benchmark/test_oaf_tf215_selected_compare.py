"""Exercise the selected driver through its real smoke and song control flow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.oaf_tf215 import compare
from experiments.oaf_tf215 import selected_compare as selected
from experiments.oaf_tf215.runtime_identity import PROBE_SCHEMA
from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex


def _inputs(tmp_path):
    return {
        "selection_path": tmp_path / "selection.json",
        "expected_selection_sha256": "1" * 64,
        "baseline_run": tmp_path / "run.json",
        "python311": tmp_path / "python3.11",
        "runtime_root": tmp_path / "runtime",
        "checkpoint_cache": tmp_path / "checkpoint",
        "input_root": tmp_path,
        "output_root": tmp_path / "experiment",
    }


@pytest.fixture
def fake_experiment(monkeypatch, tmp_path):
    inputs = _inputs(tmp_path)
    scratch = inputs["runtime_root"]
    scratch.mkdir()
    (scratch / "worker.py").write_text("")
    selection = {
        "schema": "crux.oaf-tf215-selection/v1",
        "items": [{"simfile_id": value} for value in (389, 391, 27, 241, 207)],
    }
    inputs["expected_selection_sha256"] = sha256_hex(canonical_json_bytes(selection))
    bindings = tuple(
        (
            SimpleNamespace(path=tmp_path / f"{item['simfile_id']}.wav"),
            SimpleNamespace(prediction=SimpleNamespace(events=())),
        )
        for item in selection["items"]
    )
    monkeypatch.setattr(selected, "verify_selection", lambda **_: (selection, bindings))
    monkeypatch.setattr(compare, "_prepare_checkpoint", lambda *args: {"checkpoint": str(tmp_path)})
    monkeypatch.setattr(
        selected, "collect_probe", lambda *args, **kwargs: {"schema": PROBE_SCHEMA, "status": "ok"}
    )
    monkeypatch.setattr(
        selected, "build_runtime_identity", lambda *args, **kwargs: {"schema": "test-runtime/v1"}
    )
    return inputs


def _worker_factory(inputs, *, fail_second=False, vary_fresh=False):
    workers = []

    class Worker:
        ready = {
            "type": "ready",
            "backend_id": "magenta-egmd-tf1-94529798-8hit-v1",
            "restored_tensor_count": 78,
        }

        def __init__(self, cwd, timeout, environment):
            self.cwd = cwd
            self.timeout = timeout
            self.environment = environment
            self.closed = False
            self.requests = []

        def request(self, filename):
            assert not self.closed
            self.requests.append(filename)
            if filename == "389.wav":
                if fail_second and len(self.requests) == 2:
                    return {"error": {"message": "synthetic request failure"}}
                if vary_fresh and len(workers) == 4:
                    return {"events": []}
                return {
                    "events": [
                        {
                            "time_sec": 0.18,
                            "confidence": 0.5,
                            "native_class_id": "midi_38",
                            "native_midi_note": 38,
                            "model_output_bin": 17,
                            "upstream_8hit_group_id": "snare",
                            "velocity_midi": 64,
                        }
                    ]
                }
            return {"events": []}

        def close(self):
            assert not self.closed
            self.closed = True

    def start(command, *, cwd, env, timeout_seconds, close_timeout_seconds=30.0):
        assert command[:2] == [str(inputs["python311"]), "-s"]
        assert close_timeout_seconds == 30.0
        assert env["TF_ENABLE_ONEDNN_OPTS"] == "0"
        if workers:
            assert workers[-1].closed
        worker = Worker(cwd, timeout_seconds, env)
        workers.append(worker)
        return worker

    return workers, start


def _execution_dir(root: Path) -> Path:
    (path,) = root.rglob("comparison.json")
    return path.parent


def test_run_retains_binary64_attempts_and_accepts_a_divergent_smoke(fake_experiment):
    workers, start = _worker_factory(fake_experiment)
    result = selected.run_selected_comparison(**fake_experiment, process_factory=start)
    assert result["status"] == "complete"
    assert [worker.requests for worker in workers[:2]] == [
        ["canonical.wav", "canonical.wav"],
        ["canonical.wav"],
    ]
    assert [worker.requests for worker in workers[2:]] == [
        requests
        for identifier in (389, 391, 27, 241, 207)
        for requests in ([f"{identifier}.wav"] * 2, [f"{identifier}.wav"])
    ]
    assert workers[0].timeout == 300.0
    assert all(worker.timeout == 3600.0 and worker.closed for worker in workers[2:])
    assert all(worker.environment == workers[0].environment for worker in workers)

    directory = _execution_dir(fake_experiment["output_root"])
    assert directory.name == result["execution_id"]
    assert directory.parent.name == result["runtime_sha256"]
    assert directory.parent.parent.name == result["descriptor_sha256"]
    assert json.loads((directory / "smoke.json").read_text())["status"] == "oracle_divergent"
    attempts = [
        json.loads(line) for line in (directory / "native-events.jsonl").read_text().splitlines()
    ]
    assert len(attempts) == 15
    assert attempts[0]["events"][0]["time_sec_binary64"] == "3fc70a3d70a3d70a"
    assert attempts[0]["worker_ready"]["restored_tensor_count"] == 78
    assert [row["simfile_id"] for row in result["items"]] == [389, 391, 27, 241, 207]
    assert all(row["repeatable"] for row in result["items"])

    _, fresh_start = _worker_factory(fake_experiment)
    again = selected.run_selected_comparison(**fake_experiment, process_factory=fresh_start)
    assert again["execution_id"] != result["execution_id"]
    assert len(list(fake_experiment["output_root"].rglob("comparison.json"))) == 2


@pytest.mark.parametrize("nonregression", [True, False])
def test_panel_uses_persistent_worker_and_replays_only_anchors(
    fake_experiment, monkeypatch, nonregression
):
    inputs = fake_experiment
    root = inputs["input_root"]
    identifiers = selected.PANEL_SIMFILE_IDS
    baseline = b"sealed test run"
    inputs["baseline_run"].write_bytes(baseline)
    selection = {
        "baseline_identity": {"run_id": "panel-test"},
        "baseline_run_sha256": sha256_hex(baseline),
        "items": [{"simfile_id": song} for song in identifiers],
    }
    bindings = tuple(
        (
            SimpleNamespace(path=root / f"{song}.wav", audio_frame_count=44100, sample_rate=44100),
            SimpleNamespace(prediction=SimpleNamespace(events=())),
        )
        for song in identifiers
    )
    monkeypatch.setattr(
        selected, "verify_selection", lambda *, panel, **_: (selection, bindings) if panel else None
    )
    monkeypatch.setattr(selected, "_panel_references", lambda *_: dict.fromkeys(identifiers))
    monkeypatch.setattr(
        selected,
        "parse_oaf_corpus_run",
        lambda *_, **__: {
            "items": [
                {"simfile_id": song, "wall_time_sec": 2, "execution_disposition": "inferred"}
                for song in identifiers
            ]
        },
    )
    monkeypatch.setattr(selected, "score_panel_song", lambda *_: {"scored": True})

    def aggregate(scores):
        assert list(scores) == [{"scored": True}] * len(identifiers)
        return {"micro_f1_nonregression": nonregression}

    monkeypatch.setattr(selected, "aggregate_panel_scores", aggregate)
    workers, start = _worker_factory(inputs)
    result = selected.run_selected_comparison(
        **inputs, process_factory=start, panel_manifests=(root / "reference", root / "timing")
    )
    assert result["status"] == "complete"
    assert result["panel_gate_passed"] is nonregression
    anchors = [f"{song}.wav" for song in selected.PANEL_ANCHORS]
    assert workers[2].requests == [f"{song}.wav" for song in sorted(identifiers)] + anchors
    assert workers[3].requests == anchors
    assert all(worker.closed for worker in workers)
    assert result["panel_runtime"]["all_matched"]["baseline_rtf"] == 2
    assert result["panel_runtime"]["after_first_scheduled_song"]["songs"] == len(identifiers) - 1
    for row in result["items"]:
        anchor = row["simfile_id"] in selected.PANEL_ANCHORS
        assert row["completed_attempts"] == (3 if anchor else 1)
        assert row["repeatable"] is (True if anchor else None)


@pytest.mark.parametrize(
    ("status", "options", "saved"),
    [
        ("failed", {"fail_second": True}, 13),
        ("non_repeatable", {"vary_fresh": True}, 15),
    ],
)
def test_run_keeps_failed_rows_and_partial_attempts(fake_experiment, status, options, saved):
    workers, start = _worker_factory(fake_experiment, **options)
    result = selected.run_selected_comparison(**fake_experiment, process_factory=start)
    assert result["status"] == status
    assert result["items"][0]["status"] == status
    assert all(row["status"] == "complete" for row in result["items"][1:])
    assert all(worker.closed for worker in workers)
    directory = _execution_dir(fake_experiment["output_root"])
    assert len((directory / "native-events.jsonl").read_text().splitlines()) == saved


def test_a_blocked_probe_writes_diagnostics_without_starting_any_worker(
    fake_experiment, monkeypatch
):
    blocked = {"schema": PROBE_SCHEMA, "status": "blocked", "reasons": ["unsupported_platform"]}
    monkeypatch.setattr(selected, "collect_probe", lambda *args, **kwargs: blocked)
    workers, start = _worker_factory(fake_experiment)
    result = selected.run_selected_comparison(**fake_experiment, process_factory=start)
    assert result["status"] == "blocked"
    assert result["probe"] == blocked
    assert result["runtime_sha256"] is None
    assert all(row["status"] == "unattempted" for row in result["items"])
    assert not workers
    assert _execution_dir(fake_experiment["output_root"]).name == result["execution_id"]


def test_runtime_changes_after_measurement_fail_the_record(fake_experiment, monkeypatch):
    hashes = iter(
        ({"schema": "test-runtime/v1", "cpu": 1}, {"schema": "test-runtime/v1", "cpu": 2})
    )
    monkeypatch.setattr(selected, "build_runtime_identity", lambda *args, **kwargs: next(hashes))
    _, start = _worker_factory(fake_experiment)
    result = selected.run_selected_comparison(**fake_experiment, process_factory=start)
    assert result["status"] == "failed"
    assert result["failed_stage"] == "runtime_recheck"
    assert all(row["status"] == "complete" for row in result["items"])


def test_cli_exit_follows_measurement_completion(fake_experiment, monkeypatch, capsys):
    _, start = _worker_factory(fake_experiment)
    monkeypatch.setattr(selected.WorkerProcess, "start", start)
    argv = ["run"]
    for key, value in fake_experiment.items():
        flag = key.removeprefix("expected_").replace("_", "-")
        argv.extend((f"--{flag}", str(value)))
    assert selected.main(argv) == 0
    message = json.loads(capsys.readouterr().out)
    assert message["status"] == "complete"
    assert (Path(message["output_dir"]) / "comparison.json").is_file()

    monkeypatch.setattr(
        selected,
        "collect_probe",
        lambda *args, **kwargs: {"status": "blocked", "reasons": ["unsupported_platform"]},
    )
    assert selected.main(argv) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
