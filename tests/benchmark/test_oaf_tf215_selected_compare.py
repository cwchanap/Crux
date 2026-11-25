"""Exercise the selected driver through its real smoke and song control flow."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.oaf_tf215 import compare, panel_tf1
from experiments.oaf_tf215 import selected_compare as selected
from runtime.oaf_tf1.model import load_model_config
from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex
from src.benchmark.backends.base import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.input_view import load_direct_audio
from src.benchmark.oaf_corpus_run import _expected_oaf_descriptor
from src.benchmark.oaf_smoke_oracle import read_smoke_oracle
from src.benchmark.oaf_tf215_runtime import PROBE_SCHEMA
from src.benchmark.worker_process import WorkerProcessError


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


def _test_audio(root, song):
    path = root / f"{song}.wav"
    content = path.name.encode()
    path.write_bytes(content)
    digest = sha256_hex(content)
    return CanonicalAudio(path, str(song), digest, "test", digest, len(content), 44100, 1, 2, 44100)


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
            _test_audio(tmp_path, item["simfile_id"]),
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


def _worker_factory(inputs, *, fail_second=False, vary_fresh=False, check_audio=None):
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
            if filename != "canonical.wav" and check_audio is not None:
                check_audio(self.cwd / filename)
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


def test_selected_replays_read_the_snapshot_after_source_replacement(fake_experiment):
    root = fake_experiment["input_root"]

    def check_audio(path):
        (root / path.name).write_bytes(b"x" * len(path.name))
        assert path.read_bytes() == path.name.encode()

    _, start = _worker_factory(fake_experiment, check_audio=check_audio)
    result = selected.run_selected_comparison(**fake_experiment, process_factory=start)
    assert result["status"] == "complete"
    output = Path(result["output_dir"])
    assert not list(output.glob("sealed-wavs-*"))


def test_selected_rejects_modified_wav_before_song_requests(fake_experiment):
    (fake_experiment["input_root"] / "389.wav").write_bytes(b"xxxxxxx")
    workers, start = _worker_factory(fake_experiment)
    result = selected.run_selected_comparison(**fake_experiment, process_factory=start)
    assert result["status"] == "failed"
    assert result["failed_stage"] == "sealed_wavs"
    assert len(workers) == 2


def test_run_panel_rejects_an_arbitrary_smoke_command(fake_experiment, monkeypatch):
    monkeypatch.setattr(
        selected,
        "run_selected_comparison",
        lambda **_: {"status": "complete", "output_dir": "unused"},
    )
    argv = ["run-panel"]
    for key, value in fake_experiment.items():
        flag = key.removeprefix("expected_").replace("_", "-")
        argv.extend((f"--{flag}", str(value)))
    argv.extend(("--reference-manifest", "reference.jsonl", "--timing-manifest", "timing.jsonl"))
    argv.extend(
        ("--tf1-smoke-prediction", "copied.jsonl", "--tf1-smoke-command", "cp", "saved.jsonl")
    )
    with pytest.raises(SystemExit) as error:
        selected.main(argv)
    assert error.value.code == 2


@pytest.mark.parametrize("outcome", ["exact", "divergent", "changed_boot"])
def test_paired_tf1_uses_the_reviewed_launcher_for_smoke_and_baseline(
    tmp_path, monkeypatch, outcome
):
    oracle_content = (selected.SMOKE_ROOT / "smoke-oracle.json").read_bytes()
    config = load_model_config(compare.MODEL_CONFIG_PATH)
    audio = load_direct_audio(
        selected.SMOKE_ROOT / "canonical.wav",
        source_audio_id=compare.SOURCE_AUDIO_ID,
        input_view_id=compare.SOURCE_AUDIO_ID,
        max_input_audio_frames=config.max_input_audio_frames,
    )

    def value(hex_value):
        return struct.unpack(">d", bytes.fromhex(hex_value))[0]

    events = tuple(
        NativeEvent(
            value(row.time_sec_binary64),
            row.native_class_id,
            row.model_output_bin,
            row.native_midi_note,
            {"upstream_8hit_group_id": row.upstream_8hit_group_id},
            None if row.confidence_binary64 is None else value(row.confidence_binary64),
            row.velocity_midi,
        )
        for row in read_smoke_oracle(oracle_content).native_events
    )
    if outcome == "divergent":
        events = events[:-1]
    prediction = NativePrediction(audio, _expected_oaf_descriptor(config), events)

    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "from pathlib import Path\n"
        "def install(root, repo):\n"
        "    assert root.parent == Path(__file__).parent\n"
        "    root.mkdir()\n"
        "def chroot_process_factory(root):\n"
        "    assert root.is_dir()\n"
        "    return lambda _: root\n"
    )
    launcher_sha = sha256_hex(launcher.read_bytes())
    monkeypatch.setattr(panel_tf1, "TF1_LAUNCHER_SHA256", launcher_sha)
    reference = tmp_path / "reference.jsonl"
    timing = tmp_path / "timing.jsonl"
    for path, digest_attr in (
        (reference, "FRESH_REFERENCE_MANIFEST_SHA256"),
        (timing, "FRESH_TIMING_MANIFEST_SHA256"),
    ):
        path.write_bytes(path.name.encode())
        monkeypatch.setattr(panel_tf1, digest_attr, sha256_hex(path.read_bytes()))
    boot_path = tmp_path / "boot_id"
    boot_id = "dfcab66e-1856-4b3e-b361-450357d3e936"
    boot_path.write_text(boot_id + "\n")
    monkeypatch.setattr(panel_tf1, "BOOT_ID_PATH", boot_path)
    monkeypatch.setattr(compare, "_prepare_checkpoint", lambda *_: {"checkpoint": tmp_path})
    factories = []

    def create_backend(*, process_factory, checkpoint_dir, **options):
        assert process_factory(None).is_dir()
        assert checkpoint_dir == tmp_path
        factories.append(process_factory)
        return SimpleNamespace(transcribe=lambda _: prediction, close=lambda: None)

    def run_oaf_corpus(request, *, backend_factory):
        assert request.include_simfile_ids == tuple(sorted(selected.PANEL_SIMFILE_IDS))
        assert request.backend == "oaf"
        assert not request.resume
        backend_factory(input_root=tmp_path).close()
        path = request.output_dir / "run.json"
        path.parent.mkdir()
        path.write_bytes(b"fresh measurements")
        if outcome == "changed_boot":
            boot_path.write_text("00000000-0000-0000-0000-000000000001\n")
        return SimpleNamespace(exit_code=0, run_id=panel_tf1.PANEL_RUN_ID, run_path=path)

    monkeypatch.setattr(panel_tf1, "create_backend", create_backend)
    monkeypatch.setattr(panel_tf1, "run_oaf_corpus", run_oaf_corpus)
    arguments = {
        "launcher_path": launcher,
        "reference_manifest": reference,
        "timing_manifest": timing,
        "cache_dir": tmp_path,
        "checkpoint_cache": tmp_path,
        "directory": tmp_path,
    }
    if outcome == "divergent":
        with pytest.raises(selected.SelectionError, match="oracle native event count mismatch"):
            panel_tf1.measure_tf1_panel(**arguments)
        assert len(factories) == 1
        assert not (tmp_path / panel_tf1.SMOKE_PREDICTION_NAME).exists()
        return
    if outcome == "changed_boot":
        with pytest.raises(selected.SelectionError, match="boot ID changed during the TF1 panel"):
            panel_tf1.measure_tf1_panel(**arguments)
        return
    proof = panel_tf1.measure_tf1_panel(**arguments)
    assert len(factories) == 2 and factories[0] is factories[1]
    assert proof.boot_id == boot_id
    assert proof.launcher_sha256 == launcher_sha
    assert proof.run_sha256 == sha256_hex(b"fresh measurements")
    assert proof.smoke_oracle_sha256 == sha256_hex(oracle_content)
    assert proof.smoke_prediction_sha256 == sha256_hex(
        (tmp_path / panel_tf1.SMOKE_PREDICTION_NAME).read_bytes()
    )


def test_paired_runner_rejects_unreviewed_launcher_before_loading_it(tmp_path):
    launcher = tmp_path / "copy-prediction.py"
    launcher.write_text("raise AssertionError('the unreviewed launcher ran')\n")
    with pytest.raises(selected.SelectionError, match="launcher digest does not match"):
        panel_tf1.measure_tf1_panel(
            launcher_path=launcher,
            reference_manifest=tmp_path / "reference",
            timing_manifest=tmp_path / "timing",
            cache_dir=tmp_path,
            checkpoint_cache=tmp_path,
            directory=tmp_path,
        )


def test_measure_panel_cli_seals_boot_and_run_digest_before_candidate(
    tmp_path, monkeypatch, capsys
):
    def measure_tf1_panel(**options):
        run_path = options["directory"] / "run.json"
        run_path.write_bytes(b"fresh TF1 panel")
        return panel_tf1.Tf1PanelMeasurement(
            run_path, "same-boot", sha256_hex(run_path.read_bytes()), "1" * 64, "2" * 64, "3" * 64
        )

    def prepare_panel_selection(*, baseline_run, **options):
        return {"baseline_run_sha256": sha256_hex(baseline_run.read_bytes()), "items": []}

    def run_selected_comparison(**options):
        seal = json.loads(options["selection_path"].read_bytes())
        assert seal["tf1_measurement"]["boot_id"] == "same-boot"
        assert seal["tf1_measurement"]["baseline_run_sha256"] == seal["baseline_run_sha256"]
        assert seal["baseline_run_sha256"] == sha256_hex(options["baseline_run"].read_bytes())
        assert selected.compute_selection_sha256(seal) == options["expected_selection_sha256"]
        assert options["_tf1_measurement"].run_path == options["baseline_run"]
        return {"status": "complete", "output_dir": "candidate", "panel_gate_passed": True}

    monkeypatch.setattr(panel_tf1, "measure_tf1_panel", measure_tf1_panel)
    monkeypatch.setattr(selected, "prepare_panel_selection", prepare_panel_selection)
    monkeypatch.setattr(selected, "run_selected_comparison", run_selected_comparison)
    argv = ["measure-panel"]
    for flag in (
        "tf1-rootfs-launcher",
        "reference-manifest",
        "timing-manifest",
        "cache-dir",
        "python311",
        "runtime-root",
        "checkpoint-cache",
        "output-root",
    ):
        argv.extend((f"--{flag}", str(tmp_path / flag)))
    assert selected.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["panel_gate_passed"] is True


def _prepare_panel_experiment(inputs, monkeypatch, *, evidence="same_boot", disposition="inferred"):
    root = inputs["input_root"]
    identifiers = selected.PANEL_SIMFILE_IDS
    baseline = b"sealed test run"
    inputs["baseline_run"].write_bytes(baseline)
    measurement = panel_tf1.Tf1PanelMeasurement(
        inputs["baseline_run"],
        "tf1-boot",
        sha256_hex(b"another run" if evidence == "wrong_run" else baseline),
        panel_tf1.TF1_LAUNCHER_SHA256,
        "2" * 64,
        "3" * 64,
    )
    candidate_boot = "different-boot" if evidence == "other_boot" else measurement.boot_id
    monkeypatch.setattr(panel_tf1, "read_boot_id", lambda: candidate_boot)
    selection = {
        "baseline_identity": {"run_id": "panel-test"},
        "baseline_run_sha256": sha256_hex(baseline),
        "tf1_measurement": measurement.binding(),
        "items": [{"simfile_id": song} for song in identifiers],
    }
    bindings = tuple(
        (
            _test_audio(root, song),
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
                {"simfile_id": song, "wall_time_sec": 2, "execution_disposition": disposition}
                for song in identifiers
            ]
        },
    )
    score = {"tp": 1, "fp": 0, "fn": 0, "unmapped": 0}
    monkeypatch.setattr(
        selected, "score_panel_song", lambda *_: {"baseline": score, "candidate": score}
    )
    return {
        **inputs,
        "panel_manifests": (root / "reference", root / "timing"),
        "_tf1_measurement": None if evidence == "sealed_only" else measurement,
    }, candidate_boot


@pytest.mark.parametrize(
    ("nonregression", "evidence", "disposition"),
    [
        (True, "same_boot", "inferred"),
        (False, "same_boot", "inferred"),
        (True, "other_boot", "inferred"),
        (True, "sealed_only", "inferred"),
        (True, "wrong_run", "inferred"),
        (True, "sealed_only", "resumed"),
    ],
)
def test_panel_uses_persistent_worker_and_replays_only_anchors(
    fake_experiment, monkeypatch, nonregression, evidence, disposition
):
    inputs, candidate_boot = _prepare_panel_experiment(
        fake_experiment, monkeypatch, evidence=evidence, disposition=disposition
    )
    identifiers = selected.PANEL_SIMFILE_IDS

    def aggregate(scores):
        assert len(list(scores)) == len(identifiers)
        return {"micro_f1_nonregression": nonregression}

    monkeypatch.setattr(selected, "aggregate_panel_scores", aggregate)
    workers, start = _worker_factory(inputs)
    result = selected.run_selected_comparison(**inputs, process_factory=start)
    if disposition == "resumed" or evidence == "wrong_run":
        assert result["status"] == "failed"
        assert result["failed_stage"] == "selection"
        assert not workers
        assert not result["panel_gate_passed"]
        return
    assert result["status"] == "complete"
    assert result["panel_gate_passed"] is (nonregression and evidence == "same_boot")
    if evidence != "sealed_only":
        assert result["candidate_boot_id"] == candidate_boot
        assert result["tf1_smoke"]["boot_id"] == inputs["_tf1_measurement"].boot_id
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
    ("worker_number", "stage", "error", "anchor_attempts", "runtime_changed"),
    [
        (1, "worker_start", "panel interpreter could not be started", 0, False),
        (1, "worker_close", "panel worker close timed out", 2, False),
        (2, "worker_start", "panel interpreter could not be started", 2, False),
        (2, "restore", "worker restored tensor count is invalid", 2, False),
        (2, "worker_close", "panel worker close timed out", 3, False),
        (2, "worker_close", "panel worker close timed out", 3, True),
    ],
)
def test_panel_retains_completed_evidence_when_a_worker_fails(
    fake_experiment, monkeypatch, worker_number, stage, error, anchor_attempts, runtime_changed
):
    inputs, _ = _prepare_panel_experiment(fake_experiment, monkeypatch)
    if runtime_changed:
        identities = iter(({"schema": "test-runtime/v1"}, {"schema": "test-runtime/v1", "cpu": 2}))
        monkeypatch.setattr(
            selected, "build_runtime_identity", lambda *args, **kwargs: next(identities)
        )
    workers, start = _worker_factory(inputs)

    def start_with_failure(*args, **kwargs):
        # Two smoke workers start before the first panel worker.
        if len(workers) != worker_number + 1:
            return start(*args, **kwargs)
        if stage == "worker_start":
            raise WorkerProcessError(error) from FileNotFoundError(inputs["python311"])
        worker = start(*args, **kwargs)
        if stage == "restore":
            worker.ready = {**worker.ready, "restored_tensor_count": 77}
        else:
            close = worker.close

            def close_with_failure():
                close()
                raise WorkerProcessError(error)

            worker.close = close_with_failure
        return worker

    result = selected.run_selected_comparison(**inputs, process_factory=start_with_failure)

    assert result["status"] == "failed"
    assert result["failed_stage"] == stage
    assert result["failed_worker"] == worker_number
    assert error in result["error"]
    assert result["panel_gate_passed"] is False
    assert all(worker.closed for worker in workers)
    assert len(workers) == worker_number + 2 - (stage == "worker_start")
    scored_songs = len(selected.PANEL_SIMFILE_IDS) if anchor_attempts else 0
    assert result["panel_scores"]["scored_song_count"] == scored_songs
    if scored_songs:
        assert result["panel_scores"]["micro_f1_nonregression"] is True
        assert result["panel_runtime"]["all_matched"]["songs"] == scored_songs
        assert result["panel_runtime"]["all_matched"]["baseline_rtf"] == 2
        if runtime_changed:
            assert result["runtime_after_sha256"] != result["runtime_sha256"]
            assert (
                result["runtime_recheck_error"] == "runtime fingerprint changed during measurement"
            )
        else:
            assert result["runtime_after_sha256"] == result["runtime_sha256"]
            assert "runtime_recheck_error" not in result
    else:
        assert "runtime_after_sha256" not in result
    for row in result["items"]:
        anchor = row["simfile_id"] in selected.PANEL_ANCHORS
        expected_attempts = anchor_attempts if anchor else int(scored_songs > 0)
        assert row["completed_attempts"] == expected_attempts
        assert row["repeatable"] is (True if anchor and anchor_attempts == 3 else None)
    directory = _execution_dir(inputs["output_root"])
    replay_attempts = max(0, anchor_attempts - 1) * len(selected.PANEL_ANCHORS)
    assert len((directory / "native-events.jsonl").read_text().splitlines()) == (
        scored_songs + replay_attempts
    )


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


def test_prepare_cli_can_retry_after_a_seal_publication_failure(tmp_path, monkeypatch, capsys):
    input_root = tmp_path / "wavs"
    output_parent = tmp_path / "reports"
    output_parent.write_bytes(b"not a directory")
    output = output_parent / "selection.json"

    def prepare(**options):
        options["input_root"].mkdir()
        (options["input_root"] / "27.wav").write_bytes(b"canonical audio")
        return {"seal": "test"}

    monkeypatch.setattr(selected, "prepare_fresh_selection", prepare)
    args = ["prepare-fresh", "--input-root", str(input_root), "--output", str(output)]
    for flag in ("baseline-run", "reference-manifest", "timing-manifest", "cache-dir"):
        args.extend([f"--{flag}", str(tmp_path / flag)])
    assert selected.main(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert not input_root.exists()

    output_parent.unlink()
    assert selected.main(args) == 0
    assert json.loads(output.read_text()) == {"seal": "test"}
    assert (input_root / "27.wav").exists()
