"""Measure the frozen panel and its smoke on the current Colab boot."""

from __future__ import annotations

import runpy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from experiments.oaf_tf215 import compare
from experiments.oaf_tf215.selected_inputs import (
    FRESH_REFERENCE_MANIFEST_SHA256,
    FRESH_TIMING_MANIFEST_SHA256,
    PANEL_RUN_ID,
    PANEL_SIMFILE_IDS,
    SelectionBlocked,
    SelectionError,
)
from runtime.oaf_tf1.model import load_model_config
from src.benchmark.artifact_io import publish_immutable_file, read_regular_file_no_follow
from src.benchmark.backend_identity import sha256_hex
from src.benchmark.backends.oaf import create_backend
from src.benchmark.input_view import load_direct_audio
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.oaf_corpus_run import OafCorpusRunRequest, run_oaf_corpus
from src.benchmark.oaf_smoke_oracle import assert_smoke_oracle_matches, read_smoke_oracle
from src.benchmark.prediction_artifact import read_prediction_artifact, render_prediction_artifact

TF1_LAUNCHER_SHA256 = "c503911ca6eab92b93fff71f76c4e397209ed3cf43105d306ee28c5591d1553c"
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
SMOKE_ROOT = compare.REPOSITORY_ROOT / "tests/fixtures/oaf_tf1_smoke"
SMOKE_PREDICTION_NAME = "tf1-smoke-prediction.jsonl"


@dataclass(frozen=True)
class Tf1PanelMeasurement:
    run_path: Path
    boot_id: str
    run_sha256: str
    launcher_sha256: str
    smoke_prediction_sha256: str
    smoke_oracle_sha256: str

    def binding(self) -> dict[str, str]:
        return {
            "boot_id": self.boot_id,
            "baseline_run_sha256": self.run_sha256,
            "launcher_sha256": self.launcher_sha256,
            "smoke_prediction_sha256": self.smoke_prediction_sha256,
            "smoke_oracle_sha256": self.smoke_oracle_sha256,
        }


def read_boot_id() -> str:
    return str(UUID(BOOT_ID_PATH.read_text(encoding="ascii").strip()))


def measure_tf1_panel(
    *,
    launcher_path: Path,
    reference_manifest: Path,
    timing_manifest: Path,
    cache_dir: Path,
    checkpoint_cache: Path,
    directory: Path,
) -> Tf1PanelMeasurement:
    """Install the reviewed rootfs and run both TF1 measurements locally."""
    launcher = read_regular_file_no_follow(launcher_path)
    launcher_sha256 = sha256_hex(launcher)
    if launcher_sha256 != TF1_LAUNCHER_SHA256:
        raise SelectionError("TF1 rootfs launcher digest does not match the reviewed version")
    for path, expected in (
        (reference_manifest, FRESH_REFERENCE_MANIFEST_SHA256),
        (timing_manifest, FRESH_TIMING_MANIFEST_SHA256),
    ):
        if sha256_hex(read_regular_file_no_follow(path)) != expected:
            raise SelectionError(f"panel manifest digest does not match: {path}")

    boot_id = read_boot_id()
    config = load_model_config(compare.MODEL_CONFIG_PATH)
    try:
        checkpoint = Path(compare._prepare_checkpoint(config, checkpoint_cache)["checkpoint"])
    except compare.CheckpointAcquisitionError as error:
        raise SelectionBlocked(str(error)) from error
    oracle_bytes = read_regular_file_no_follow(SMOKE_ROOT / "smoke-oracle.json")
    audio = load_direct_audio(
        SMOKE_ROOT / "canonical.wav",
        source_audio_id=compare.SOURCE_AUDIO_ID,
        input_view_id=compare.SOURCE_AUDIO_ID,
        max_input_audio_frames=config.max_input_audio_frames,
    )
    with TemporaryDirectory(prefix="tf1-rootfs-", dir=directory) as staging:
        rootfs = Path(staging) / "rootfs"
        script = rootfs.with_name("tf1_rootfs.py")
        script.write_bytes(launcher)
        reviewed = runpy.run_path(str(script))
        reviewed["install"](rootfs, compare.REPOSITORY_ROOT)
        backend_factory = partial(
            create_backend,
            checkpoint_dir=checkpoint,
            process_factory=reviewed["chroot_process_factory"](rootfs),
        )
        backend = backend_factory(
            input_root=audio.path.parent, timeout_seconds=compare.WORKER_START_TIMEOUT_SECONDS
        )
        try:
            prediction, _ = map_oaf_prediction(backend.transcribe(audio))
            content = render_prediction_artifact(prediction)
            assert_smoke_oracle_matches(
                read_prediction_artifact(content).prediction, read_smoke_oracle(oracle_bytes)
            )
        except AssertionError as error:
            raise SelectionError(f"TF1 smoke did not match the frozen oracle: {error}") from error
        finally:
            backend.close()
        smoke = publish_immutable_file(directory / SMOKE_PREDICTION_NAME, content)
        outcome = run_oaf_corpus(
            OafCorpusRunRequest(
                reference_manifest_path=reference_manifest,
                timing_manifest_path=timing_manifest,
                cache_dir=cache_dir,
                output_dir=directory / "tf1",
                include_simfile_ids=PANEL_SIMFILE_IDS,
            ),
            backend_factory=backend_factory,
        )
    if outcome.exit_code != 0 or outcome.run_id != PANEL_RUN_ID or outcome.run_path is None:
        raise SelectionError(
            f"frozen TF1 panel did not complete: {outcome.failure_detail or outcome.overall_status}"
        )
    run_sha256 = sha256_hex(read_regular_file_no_follow(outcome.run_path))
    if read_boot_id() != boot_id:
        raise SelectionError("measurement VM boot ID changed during the TF1 panel")
    return Tf1PanelMeasurement(
        outcome.run_path,
        boot_id,
        run_sha256,
        launcher_sha256,
        smoke.sha256,
        sha256_hex(oracle_bytes),
    )
