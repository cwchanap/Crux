import sys
from types import ModuleType

import httpx
import numpy as np
import pytest
import soundfile as sf

from src.app.transcriber import DrumTranscriber


@pytest.mark.asyncio
async def test_transcribe_fallback_uses_onset_detection(monkeypatch, tmp_path):
    # Ensure model building is skipped and fallback path is used
    monkeypatch.setattr(DrumTranscriber, "_build_model", lambda self: None, raising=True)

    dt = DrumTranscriber(load_model=False, sample_rate=16000)

    # Create a tiny silent audio file that librosa can load
    sr = 16000
    audio = np.zeros(sr // 2, dtype=np.float32)  # 0.5s silence
    wav_path = tmp_path / "silence.wav"
    sf.write(wav_path, audio, sr)

    # Prepare job store
    job_id = "job-fallback"
    jobs_store = {
        job_id: {
            "status": "pending",
            "progress": 0,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }
    }

    midi_bytes = await dt.transcribe(str(wav_path), job_id, jobs_store)
    assert isinstance(midi_bytes, (bytes, bytearray))
    assert midi_bytes[:4] == b"MThd"
    # Progress should have advanced
    assert jobs_store[job_id]["progress"] >= 70


def test_init_falls_back_when_model_download_fails(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        DrumTranscriber,
        "_resolve_existing_path",
        classmethod(lambda cls, relative_path: None),
        raising=True,
    )

    def fake_stream(*args, **kwargs):
        raise httpx.HTTPStatusError(
            "404 Client Error",
            request=httpx.Request("GET", "http://fake"),
            response=httpx.Response(404),
        )

    monkeypatch.setattr("src.app.transcriber.httpx.stream", fake_stream)

    transcriber = DrumTranscriber(load_model=True)

    assert transcriber.model_path is None
    assert transcriber.model is None


def test_init_uses_shared_root_workspace_model_from_worktree(monkeypatch, tmp_path):
    repo_root = tmp_path / "Crux"
    worktree = repo_root / ".worktrees" / "feature"
    shared_model = repo_root / "models" / "e-gmd" / "tf2_model.weights.h5"
    worktree.mkdir(parents=True)
    shared_model.parent.mkdir(parents=True)
    shared_model.write_text("weights", encoding="utf-8")
    monkeypatch.chdir(worktree)

    def fail_download(self):
        raise AssertionError("should not download when shared root model exists")

    monkeypatch.setattr(DrumTranscriber, "_download_model", fail_download, raising=True)
    monkeypatch.setattr(DrumTranscriber, "_build_model", lambda self: None, raising=True)

    transcriber = DrumTranscriber(load_model=True)

    assert transcriber.model_path == str(shared_model)
    assert transcriber.model is None


def test_compute_spectrogram_shape():
    dt = DrumTranscriber(load_model=False, sample_rate=22050)
    # 0.1s of ones
    audio = np.ones(2205, dtype=np.float32)
    spec = dt._compute_spectrogram_for_model(audio, sr=22050)
    assert spec.ndim == 2
    # 229 mel bins expected per implementation
    assert spec.shape[1] == 229
    assert np.isfinite(spec).all()


def test_build_model_returns_none_on_import_error(monkeypatch, tmp_path):
    """If tensorflow is not installed, _build_model should return None (fallback), not raise."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        DrumTranscriber,
        "_resolve_existing_path",
        classmethod(lambda cls, relative_path: tmp_path / "fake.weights.h5"),
        raising=True,
    )
    # Make the weights file exist so _build_model tries to import the model module.
    (tmp_path / "fake.weights.h5").write_bytes(b"fake")

    # Patch the import to raise ImportError as if tensorflow is missing.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "src.app.tf2_magenta_model":
            raise ImportError("no module named 'tensorflow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    transcriber = DrumTranscriber(load_model=True)

    assert transcriber.model is None


def test_init_falls_back_when_tf1_checkpoint_conversion_raises(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.ckpt-10000"
    # TF1 checkpoints store data as .index + .data-* shards — no base file.
    checkpoint.with_name(f"{checkpoint.name}.index").write_text("index", encoding="utf-8")

    monkeypatch.setattr(
        DrumTranscriber,
        "_resolve_existing_path",
        classmethod(lambda cls, relative_path: None),
        raising=True,
    )
    monkeypatch.setattr(
        DrumTranscriber, "_download_model", lambda self: str(checkpoint), raising=True
    )

    fake_module = ModuleType("src.app.tf2_magenta_model")

    class FakeModel:
        def load_weights(self, path):  # noqa: ANN001
            raise AssertionError(f"unexpected TF2 weight load: {path}")

    fake_module.create_drum_model = lambda: FakeModel()

    def raise_conversion_error(checkpoint_path, model):  # noqa: ANN001
        raise RuntimeError(f"Non-H5 checkpoint loading not implemented: {checkpoint_path}")

    fake_module.load_tf1_checkpoint_to_tf2 = raise_conversion_error
    monkeypatch.setitem(sys.modules, "src.app.tf2_magenta_model", fake_module)

    transcriber = DrumTranscriber(load_model=True)

    assert transcriber.model_path == str(checkpoint)
    assert transcriber.model is None


def test_build_model_loads_cached_tf1_checkpoint_without_base_file(monkeypatch, tmp_path):
    """Regression test: cached TF1 checkpoints only have .index/.data-* shards.
    _build_model must recognise the checkpoint even when no literal base file exists."""
    checkpoint = tmp_path / "model.ckpt-10000"
    checkpoint.with_name(f"{checkpoint.name}.index").write_text("index", encoding="utf-8")

    monkeypatch.setattr(
        DrumTranscriber,
        "_resolve_existing_path",
        classmethod(lambda cls, relative_path: None),
        raising=True,
    )
    monkeypatch.setattr(
        DrumTranscriber, "_download_model", lambda self: str(checkpoint), raising=True
    )

    fake_module = ModuleType("src.app.tf2_magenta_model")

    class FakeModel:
        pass

    fake_module.create_drum_model = lambda: FakeModel()
    fake_module.load_tf1_checkpoint_to_tf2 = lambda path, model: model
    monkeypatch.setitem(sys.modules, "src.app.tf2_magenta_model", fake_module)

    transcriber = DrumTranscriber(load_model=True)

    assert transcriber.model is not None
