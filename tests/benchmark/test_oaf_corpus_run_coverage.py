from __future__ import annotations

from pathlib import Path

import pytest

from runtime.oaf_tf1.model import load_model_config
from src.benchmark.corpus_cache import ResolvedSourceAudio
from src.benchmark.oaf_corpus_run import _materialize_oaf_full_mix


def test_materialize_oaf_full_mix_rejects_non_oaf_model_config(tmp_path: Path) -> None:
    """The OaF full-mix materializer validates config type before delegating."""
    source = ResolvedSourceAudio(
        path=tmp_path / "source.wav",
        source_audio_id="42/bgm.wav",
        source_audio_sha256="0" * 64,
        duration_sec=1.0,
    )
    output_path = tmp_path / "output" / "full-mix.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pytest.raises(TypeError, match="config must be OafModelConfig"):
        _materialize_oaf_full_mix(
            source,
            output_path,
            input_root=tmp_path,
            config="not-an-oaf-config",
        )


def test_materialize_oaf_full_mix_delegates_to_neutral_materializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid OafModelConfig delegates to the model-neutral materializer."""
    import src.benchmark.oaf_corpus_run as run_module

    source = ResolvedSourceAudio(
        path=tmp_path / "source.wav",
        source_audio_id="42/bgm.wav",
        source_audio_sha256="0" * 64,
        duration_sec=1.0,
    )
    output_path = tmp_path / "output" / "full-mix.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = load_model_config()
    sentinel = object()

    def fake_materialize(*args: object, **kwargs: object) -> object:
        return sentinel

    monkeypatch.setattr(run_module, "materialize_full_mix_audio", fake_materialize)
    result = _materialize_oaf_full_mix(
        source,
        output_path,
        input_root=tmp_path,
        config=config,
    )

    assert result is sentinel
