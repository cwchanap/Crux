import importlib
import re
import sys
import types

import pytest


@pytest.fixture()
def export_module(monkeypatch):
    fake_tf = types.ModuleType("tensorflow")

    # Minimal fake Keras surface used by export_to_tfjs
    class _FakeActivation:
        """Record ``(name)`` so tests can assert the wrapper passes names."""

        def __init__(self, activation, name=None):
            self.name = name

        def __call__(self, tensor):
            return tensor

    class _FakeInput:
        pass

    _fake_layers = types.SimpleNamespace(Activation=_FakeActivation, Input=_FakeInput)
    _fake_keras = types.SimpleNamespace(
        Model=object,
        Input=lambda *a, **kw: None,
        layers=_fake_layers,
    )
    fake_tf.keras = _fake_keras

    fake_magenta_model = types.ModuleType("src.app.tf2_magenta_model")
    fake_magenta_model.create_drum_model = lambda checkpoint_path=None: None

    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setitem(sys.modules, "src.app.tf2_magenta_model", fake_magenta_model)
    sys.modules.pop("export_to_tfjs", None)

    module = importlib.import_module("export_to_tfjs")
    yield module

    sys.modules.pop("export_to_tfjs", None)


def test_build_parser_accepts_save_keras_h5_option(export_module):
    parser = export_module.build_parser()

    args = parser.parse_args(["--save-keras-h5", "custom.keras.h5"])

    assert args.save_keras_h5 == "custom.keras.h5"


def test_export_to_tfjs_saves_to_explicit_h5_path(export_module, monkeypatch, tmp_path):
    class FakeModel:
        def __init__(self):
            self.saved_args = None

        def save(self, path: str, include_optimizer: bool = True):
            self.saved_args = (path, include_optimizer)

    fake_model = FakeModel()

    monkeypatch.setattr(export_module, "build_functional_model", lambda _weights_path: fake_model)

    output_dir = tmp_path / "web_model"
    keras_h5_path = tmp_path / "custom.keras.h5"
    export_module.export_to_tfjs(None, output_dir, keras_h5_path)

    assert fake_model.saved_args == (str(keras_h5_path), False)


def test_main_fails_fast_for_missing_weights_path(export_module, monkeypatch, caplog, tmp_path):
    missing_weights = tmp_path / "does_not_exist.weights.h5"
    output_dir = tmp_path / "web_model"
    export_called = False

    def _fake_export(*_args, **_kwargs):
        nonlocal export_called
        export_called = True

    monkeypatch.setattr(export_module, "export_to_tfjs", _fake_export)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_to_tfjs.py",
            "--weights",
            str(missing_weights),
            "--output-dir",
            str(output_dir),
        ],
    )

    with caplog.at_level("ERROR", logger="export_to_tfjs"):
        with pytest.raises(FileNotFoundError, match=re.escape(str(missing_weights))):
            export_module.main()

    assert not export_called
    assert str(missing_weights) in caplog.text
