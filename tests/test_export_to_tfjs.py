import importlib
import sys
import types

import pytest


@pytest.fixture()
def export_module(monkeypatch):
    fake_tf = types.ModuleType("tensorflow")
    fake_tf.keras = types.SimpleNamespace(Model=object)

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


def test_export_to_tfjs_uses_requested_h5_path_on_fallback(export_module, monkeypatch, tmp_path):
    class FakeModel:
        def __init__(self):
            self.saved_args = None

        def save(self, path: str, include_optimizer: bool = True):
            self.saved_args = (path, include_optimizer)

    fake_model = FakeModel()
    fake_tfjs = types.ModuleType("tensorflowjs")
    fake_tfjs.converters = types.SimpleNamespace(
        save_keras_model=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    monkeypatch.setattr(export_module, "build_functional_model", lambda _weights_path: fake_model)
    monkeypatch.setitem(sys.modules, "tensorflowjs", fake_tfjs)

    output_dir = tmp_path / "web_model"
    keras_h5_path = tmp_path / "custom.keras.h5"
    export_module.export_to_tfjs(None, output_dir, keras_h5_path)

    assert fake_model.saved_args == (str(keras_h5_path), False)
