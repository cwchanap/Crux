import importlib
import sys
import types

import pytest


@pytest.fixture()
def tf2_model_module(monkeypatch):
    class _PlaceholderLayer:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, *_args, **_kwargs):
            return None

    fake_layers = types.ModuleType("tensorflow.keras.layers")
    for name in (
        "BatchNormalization",
        "Bidirectional",
        "Concatenate",
        "Conv2D",
        "Dense",
        "Dropout",
        "LSTM",
        "MaxPool2D",
        "Reshape",
    ):
        setattr(fake_layers, name, _PlaceholderLayer)

    fake_keras = types.ModuleType("tensorflow.keras")
    fake_keras.Model = object
    fake_keras.layers = fake_layers

    fake_tf = types.ModuleType("tensorflow")
    fake_tf.keras = fake_keras
    fake_tf.nn = types.SimpleNamespace(relu=lambda value: value)
    fake_tf.shape = lambda value: value
    fake_tf.reshape = lambda value, _shape: value
    fake_tf.zeros = lambda _shape: None

    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", fake_keras)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.layers", fake_layers)
    sys.modules.pop("src.app.tf2_magenta_model", None)

    module = importlib.import_module("src.app.tf2_magenta_model")
    yield module

    sys.modules.pop("src.app.tf2_magenta_model", None)


def test_load_tf1_checkpoint_to_tf2_accepts_checkpoint_path_first(tf2_model_module):
    class FakeModel:
        def __init__(self):
            self.loaded_paths = []

        def load_weights(self, path):
            self.loaded_paths.append(path)

    model = FakeModel()

    result = tf2_model_module.load_tf1_checkpoint_to_tf2("model.weights.h5", model)

    assert result is model
    assert model.loaded_paths == ["model.weights.h5"]


def test_load_tf1_checkpoint_to_tf2_wraps_load_failures(tf2_model_module):
    class FakeModel:
        def load_weights(self, _path):
            raise ValueError("boom")

    with pytest.raises(RuntimeError, match="Failed to load weights from model.weights.h5"):
        tf2_model_module.load_tf1_checkpoint_to_tf2("model.weights.h5", FakeModel())
