import os
import sys
import types
from pathlib import Path

from click.testing import CliRunner


def test_cli_error_when_checkpoint_missing(tmp_path: Path, monkeypatch):
    # Stub heavy modules before importing CLI
    monkeypatch.setitem(sys.modules, "tensorflow", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "src.app.tf2_magenta_model",
        types.SimpleNamespace(create_drum_model=lambda *a, **k: object()),
    )
    from src.cli.convert import main as cli_main  # import after stubbing

    runner = CliRunner()
    # Do not create .index file, so it should exit with code 1
    result = runner.invoke(
        cli_main,
        [
            "--checkpoint-path",
            str(tmp_path / "nope.ckpt"),
            "--output-path",
            str(tmp_path / "out.weights.h5"),
            "--no-test",
        ],
    )
    assert result.exit_code == 1


def test_cli_success_with_fake_conversion(tmp_path: Path, monkeypatch):
    runner = CliRunner()

    # Make checkpoint look present by creating .index file
    ckpt_base = tmp_path / "model.ckpt-12345"
    ckpt_base.with_suffix(".index").write_text("")

    # Provide a fake conversion function returning a fake model
    class FakeModel:
        def __call__(self, x):  # noqa: D401, ANN001
            return {"onset_probs": object(), "velocity_values": object()}

        def save_weights(self, path):  # noqa: D401, ANN001
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("weights")

    def fake_convert(checkpoint_path: str, output_path: str):  # noqa: D401
        # Simulate writing weights
        fm = FakeModel()
        fm.save_weights(output_path)
        return fm

    # Stub heavy modules before importing CLI
    monkeypatch.setitem(sys.modules, "tensorflow", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "src.app.tf2_magenta_model",
        types.SimpleNamespace(create_drum_model=lambda *a, **k: object()),
    )

    import src.cli.convert as convert_mod

    monkeypatch.setattr(convert_mod, "convert_tf1_checkpoint_to_tf2", fake_convert, raising=True)
    # Patch os.path.exists inside module to simulate checkpoint presence
    real_exists = os.path.exists

    def fake_exists(p):  # noqa: D401, ANN001
        if p == str(ckpt_base) + ".index":
            return True
        return real_exists(p)

    monkeypatch.setattr(convert_mod.os.path, "exists", fake_exists, raising=False)

    result = runner.invoke(
        convert_mod.main,
        [
            "--checkpoint-path",
            str(ckpt_base),
            "--output-path",
            str(tmp_path / "out.weights.h5"),
            "--no-test",
        ],
    )

    assert result.exit_code == 0
    # Verify weights file created as the success signal
    out_path = tmp_path / "out.weights.h5"
    assert out_path.exists()
