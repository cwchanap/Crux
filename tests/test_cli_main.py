from click.testing import CliRunner

from src.cli.main import main


def test_root_cli_lists_benchmark_group():
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "benchmark" in result.output


def test_convert_checkpoint_command_path_exists():
    result = CliRunner().invoke(main, ["convert", "--help"])

    assert result.exit_code == 0
    assert "checkpoint" in result.output
