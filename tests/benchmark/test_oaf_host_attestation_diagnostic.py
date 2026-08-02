from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmark.backend_attestation import AttestationError
from tools.hpa320 import oaf_host_attestation_diagnostic as diagnostic
from tools.hpa320.oaf_host_attestation import HostAttestationError
from tools.hpa320.oaf_host_attestation_diagnostic import safe_exception_chain


def _controlled_failure() -> HostAttestationError:
    try:
        raise AttestationError("logical CPU fingerprints are inconsistent")
    except AttestationError as error:
        try:
            raise HostAttestationError("native-host attestation publication failed") from error
        except HostAttestationError as wrapped:
            return wrapped


def _uncontrolled_failure() -> HostAttestationError:
    try:
        raise RuntimeError("secret-bearing implementation detail")
    except RuntimeError as error:
        try:
            raise HostAttestationError("native-host attestation publication failed") from error
        except HostAttestationError as wrapped:
            return wrapped


def test_safe_exception_chain_preserves_controlled_root_cause() -> None:
    assert safe_exception_chain(_controlled_failure()) == (
        "HostAttestationError: native-host attestation publication failed"
        " <- AttestationError: logical CPU fingerprints are inconsistent"
    )


def test_safe_exception_chain_redacts_uncontrolled_root_cause() -> None:
    assert safe_exception_chain(_uncontrolled_failure()) == (
        "HostAttestationError: native-host attestation publication failed <- RuntimeError"
    )


def test_main_reports_the_safe_failure_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_publisher(*, phase: str, output_directory: Path) -> object:
        assert phase == "bootstrap"
        assert output_directory == tmp_path / "diagnostic"
        raise _controlled_failure()

    monkeypatch.setattr(diagnostic, "publish_github_host_attestation", fail_publisher)

    result = diagnostic.main(
        ["--phase", "bootstrap", "--output", str(tmp_path / "diagnostic")]
    )

    assert result == 2
    assert capsys.readouterr().err == (
        "HostAttestationError: native-host attestation publication failed"
        " <- AttestationError: logical CPU fingerprints are inconsistent\n"
    )


def test_main_removes_output_and_refuses_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "diagnostic"

    def successful_publisher(*, phase: str, output_directory: Path) -> object:
        assert phase == "bootstrap"
        assert output_directory == output
        output_directory.mkdir()
        (output_directory / "unexpected.json").write_text("{}\n", encoding="utf-8")
        return object()

    monkeypatch.setattr(
        diagnostic,
        "publish_github_host_attestation",
        successful_publisher,
    )

    result = diagnostic.main(["--phase", "bootstrap", "--output", str(output)])

    assert result == 3
    assert not output.exists()
    assert capsys.readouterr().err == (
        "native-host diagnostic unexpectedly succeeded; refusing to continue\n"
    )
