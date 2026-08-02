from __future__ import annotations

from src.benchmark.backend_attestation import AttestationError
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
