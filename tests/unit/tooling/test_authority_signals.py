"""Unit tests for tightened verify/authority signal detection."""

from __future__ import annotations

from tools.test_architecture.authority_signals import has_verify_authority_signal


def test_verified_token_does_not_match_unverified() -> None:
    assert not has_verify_authority_signal(
        'assert snapshot.verification.value == "UNVERIFIED"'
    )
    assert has_verify_authority_signal(
        "assert result.verification_record_uri is not None"
    )


def test_capability_id_verify_invoke_counts() -> None:
    assert has_verify_authority_signal(
        'CapabilityRequest(capability_id="polynomial.map.inverse.verify", input={})'
    )
    assert not has_verify_authority_signal(
        'related = "modular.polynomial_residue_image.verify"'
    )


def test_verification_service_calls_count() -> None:
    assert has_verify_authority_signal("runtime.services.verification.verify_witness(")
    assert has_verify_authority_signal("assert installation.checker_id is not None")
