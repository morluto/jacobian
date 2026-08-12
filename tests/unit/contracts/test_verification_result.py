from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.checkers import CheckerDecision
from jacobian.contracts.results import VerificationResult


def test_timeout_cannot_carry_a_verified_false_conclusion() -> None:
    with pytest.raises(ValidationError):
        VerificationResult.model_validate(
            {
                "schema_version": "1",
                "execution": {"status": "TIMEOUT", "runtime_ms": 10},
                "input": {"status": "ACCEPTED"},
                "conclusion": "FALSE",
                "verification_record_uri": "artifact://sha256/" + "f" * 64,
                "evidence_uris": ["artifact://sha256/" + "c" * 64],
            }
        )


def test_timeout_is_represented_as_unknown_unverified_execution() -> None:
    result = VerificationResult.model_validate(
        {
            "schema_version": "1",
            "execution": {"status": "TIMEOUT", "runtime_ms": 10},
            "input": {"status": "ACCEPTED"},
            "conclusion": "UNKNOWN",
        }
    )

    assert result.model_dump(mode="json")["execution"]["status"] == "TIMEOUT"


def test_verification_record_requires_a_decisive_conclusion() -> None:
    with pytest.raises(ValidationError, match="decisive mathematical conclusion"):
        VerificationResult.model_validate(
            {
                "execution": {"status": "COMPLETED"},
                "input": {"status": "ACCEPTED"},
                "conclusion": "UNKNOWN",
                "verification_record_uri": "artifact://sha256/" + "f" * 64,
            }
        )


def test_verified_result_requires_claim_and_semantics_bindings() -> None:
    for missing in ("claim_digest", "semantics_digest"):
        payload = _verified_result()
        payload.pop(missing)
        with pytest.raises(ValidationError):
            VerificationResult.model_validate(payload)


@pytest.mark.parametrize("method", ["DIRECT_WITNESS", "EXHAUSTIVE_FINITE"])
def test_verified_result_requires_candidate_binding(method: str) -> None:
    payload = _verified_result(
        method=method,
        coverage=("NOT_APPLICABLE" if method == "DIRECT_WITNESS" else "EXHAUSTIVE"),
    )
    payload.pop("candidate_digest")

    with pytest.raises(ValidationError):
        VerificationResult.model_validate(payload)


@pytest.mark.parametrize(
    ("arithmetic", "coverage"),
    [
        ("SYMBOLIC", "NOT_APPLICABLE"),
        ("VERIFIED_INTERVAL", "NOT_APPLICABLE"),
        ("EXACT_ALGEBRAIC", "EXHAUSTIVE"),
    ],
)
def test_checked_certificates_support_non_rational_proof_mechanisms(
    arithmetic: str,
    coverage: str,
) -> None:
    VerificationResult.model_validate(
        _verified_result(
            arithmetic=arithmetic,
            method="CHECKED_CERTIFICATE",
            coverage=coverage,
        )
    )
    decision = CheckerDecision.model_validate(
        {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": arithmetic,
            "method": "CHECKED_CERTIFICATE",
            "coverage": coverage,
        }
    )

    assert decision.accepted is True


def test_checker_relationship_requires_exact_artifact_endpoints() -> None:
    decision = {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "CHECKED_CERTIFICATE",
        "coverage": "EXHAUSTIVE",
        "relation_id": "example.relation.checked",
    }

    with pytest.raises(ValidationError):
        CheckerDecision.model_validate(decision)

    accepted = CheckerDecision.model_validate(
        {
            **decision,
            "relationship_source_artifact_uris": ["artifact://sha256/" + "1" * 64],
            "relationship_target_artifact_uris": ["artifact://sha256/" + "2" * 64],
        }
    )

    assert accepted.relationship_source_artifact_uris == (
        "artifact://sha256/" + "1" * 64,
    )


def _verified_result(
    *,
    conclusion: str = "TRUE",
    arithmetic: str = "EXACT_INTEGER",
    method: str = "DIRECT_WITNESS",
    coverage: str = "NOT_APPLICABLE",
) -> dict[str, object]:
    return {
        "schema_version": "1",
        "execution": {"status": "COMPLETED"},
        "input": {"status": "ACCEPTED"},
        "conclusion": conclusion,
        "claim_digest": "sha256:" + "c" * 64,
        "semantics_digest": "sha256:" + "d" * 64,
        "candidate_digest": "sha256:" + "e" * 64,
        "evidence_uris": ["artifact://sha256/" + "f" * 64],
        "verification_record_uri": "artifact://sha256/" + "0" * 64,
    }
