from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.checkers import CheckerDecision
from jacobian.contracts.results import ResultEnvelope, validate_result_envelope


def test_timeout_cannot_carry_a_verified_false_conclusion() -> None:
    with pytest.raises(ValidationError):
        ResultEnvelope.model_validate(
            {
                "schema_version": "1",
                "execution": {"status": "TIMEOUT", "runtime_ms": 10},
                "input": {"status": "ACCEPTED"},
                "conclusion": "FALSE",
                "assurance": {
                    "arithmetic": "EXACT_INTEGER",
                    "method": "DIRECT_WITNESS",
                    "coverage": "NOT_APPLICABLE",
                    "verification": "VERIFIED",
                    "checker_id": "checker://sha256/" + "a" * 64,
                    "checker_digest": "sha256:" + "b" * 64,
                },
                "evidence_uris": ["artifact://sha256/" + "c" * 64],
            }
        )


def test_timeout_is_represented_as_unknown_unverified_execution() -> None:
    result = ResultEnvelope.model_validate(
        {
            "schema_version": "1",
            "execution": {"status": "TIMEOUT", "runtime_ms": 10},
            "input": {"status": "ACCEPTED"},
            "conclusion": "UNKNOWN",
            "assurance": {
                "arithmetic": "SYMBOLIC",
                "method": "BOUNDED_SEARCH",
                "coverage": "BOUNDED",
                "verification": "UNVERIFIED",
            },
        }
    )

    assert result.model_dump(mode="json")["execution"]["status"] == "TIMEOUT"


def test_unverified_result_cannot_smuggle_a_verification_record() -> None:
    with pytest.raises(ValidationError, match="unverified result"):
        ResultEnvelope.model_validate(
            {
                "execution": {"status": "COMPLETED"},
                "input": {"status": "ACCEPTED"},
                "conclusion": "UNKNOWN",
                "assurance": {
                    "arithmetic": "SYMBOLIC",
                    "method": "HEURISTIC",
                    "coverage": "NOT_APPLICABLE",
                    "verification": "UNVERIFIED",
                },
                "verification_record_uri": "artifact://sha256/" + "f" * 64,
            }
        )


@pytest.mark.parametrize(
    ("conclusion", "arithmetic", "method", "coverage"),
    [
        ("UNKNOWN", "EXACT_INTEGER", "DIRECT_WITNESS", "NOT_APPLICABLE"),
        ("NOT_APPLICABLE", "EXACT_INTEGER", "DIRECT_WITNESS", "NOT_APPLICABLE"),
        ("TRUE", "FLOATING_HEURISTIC", "DIRECT_WITNESS", "NOT_APPLICABLE"),
        ("TRUE", "EXACT_INTEGER", "DIRECT_WITNESS", "SAMPLED"),
        ("TRUE", "EXACT_INTEGER", "EXHAUSTIVE_FINITE", "BOUNDED"),
    ],
)
def test_verified_result_rejects_non_replayable_assurance(
    conclusion: str,
    arithmetic: str,
    method: str,
    coverage: str,
) -> None:
    with pytest.raises(ValidationError):
        ResultEnvelope.model_validate(
            _verified_result(
                conclusion=conclusion,
                arithmetic=arithmetic,
                method=method,
                coverage=coverage,
            )
        )


def test_verified_result_requires_claim_and_semantics_bindings() -> None:
    for missing in ("claim_digest", "semantics_digest"):
        payload = _verified_result()
        payload.pop(missing)
        with pytest.raises(ValidationError):
            ResultEnvelope.model_validate(payload)


@pytest.mark.parametrize("method", ["DIRECT_WITNESS", "EXHAUSTIVE_FINITE"])
def test_verified_result_requires_candidate_binding(method: str) -> None:
    payload = _verified_result(
        method=method,
        coverage=("NOT_APPLICABLE" if method == "DIRECT_WITNESS" else "EXHAUSTIVE"),
    )
    payload.pop("candidate_digest")

    with pytest.raises(ValidationError):
        ResultEnvelope.model_validate(payload)


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
    result = ResultEnvelope.model_validate(
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

    assert result.assurance.arithmetic.value == arithmetic
    assert decision.accepted is True


def test_trust_boundary_revalidates_model_construct_instances() -> None:
    malformed = ResultEnvelope.model_construct(
        **_verified_result(
            conclusion="UNKNOWN",
            arithmetic="FLOATING_HEURISTIC",
        )
    )

    with pytest.raises(ValidationError):
        validate_result_envelope(malformed)


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
        "assurance": {
            "arithmetic": arithmetic,
            "method": method,
            "coverage": coverage,
            "verification": "VERIFIED",
            "checker_id": "checker://sha256/" + "a" * 64,
            "checker_digest": "sha256:" + "b" * 64,
        },
        "claim_digest": "sha256:" + "c" * 64,
        "semantics_digest": "sha256:" + "d" * 64,
        "candidate_digest": "sha256:" + "e" * 64,
        "evidence_uris": ["artifact://sha256/" + "f" * 64],
        "verification_record_uri": "artifact://sha256/" + "0" * 64,
    }
