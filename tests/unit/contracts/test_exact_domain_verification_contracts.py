from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.exact_domain_verification import (
    ExactComputedVerificationOutput,
)

_ARTIFACT_URI = "artifact://sha256/" + "a" * 64
_CHECKER_URI = "checker://sha256/" + "b" * 64


def _output(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "REJECTED",
        "conclusion": "UNKNOWN",
        "operation_id": "integer.compute.prime_factorization",
        "checker_id": _CHECKER_URI,
        "detail": "not accepted",
    }
    payload.update(updates)
    return payload


def test_verified_exact_output_requires_true_conclusion_and_record() -> None:
    verified = ExactComputedVerificationOutput.model_validate(
        _output(
            status="VERIFIED",
            conclusion="TRUE",
            verification_record_uri=_ARTIFACT_URI,
        )
    )
    assert verified.verification_record_uri == _ARTIFACT_URI

    for invalid in (
        _output(status="VERIFIED", conclusion="TRUE"),
        _output(status="VERIFIED", verification_record_uri=_ARTIFACT_URI),
        _output(conclusion="TRUE", verification_record_uri=_ARTIFACT_URI),
    ):
        with pytest.raises(ValidationError):
            ExactComputedVerificationOutput.model_validate(invalid)


def test_rejected_exact_output_may_still_identify_examined_witness() -> None:
    rejected = ExactComputedVerificationOutput.model_validate(
        _output(witness_uri=_ARTIFACT_URI)
    )

    assert rejected.status == "REJECTED"
    assert rejected.witness_uri == _ARTIFACT_URI
