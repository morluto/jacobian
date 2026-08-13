from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.checker_identity import build_checker_manifest
from jacobian.contracts.checkers import CheckerDecision
from jacobian.contracts.evidence import EvidenceBindings
from jacobian.contracts.exact_domain_verification import (
    ExactComputedVerificationOutput,
    InlineExactVerificationRecord,
)
from jacobian.contracts.results import Arithmetic, Conclusion, Coverage, Method

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
            claim_digest="sha256:" + "c" * 64,
            semantics_digest="sha256:" + "d" * 64,
            candidate_digest="sha256:" + "e" * 64,
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


def _inline_v4_record() -> InlineExactVerificationRecord:
    manifest = build_checker_manifest(
        "jacobian_checkers.reject:check",
        provider_runtime=None,
        passive_contract_uris=(),
    )
    digest = "sha256:" + "c" * 64
    return InlineExactVerificationRecord(
        witness_format="tests.inline",
        operation_id="integer.compute.gcd",
        checker_id=_CHECKER_URI,
        implementation_digest=manifest.implementation_digest(),
        checker_manifest=manifest,
        environment_digest=digest,
        input_schema_uri=_ARTIFACT_URI,
        candidate_schema_uri=_ARTIFACT_URI,
        semantics_uri=_ARTIFACT_URI,
        bindings=EvidenceBindings(
            claim_digest=digest,
            semantics_digest=digest,
            candidate_digest=digest,
        ),
        decision=CheckerDecision(
            accepted=True,
            conclusion=Conclusion.TRUE,
            arithmetic=Arithmetic.EXACT_INTEGER,
            method=Method.DIRECT_WITNESS,
            coverage=Coverage.NOT_APPLICABLE,
        ),
        request_digest=digest,
    )


def test_inline_v4_record_binds_the_exact_checker_manifest() -> None:
    record = _inline_v4_record()

    assert record.record_schema_version == "4"
    assert (
        record.implementation_digest == record.checker_manifest.implementation_digest()
    )

    payload = record.model_dump(mode="json")
    with pytest.raises(ValidationError):
        InlineExactVerificationRecord.model_validate(
            payload | {"checker_manifest": None}
        )
    with pytest.raises(ValidationError, match="digest must match"):
        InlineExactVerificationRecord.model_validate(
            payload | {"implementation_digest": "sha256:" + "0" * 64}
        )
