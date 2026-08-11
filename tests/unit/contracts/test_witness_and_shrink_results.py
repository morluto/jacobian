from __future__ import annotations

import pytest
from pydantic import ValidationError
from tests.support.artifacts import artifact_uri as _uri

from jacobian.contracts.shrinking import ShrinkResult, ShrinkStep
from jacobian.contracts.witness_search import (
    PluginWitnessResponse,
    WitnessFindResult,
)


def _digest(fill: str) -> str:
    return "sha256:" + fill * 64


def _verified_result(*, evidence_uri: str) -> dict[str, object]:
    return {
        "schema_version": "1",
        "execution": {"status": "COMPLETED"},
        "input": {"status": "ACCEPTED"},
        "conclusion": "TRUE",
        "assurance": {
            "arithmetic": "EXACT_INTEGER",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "EXHAUSTIVE",
            "verification": "VERIFIED",
            "checker_id": "checker://sha256/" + "a" * 64,
            "checker_digest": _digest("b"),
        },
        "claim_digest": _digest("c"),
        "semantics_digest": _digest("d"),
        "candidate_digest": _digest("e"),
        "evidence_uris": [evidence_uri],
        "verification_record_uri": _uri("0"),
    }


def test_none_certified_requires_the_named_certificate_to_be_verified() -> None:
    named_certificate = _uri("f")
    other_evidence = _uri("1")

    with pytest.raises(ValidationError):
        WitnessFindResult.model_validate(
            {
                "status": "NONE_CERTIFIED",
                "result": _verified_result(evidence_uri=other_evidence),
                "claim_uri": _uri("2"),
                "candidate_uri": _uri("3"),
                "plugin_id": _uri("4"),
                "certificate_uri": named_certificate,
            }
        )


def test_plugin_may_only_propose_none_certified_with_a_certificate() -> None:
    with pytest.raises(ValidationError):
        PluginWitnessResponse.model_validate(
            {
                "status": "NONE_CERTIFIED",
                "arithmetic": "EXACT_INTEGER",
                "coverage": "EXHAUSTIVE",
            }
        )

    proposed = PluginWitnessResponse.model_validate(
        {
            "status": "NONE_CERTIFIED",
            "certificate_uri": _uri("7"),
            "arithmetic": "EXACT_INTEGER",
            "coverage": "EXHAUSTIVE",
        }
    )
    assert proposed.certificate_uri == _uri("7")


def test_accepted_shrink_step_requires_a_completed_checker_record() -> None:
    with pytest.raises(ValidationError, match="accepted shrink step"):
        ShrinkStep(
            index=0,
            reducer="delete_vertex",
            from_uri=_uri("1"),
            proposed_uri=_uri("2"),
            accepted=True,
            execution_status="COMPLETED",
            input_status="ACCEPTED",
        )

    with pytest.raises(ValidationError, match="rejected shrink step"):
        ShrinkStep(
            index=0,
            reducer="delete_vertex",
            from_uri=_uri("1"),
            proposed_uri=_uri("2"),
            accepted=False,
            execution_status="COMPLETED",
            input_status="REJECTED",
            verification_record_uri=_uri("3"),
        )


@pytest.mark.parametrize("minimality", ["ONE_STEP", "BOUNDED_GLOBAL", "PROVED_GLOBAL"])
def test_v01_rejects_unsupported_minimality_claims(minimality: str) -> None:
    final_target = _uri("5")

    with pytest.raises(ValidationError):
        ShrinkResult.model_validate(
            {
                "execution": {"status": "COMPLETED"},
                "input": {"status": "ACCEPTED"},
                "result": _verified_result(evidence_uri=final_target),
                "target_kind": "candidate",
                "initial_target_uri": _uri("6"),
                "final_target_uri": final_target,
                "minimality": minimality,
                "evaluations": 1,
            }
        )
