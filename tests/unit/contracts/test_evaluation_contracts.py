from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.evaluation import EvaluationBatchResult, EvaluationItem


def _uri(fill: str) -> str:
    return "artifact://sha256/" + fill * 64


def _digest(fill: str) -> str:
    return "sha256:" + fill * 64


def _unverified_result() -> dict[str, object]:
    return {
        "execution": {"status": "COMPLETED"},
        "input": {"status": "ACCEPTED"},
        "conclusion": "TRUE",
        "assurance": {
            "arithmetic": "EXACT_INTEGER",
            "method": "HEURISTIC",
            "coverage": "BOUNDED",
            "verification": "UNVERIFIED",
        },
    }


def _verified_result() -> dict[str, object]:
    return {
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
        "evidence_uris": [_uri("f")],
        "verification_record_uri": _uri("0"),
    }


def test_evaluator_item_cannot_promote_its_result_to_verified() -> None:
    with pytest.raises(ValidationError, match="cannot grant verified"):
        EvaluationItem(
            candidate_uri=_uri("1"),
            result=_verified_result(),
        )


def test_evaluation_batch_binds_admission_to_items_and_runtime_digests() -> None:
    item = {
        "candidate_uri": _uri("1"),
        "result": _unverified_result(),
    }
    base = {
        "execution": {"status": "COMPLETED"},
        "claim_uri": _uri("2"),
        "plugin_id": _uri("3"),
        "profile": "EXACT_CANDIDATE",
        "seed": 0,
    }
    with pytest.raises(ValidationError, match="requires result items"):
        EvaluationBatchResult.model_validate({**base, "input": {"status": "ACCEPTED"}})
    with pytest.raises(ValidationError, match="cannot carry evaluation evidence"):
        EvaluationBatchResult.model_validate(
            {
                **base,
                "input": {"status": "REJECTED", "errors": ["invalid batch"]},
                "items": [item],
            }
        )
