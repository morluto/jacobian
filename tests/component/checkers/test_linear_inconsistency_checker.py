from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from tests.support.artifacts import artifact_uri as _uri
from tests.support.artifacts import canonical_digest as _digest
from tests.support.rationals import rational_payload as _q

from jacobian.contracts.exact_domain_verification import inline_exact_value_digest
from jacobian.contracts.linear import (
    LinearRationalInconsistencyResult,
    LinearRationalSystem,
)
from jacobian_checkers.linear import check_rational_inconsistency


def _artifact(
    character: str,
    payload: dict[str, Any],
    *,
    semantics_uri: str,
    parents: list[str],
) -> dict[str, Any]:
    return {
        "artifact_uri": _uri(character),
        "object_digest": "sha256:" + character * 64,
        "payload_digest": _digest(payload),
        "schema_uri": _uri(chr(ord(character) + 1)),
        "semantics_uri": semantics_uri,
        "parents": parents,
        "payload": payload,
    }


def _request() -> dict[str, Any]:
    system = LinearRationalSystem.model_validate(
        {
            "variables": ["x", "y"],
            "coefficients": {"entries": [[_q(1), _q(1)], [_q(2), _q(2)]]},
            "rhs": [_q(1), _q(3)],
        }
    ).model_dump(mode="json")
    request_payload = {
        "system": system,
        "resource_budget": {"budget_version": "1", "wall_seconds": 5},
    }
    result = LinearRationalInconsistencyResult(
        left_witness=(_q(-2), _q(1)),
        rhs_pairing=_q(1),
    ).model_dump(mode="json")
    semantics = _artifact(
        "e",
        {"kind": "linear.rational_inconsistency"},
        semantics_uri=_uri("0"),
        parents=[],
    )
    claim = {
        "schema_uri": _uri("a"),
        "semantics_uri": semantics["artifact_uri"],
        "payload": request_payload,
    }
    candidate = {
        "schema_uri": _uri("c"),
        "semantics_uri": semantics["artifact_uri"],
        "payload": result,
    }
    bindings = {
        "claim_digest": inline_exact_value_digest(**claim),
        "semantics_digest": semantics["object_digest"],
        "candidate_digest": inline_exact_value_digest(**candidate),
        "scope_digest": None,
        "encoding_digest": None,
    }
    return {
        "request_version": "2",
        "operation_id": "linear.rational_inconsistency.compute",
        "claim": claim,
        "candidate": candidate,
        "semantics": semantics,
        "scope": None,
        "expected_bindings": bindings,
    }


def test_checker_accepts_exact_left_nullspace_inconsistency_witness() -> None:
    decision = check_rational_inconsistency(_request())

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


@pytest.mark.parametrize(
    "mutation", ("wrong_witness", "wrong_pairing", "changed_system")
)
def test_checker_rejects_tampered_inline_inconsistency(mutation: str) -> None:
    request = _request()
    if mutation == "wrong_witness":
        request["candidate"]["payload"]["left_witness"][0] = _q(-3)
    elif mutation == "wrong_pairing":
        request["candidate"]["payload"]["rhs_pairing"] = _q(2)
    else:
        request["claim"]["payload"]["system"]["rhs"][0] = _q(2)

    decision = check_rational_inconsistency(deepcopy(request))

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
