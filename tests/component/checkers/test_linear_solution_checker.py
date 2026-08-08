from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from tests.support.artifacts import artifact_uri as _uri
from tests.support.artifacts import canonical_digest as _digest
from tests.support.rationals import rational_payload as _q

from jacobian.contracts.exact_domain_verification import inline_exact_value_digest
from jacobian.contracts.linear import (
    LinearRationalSolutionResult,
    LinearRationalSystem,
)
from jacobian_checkers.linear import check_rational_solution


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
            "coefficients": {"entries": [[_q(2), _q(1)], [_q(1), _q(-1)]]},
            "rhs": [_q(5), _q(1)],
        }
    ).model_dump(mode="json")
    request_payload = {
        "system": system,
        "resource_budget": {"budget_version": "1", "wall_seconds": 5},
    }
    result = LinearRationalSolutionResult(
        values=(_q(2), _q(1)),
    ).model_dump(mode="json")
    semantics = _artifact(
        "e",
        {"kind": "linear.rational_solution"},
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
        "operation_id": "linear.rational_solution.compute",
        "claim": claim,
        "candidate": candidate,
        "semantics": semantics,
        "scope": None,
        "expected_bindings": bindings,
    }


def test_checker_accepts_vector_satisfying_every_exact_equation() -> None:
    decision = check_rational_solution(_request())

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["arithmetic"] == "EXACT_RATIONAL"


@pytest.mark.parametrize("mutation", ("wrong_value", "changed_system", "extra_field"))
def test_checker_rejects_tampered_inline_solution(mutation: str) -> None:
    request = _request()
    if mutation == "wrong_value":
        request["candidate"]["payload"]["values"][0] = _q(3)
    elif mutation == "changed_system":
        request["claim"]["payload"]["system"]["rhs"][0] = _q(6)
    else:
        request["candidate"]["payload"]["unexpected"] = True

    decision = check_rational_solution(deepcopy(request))

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_forged_inline_binding() -> None:
    request = _request()
    request["expected_bindings"]["candidate_digest"] = "sha256:" + "9" * 64

    decision = check_rational_solution(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
