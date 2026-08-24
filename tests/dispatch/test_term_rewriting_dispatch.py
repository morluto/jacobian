"""Dispatch input-parsing boundaries for first-order term rewriting."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.dispatch import parse_operation_input
from jacobian.math.term_rewriting._models import (
    CriticalPairsRequest,
    CriticalPairsResult,
    Term,
)
from jacobian.math.term_rewriting._operations import compute_critical_pairs
from jacobian.math.term_rewriting.values import MAX_VARIABLE_LABEL


def _var(symbol: int) -> Term:
    return Term(is_variable=True, symbol=symbol)


def _app(symbol: int, *children: Term) -> Term:
    return Term(is_variable=False, symbol=symbol, children=tuple(children))


def test_transport_depth_boundary_admits_the_deepest_unary_chain():
    # f^30(x) -> x is the deepest rule strict JSON transport carries:
    # each unary node costs one object level plus one children array
    # level inside the request. It must parse end-to-end through math.run
    # input parsing and replay as a complete profile.
    def unary_chain(function_nodes: int) -> Term:
        term = _var(0)
        for _ in range(function_nodes):
            term = _app(0, term)
        return term

    payload = {
        "signature": {"arities": [1]},
        "rules": [
            {
                "lhs": unary_chain(30).model_dump(mode="json"),
                "rhs": {"is_variable": True, "symbol": 0},
            }
        ],
    }
    request = parse_operation_input(CriticalPairsRequest, payload)
    result = compute_critical_pairs(request)
    assert len(result.profile.candidates) == 29
    assert all(candidate.unifiable for candidate in result.profile.candidates)
    assert CriticalPairsResult.model_validate_json(result.model_dump_json()) == (result)


def test_variable_label_bound_is_the_interoperable_integer_maximum():
    # The widest interoperable label admits through math.run input
    # parsing; one beyond it is rejected by the symbol bound itself.
    widest = MAX_VARIABLE_LABEL
    payload = {
        "signature": {"arities": [1]},
        "rules": [
            {
                "lhs": {
                    "is_variable": False,
                    "symbol": 0,
                    "children": [{"is_variable": True, "symbol": widest}],
                },
                "rhs": {"is_variable": True, "symbol": widest},
            }
        ],
    }
    request = parse_operation_input(CriticalPairsRequest, payload)
    result = compute_critical_pairs(request)
    assert result.profile.candidates == ()
    assert result.profile.pairs == ()
    with pytest.raises(ValidationError, match=str(widest)):
        CriticalPairsRequest.model_validate(
            {
                "signature": {"arities": [1]},
                "rules": [
                    {
                        "lhs": {
                            "is_variable": False,
                            "symbol": 0,
                            "children": [{"is_variable": True, "symbol": widest + 1}],
                        },
                        "rhs": {"is_variable": True, "symbol": widest + 1},
                    }
                ],
            }
        )
