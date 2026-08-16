"""Tests for bounded finite-model finding."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.finite_model import (
    FiniteModelFindRequest,
    FiniteModelSignature,
    FunctionSymbol,
)
from jacobian.domains.finite_model.operations import compute_finite_model_find


def test_associativity_satisfiable_on_order_2():
    """A binary function on 2 elements can be associative."""
    request = FiniteModelFindRequest.model_validate({
        "signature": {
            "functions": [{"name": "f", "arity": 2}],
            "relations": [],
        },
        "axioms": [
            {"name": "associativity", "smtlib": "(forall x y z: f(f(x,y),z) = f(x,f(y,z)))"},
        ],
        "carrier_order": 2,
        "timeout_ms": 5000,
    })
    result = compute_finite_model_find(request)
    assert result.status == "SATISFIABLE"
    assert len(result.function_tables) == 1
    assert all(
        0 <= v < 2 for v in result.function_tables[0].values
    )


def test_associativity_satisfiable_on_order_3():
    """A binary function on 3 elements can be associative."""
    request = FiniteModelFindRequest.model_validate({
        "signature": {
            "functions": [{"name": "mul", "arity": 2}],
            "relations": [],
        },
        "axioms": [
            {"name": "associativity", "smtlib": "forall"},
        ],
        "carrier_order": 3,
        "timeout_ms": 5000,
    })
    result = compute_finite_model_find(request)
    assert result.status == "SATISFIABLE"


def test_commutativity_satisfiable_on_order_2():
    """A commutative binary function on 2 elements exists."""
    request = FiniteModelFindRequest.model_validate({
        "signature": {
            "functions": [{"name": "g", "arity": 2}],
            "relations": [],
        },
        "axioms": [
            {"name": "commutativity", "smtlib": "forall"},
        ],
        "carrier_order": 2,
        "timeout_ms": 5000,
    })
    result = compute_finite_model_find(request)
    assert result.status == "SATISFIABLE"


def test_idempotency_satisfiable_on_order_2():
    """An idempotent binary function on 2 elements exists."""
    request = FiniteModelFindRequest.model_validate({
        "signature": {
            "functions": [{"name": "h", "arity": 2}],
            "relations": [],
        },
        "axioms": [
            {"name": "idempotency", "smtlib": "forall"},
        ],
        "carrier_order": 2,
        "timeout_ms": 5000,
    })
    result = compute_finite_model_find(request)
    assert result.status == "SATISFIABLE"


def test_operation_discoverable():
    """The operation should be discoverable via the factory."""
    from jacobian.domains.finite_model import finite_model_operations

    ops = finite_model_operations()
    assert any(op.operation_id == "model.find.finite" for op in ops)


def test_duplicate_function_names_rejected():
    """Duplicate function names should fail validation."""
    with pytest.raises(ValidationError, match="duplicate"):
        FiniteModelSignature.model_validate({
            "functions": [
                {"name": "f", "arity": 2},
                {"name": "f", "arity": 1},
            ]
        })


def test_duplicate_relation_names_rejected():
    """Duplicate relation names should fail validation."""
    with pytest.raises(ValidationError, match="duplicate"):
        FiniteModelSignature.model_validate({
            "functions": [{"name": "f", "arity": 0}],
            "relations": [
                {"name": "R", "arity": 1},
                {"name": "R", "arity": 2},
            ],
        })


def test_arity_bounded():
    """Function arity should be bounded."""
    with pytest.raises(ValidationError):
        FunctionSymbol.model_validate({"name": "f", "arity": 5})
