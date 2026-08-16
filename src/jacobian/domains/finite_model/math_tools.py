"""MathTool declarations for bounded finite-model finding."""

from __future__ import annotations

from jacobian.contracts.finite_model import (
    FiniteModelFindRequest,
    FiniteModelFindResult,
)
from jacobian.domains._examples import example
from jacobian.domains.finite_model.operations import compute_finite_model_find
from jacobian.math_tools import MathTool


FINITE_MODEL_OPERATIONS: tuple[MathTool, ...] = (
    MathTool(
        operation_id="model.find.finite",
        version="1",
        title="Find one finite model for a bounded first-order claim",
        description=(
            "Use Z3 SMT to find one finite model (or countermodel) for a "
            "bounded first-order or equational claim with a carrier-order "
            "bound.  Returns a satisfying assignment of function tables or "
            "an unsatisfiable/unknown certificate."
        ),
        request_type=FiniteModelFindRequest,
        result_type=FiniteModelFindResult,
        run=compute_finite_model_find,
        tags=(
            "logic",
            "finite-model",
            "smt",
            "z3",
            "bounded",
            "first-order",
            "equational",
        ),
        examples=(
            example(
                "associative_semigroup",
                "Find a model satisfying associativity for a binary function on 3 elements.",
                {
                    "signature": {
                        "functions": [{"name": "f", "arity": 2}],
                        "relations": [],
                    },
                    "axioms": [
                        {
                            "name": "associativity",
                            "smtlib": "(forall x y z: f(f(x,y),z) = f(x,f(y,z)))",
                        }
                    ],
                    "carrier_order": 3,
                },
            ),
        ),
    ),
)

__all__ = ["FINITE_MODEL_OPERATIONS"]
