"""Universal-algebra operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.universal_algebra._models import (
    CongruenceRequest,
    CongruenceResult,
    EquationProfileRequest,
    EquationProfileResult,
    EvaluateRequest,
    EvaluateResult,
    HomomorphismProfileRequest,
    HomomorphismProfileResult,
    QuotientRequest,
    SubalgebraRequest,
    SubalgebraResult,
)
from jacobian.math.universal_algebra._operations import (
    compute_congruence,
    compute_equation_profile,
    compute_evaluate,
    compute_generated_subalgebra,
    compute_homomorphism_profile,
    compute_quotient,
)
from jacobian.math.universal_algebra.values import FiniteAlgebraHomomorphism


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


# A 2-element Boolean algebra: carrier {0, 1}, operations AND (binary), OR (binary).
# Table for AND: 0∧0=0, 0∧1=0, 1∧0=0, 1∧1=1. Table for OR: 0OR0=0, 0OR1=1, 1OR0=1, 1OR1=1.
_ALGEBRA = {
    "carrier": ["0", "1"],
    "operations": [
        {"operation_id": "and", "arity": 2},
        {"operation_id": "or", "arity": 2},
    ],
    "tables": [[0, 0, 0, 1], [0, 1, 1, 1]],
}


# Term: AND(x0, x1) — application of operation 0 (and) with two variable children.
# Flat term: node 0 = variable 0, node 1 = variable 1, node 2 = application of op 0 with children (0, 1).
_TERM = {
    "nodes": [
        {"kind": "variable", "variable_id": 0},
        {"kind": "variable", "variable_id": 1},
        {
            "kind": "application",
            "operation": 0,
            "children": [0, 1],
        },
    ],
    "root": 2,
}


UNIVERSAL_ALGEBRA_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "universal_algebra.term.evaluate.compute",
        "Evaluate a source-bound term under a complete assignment",
        "Return the exact carrier value t^A(alpha) for a finite algebra A and "
        "a complete assignment alpha. Every accepted call is deterministic "
        "and complete.",
        EvaluateRequest,
        EvaluateResult,
        compute_evaluate,
        "universal-algebra",
        "term-evaluation",
        "exact",
        examples=(
            example(
                "and_01",
                "Evaluate AND(x0, x1) with x0=0, x1=1 in a 2-element Boolean algebra.",
                {"algebra": _ALGEBRA, "term": _TERM, "assignment": [0, 1]},
            ),
        ),
        version="2",
    ),
    _op(
        "universal_algebra.equation.profile.compute",
        "Evaluate s = t over all assignments",
        "Return HOLDS with the satisfying assignment count, or FAILS with "
        "the first counterassignment and exact left/right values. This "
        "generalizes magma identity calculation to an arbitrary finite "
        "signature.",
        EquationProfileRequest,
        EquationProfileResult,
        compute_equation_profile,
        "universal-algebra",
        "equation-profile",
        "exact",
        examples=(
            example(
                "idempotence_and",
                "Check AND(x,x) = x in the 2-element Boolean algebra.",
                {
                    "algebra": _ALGEBRA,
                    "left": {
                        "nodes": [
                            {"kind": "variable", "variable_id": 0},
                            {
                                "kind": "application",
                                "operation": 0,
                                "children": [0, 0],
                            },
                        ],
                        "root": 1,
                    },
                    "right": {
                        "nodes": [{"kind": "variable", "variable_id": 0}],
                        "root": 0,
                    },
                    "variable_count": 1,
                },
            ),
        ),
        version="2",
    ),
    _op(
        "universal_algebra.subalgebra.generated.compute",
        "Compute the least subalgebra containing the generating set",
        "Return the least subalgebra containing the supplied carrier subset by "
        "finite closure under all basic operations and nullary constants. "
        "Output includes the canonical closed carrier subset and closure rounds.",
        SubalgebraRequest,
        SubalgebraResult,
        compute_generated_subalgebra,
        "universal-algebra",
        "subalgebra",
        "exact",
        examples=(
            example(
                "generated_by_0",
                "Generated subalgebra of {0} in the 2-element Boolean algebra.",
                {"algebra": _ALGEBRA, "generators": [0]},
            ),
        ),
        version="2",
    ),
    _op(
        "universal_algebra.map.homomorphism_profile.compute",
        "Profile a supplied finite-algebra carrier map",
        "Check every basic-operation table cell under one total carrier map. "
        "Return a reusable checked homomorphism with canonical kernel and image, "
        "or the first exact preservation obstruction in deterministic signature "
        "and source-tuple order.",
        HomomorphismProfileRequest,
        HomomorphismProfileResult,
        compute_homomorphism_profile,
        "universal-algebra",
        "homomorphism",
        "carrier-map",
        "exact",
        examples=(
            example(
                "boolean_identity_map",
                "Check the identity carrier map between two copies of the "
                "2-element Boolean algebra; source and target operation "
                "identifiers and arities must match exactly and the map must "
                "cover every source carrier position.",
                {
                    "carrier_map": {
                        "source": _ALGEBRA,
                        "target": _ALGEBRA,
                        "mapping": [0, 1],
                    }
                },
            ),
        ),
    ),
    _op(
        "universal_algebra.congruence.check.compute",
        "Check whether a carrier partition is a congruence",
        "Return whether a carrier partition is a compatible equivalence "
        "relation (congruence). A congruence theta satisfies: if x_j theta "
        "y_j for every argument j, then f(x_1,...,x_r) theta f(y_1,...,y_r) "
        "for every basic operation.",
        CongruenceRequest,
        CongruenceResult,
        compute_congruence,
        "universal-algebra",
        "congruence",
        "exact",
        examples=(
            example(
                "trivial_congruence",
                "The universal partition {{0, 1}} is a congruence.",
                {"algebra": _ALGEBRA, "partition": [[0, 1]]},
            ),
        ),
        version="2",
    ),
    _op(
        "universal_algebra.quotient.compute",
        "Compute the quotient algebra A/theta",
        "Return the canonical checked homomorphism from a finite algebra onto "
        "the quotient induced by a congruence. The target carrier is the set "
        "of blocks, and the retained source, target, and mapping pass directly "
        "to homomorphism-profile consumers.",
        QuotientRequest,
        FiniteAlgebraHomomorphism,
        compute_quotient,
        "universal-algebra",
        "quotient",
        "exact",
        examples=(
            example(
                "trivial_quotient",
                "The quotient by the universal congruence is a one-element algebra.",
                {"algebra": _ALGEBRA, "partition": [[0, 1]]},
            ),
        ),
        version="3",
    ),
)

TOOLS = UNIVERSAL_ALGEBRA_OPERATIONS

__all__ = ["TOOLS"]
