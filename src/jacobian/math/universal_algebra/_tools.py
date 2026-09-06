"""Universal-algebra operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.universal_algebra import operations as native
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
from jacobian.math.universal_algebra.values import FiniteAlgebraHomomorphism


def compute_evaluate(request: EvaluateRequest) -> EvaluateResult:
    return EvaluateResult(
        algebra=request.algebra,
        term=request.term,
        assignment=request.assignment,
        value=native.evaluate_term(
            request.algebra,
            request.term,
            dict(enumerate(request.assignment)),
        ),
    )


def compute_equation_profile(request: EquationProfileRequest) -> EquationProfileResult:
    return native.equation_profile(
        request.algebra, request.left, request.right, request.variable_count
    )


def compute_generated_subalgebra(request: SubalgebraRequest) -> SubalgebraResult:
    return native.generated_subalgebra(request.algebra, request.generators)


def compute_homomorphism_profile(
    request: HomomorphismProfileRequest,
) -> HomomorphismProfileResult:
    return native.homomorphism_profile(request.carrier_map)


def compute_congruence(request: CongruenceRequest) -> CongruenceResult:
    return native.congruence_check(request.algebra, request.partition)


def compute_quotient(request: QuotientRequest) -> FiniteAlgebraHomomorphism:
    return native.quotient(request.algebra, request.partition)


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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="universal_algebra.term.evaluate.compute",
        title="Evaluate a source-bound term under a complete assignment",
        description="Return the exact carrier value t^A(alpha) for a finite algebra A and "
        "a complete assignment alpha. Every accepted call is deterministic "
        "and complete.",
        request_type=EvaluateRequest,
        result_type=EvaluateResult,
        run=compute_evaluate,
        tags=("universal-algebra", "term-evaluation", "exact"),
        examples=(
            OperationExample(
                name="and_01",
                description="Evaluate AND(x0, x1) with x0=0, x1=1 in a 2-element Boolean algebra.",
                input={"algebra": _ALGEBRA, "term": _TERM, "assignment": [0, 1]},
            ),
        ),
    ),
    MathTool(
        operation_id="universal_algebra.equation.profile.compute",
        title="Evaluate s = t over all assignments",
        description="Return HOLDS with the satisfying assignment count, or FAILS with "
        "the first counterassignment and exact left/right values. This "
        "generalizes magma identity calculation to an arbitrary finite "
        "signature.",
        request_type=EquationProfileRequest,
        result_type=EquationProfileResult,
        run=compute_equation_profile,
        tags=("universal-algebra", "equation-profile", "exact"),
        examples=(
            OperationExample(
                name="idempotence_and",
                description="Check AND(x,x) = x in the 2-element Boolean algebra.",
                input={
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
    ),
    MathTool(
        operation_id="universal_algebra.subalgebra.generated.compute",
        title="Compute the least subalgebra containing the generating set",
        description="Return the least subalgebra containing the supplied carrier subset by "
        "finite closure under all basic operations and nullary constants. "
        "Output includes the canonical closed carrier subset and closure rounds.",
        request_type=SubalgebraRequest,
        result_type=SubalgebraResult,
        run=compute_generated_subalgebra,
        tags=("universal-algebra", "subalgebra", "exact"),
        examples=(
            OperationExample(
                name="generated_by_0",
                description="Generated subalgebra of {0} in the 2-element Boolean algebra.",
                input={"algebra": _ALGEBRA, "generators": [0]},
            ),
        ),
    ),
    MathTool(
        operation_id="universal_algebra.map.homomorphism_profile.compute",
        title="Profile a supplied finite-algebra carrier map",
        description="Check every basic-operation table cell under one total carrier map. "
        "Return a reusable checked homomorphism with canonical kernel and image, "
        "or the first exact preservation obstruction in deterministic signature "
        "and source-tuple order.",
        request_type=HomomorphismProfileRequest,
        result_type=HomomorphismProfileResult,
        run=compute_homomorphism_profile,
        tags=("universal-algebra", "homomorphism", "carrier-map", "exact"),
        examples=(
            OperationExample(
                name="boolean_identity_map",
                description="Check the identity carrier map between two copies of the "
                "2-element Boolean algebra; source and target operation "
                "identifiers and arities must match exactly and the map must "
                "cover every source carrier position.",
                input={
                    "carrier_map": {
                        "source": _ALGEBRA,
                        "target": _ALGEBRA,
                        "mapping": [0, 1],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="universal_algebra.congruence.check.compute",
        title="Check whether a carrier partition is a congruence",
        description="Return whether a carrier partition is a compatible equivalence "
        "relation (congruence). A congruence theta satisfies: if x_j theta "
        "y_j for every argument j, then f(x_1,...,x_r) theta f(y_1,...,y_r) "
        "for every basic operation.",
        request_type=CongruenceRequest,
        result_type=CongruenceResult,
        run=compute_congruence,
        tags=("universal-algebra", "congruence", "exact"),
        examples=(
            OperationExample(
                name="trivial_congruence",
                description="The universal partition {{0, 1}} is a congruence.",
                input={"algebra": _ALGEBRA, "partition": [[0, 1]]},
            ),
        ),
    ),
    MathTool(
        operation_id="universal_algebra.quotient.compute",
        title="Compute the quotient algebra A/theta",
        description="Return the canonical checked homomorphism from a finite algebra onto "
        "the quotient induced by a congruence. The target carrier is the set "
        "of blocks, and the retained source, target, and mapping pass directly "
        "to homomorphism-profile consumers.",
        request_type=QuotientRequest,
        result_type=FiniteAlgebraHomomorphism,
        run=compute_quotient,
        tags=("universal-algebra", "quotient", "exact"),
        examples=(
            OperationExample(
                name="trivial_quotient",
                description="The quotient by the universal congruence is a one-element algebra.",
                input={"algebra": _ALGEBRA, "partition": [[0, 1]]},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
