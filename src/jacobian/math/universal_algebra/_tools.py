"""Universal-algebra operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.universal_algebra import operations as native
from jacobian.math.universal_algebra._models import (
    CongruenceRequest,
    CongruenceResult,
    CountermodelFindRequest,
    CountermodelFindResult,
    EquationProfileRequest,
    EquationProfileResult,
    EvaluateRequest,
    EvaluateResult,
    HomomorphismProfileRequest,
    HomomorphismProfileResult,
    ImplicationCountermodelCheckRequest,
    ImplicationCountermodelCheckResult,
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


def compute_implication_countermodel_check(
    request: ImplicationCountermodelCheckRequest,
) -> ImplicationCountermodelCheckResult:
    return native.implication_countermodel_check(
        request.algebra, request.premises, request.target
    )


def compute_countermodel_find(
    request: CountermodelFindRequest,
) -> CountermodelFindResult:
    return native.countermodel_find(
        request.premises,
        request.target,
        request.min_order,
        request.max_order,
        request.table_budget,
        request.break_symmetry,
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


# Left-projection magma on {0, 1}: x*y = x. Associative but not commutative,
# so it is a countermodel to (associativity => commutativity).
_COUNTERMODEL_MAGMA = {
    "carrier": ["0", "1"],
    "operations": [{"operation_id": "mul", "arity": 2}],
    "tables": [[0, 0, 1, 1]],
}


def _magma_variable(variable_id: int) -> dict[str, object]:
    return {"nodes": [{"kind": "variable", "variable_id": variable_id}], "root": 0}


# (x*y)*z over operation 0 with flat node indices 0..4.
_ASSOCIATIVE_LEFT = {
    "nodes": [
        {"kind": "variable", "variable_id": 0},
        {"kind": "variable", "variable_id": 1},
        {"kind": "variable", "variable_id": 2},
        {"kind": "application", "operation": 0, "children": [0, 1]},
        {"kind": "application", "operation": 0, "children": [3, 2]},
    ],
    "root": 4,
}


# x*(y*z) over operation 0 with flat node indices 0..4.
_ASSOCIATIVE_RIGHT = {
    "nodes": [
        {"kind": "variable", "variable_id": 0},
        {"kind": "variable", "variable_id": 1},
        {"kind": "variable", "variable_id": 2},
        {"kind": "application", "operation": 0, "children": [1, 2]},
        {"kind": "application", "operation": 0, "children": [0, 3]},
    ],
    "root": 4,
}


# x*y over operation 0.
_PRODUCT_XY = {
    "nodes": [
        {"kind": "variable", "variable_id": 0},
        {"kind": "variable", "variable_id": 1},
        {"kind": "application", "operation": 0, "children": [0, 1]},
    ],
    "root": 2,
}


# y*x over operation 0.
_PRODUCT_YX = {
    "nodes": [
        {"kind": "variable", "variable_id": 0},
        {"kind": "variable", "variable_id": 1},
        {"kind": "application", "operation": 0, "children": [1, 0]},
    ],
    "root": 2,
}


# Benchmark premise (x*y) = (((y*x)*x)*y) over operation 0.
_BENCHMARK_PREMISE_LEFT = {
    "nodes": [
        {"kind": "variable", "variable_id": 0},
        {"kind": "variable", "variable_id": 1},
        {"kind": "application", "operation": 0, "children": [0, 1]},
    ],
    "root": 2,
}
_BENCHMARK_PREMISE_RIGHT = {
    "nodes": [
        {"kind": "variable", "variable_id": 0},
        {"kind": "variable", "variable_id": 1},
        {"kind": "application", "operation": 0, "children": [1, 0]},
        {"kind": "application", "operation": 0, "children": [2, 0]},
        {"kind": "application", "operation": 0, "children": [3, 1]},
    ],
    "root": 4,
}


# Benchmark target ((x*y)*y) = ((y*x)*x) over operation 0.
_BENCHMARK_TARGET_LEFT = {
    "nodes": [
        {"kind": "variable", "variable_id": 0},
        {"kind": "variable", "variable_id": 1},
        {"kind": "application", "operation": 0, "children": [0, 1]},
        {"kind": "application", "operation": 0, "children": [2, 1]},
    ],
    "root": 3,
}
_BENCHMARK_TARGET_RIGHT = {
    "nodes": [
        {"kind": "variable", "variable_id": 0},
        {"kind": "variable", "variable_id": 1},
        {"kind": "application", "operation": 0, "children": [1, 0]},
        {"kind": "application", "operation": 0, "children": [2, 0]},
    ],
    "root": 3,
}


# Idempotent law x*x = x over operation 0 (used for the exhaustion example).
_IDEMPOTENT_LEFT = {
    "nodes": [
        {"kind": "variable", "variable_id": 0},
        {"kind": "application", "operation": 0, "children": [0, 0]},
    ],
    "root": 1,
}
_IDEMPOTENT_RIGHT = {
    "nodes": [{"kind": "variable", "variable_id": 0}],
    "root": 0,
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
        operation_id="universal_algebra.implication.countermodel.check",
        title="Check one finite magma as an equation-implication countermodel",
        description="Exhaust every premise equation and the target equation over "
        "their complete finite assignment spaces in one magma with exactly one "
        "binary operation. Return every premise profile with its first violation, "
        "the target profile with its refutation, and whether the magma is a "
        "countermodel (all premises hold universally and the target fails). "
        "Duplicate premises normalize to their first occurrence; requests whose "
        "complete term-evaluation work exceeds the enumeration budget are rejected.",
        request_type=ImplicationCountermodelCheckRequest,
        result_type=ImplicationCountermodelCheckResult,
        run=compute_implication_countermodel_check,
        tags=("universal-algebra", "countermodel", "exact"),
        discovery_terms=("countermodel", "equation implication", "finite magma"),
        examples=(
            OperationExample(
                name="left_projection_countermodel",
                description="Check the 2-element left-projection magma against "
                "associativity as premise and commutativity as target; the magma "
                "must carry exactly one binary operation and the complete "
                "assignment work must fit the enumeration budget.",
                input={
                    "algebra": _COUNTERMODEL_MAGMA,
                    "premises": [
                        {
                            "left": _ASSOCIATIVE_LEFT,
                            "right": _ASSOCIATIVE_RIGHT,
                        }
                    ],
                    "target": {"left": _PRODUCT_XY, "right": _PRODUCT_YX},
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
    MathTool(
        operation_id="universal_algebra.magma_implication.countermodel.find",
        title="Find a finite-magma equation-implication countermodel",
        description="Enumerate finite magma tables in deterministic order "
        "(orders increasingly, tables row-major with ascending cells) and "
        "check each one against premise equations and a target equation with "
        "the exact countermodel checker. FOUND carries the first countermodel "
        "order and certificate with every smaller searched order exhausted; "
        "EXHAUSTED_UP_TO_BOUND carries the receipt that every table of every "
        "declared order was examined; UNKNOWN carries the spent budget and "
        "current order. A truncated search never yields a negative conclusion.",
        request_type=CountermodelFindRequest,
        result_type=CountermodelFindResult,
        run=compute_countermodel_find,
        tags=("universal-algebra", "countermodel", "exact"),
        discovery_terms=("countermodel", "equation implication", "finite magma"),
        examples=(
            OperationExample(
                name="benchmark_implication_countermodel",
                description="The held-out benchmark implication fails already "
                "in carrier orders 1 and 2: bounded search finds the first "
                "countermodel table with its checker certificate.",
                input={
                    "premises": [
                        {
                            "left": _BENCHMARK_PREMISE_LEFT,
                            "right": _BENCHMARK_PREMISE_RIGHT,
                        }
                    ],
                    "target": {
                        "left": _BENCHMARK_TARGET_LEFT,
                        "right": _BENCHMARK_TARGET_RIGHT,
                    },
                    "min_order": 1,
                    "max_order": 2,
                    "table_budget": 1000,
                    "break_symmetry": False,
                },
            ),
            OperationExample(
                name="idempotence_exhaustion",
                description="Idempotence implies itself: all 17 tables of "
                "orders 1 and 2 are examined with no countermodel.",
                input={
                    "premises": [
                        {"left": _IDEMPOTENT_LEFT, "right": _IDEMPOTENT_RIGHT}
                    ],
                    "target": {"left": _IDEMPOTENT_LEFT, "right": _IDEMPOTENT_RIGHT},
                    "min_order": 1,
                    "max_order": 2,
                    "table_budget": 1000,
                    "break_symmetry": False,
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
