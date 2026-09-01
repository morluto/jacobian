"""Majorization operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.analysis.majorization._models import (
    BirkhoffDecompositionRequest,
    BirkhoffDecompositionResult,
    DoublyStochasticCheckRequest,
    DoublyStochasticCheckResult,
    MajorizationCheckRequest,
    MajorizationCheckResult,
    SchurHornCheckRequest,
    SchurHornCheckResult,
    TTransformSequenceRequest,
    TTransformSequenceResult,
    WeakMajorizationCheckRequest,
    WeakMajorizationCheckResult,
)
from jacobian.math.analysis.majorization.operations import (
    birkhoff_decomposition,
    doubly_stochastic_check,
    majorization_check,
    schur_horn_check,
    t_transform_sequence,
    weak_majorization_check,
)


def _majorization_check(request: MajorizationCheckRequest) -> MajorizationCheckResult:
    return majorization_check(request.x, request.y)


def _weak_majorization_check(
    request: WeakMajorizationCheckRequest,
) -> WeakMajorizationCheckResult:
    return weak_majorization_check(request.x, request.y, request.direction)


def _t_transform_sequence(
    request: TTransformSequenceRequest,
) -> TTransformSequenceResult:
    return t_transform_sequence(request.x, request.y)


def _doubly_stochastic_check(
    request: DoublyStochasticCheckRequest,
) -> DoublyStochasticCheckResult:
    return doubly_stochastic_check(request.matrix)


def _birkhoff_decomposition(
    request: BirkhoffDecompositionRequest,
) -> BirkhoffDecompositionResult:
    return birkhoff_decomposition(request.matrix)


def _schur_horn_check(request: SchurHornCheckRequest) -> SchurHornCheckResult:
    return schur_horn_check(request.eigenvalues, request.diagonal)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="majorization.check.compute",
        title="Check majorization relation",
        description="Check if vector x majorizes vector y (ordinary majorization): "
        "after sorting both in nonincreasing order, verify prefix-sum "
        "inequalities and total-sum equality.",
        request_type=MajorizationCheckRequest,
        result_type=MajorizationCheckResult,
        run=_majorization_check,
        tags=("linear-algebra", "majorization", "exact"),
        examples=(
            OperationExample(
                name="majorizes",
                description="Check that (3, 1) majorizes (2, 2) with labelled rational vectors.",
                input={
                    "x": {
                        "labels": ["a", "b"],
                        "values": [{"num": "3", "den": "1"}, {"num": "1", "den": "1"}],
                    },
                    "y": {
                        "labels": ["a", "b"],
                        "values": [{"num": "2", "den": "1"}, {"num": "2", "den": "1"}],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="majorization.weak_check.compute",
        title="Check weak majorization",
        description="Check weak majorization: sub (x weakly submajorizes y) or "
        "super (x weakly supermajorizes y) without total-sum equality.",
        request_type=WeakMajorizationCheckRequest,
        result_type=WeakMajorizationCheckResult,
        run=_weak_majorization_check,
        tags=("linear-algebra", "majorization", "exact"),
        examples=(
            OperationExample(
                name="weak_sub",
                description="Check weak submajorization for labelled rational vectors.",
                input={
                    "x": {
                        "labels": ["a", "b"],
                        "values": [{"num": "4", "den": "1"}, {"num": "1", "den": "1"}],
                    },
                    "y": {
                        "labels": ["a", "b"],
                        "values": [{"num": "2", "den": "1"}, {"num": "2", "den": "1"}],
                    },
                    "direction": "sub",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="majorization.t_transform.compute",
        title="Compute T-transform sequence",
        description="Compute an exact T-transform sequence from x to y when x "
        "majorizes y. Returns steps, intermediate vectors, and the "
        "composed doubly stochastic matrix.",
        request_type=TTransformSequenceRequest,
        result_type=TTransformSequenceResult,
        run=_t_transform_sequence,
        tags=("linear-algebra", "majorization", "exact"),
        examples=(
            OperationExample(
                name="t_transform_4_0_to_2_2",
                description="Compute T-transform sequence from (4,0) to (2,2) where (4,0) majorizes (2,2).",
                input={
                    "x": {
                        "labels": ["a", "b"],
                        "values": [{"num": "4", "den": "1"}, {"num": "0", "den": "1"}],
                    },
                    "y": {
                        "labels": ["a", "b"],
                        "values": [{"num": "2", "den": "1"}, {"num": "2", "den": "1"}],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="majorization.doubly_stochastic.check",
        title="Check doubly stochastic matrix",
        description="Check if a rational square matrix is doubly stochastic "
        "(non-negative, rows and columns sum to 1).",
        request_type=DoublyStochasticCheckRequest,
        result_type=DoublyStochasticCheckResult,
        run=_doubly_stochastic_check,
        tags=("linear-algebra", "majorization", "exact"),
        examples=(
            OperationExample(
                name="identity",
                description="Check the 2x2 identity matrix.",
                input={
                    "matrix": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="majorization.birkhoff_decomposition.compute",
        title="Birkhoff-von Neumann decomposition",
        description="Decompose a doubly stochastic matrix into a convex combination "
        "of permutation matrices using the greedy matching algorithm.",
        request_type=BirkhoffDecompositionRequest,
        result_type=BirkhoffDecompositionResult,
        run=_birkhoff_decomposition,
        tags=("linear-algebra", "majorization", "exact"),
        examples=(
            OperationExample(
                name="birkhoff_2x2_average",
                description="Decompose the 2x2 averaging matrix [[1/2,1/2],[1/2,1/2]] which is doubly stochastic.",
                input={
                    "matrix": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": "1", "den": "2"}, {"num": "1", "den": "2"}],
                            [{"num": "1", "den": "2"}, {"num": "1", "den": "2"}],
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="majorization.schur_horn.check",
        title="Check Schur-Horn feasibility",
        description="Check if a diagonal vector is realizable as the diagonal of a "
        "Hermitian matrix with given eigenvalues (Schur-Horn theorem).",
        request_type=SchurHornCheckRequest,
        result_type=SchurHornCheckResult,
        run=_schur_horn_check,
        tags=("linear-algebra", "majorization", "exact"),
        examples=(
            OperationExample(
                name="feasible",
                description="Check if (1, 0) is feasible for eigenvalues (2, -1).",
                input={
                    "eigenvalues": [
                        {"num": "2", "den": "1"},
                        {"num": "-1", "den": "1"},
                    ],
                    "diagonal": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
