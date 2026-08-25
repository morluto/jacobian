"""Majorization operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.majorization._models import (
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
from jacobian.math.majorization._operations import (
    compute_birkhoff_decomposition,
    compute_doubly_stochastic_check,
    compute_majorization_check,
    compute_schur_horn_check,
    compute_t_transform_sequence,
    compute_weak_majorization_check,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "majorization.check.compute",
        "Check majorization relation",
        "Check if vector x majorizes vector y (ordinary majorization): "
        "after sorting both in nonincreasing order, verify prefix-sum "
        "inequalities and total-sum equality.",
        MajorizationCheckRequest,
        MajorizationCheckResult,
        compute_majorization_check,
        "linear-algebra",
        "majorization",
        "exact",
        examples=(
            example(
                "majorizes",
                "Check that (3, 1) majorizes (2, 2) with labelled rational vectors.",
                {
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
    _op(
        "majorization.weak_check.compute",
        "Check weak majorization",
        "Check weak majorization: sub (x weakly submajorizes y) or "
        "super (x weakly supermajorizes y) without total-sum equality.",
        WeakMajorizationCheckRequest,
        WeakMajorizationCheckResult,
        compute_weak_majorization_check,
        "linear-algebra",
        "majorization",
        "exact",
        examples=(
            example(
                "weak_sub",
                "Check weak submajorization for labelled rational vectors.",
                {
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
    _op(
        "majorization.t_transform.compute",
        "Compute T-transform sequence",
        "Compute an exact T-transform sequence from x to y when x "
        "majorizes y. Returns steps, intermediate vectors, and the "
        "composed doubly stochastic matrix.",
        TTransformSequenceRequest,
        TTransformSequenceResult,
        compute_t_transform_sequence,
        "linear-algebra",
        "majorization",
        "exact",
        examples=(
            example(
                "t_transform_4_0_to_2_2",
                "Compute T-transform sequence from (4,0) to (2,2) where (4,0) majorizes (2,2).",
                {
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
    _op(
        "majorization.doubly_stochastic.check",
        "Check doubly stochastic matrix",
        "Check if a rational square matrix is doubly stochastic "
        "(non-negative, rows and columns sum to 1).",
        DoublyStochasticCheckRequest,
        DoublyStochasticCheckResult,
        compute_doubly_stochastic_check,
        "linear-algebra",
        "majorization",
        "exact",
        examples=(
            example(
                "identity",
                "Check the 2x2 identity matrix.",
                {
                    "matrix": {
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                        ],
                    },
                },
            ),
        ),
    ),
    _op(
        "majorization.birkhoff_decomposition.compute",
        "Birkhoff-von Neumann decomposition",
        "Decompose a doubly stochastic matrix into a convex combination "
        "of permutation matrices using the greedy matching algorithm.",
        BirkhoffDecompositionRequest,
        BirkhoffDecompositionResult,
        compute_birkhoff_decomposition,
        "linear-algebra",
        "majorization",
        "exact",
        examples=(
            example(
                "birkhoff_2x2_average",
                "Decompose the 2x2 averaging matrix [[1/2,1/2],[1/2,1/2]] which is doubly stochastic.",
                {
                    "matrix": {
                        "entries": [
                            [{"num": "1", "den": "2"}, {"num": "1", "den": "2"}],
                            [{"num": "1", "den": "2"}, {"num": "1", "den": "2"}],
                        ],
                    },
                },
            ),
        ),
    ),
    _op(
        "majorization.schur_horn.check",
        "Check Schur-Horn feasibility",
        "Check if a diagonal vector is realizable as the diagonal of a "
        "Hermitian matrix with given eigenvalues (Schur-Horn theorem).",
        SchurHornCheckRequest,
        SchurHornCheckResult,
        compute_schur_horn_check,
        "linear-algebra",
        "majorization",
        "exact",
        examples=(
            example(
                "feasible",
                "Check if (1, 0) is feasible for eigenvalues (2, -1).",
                {
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
