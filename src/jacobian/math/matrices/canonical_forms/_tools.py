"""Exact canonical-form operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.matrices.canonical_forms._models import (
    MatrixPolynomialEvaluationRequest,
    MatrixPolynomialEvaluationResult,
    MinimalPolynomialResult,
    PrimaryDecompositionResult,
    RationalCanonicalFormResult,
    SquareMatrixRequest,
)
from jacobian.math.matrices.canonical_forms._operations import (
    compute_matrix_polynomial_evaluation,
    compute_minimal_polynomial,
    compute_primary_decomposition,
    compute_rational_canonical_form,
)


def canonical_form_operation[
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


TOOLS: MathTools = (
    canonical_form_operation(
        "matrix.polynomial.evaluate.compute",
        "Evaluate an exact rational polynomial at a square matrix",
        "Compute f(A) over QQ by bounded exact Horner evaluation. The result "
        "retains the source matrix and canonical one-variable rational polynomial "
        "and independently replays the defining evaluation.",
        MatrixPolynomialEvaluationRequest,
        MatrixPolynomialEvaluationResult,
        compute_matrix_polynomial_evaluation,
        "matrix",
        "polynomial",
        "functional-calculus",
        "exact",
        examples=(
            example(
                "rotation_annihilator",
                "Evaluate t^2 + 1 at the rational quarter-turn matrix, obtaining "
                "the zero matrix; the matrix must be square and the polynomial "
                "must declare exactly one variable over QQ.",
                {
                    "matrix": {
                        "entries": [
                            [
                                {"num": "0", "den": "1"},
                                {"num": "-1", "den": "1"},
                            ],
                            [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                        ]
                    },
                    "polynomial": {
                        "variables": ["t"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    },
                },
            ),
        ),
    ),
    canonical_form_operation(
        "matrix.minimal_polynomial.compute",
        "Compute the exact minimal polynomial of a square rational matrix",
        "Compute the monic minimal polynomial over QQ by the Krylov/nullspace "
        "method, returning the exact minimal and characteristic polynomials.",
        SquareMatrixRequest,
        MinimalPolynomialResult,
        compute_minimal_polynomial,
        "matrix",
        "minimal-polynomial",
        "exact",
        examples=(
            example(
                "nilpotent_block",
                "Minimal polynomial of a 2x2 nilpotent Jordan block is t^2.",
                {
                    "matrix": {
                        "entries": [
                            [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}],
                        ],
                    },
                },
            ),
        ),
    ),
    canonical_form_operation(
        "matrix.rational_canonical_form.compute",
        "Compute the exact rational (Frobenius) canonical form",
        "Compute the invariant factors, characteristic polynomial, and minimal "
        "polynomial of a square rational matrix via Smith normal form of tI - A "
        "over QQ[t].",
        SquareMatrixRequest,
        RationalCanonicalFormResult,
        compute_rational_canonical_form,
        "matrix",
        "rational-canonical-form",
        "exact",
        examples=(
            example(
                "diagonal_distinct",
                "Rational canonical form of diag(2,3) has one invariant factor (t-2)(t-3).",
                {
                    "matrix": {
                        "entries": [
                            [{"num": "2", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "3", "den": "1"}],
                        ],
                    },
                },
            ),
        ),
    ),
    canonical_form_operation(
        "matrix.primary_decomposition.compute",
        "Decompose the minimal polynomial into irreducible-power components",
        "Factor the minimal polynomial over QQ into its irreducible-power "
        "components and return each monic component polynomial.",
        SquareMatrixRequest,
        PrimaryDecompositionResult,
        compute_primary_decomposition,
        "matrix",
        "primary-decomposition",
        "exact",
        examples=(
            example(
                "diagonal_distinct",
                "Primary decomposition of diag(2,3) gives (t-2) and (t-3).",
                {
                    "matrix": {
                        "entries": [
                            [{"num": "2", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "3", "den": "1"}],
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
