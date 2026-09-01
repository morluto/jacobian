"""Exact canonical-form operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.matrices.canonical_forms._models import (
    MatrixPolynomialEvaluationRequest,
    MatrixPolynomialEvaluationResult,
    MinimalPolynomialResult,
    PrimaryDecompositionResult,
    RationalCanonicalFormResult,
    SquareMatrixRequest,
)
from jacobian.math.matrices.canonical_forms.operations import (
    _minimal_polynomial_components,
    _primary_decomposition_components,
    _rational_canonical_components,
    evaluate_matrix_polynomial_value,
)
from jacobian.math.matrices.values import RationalMatrix
from jacobian.math.polynomials.values import RationalPolynomial


def compute_matrix_polynomial_evaluation(
    matrix: RationalMatrix,
    polynomial: RationalPolynomial,
) -> MatrixPolynomialEvaluationResult:
    return MatrixPolynomialEvaluationResult._from_kernel(
        matrix=matrix,
        polynomial=polynomial,
        value=evaluate_matrix_polynomial_value(matrix, polynomial),
    )


def compute_minimal_polynomial(matrix: RationalMatrix) -> MinimalPolynomialResult:
    minimal, characteristic = _minimal_polynomial_components(matrix)
    return MinimalPolynomialResult._from_kernel(
        matrix=matrix,
        minimal_polynomial=minimal,
        characteristic_polynomial=characteristic,
    )


def compute_rational_canonical_form(
    matrix: RationalMatrix,
) -> RationalCanonicalFormResult:
    invariant_factors, characteristic, minimal = _rational_canonical_components(matrix)
    return RationalCanonicalFormResult._from_kernel(
        matrix=matrix,
        invariant_factors=invariant_factors,
        characteristic_polynomial=characteristic,
        minimal_polynomial=minimal,
    )


def compute_primary_decomposition(
    matrix: RationalMatrix,
) -> PrimaryDecompositionResult:
    components, minimal = _primary_decomposition_components(matrix)
    return PrimaryDecompositionResult._from_kernel(
        matrix=matrix,
        components=components,
        minimal_polynomial=minimal,
    )


def _run_matrix_polynomial_evaluation(
    request: MatrixPolynomialEvaluationRequest,
) -> MatrixPolynomialEvaluationResult:
    return compute_matrix_polynomial_evaluation(request.matrix, request.polynomial)


def _run_minimal_polynomial(request: SquareMatrixRequest) -> MinimalPolynomialResult:
    return compute_minimal_polynomial(request.matrix)


def _run_rational_canonical_form(
    request: SquareMatrixRequest,
) -> RationalCanonicalFormResult:
    return compute_rational_canonical_form(request.matrix)


def _run_primary_decomposition(
    request: SquareMatrixRequest,
) -> PrimaryDecompositionResult:
    return compute_primary_decomposition(request.matrix)


TOOLS: MathTools = (
    MathTool(
        operation_id="matrix.polynomial.evaluate.compute",
        title="Evaluate an exact rational polynomial at a square matrix",
        description="Compute f(A) over QQ by bounded exact Horner evaluation. The result "
        "retains the source matrix and canonical one-variable rational polynomial "
        "alongside the exact evaluated matrix.",
        request_type=MatrixPolynomialEvaluationRequest,
        result_type=MatrixPolynomialEvaluationResult,
        run=_run_matrix_polynomial_evaluation,
        tags=("matrix", "polynomial", "functional-calculus", "exact"),
        examples=(
            OperationExample(
                name="rotation_annihilator",
                description="Evaluate t^2 + 1 at the rational quarter-turn matrix, obtaining "
                "the zero matrix; the matrix must be square and the polynomial "
                "must declare exactly one variable over QQ.",
                input={
                    "matrix": {
                        "domain": "QQ",
                        "entries": [
                            [
                                {"num": "0", "den": "1"},
                                {"num": "-1", "den": "1"},
                            ],
                            [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                        ],
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
    MathTool(
        operation_id="matrix.minimal_polynomial.compute",
        title="Compute the exact minimal polynomial of a square rational matrix",
        description="Compute the monic minimal polynomial over QQ by the Krylov/nullspace "
        "method, returning the exact minimal and characteristic polynomials.",
        request_type=SquareMatrixRequest,
        result_type=MinimalPolynomialResult,
        run=_run_minimal_polynomial,
        tags=("matrix", "minimal-polynomial", "exact"),
        examples=(
            OperationExample(
                name="nilpotent_block",
                description="Minimal polynomial of a 2x2 nilpotent Jordan block is t^2.",
                input={
                    "matrix": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}],
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.rational_canonical_form.compute",
        title="Compute the exact rational (Frobenius) canonical form",
        description="Compute the invariant factors, characteristic polynomial, and minimal "
        "polynomial of a square rational matrix via Smith normal form of tI - A "
        "over QQ[t].",
        request_type=SquareMatrixRequest,
        result_type=RationalCanonicalFormResult,
        run=_run_rational_canonical_form,
        tags=("matrix", "rational-canonical-form", "exact"),
        examples=(
            OperationExample(
                name="diagonal_distinct",
                description="Rational canonical form of diag(2,3) has one invariant factor (t-2)(t-3).",
                input={
                    "matrix": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": "2", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "3", "den": "1"}],
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.primary_decomposition.compute",
        title="Decompose the minimal polynomial into irreducible-power components",
        description="Factor the minimal polynomial over QQ into its irreducible-power "
        "components and return each monic component polynomial.",
        request_type=SquareMatrixRequest,
        result_type=PrimaryDecompositionResult,
        run=_run_primary_decomposition,
        tags=("matrix", "primary-decomposition", "exact"),
        examples=(
            OperationExample(
                name="diagonal_distinct",
                description="Primary decomposition of diag(2,3) gives (t-2) and (t-3).",
                input={
                    "matrix": {
                        "domain": "QQ",
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
