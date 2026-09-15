"""Exact canonical-form operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.matrices.canonical_forms._models import (
    CentralizerResult,
    InvariantFactorProfileResult,
    MatrixPolynomialEvaluationRequest,
    MatrixPolynomialEvaluationResult,
    MatrixPolynomialRemainderRequest,
    MatrixPolynomialRemainderResult,
    MinimalPolynomialResult,
    PrimaryDecompositionResult,
    RationalCanonicalFormResult,
    SimilarityRequest,
    SimilarityResult,
    SquareMatrixRequest,
)
from jacobian.math.matrices.canonical_forms.operations import (
    _minimal_polynomial_components,
    _primary_decomposition_components,
    _rational_canonical_components,
    centralizer_basis,
    decide_similarity,
    evaluate_matrix_polynomial_value,
    invariant_factor_profile,
    reduce_matrix_polynomial,
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


def compute_matrix_polynomial_remainder(
    matrix: RationalMatrix,
    polynomial: RationalPolynomial,
) -> MatrixPolynomialRemainderResult:
    minimal, quotient, remainder = reduce_matrix_polynomial(matrix, polynomial)
    return MatrixPolynomialRemainderResult._from_kernel(
        matrix=matrix,
        polynomial=polynomial,
        minimal_polynomial=minimal,
        quotient=quotient,
        remainder=remainder,
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


def _run_matrix_polynomial_remainder(
    request: MatrixPolynomialRemainderRequest,
) -> MatrixPolynomialRemainderResult:
    return compute_matrix_polynomial_remainder(request.matrix, request.polynomial)


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


def _run_invariant_factor_profile(
    request: SquareMatrixRequest,
) -> InvariantFactorProfileResult:
    return invariant_factor_profile(request.matrix)


def _run_similarity(request: SimilarityRequest) -> SimilarityResult:
    return decide_similarity(request.left, request.right)


def _run_centralizer(request: SquareMatrixRequest) -> CentralizerResult:
    return centralizer_basis(request.matrix)


TOOLS: MathTools = (
    MathTool(
        operation_id="matrix.polynomial.remainder.compute",
        title="Reduce a rational polynomial modulo a matrix minimal polynomial",
        description=(
            "Compute the exact Euclidean quotient and remainder of a univariate "
            "QQ polynomial by the source matrix's minimal polynomial. The "
            "returned source variable is retained and the remainder degree is "
            "strictly smaller than the minimal-polynomial degree."
        ),
        request_type=MatrixPolynomialRemainderRequest,
        result_type=MatrixPolynomialRemainderResult,
        run=_run_matrix_polynomial_remainder,
        tags=("matrix", "polynomial", "minimal-polynomial", "remainder", "exact"),
        examples=(
            OperationExample(
                name="nilpotent_reduction",
                description=(
                    "Reduce t^3+2t+1 modulo the minimal polynomial t^2 of a "
                    "nilpotent Jordan block; the matrix must be nonempty and "
                    "square and the polynomial must be univariate."
                ),
                input={
                    "matrix": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}],
                        ],
                    },
                    "polynomial": {
                        "variables": ["t"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [3],
                                },
                                {
                                    "coefficient": {"num": "2", "den": "1"},
                                    "exponents": [1],
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
    MathTool(
        operation_id="matrix.invariant_factors.compute",
        title="Compute the invariant-factor profile of a square rational matrix",
        description="Return the complete monic divisibility chain with the product "
        "relation against the characteristic polynomial and the terminal "
        "minimal-polynomial relation over QQ.",
        request_type=SquareMatrixRequest,
        result_type=InvariantFactorProfileResult,
        run=_run_invariant_factor_profile,
        tags=("matrix", "invariant-factors", "exact", "complete"),
        examples=(
            OperationExample(
                name="diagonal_distinct_factors",
                description="Invariant factors of diag(2,3) are a single (t-2)(t-3); the matrix must be square.",
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
        operation_id="matrix.similarity.decide",
        title="Decide similarity of two rational matrices",
        description="Decide similarity from complete invariant-factor data; matrices are "
        "similar exactly when canonical representatives agree over QQ.",
        request_type=SimilarityRequest,
        result_type=SimilarityResult,
        run=_run_similarity,
        tags=("matrix", "similarity", "exact"),
        examples=(
            OperationExample(
                name="similar_diagonal_permutation",
                description="diag(2,3) is similar to diag(3,2); both matrices must be square of equal order.",
                input={
                    "left": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": "2", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "3", "den": "1"}],
                        ],
                    },
                    "right": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": "3", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "2", "den": "1"}],
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.centralizer.compute",
        title="Compute an exact centralizer basis",
        description="Return a complete basis of {X : AX = XA} as a nullspace of "
        "I(x)A - A^T(x)I; the basis always contains the identity.",
        request_type=SquareMatrixRequest,
        result_type=CentralizerResult,
        run=_run_centralizer,
        tags=("matrix", "centralizer", "exact", "complete"),
        examples=(
            OperationExample(
                name="scalar_centralizer",
                description="Centralizer of diag(2,3) has dimension 2; the matrix must be square.",
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
