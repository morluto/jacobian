"""Thin SymPy projections for exact matrix operations."""

from __future__ import annotations

from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math import matrices
from jacobian.math.matrices import _conversions as conversions
from jacobian.math.matrices._operation_models import (
    CharacteristicPolynomialResult,
    IntegerMatrixRequest,
    MatrixAdjugateResult,
    MatrixDeterminantRequest,
    MatrixDeterminantResult,
    MatrixInverseResult,
    MatrixKroneckerProductRequest,
    MatrixKroneckerProductResult,
    MatrixPartialTraceRequest,
    MatrixPartialTraceResult,
    MatrixPermanentRequest,
    MatrixPermanentResult,
    MatrixProductResult,
    MatrixRankRequest,
    MatrixRankResult,
    MatrixTraceResult,
    NonsingularIntegerMatrixRequest,
    NullspaceResult,
    RationalLinearSolveRequest,
    RationalLinearSolveResult,
    RationalMatrixProductRequest,
    RationalMatrixRequest,
    RrefResult,
    SquareIntegerMatrixRequest,
    SquareRationalMatrixRequest,
)
from jacobian.math.matrices.operations import MatrixSingularError
from jacobian.math.matrices.values import RationalMatrix, SmithNormalForm


def compute_determinant(
    request: MatrixDeterminantRequest,
) -> MatrixDeterminantResult:
    determinant = matrices.determinant(
        conversions.rational_matrix_to_sympy(request.matrix)
    )
    return MatrixDeterminantResult(
        determinant=conversions.rational_from_sympy(determinant)
    )


def _rref_replay(matrix: RationalMatrix) -> tuple[Any, tuple[int, ...]]:
    """Return the exact RREF and pivot columns of a retained source matrix."""
    reduced, pivots = matrices.rref(conversions.rational_matrix_to_sympy(matrix))
    return reduced, tuple(int(pivot) for pivot in pivots)


def _rank_replay(matrix: RationalMatrix) -> tuple[int, tuple[int, ...]]:
    """Return the exact rank and RREF pivot columns of a retained source matrix."""
    rank, pivots = matrices.rank(conversions.rational_matrix_to_sympy(matrix))
    return int(rank), tuple(int(pivot) for pivot in pivots)


def _system_rank_replay(
    matrix: RationalMatrix, rhs: tuple[CanonicalRational, ...]
) -> tuple[int, int]:
    """Return the exact coefficient and augmented ranks of a retained system."""
    import sympy

    source = conversions.rational_matrix_to_sympy(matrix)
    column = sympy.Matrix([sympy.Rational(value.as_fraction()) for value in rhs])
    coefficient_rank, _pivots = matrices.rank(source)
    augmented_rank, _augmented_pivots = matrices.rank(source.row_join(column))
    return coefficient_rank, augmented_rank


def compute_rank(request: MatrixRankRequest) -> MatrixRankResult:
    rank, pivot_columns = matrices.rank(
        conversions.rational_matrix_to_sympy(request.matrix)
    )
    return MatrixRankResult._from_kernel(
        matrix=request.matrix,
        rank=rank,
        pivot_columns=tuple(int(column) for column in pivot_columns),
    )


def compute_rref(request: RationalMatrixRequest) -> RrefResult:
    reduced, pivots = matrices.rref(
        conversions.rational_matrix_to_sympy(request.matrix)
    )
    columns = reduced.cols
    pivot_columns = tuple(int(column) for column in pivots)
    return RrefResult._from_kernel(
        matrix=request.matrix,
        reduced_matrix=conversions.rational_matrix_from_sympy(reduced),
        rank=len(pivot_columns),
        pivot_columns=pivot_columns,
        free_columns=tuple(
            column for column in range(columns) if column not in pivot_columns
        ),
    )


def compute_nullspace(request: RationalMatrixRequest) -> NullspaceResult:
    import sympy

    matrix = conversions.rational_matrix_to_sympy(request.matrix)
    reduced, pivots = matrices.rref(matrix)
    pivot_columns = tuple(int(column) for column in pivots)
    free_columns = tuple(
        column for column in range(matrix.cols) if column not in pivot_columns
    )
    pivot_row_by_column = {
        pivot_column: row for row, pivot_column in enumerate(pivot_columns)
    }
    basis: list[tuple[CanonicalRational, ...]] = []
    for free_column in free_columns:
        vector = [sympy.S.Zero] * matrix.cols
        vector[free_column] = sympy.S.One
        for pivot_column, row in pivot_row_by_column.items():
            vector[pivot_column] = -reduced[row, free_column]
        basis.append(tuple(conversions.rational_from_sympy(value) for value in vector))
    return NullspaceResult._from_kernel(
        matrix=request.matrix,
        ambient_dimension=matrix.cols,
        rank=len(pivot_columns),
        nullity=len(basis),
        basis_vectors=tuple(basis),
        free_columns=free_columns,
    )


def compute_characteristic_polynomial(
    request: SquareRationalMatrixRequest,
) -> CharacteristicPolynomialResult:
    polynomial = matrices.characteristic_polynomial(
        conversions.rational_matrix_to_sympy(request.matrix), "lambda"
    )
    return CharacteristicPolynomialResult(
        degree=polynomial.degree(),
        coefficients_descending=tuple(
            conversions.rational_from_sympy(coefficient)
            for coefficient in polynomial.all_coeffs()
        ),
    )


def compute_smith_normal_form(
    request: IntegerMatrixRequest,
) -> SmithNormalForm:
    raw = matrices.smith_normal_form(
        conversions.integer_matrix_to_sympy(request.matrix)
    )
    return conversions.smith_normal_form_from_sympy(raw)


def compute_inverse(request: NonsingularIntegerMatrixRequest) -> MatrixInverseResult:
    try:
        inverse = matrices.inverse(conversions.integer_matrix_to_sympy(request.matrix))
    except MatrixSingularError as exc:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="matrix.singular_matrix",
            message="matrix is singular; inverse does not exist",
        ) from exc
    return MatrixInverseResult(inverse=conversions.rational_matrix_from_sympy(inverse))


def compute_trace(request: SquareIntegerMatrixRequest) -> MatrixTraceResult:
    return MatrixTraceResult(
        trace=format_canonical_integer(
            matrices.trace(conversions.integer_matrix_to_sympy(request.matrix))
        )
    )


def compute_product(request: RationalMatrixProductRequest) -> MatrixProductResult:
    left = conversions.rational_matrix_to_sympy(request.left)
    right = conversions.rational_matrix_to_sympy(request.right)
    product = matrices.multiply(left, right)
    return MatrixProductResult(
        product=conversions.rational_matrix_from_sympy(product),
        left_rows=left.rows,
        inner_dimension=left.cols,
        right_columns=right.cols,
    )


def compute_rational_linear_solve(
    request: RationalLinearSolveRequest,
) -> RationalLinearSolveResult:
    import sympy

    source = conversions.rational_matrix_to_sympy(request.matrix)
    rhs = sympy.Matrix([sympy.Rational(value.as_fraction()) for value in request.rhs])
    try:
        solution, parameters = matrices.solve_linear_system(source, rhs)
    except ValueError:
        return RationalLinearSolveResult._from_kernel(
            matrix=request.matrix,
            rhs=request.rhs,
            outcome="INCONSISTENT",
        )
    if parameters.rows:
        return RationalLinearSolveResult._from_kernel(
            matrix=request.matrix,
            rhs=request.rhs,
            outcome="NON_UNIQUE",
        )
    return RationalLinearSolveResult._from_kernel(
        matrix=request.matrix,
        rhs=request.rhs,
        outcome="UNIQUE",
        solution=tuple(conversions.rational_from_sympy(value) for value in solution),
    )


def compute_adjugate(request: SquareIntegerMatrixRequest) -> MatrixAdjugateResult:
    adjugate = matrices.adjugate(conversions.integer_matrix_to_sympy(request.matrix))
    return MatrixAdjugateResult(
        adjugate=conversions.integer_matrix_from_sympy(adjugate)
    )


def compute_permanent(request: MatrixPermanentRequest) -> MatrixPermanentResult:
    value = matrices.permanent(conversions.rational_matrix_to_sympy(request.matrix))
    return MatrixPermanentResult(
        permanent=conversions.rational_from_sympy(value),
    )


def compute_kronecker_product(
    request: MatrixKroneckerProductRequest,
) -> MatrixKroneckerProductResult:
    left = conversions.rational_matrix_to_sympy(request.left)
    right = conversions.rational_matrix_to_sympy(request.right)
    product = matrices.kronecker_product(left, right)
    return MatrixKroneckerProductResult(
        product=conversions.rational_matrix_from_sympy(product),
        left_rows=left.rows,
        left_columns=left.cols,
        right_rows=right.rows,
        right_columns=right.cols,
    )


def compute_partial_trace(
    request: MatrixPartialTraceRequest,
) -> MatrixPartialTraceResult:
    matrix = conversions.rational_matrix_to_sympy(request.matrix)
    reduced = matrices.partial_trace(
        matrix,
        request.traced_dimension,
        request.kept_dimension,
    )
    return MatrixPartialTraceResult(
        reduced_matrix=conversions.rational_matrix_from_sympy(reduced),
        traced_dimension=request.traced_dimension,
        kept_dimension=request.kept_dimension,
    )


def verify_rref_result(result: RrefResult) -> bool:
    """Independently replay one bounded RREF claim from its retained source."""

    expected, pivots = _rref_replay(result.matrix)
    return tuple(int(pivot) for pivot in pivots) == result.pivot_columns and (
        conversions.rational_matrix_to_sympy(result.reduced_matrix) == expected
    )


def verify_rank_result(result: MatrixRankResult) -> bool:
    """Independently replay one bounded rank and pivot claim."""

    rank, pivots = _rank_replay(result.matrix)
    return rank == result.rank and pivots == result.pivot_columns


def verify_nullspace_result(result: NullspaceResult) -> bool:
    """Check a bounded fundamental nullspace basis against its source."""

    matrix = conversions.rational_matrix_to_sympy(result.matrix)
    _reduced, pivots = matrices.rref(matrix)
    pivot_columns = tuple(int(column) for column in pivots)
    free_columns = tuple(
        column for column in range(matrix.cols) if column not in pivot_columns
    )
    if (
        result.rank != len(pivot_columns)
        or result.free_columns != free_columns
        or result.nullity != len(free_columns)
    ):
        return False
    for index, vector in enumerate(result.basis_vectors):
        components = [value.as_fraction() for value in vector]
        if any(
            sum(
                coefficient.as_fraction() * component
                for coefficient, component in zip(row, components, strict=True)
            )
            != 0
            for row in result.matrix.entries
        ):
            return False
        own = free_columns[index]
        if components[own] != 1 or any(
            components[column] != 0 for column in free_columns if column != own
        ):
            return False
    return True


def verify_rational_linear_solve_result(result: RationalLinearSolveResult) -> bool:
    """Replay one bounded linear-system outcome and optional exact witness."""

    coefficient_rank, augmented_rank = _system_rank_replay(result.matrix, result.rhs)
    columns = len(result.matrix.entries[0])
    if result.outcome == "UNIQUE":
        if result.solution is None or coefficient_rank != columns:
            return False
        return all(
            sum(
                coefficient.as_fraction() * component.as_fraction()
                for coefficient, component in zip(row, result.solution, strict=True)
            )
            == bound.as_fraction()
            for row, bound in zip(result.matrix.entries, result.rhs, strict=True)
        )
    if result.outcome == "INCONSISTENT":
        return result.solution is None and coefficient_rank < augmented_rank
    return (
        result.solution is None
        and coefficient_rank < columns
        and coefficient_rank == augmented_rank
    )
