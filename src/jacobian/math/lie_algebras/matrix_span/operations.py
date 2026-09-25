"""Exact construction from an independent commutator-closed matrix span."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import factorial

from jacobian._exact import CanonicalRational
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    StructureConstant,
)
from jacobian.math.lie_algebras.matrix_span._models import (
    MAX_MATRIX_SPAN_OUTPUT_BYTES,
    MAX_MATRIX_SPAN_RESULT_DIGITS,
    LieMatrixSpanRealization,
    LieMatrixSpanRequest,
)
from jacobian.math.matrices.operations import rref_result
from jacobian.math.matrices.values import RationalMatrix, rational_matrix_from_fractions


def _admit(request: LieMatrixSpanRequest) -> tuple[int, int, int]:
    dimension = len(request.matrices)
    order = request.matrices[0].row_count
    if not 1 <= order <= 8 or any(
        matrix.row_count < 1 or matrix.row_count > 8 for matrix in request.matrices
    ):
        raise OperationDomainValidationError(
            location=("matrices",), code="lie_algebra.matrix_span_order",
            message="matrix order must be 1..8",
        )
    if any(
        matrix.row_count != order or matrix.column_count != order
        for matrix in request.matrices
    ):
        raise OperationDomainValidationError(
            location=("matrices",),
            code="lie_algebra.matrix_span_shape",
            message="all matrices must have the same square order",
        )
    input_digits = max(
        max(decimal_digit_width(entry.num), decimal_digit_width(entry.den))
        for matrix in request.matrices
        for row in matrix.entries
        for entry in row
    )
    if input_digits > 64:
        raise OperationResourceAdmissionError(
            location=("matrices",),
            code="lie_algebra.matrix_span_input_height",
            message="input matrix entries are limited to 64 decimal digits",
        )
    pairs = dimension * (dimension - 1) // 2
    # Bound products, sums, exact row reduction and coordinate solving before
    # any matrix multiplication or RREF expansion. Cramer's rule gives a
    # conservative determinant-height ceiling for the d by d pivot system.
    # A sum of 2*order rational products may require a product of their
    # denominators; charge that full rational-height growth, not integer growth.
    commutator_digits = 2 * order * input_digits + decimal_digit_width(2 * order) + 2
    determinant_digits = (
        dimension * dimension * input_digits
        + decimal_digit_width(factorial(dimension))
        + 2
    )
    replacement_digits = (
        dimension * (dimension - 1) * input_digits
        + dimension * commutator_digits
        + decimal_digit_width(factorial(dimension))
        + 2
    )
    coordinate_digits = determinant_digits + replacement_digits + 2
    # Output coefficients are rational coordinates in the pivot basis. Refuse
    # conservatively before any RREF or commutator expansion.
    if coordinate_digits > MAX_MATRIX_SPAN_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("matrices",), code="lie_algebra.matrix_span_result_height",
            message="induced structure constants may exceed the 64-digit result bound",
        )
    work = (
        dimension * order**2 * dimension * input_digits
        + 2 * pairs * order**3
        + pairs * dimension**3 * max(commutator_digits, coordinate_digits)
        + pairs * order**2 * dimension
    )
    output_bytes = 4096 + dimension * order**2 * 160 + pairs * dimension * 160
    if (
        work > 25_000_000
        or coordinate_digits > 32_768
        or output_bytes > MAX_MATRIX_SPAN_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("matrices",),
            code="lie_algebra.matrix_span_work_bound",
            message="matrix span exceeds the admitted exact work or output budget",
        )
    return order, pairs, commutator_digits


def _commutator(left: RationalMatrix, right: RationalMatrix) -> tuple[Fraction, ...]:
    order = left.row_count
    a = tuple(tuple(value.as_fraction() for value in row) for row in left.entries)
    b = tuple(tuple(value.as_fraction() for value in row) for row in right.entries)
    values = []
    for i in range(order):
        for j in range(order):
            values.append(
                sum(
                    (a[i][k] * b[k][j] - b[i][k] * a[k][j] for k in range(order)),
                    Fraction(0),
                )
            )
    return tuple(values)


def _coordinates(
    rows: tuple[tuple[Fraction, ...], ...],
    pivots: tuple[int, ...],
    target: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    """Solve the pivot-coordinate system by bounded exact elimination."""
    size = len(rows)
    augmented = [
        [rows[col][pivots[row]] for col in range(size)] + [target[pivots[row]]]
        for row in range(size)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column]), None
        )
        if pivot is None:
            raise OperationDomainValidationError(
                location=("matrices",),
                code="lie_algebra.matrix_span_dependent",
                message="input matrices must be linearly independent",
            )
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            scale = augmented[row][column]
            if scale:
                augmented[row] = [
                    a - scale * b
                    for a, b in zip(augmented[row], augmented[column], strict=True)
                ]
    return tuple(augmented[row][-1] for row in range(size))


def lie_algebra_from_matrix_span(
    request: LieMatrixSpanRequest,
) -> LieMatrixSpanRealization:
    """Construct the induced Lie algebra from an independent closed QQ span."""
    order, _pair_count, _intermediate_digit_bound = _admit(request)
    dimension = len(request.matrices)
    rows = tuple(
        tuple(
            entry.as_fraction() for matrix_row in matrix.entries for entry in matrix_row
        )
        for matrix in request.matrices
    )
    row_matrix = rational_matrix_from_fractions(rows, column_count=order**2)
    reduced = rref_result(row_matrix)
    if reduced.rank != dimension:
        raise OperationDomainValidationError(
            location=("matrices",),
            code="lie_algebra.matrix_span_dependent",
            message="input matrices must be linearly independent",
        )
    pivots = reduced.pivot_columns
    constants: list[StructureConstant] = []
    labels = tuple(f"M{index}" for index in range(dimension))
    for i, j in combinations(range(dimension), 2):
        commutator = _commutator(request.matrices[i], request.matrices[j])
        coordinates = _coordinates(rows, pivots, commutator)
        if any(
            sum(
                (
                    rows[index][position] * coordinates[index]
                    for index in range(dimension)
                ),
                Fraction(0),
            )
            != commutator[position]
            for position in range(order**2)
        ):
            raise OperationDomainValidationError(
                location=("matrices",),
                code="lie_algebra.matrix_span_not_closed",
                message="the supplied matrix span is not closed under commutator",
            )
        for k, coefficient in enumerate(coordinates):
            if (
                decimal_digit_width(coefficient.numerator)
                > MAX_MATRIX_SPAN_RESULT_DIGITS
                or decimal_digit_width(coefficient.denominator)
                > MAX_MATRIX_SPAN_RESULT_DIGITS
            ):
                raise OperationResourceAdmissionError(
                    location=("matrices", i, j, k),
                    code="lie_algebra.matrix_span_result_height",
                    message="induced structure constants exceed the 64-digit result bound",
                )
            if coefficient:
                constants.append(
                    StructureConstant(
                        i=i,
                        j=j,
                        k=k,
                        coefficient=CanonicalRational.from_fraction(coefficient),
                    )
                )
    algebra = FiniteDimensionalLieAlgebra(
        basis=labels, structure_constants=tuple(constants)
    )
    return LieMatrixSpanRealization(algebra=algebra, matrix_basis=request.matrices)


__all__ = ["lie_algebra_from_matrix_span"]
