"""Exact construction from an independent commutator-closed matrix span."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import factorial, gcd

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


def _require_qq_matrix_domains(matrices: tuple[RationalMatrix, ...]) -> None:
    if any(matrix.domain != "QQ" for matrix in matrices):
        raise OperationDomainValidationError(
            location=("matrices",),
            code="lie_algebra.matrix_span_domain",
            message="matrix span entries must use the QQ matrix domain",
        )


def _admit(request: LieMatrixSpanRequest) -> tuple[int, int, int]:
    if not isinstance(request, LieMatrixSpanRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="lie_algebra.matrix_span_request",
            message="request must be a typed matrix-span value",
        )
    matrices = getattr(request, "matrices", None)
    if type(matrices) is not tuple or not 1 <= len(matrices) <= 8:
        raise OperationDomainValidationError(
            location=("matrices",),
            code="lie_algebra.matrix_span_dimension",
            message="matrix span dimension must be 1..8",
        )
    if any(not isinstance(matrix, RationalMatrix) for matrix in matrices):
        raise OperationDomainValidationError(
            location=("matrices",),
            code="lie_algebra.matrix_span_shape",
            message="every span element must be a rational matrix",
        )
    _require_qq_matrix_domains(matrices)
    order = getattr(matrices[0], "row_count", None)
    if type(order) is not int:
        raise OperationDomainValidationError(
            location=("matrices", 0),
            code="lie_algebra.matrix_span_shape",
            message="matrix shape is malformed",
        )
    dimension = len(matrices)
    if not 1 <= order <= 8 or any(
        type(getattr(matrix, "row_count", None)) is not int
        or not 1 <= matrix.row_count <= 8
        for matrix in matrices
    ):
        raise OperationDomainValidationError(
            location=("matrices",),
            code="lie_algebra.matrix_span_order",
            message="matrix order must be 1..8",
        )
    if any(
        matrix.row_count != order
        or type(getattr(matrix, "column_count", None)) is not int
        or matrix.column_count != order
        or type(getattr(matrix, "entries", None)) is not tuple
        or len(matrix.entries) != order
        or any(type(row) is not tuple or len(row) != order for row in matrix.entries)
        for matrix in matrices
    ):
        raise OperationDomainValidationError(
            location=("matrices",),
            code="lie_algebra.matrix_span_shape",
            message="all matrices must have the same square order",
        )
    for matrix_index, matrix in enumerate(matrices):
        for row_index, row in enumerate(matrix.entries):
            for column_index, entry in enumerate(row):
                if not isinstance(entry, CanonicalRational):
                    raise OperationDomainValidationError(
                        location=("matrices", matrix_index, "entries", row_index, column_index),
                        code="lie_algebra.matrix_span_shape",
                        message="matrix entries must be canonical rational scalars",
                    )
                if (
                    type(entry.num) is not int
                    or type(entry.den) is not int
                    or entry.den <= 0
                    or gcd(abs(entry.num), entry.den) != 1
                ):
                    raise OperationDomainValidationError(
                        location=("matrices", matrix_index, "entries", row_index, column_index),
                        code="lie_algebra.matrix_span_rational",
                        message="matrix entries must be reduced canonical rationals",
                    )
    max_numerator = max(
        abs(entry.num)
        for matrix in matrices
        for row in matrix.entries
        for entry in row
    )
    max_denominator = max(
        entry.den
        for matrix in matrices
        for row in matrix.entries
        for entry in row
    )
    input_numerator_digits = decimal_digit_width(max_numerator)
    input_denominator_digits = decimal_digit_width(max_denominator)
    # Treat unit factors as zero growth; any non-unit factor can carry across a
    # decimal boundary, including one-digit factors such as 7 and 8.
    input_numerator_growth = (
        0 if max_numerator <= 1 else input_numerator_digits
    )
    input_denominator_growth = 0 if max_denominator == 1 else input_denominator_digits
    input_digits = max(input_numerator_digits, input_denominator_digits)
    if input_digits > 64:
        raise OperationResourceAdmissionError(
            location=("matrices",),
            code="lie_algebra.matrix_span_input_height",
            message="input matrix entries are limited to 64 decimal digits",
        )
    pairs = dimension * (dimension - 1) // 2
    # Bound rational products and sums before matrix expansion. Each
    # commutator entry sums 2*order products; unrelated product denominators
    # are safely charged as their full product. Determinant bounds then charge
    # every permutation term and its common denominator in the pivot systems.
    product_count = 2 * order
    commutator_denominator_digits = (
        1 + product_count * 2 * input_denominator_growth
    )
    commutator_numerator_digits = (
        1
        + 2 * input_numerator_growth
        + (product_count - 1) * 2 * input_denominator_growth
        + decimal_digit_width(product_count)
    )
    commutator_digits = max(
        commutator_numerator_digits, commutator_denominator_digits
    )
    determinant_terms = factorial(dimension)
    pivot_term_denominator_growth = dimension * input_denominator_growth
    determinant_denominator_digits = (
        1 + determinant_terms * pivot_term_denominator_growth
    )
    determinant_numerator_digits = (
        1
        + dimension * input_numerator_growth
        + (determinant_terms - 1) * pivot_term_denominator_growth
        + decimal_digit_width(determinant_terms)
    )
    replacement_term_denominator_growth = (
        (dimension - 1) * input_denominator_growth
        + commutator_denominator_digits
        - 1
    )
    replacement_term_numerator_growth = (
        (dimension - 1) * input_numerator_growth
        + commutator_numerator_digits
        - 1
    )
    replacement_denominator_digits = (
        1 + determinant_terms * replacement_term_denominator_growth
    )
    replacement_numerator_digits = (
        1
        + replacement_term_numerator_growth
        + (determinant_terms - 1) * replacement_term_denominator_growth
        + decimal_digit_width(determinant_terms)
    )
    coordinate_digits = (
        max(
            replacement_numerator_digits + determinant_denominator_digits,
            replacement_denominator_digits + determinant_numerator_digits,
        )
        + 1
        if pairs
        else 1
    )
    # Output coefficients are rational coordinates in the pivot basis. Refuse
    # conservatively before any RREF or commutator expansion.
    if coordinate_digits > MAX_MATRIX_SPAN_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("matrices",),
            code="lie_algebra.matrix_span_result_height",
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
