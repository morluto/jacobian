"""Exact finite-dimensional Lie-algebra basis transport over QQ."""

from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import Any

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.lie_algebras._models import (
    MAX_STRUCTURE_COEFFICIENT_DIGITS,
    FiniteDimensionalLieAlgebra,
    StructureConstant,
)
from jacobian.math.lie_algebras.basis_transport._models import (
    MAX_BASIS_CHANGE_DIMENSION,
    MAX_BASIS_CHANGE_INPUT_DIGITS,
    MAX_BASIS_CHANGE_INTERMEDIATE_DIGITS,
    MAX_BASIS_CHANGE_WORK,
    LieBasisChangeRequest,
    LieBasisChangeResult,
)
from jacobian.math.lie_algebras.operations import _admit_lie_algebra, _bracket_table
from jacobian.math.matrices.values import (
    RationalMatrix,
    rational_matrix_from_fractions,
)


def _rational_inverse(
    entries: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Invert a nonempty rational square matrix with the exact FLINT kernel."""

    from flint import fmpq, fmpq_mat

    backend = fmpq_mat(
        [[fmpq(value.numerator, value.denominator) for value in row] for row in entries]
    )
    try:
        inverse = backend.inv()
    except ZeroDivisionError as error:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="lie_algebra.singular_basis_change",
            message="basis-change matrix must be invertible",
        ) from error
    return tuple(
        tuple(
            Fraction(int(inverse[row, column].p), int(inverse[row, column].q))
            for column in range(inverse.ncols())
        )
        for row in range(inverse.nrows())
    )


def _dot(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> Fraction:
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def _bracket(
    first: tuple[Fraction, ...],
    second: tuple[Fraction, ...],
    table: dict[tuple[int, int], dict[int, Fraction]],
    dimension: int,
) -> tuple[Fraction, ...]:
    result = [Fraction(0) for _ in range(dimension)]
    for i in range(dimension):
        for j in range(dimension):
            scale = first[i] * second[j]
            if scale:
                for k, coefficient in table.get((i, j), {}).items():
                    result[k] += scale * coefficient
    return tuple(result)


def _admit_basis_change(request: LieBasisChangeRequest) -> tuple[int, int]:
    """Bound inversion, all transformed brackets, and retained exact output."""

    dimension = len(request.algebra.basis)
    if not 1 <= dimension <= MAX_BASIS_CHANGE_DIMENSION:
        raise OperationResourceAdmissionError(
            location=("algebra", "basis"),
            code="lie_algebra.dimension_bound",
            message=f"basis transport supports dimensions through {MAX_BASIS_CHANGE_DIMENSION}",
        )
    _admit_lie_algebra(request.algebra)
    fractions = tuple(
        tuple(value.as_fraction() for value in row) for row in request.matrix.entries
    )
    height = max(
        (
            max(
                decimal_digit_width(value.numerator),
                decimal_digit_width(value.denominator),
            )
            for row in fractions
            for value in row
        ),
        default=1,
    )
    if height > MAX_BASIS_CHANGE_INPUT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="lie_algebra.basis_change_scalar_bound",
            message=(
                "basis-change matrix entries are limited to "
                f"{MAX_BASIS_CHANGE_INPUT_DIGITS} decimal digits"
            ),
        )
    work = dimension**5 * height**2
    # Clear the n^2 entry denominators, then use determinant/cofactor bounds
    # for the rational inverse. The second bound admits transformed bracket
    # intermediates before the first bracket is expanded.
    common_denominator_height = dimension**2 * height
    integer_entry_height = common_denominator_height + height
    factorial_height = decimal_digit_width(factorial(dimension))
    inverse_height = (
        dimension * integer_entry_height + common_denominator_height + factorial_height
    )
    bracket_height = (
        2 * dimension**2 * height
        + MAX_STRUCTURE_COEFFICIENT_DIGITS
        + dimension.bit_length()
    )
    output_height = (
        dimension * (inverse_height + bracket_height) + dimension.bit_length()
    )
    if (
        work > MAX_BASIS_CHANGE_WORK
        or inverse_height > MAX_CANONICAL_RATIONAL_DIGITS
        or output_height > MAX_BASIS_CHANGE_INTERMEDIATE_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="lie_algebra.basis_change_work_bound",
            message="basis transport exceeds its admitted exact work or intermediate-height bound",
        )
    return work, output_height


def lie_algebra_change_basis(
    algebra: FiniteDimensionalLieAlgebra | dict[str, Any],
    basis: tuple[str, ...] | list[str],
    matrix: RationalMatrix | dict[str, Any],
) -> LieBasisChangeResult:
    """Return structure constants in a new basis and both exact coordinate maps.

    ``matrix[:, j]`` is the vector of the j-th new basis vector in the source
    basis. Thus target coordinates map to source coordinates by ``P``, while
    source coordinates map to target coordinates by ``P^-1``.
    """

    request = LieBasisChangeRequest.model_validate(
        {"algebra": algebra, "basis": basis, "matrix": matrix}
    )
    _admit_basis_change(request)
    source = request.algebra
    dimension = len(source.basis)
    p = tuple(
        tuple(value.as_fraction() for value in row) for row in request.matrix.entries
    )
    inverse = _rational_inverse(p)
    table = _bracket_table(source)
    columns = tuple(
        tuple(p[row][column] for row in range(dimension)) for column in range(dimension)
    )
    constants: list[StructureConstant] = []
    for i in range(dimension):
        for j in range(i + 1, dimension):
            old_coordinates = _bracket(columns[i], columns[j], table, dimension)
            target_coordinates = tuple(
                _dot(tuple(inverse[row][k] for k in range(dimension)), old_coordinates)
                for row in range(dimension)
            )
            for k, coefficient in enumerate(target_coordinates):
                if not coefficient:
                    continue
                if (
                    decimal_digit_width(abs(coefficient.numerator)) > 64
                    or decimal_digit_width(coefficient.denominator) > 64
                ):
                    raise OperationResourceAdmissionError(
                        location=("result", "structure_constants"),
                        code="lie_algebra.basis_change_output_bound",
                        message="transported structure constants exceed 64 decimal digits",
                    )
                try:
                    canonical = CanonicalRational.from_fraction(coefficient)
                    require_bounded_rational(
                        canonical,
                        max_digits=64,
                        label="transported structure constant",
                    )
                except ValueError as error:
                    raise OperationResourceAdmissionError(
                        location=("result", "structure_constants"),
                        code="lie_algebra.basis_change_output_bound",
                        message="transported structure constants exceed 64 decimal digits",
                    ) from error
                constants.append(
                    StructureConstant(i=i, j=j, k=k, coefficient=canonical)
                )
    target = FiniteDimensionalLieAlgebra.model_validate(
        {"basis": request.basis, "structure_constants": constants}
    )
    return LieBasisChangeResult._from_kernel(
        source,
        target,
        rational_matrix_from_fractions(p),
        rational_matrix_from_fractions(inverse),
    )


__all__ = ["lie_algebra_change_basis"]
