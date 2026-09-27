"""Exact bounded cellular complexes for crystallographic mapping tori."""

from __future__ import annotations

from itertools import combinations
from math import comb, factorial
from typing import NoReturn

from pydantic import ValidationError

from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.crystallographic._models import (
    MAX_FINITE_ORDER_EXPONENT,
    MAX_MAPPING_TORUS_ENTRY_DIGITS,
    MAX_MAPPING_TORUS_RANK,
)
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.topology.chain_complexes.operations import mapping_cone
from jacobian.math.topology.chain_complexes.values import (
    MAX_MATRIX_ENTRY_CHARS,
    ChainComplexValue,
    ChainMapValue,
    CoefficientRing,
)

_IntegerRows = tuple[tuple[int, ...], ...]
_IntegerMatrix = tuple[tuple[int, ...], ...]


def _domain_error(
    location: tuple[str | int, ...], reason: str, message: str
) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=f"crystallographic.mapping_torus.{reason}",
        message=message,
    )


def _resource_error(
    reason: str,
    message: str,
    *,
    location: tuple[str | int, ...] = ("linear_part",),
) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=location,
        code=f"crystallographic.mapping_torus.{reason}",
        message=message,
    )


def _admit_matrix(value: object) -> IntegerMatrix:
    if not isinstance(value, IntegerMatrix):
        _domain_error(
            ("linear_part",),
            "matrix_type",
            "linear_part must be a canonical integer matrix",
        )
    try:
        matrix = IntegerMatrix.model_validate(
            value.model_dump(mode="python", warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("linear_part",),
            code="crystallographic.mapping_torus.matrix_shape",
            message="linear_part must satisfy the complete integer-matrix contract",
        ) from exc
    if matrix.row_count != matrix.column_count:
        _domain_error(
            ("linear_part",),
            "matrix_not_square",
            "linear_part must be square",
        )
    if matrix.row_count > MAX_MAPPING_TORUS_RANK:
        _resource_error(
            "rank_bound",
            f"mapping-torus rank {matrix.row_count} exceeds the "
            f"{MAX_MAPPING_TORUS_RANK}-dimensional exterior-power envelope",
        )
    return matrix


def _admit_request(
    linear_part: object, finite_order_exponent: object
) -> tuple[IntegerMatrix, int]:
    matrix = _admit_matrix(linear_part)
    if not isinstance(finite_order_exponent, int) or isinstance(
        finite_order_exponent, bool
    ):
        _domain_error(
            ("finite_order_exponent",),
            "exponent_type",
            "finite_order_exponent must be an exact integer",
        )
    exponent = finite_order_exponent
    if not 1 <= exponent <= MAX_FINITE_ORDER_EXPONENT:
        _resource_error(
            "exponent_bound",
            f"finite_order_exponent must lie between 1 and {MAX_FINITE_ORDER_EXPONENT}",
            location=("finite_order_exponent",),
        )

    largest_digits = max(
        (decimal_digit_width(entry) for row in matrix.entries for entry in row),
        default=1,
    )
    if largest_digits > MAX_MAPPING_TORUS_ENTRY_DIGITS:
        _resource_error(
            "coefficient_height_bound",
            "linear-part entries exceed the 32-digit exact-power envelope",
        )

    rank = matrix.row_count
    # A k-th exterior-power entry is a k-minor: at most k! products of k
    # source entries. This bound covers every exact output coefficient before
    # any determinant or mapping-cone matrix is materialized.
    minor_digits = max(
        (
            max(1, degree * largest_digits + decimal_digit_width(factorial(degree)))
            for degree in range(rank + 1)
        ),
        default=1,
    )
    exterior_cells = comb(2 * rank, rank)
    cone_ranks = tuple(comb(rank + 1, degree) for degree in range(rank + 2))
    cone_cells = sum(
        cone_ranks[index] * cone_ranks[index + 1]
        for index in range(len(cone_ranks) - 1)
    )
    result_character_bound = exterior_cells * (minor_digits + 2) + cone_cells
    if result_character_bound > MAX_MATRIX_ENTRY_CHARS:
        _resource_error(
            "result_size_bound",
            "the exterior-power coefficient envelope exceeds the canonical "
            "chain-complex result budget",
        )

    # Binary matrix powering only materializes powers A^e with e <= m. Each
    # entry is bounded by n^(e-1) B^e. The fixed rank/exponent/height envelope
    # therefore bounds every exact intermediate before the finite-order check.
    power_digits = exponent * largest_digits + max(0, exponent - 1) * max(
        1, decimal_digit_width(max(1, rank))
    )
    if power_digits > 10_000:  # Defensive coupling if constants change.
        _resource_error(
            "power_height_bound",
            "the finite-order matrix-power envelope exceeds 10000 digits",
        )
    return matrix, exponent


def _identity(order: int) -> _IntegerRows:
    return tuple(
        tuple(1 if row == column else 0 for column in range(order))
        for row in range(order)
    )


def _multiply(left: _IntegerRows, right: _IntegerRows) -> _IntegerRows:
    order = len(left)
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(order))
            for column in range(order)
        )
        for row in range(order)
    )


def _power(matrix: _IntegerRows, exponent: int) -> _IntegerRows:
    result = _identity(len(matrix))
    factor = matrix
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _multiply(result, factor)
        remaining >>= 1
        if remaining:
            factor = _multiply(factor, factor)
    return result


def _determinant(rows: _IntegerRows) -> int:
    """Return one small exact determinant by fraction-free elimination."""

    order = len(rows)
    if order == 0:
        return 1
    matrix = [list(row) for row in rows]
    sign = 1
    previous_pivot = 1
    for pivot_index in range(order - 1):
        pivot_row = next(
            (row for row in range(pivot_index, order) if matrix[row][pivot_index] != 0),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != pivot_index:
            matrix[pivot_index], matrix[pivot_row] = (
                matrix[pivot_row],
                matrix[pivot_index],
            )
            sign = -sign
        pivot = matrix[pivot_index][pivot_index]
        for row in range(pivot_index + 1, order):
            for column in range(pivot_index + 1, order):
                numerator = (
                    matrix[row][column] * pivot
                    - matrix[row][pivot_index] * matrix[pivot_index][column]
                )
                matrix[row][column] = numerator // previous_pivot
            matrix[row][pivot_index] = 0
        previous_pivot = pivot
    return sign * matrix[-1][-1]


def _exterior_power(matrix: _IntegerRows, degree: int) -> _IntegerRows:
    rank = len(matrix)
    axes = tuple(combinations(range(rank), degree))
    return tuple(
        tuple(
            _determinant(
                tuple(
                    tuple(matrix[row][column] for column in source_axes)
                    for row in target_axes
                )
            )
            for source_axes in axes
        )
        for target_axes in axes
    )


def _mapping_components(matrix: _IntegerRows) -> tuple[_IntegerMatrix, ...]:
    components: list[_IntegerMatrix] = []
    for degree in range(len(matrix) + 1):
        exterior = _exterior_power(matrix, degree)
        components.append(
            tuple(
                tuple(
                    entry - (1 if row == column else 0)
                    for column, entry in enumerate(entries)
                )
                for row, entries in enumerate(exterior)
            )
        )
    return tuple(components)


def _torus_complex(rank: int) -> ChainComplexValue:
    basis_sizes = tuple(comb(rank, degree) for degree in range(rank + 1))
    differentials = tuple(
        tuple(
            tuple(0 for _ in range(basis_sizes[degree + 1]))
            for _ in range(basis_sizes[degree])
        )
        for degree in range(rank)
    )
    return ChainComplexValue(
        coefficient_ring=CoefficientRing.INTEGER,
        degree_min=0,
        degree_max=rank,
        basis_sizes=basis_sizes,
        differential_matrices=differentials,
    )


def mapping_torus_chain_complex(
    linear_part: IntegerMatrix,
    finite_order_exponent: int,
) -> ChainComplexValue:
    """Construct the integral cellular complex of a flat mapping torus.

    For an integral matrix ``A`` and an admitted exponent ``m`` satisfying
    ``A^m = I``, the affine action ``(x,t) -> (Ax,t+1)`` defines a torsion-free
    crystallographic group and compact flat mapping torus. The returned based
    complex is the mapping cone of ``Λ* A - I`` on the standard torus cellular
    complex, and composes unchanged with integral chain-complex homology.
    """

    admitted, exponent = _admit_request(linear_part, finite_order_exponent)
    rows = tuple(tuple(entry for entry in row) for row in admitted.entries)
    if _power(rows, exponent) != _identity(admitted.row_count):
        _domain_error(
            ("finite_order_exponent",),
            "finite_order_relation",
            "finite_order_exponent does not satisfy linear_part^m = I",
        )
    torus = _torus_complex(admitted.row_count)
    return mapping_cone(
        ChainMapValue(
            source=torus,
            target=torus,
            map_matrices=_mapping_components(rows),
        )
    ).value


__all__ = ["mapping_torus_chain_complex"]
