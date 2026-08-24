"""Native exact operations on subsystem-bound rational Hermitian matrices."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math import matrices
from jacobian.math._exact_linear_algebra import symmetric_inertia
from jacobian.math.matrices import _conversions as conversions
from jacobian.math.matrices.subsystems._models import (
    NegativeQuadraticWitness,
    PsdInertia,
    PsdOrderRequest,
    PsdOrderResult,
    SubsystemKroneckerProductRequest,
    SubsystemPartialTraceRequest,
)
from jacobian.math.matrices.subsystems.values import (
    FactorizedHermitianMatrix,
    MatrixSubsystem,
    partial_trace_entries,
)
from jacobian.math.matrices.values import RationalMatrix

__all__ = [
    "kronecker_product",
    "partial_trace",
    "psd_order",
]


def _fractions(matrix: FactorizedHermitianMatrix) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(entry.as_fraction() for entry in row) for row in matrix.matrix.entries
    )


def _rational_matrix(entries: tuple[tuple[Fraction, ...], ...]) -> RationalMatrix:
    return RationalMatrix(
        entries=tuple(
            tuple(CanonicalRational.from_fraction(entry) for entry in row)
            for row in entries
        )
    )


def _factorized(
    entries: tuple[tuple[Fraction, ...], ...],
    *,
    factors: tuple[MatrixSubsystem, ...],
) -> FactorizedHermitianMatrix:
    return FactorizedHermitianMatrix(matrix=_rational_matrix(entries), factors=factors)


def _negative_direction(
    matrix: tuple[tuple[Fraction, ...], ...],
) -> tuple[Fraction, ...] | None:
    """Return one exact negative direction, if this symmetric form has one.

    Symmetric elimination gives a rational congruence to a diagonal pivot or a
    two-dimensional off-diagonal pivot.  The recursive vector is transported
    back through that congruence, so a returned vector satisfies ``v^T A v <
    0`` exactly rather than serving as a numerical heuristic.
    """

    dimension = len(matrix)
    diagonal_pivot = next(
        (index for index in range(dimension) if matrix[index][index] != 0),
        None,
    )
    if diagonal_pivot is not None:
        pivot = matrix[diagonal_pivot][diagonal_pivot]
        if pivot < 0:
            return tuple(
                Fraction(1) if index == diagonal_pivot else Fraction(0)
                for index in range(dimension)
            )
        remaining = tuple(
            index for index in range(dimension) if index != diagonal_pivot
        )
        schur = tuple(
            tuple(
                matrix[row][column]
                - matrix[row][diagonal_pivot] * matrix[diagonal_pivot][column] / pivot
                for column in remaining
            )
            for row in remaining
        )
        direction = _negative_direction(schur)
        if direction is None:
            return None
        pivot_coordinate = (
            -sum(
                (
                    matrix[diagonal_pivot][index] * value
                    for index, value in zip(remaining, direction, strict=True)
                ),
                Fraction(0),
            )
            / pivot
        )
        result = [Fraction(0)] * dimension
        result[diagonal_pivot] = pivot_coordinate
        for index, value in zip(remaining, direction, strict=True):
            result[index] = value
        return tuple(result)

    for first in range(dimension):
        for second in range(first + 1, dimension):
            off_diagonal = matrix[first][second]
            if off_diagonal == 0:
                continue
            # On this two-coordinate restriction, choose s=1 and
            # t=-(|c|+1)/(2b), giving 2*b*t + c < 0 exactly.
            result = [Fraction(0)] * dimension
            result[first] = -(abs(matrix[second][second]) + 1) / (2 * off_diagonal)
            result[second] = Fraction(1)
            return tuple(result)
    return None


def kronecker_product(
    left: FactorizedHermitianMatrix,
    right: FactorizedHermitianMatrix,
) -> FactorizedHermitianMatrix:
    """Compute one exact product while concatenating the ordered factors."""

    SubsystemKroneckerProductRequest(left=left, right=right)
    if len(left.matrix.entries) * len(right.matrix.entries) > 16:
        raise ValueError("Kronecker product dimension exceeds the 16 bound")
    factors = (*left.factors, *right.factors)
    if len(factors) > 4 or len({factor.label for factor in factors}) != len(factors):
        raise ValueError("Kronecker product requires unique factors within the bound")
    product_matrix = matrices.kronecker_product(
        conversions.rational_matrix_to_sympy(left.matrix),
        conversions.rational_matrix_to_sympy(right.matrix),
    )
    return FactorizedHermitianMatrix(
        matrix=conversions.rational_matrix_from_sympy(product_matrix),
        factors=factors,
    )


def partial_trace(
    matrix: FactorizedHermitianMatrix,
    traced_factor_labels: tuple[str, ...],
) -> FactorizedHermitianMatrix:
    """Trace named factors from a product basis, retaining source factor order."""

    SubsystemPartialTraceRequest(
        matrix=matrix,
        traced_factor_labels=traced_factor_labels,
    )
    if not traced_factor_labels or len(set(traced_factor_labels)) != len(
        traced_factor_labels
    ):
        raise ValueError("partial trace requires distinct traced subsystem labels")
    positions = {
        factor.label: position for position, factor in enumerate(matrix.factors)
    }
    if not set(traced_factor_labels) <= set(positions):
        raise ValueError("each traced subsystem label must occur in matrix.factors")
    expected_order = tuple(
        factor.label
        for factor in matrix.factors
        if factor.label in traced_factor_labels
    )
    if traced_factor_labels != expected_order:
        raise ValueError("traced subsystem labels must follow source factor order")
    kept_positions = tuple(
        position
        for position, factor in enumerate(matrix.factors)
        if factor.label not in traced_factor_labels
    )
    return _factorized(
        partial_trace_entries(matrix, traced_factor_labels),
        factors=tuple(matrix.factors[position] for position in kept_positions),
    )


def psd_order(
    left: FactorizedHermitianMatrix,
    right: FactorizedHermitianMatrix,
) -> PsdOrderResult:
    """Decide the exact rational Löwner order ``left <= right``."""

    PsdOrderRequest(left=left, right=right)
    if left.factors != right.factors:
        raise ValueError("PSD order requires exactly equal subsystem factor axes")
    difference_entries = tuple(
        tuple(
            right_entry - left_entry
            for left_entry, right_entry in zip(left_row, right_row, strict=True)
        )
        for left_row, right_row in zip(_fractions(left), _fractions(right), strict=True)
    )
    difference = _factorized(difference_entries, factors=left.factors)
    positive, negative, zero = symmetric_inertia(difference_entries)  # type: ignore[arg-type]
    is_less_or_equal = negative == 0
    witness = None
    if not is_less_or_equal:
        direction = _negative_direction(difference_entries)
        if direction is None:
            raise RuntimeError(
                "exact inertia found a negative direction without a witness"
            )
        quadratic_value = sum(
            (
                direction[row] * difference_entries[row][column] * direction[column]
                for row in range(len(direction))
                for column in range(len(direction))
            ),
            Fraction(0),
        )
        witness = NegativeQuadraticWitness(
            vector=tuple(CanonicalRational.from_fraction(value) for value in direction),
            quadratic_value=CanonicalRational.from_fraction(quadratic_value),
        )
    return PsdOrderResult(
        left=left,
        right=right,
        difference=difference,
        inertia=PsdInertia(n_positive=positive, n_negative=negative, n_zero=zero),
        is_less_or_equal=is_less_or_equal,
        negative_witness=witness,
    )
