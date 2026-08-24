"""Domain-owned finite set-system discrepancy operations."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

from sympy import ZZ
from sympy.polys.matrices import DomainMatrix

from jacobian._exact import CanonicalRational
from jacobian.math.discrepancy_theory._models import (
    MAX_ROUNDING_INTERMEDIATE_DIGITS,
    DiscrepancyEvalRequest,
    DiscrepancyEvalResult,
    DiscrepancyOptimumRequest,
    DiscrepancyOptimumResult,
    HardConstraintRoundingRequest,
    HardConstraintRoundingResult,
    HardConstraintRowLedger,
    MonitoredColumnLedger,
)


def _primitive_nullspace_direction(
    rows: list[list[int]], variable_count: int
) -> tuple[int, ...]:
    """Return the first primitive fraction-free right-nullspace direction.

    SymPy's exact ``DomainMatrix.nullspace`` orders basis rows by non-pivot
    columns. We fix the remaining scale ambiguity by dividing out the gcd and
    making the first nonzero coordinate positive.
    """

    if not rows:
        return (1,) + (0,) * (variable_count - 1)
    basis = DomainMatrix.from_list(rows, ZZ).nullspace().to_list()
    if not basis:
        raise RuntimeError("admitted retained system unexpectedly has trivial kernel")
    direction = tuple(int(value) for value in basis[0])
    divisor = math.gcd(*(abs(value) for value in direction))
    direction = tuple(value // divisor for value in direction)
    first_nonzero = next(value for value in direction if value)
    if first_nonzero < 0:
        direction = tuple(-value for value in direction)
    return direction


def _require_admitted_step_height(
    common_denominator_digits: int, direction: tuple[int, ...]
) -> int:
    """Admit the next common-denominator factor before an endpoint update."""

    direction_digits = max(
        (len(str(abs(value))) for value in direction if value),
        default=1,
    )
    admitted_direction_digits = len(direction) * len(str(len(direction))) + 1
    if direction_digits > admitted_direction_digits:
        raise RuntimeError("nullspace direction exceeds the admitted minor bound")
    next_digits = common_denominator_digits + direction_digits
    if next_digits > MAX_ROUNDING_INTERMEDIATE_DIGITS:
        raise RuntimeError("admitted rounding step exceeds the rational-height bound")
    return next_digits


def _floating_round(
    source_values: list[Fraction], request: HardConstraintRoundingRequest
) -> tuple[int, ...]:
    """Run the deterministic exact floating-variable algorithm."""

    source = request.source
    values = list(source_values)
    common_denominator_digits = 1 + sum(
        len(value.den) for value in source.values if value.den != "1"
    )
    coordinate_count = len(values)
    column_supports = tuple(set(column.coordinates) for column in source.columns)
    incidences = [0] * coordinate_count
    for support in column_supports:
        for index in support:
            incidences[index] += 1
    maximum_incidence = max(incidences, default=0)
    retention_threshold = 4 * maximum_incidence

    while True:
        fractional = tuple(index for index, value in enumerate(values) if 0 < value < 1)
        if not fractional:
            return tuple(int(value) for value in values)
        fractional_set = set(fractional)

        retained_rows: list[list[int]] = []
        for row in source.rows:
            active = fractional_set.intersection(row.coordinates)
            if active:
                if len(active) < 2:
                    raise RuntimeError(
                        "admitted integral row has one fractional coordinate"
                    )
                retained_rows.append([int(index in active) for index in fractional])
        for support in column_supports:
            active = fractional_set.intersection(support)
            if len(active) > retention_threshold:
                retained_rows.append([int(index in active) for index in fractional])
        if len(retained_rows) >= len(fractional):
            raise RuntimeError("admitted retained system is not rank deficient")

        direction = _primitive_nullspace_direction(retained_rows, len(fractional))
        common_denominator_digits = _require_admitted_step_height(
            common_denominator_digits, direction
        )
        endpoint_candidates = tuple(
            (Fraction(1) - values[index]) / coefficient
            if coefficient > 0
            else values[index] / (-coefficient)
            for index, coefficient in zip(fractional, direction, strict=True)
            if coefficient != 0
        )
        step = min(endpoint_candidates)
        if step <= 0:
            raise RuntimeError("floating endpoint step must be positive")
        for index, coefficient in zip(fractional, direction, strict=True):
            values[index] += step * coefficient
            if not 0 <= values[index] <= 1:
                raise RuntimeError("floating endpoint left the unit cube")


def compute_hard_constraint_rounding(
    request: HardConstraintRoundingRequest,
) -> HardConstraintRoundingResult:
    """Round exactly while preserving hard rows and bounding column errors.

    At each step every active hard row and every monitored column with more
    than ``4d`` fractional coordinates is retained. The retained zero-one
    system has fewer equations than fractional coordinates, so a nonzero exact
    nullspace direction reaches a cube endpoint. This is the floating-rounding
    argument used in the cited Erdős 390 proof and terminates after at most one
    step per initially fractional coordinate.
    """

    source = request.source
    source_values = [value.as_fraction() for value in source.values]
    rounded_values = _floating_round(source_values, request)

    incidences = [0] * len(source_values)
    for column in source.columns:
        for index in column.coordinates:
            incidences[index] += 1
    maximum_incidence = max(incidences, default=0)
    column_error_bound = 4 * maximum_incidence

    row_ledger = tuple(
        HardConstraintRowLedger(
            row_label=row.label,
            source_sum=CanonicalRational.from_fraction(
                sum(
                    (source_values[index] for index in row.coordinates),
                    Fraction(),
                )
            ),
            rounded_sum=sum(rounded_values[index] for index in row.coordinates),
        )
        for row in source.rows
    )
    column_ledger: list[MonitoredColumnLedger] = []
    for column in source.columns:
        source_sum = sum(
            (source_values[index] for index in column.coordinates), Fraction()
        )
        rounded_sum = sum(rounded_values[index] for index in column.coordinates)
        signed_error = Fraction(rounded_sum) - source_sum
        column_ledger.append(
            MonitoredColumnLedger(
                column_label=column.label,
                source_sum=CanonicalRational.from_fraction(source_sum),
                rounded_sum=rounded_sum,
                signed_error=CanonicalRational.from_fraction(signed_error),
                absolute_error=CanonicalRational.from_fraction(abs(signed_error)),
            )
        )
    return HardConstraintRoundingResult(
        source=source,
        rounded_values=rounded_values,
        maximum_column_incidence=maximum_incidence,
        column_error_bound=column_error_bound,
        row_ledger=row_ledger,
        column_ledger=tuple(column_ledger),
    )


def _max_absolute_imbalance(
    signed_sums: tuple[int, ...],
) -> int:
    """Return the maximum absolute value among signed sums (0 for empty)."""

    return max((abs(value) for value in signed_sums), default=0)


def compute_discrepancy(request: DiscrepancyEvalRequest) -> DiscrepancyEvalResult:
    """Compute the signed sum on every set and the maximum absolute imbalance.

    For a coloring ``c`` and a set ``S`` the signed sum is
    ``sum(c[i] for i in S)``. The maximum absolute imbalance is the maximum
    of the absolute signed sums across all sets; it is zero when the family
    is empty.
    """
    signed_sums = tuple(
        sum(request.coloring[element] for element in subset)
        for subset in request.set_system.sets
    )
    return DiscrepancyEvalResult(
        signed_sums=signed_sums,
        max_absolute_imbalance=_max_absolute_imbalance(signed_sums),
    )


def compute_optimal_discrepancy(
    request: DiscrepancyOptimumRequest,
) -> DiscrepancyOptimumResult:
    """Search over all 2^n colorings for the minimum maximum discrepancy.

    The ground set size is bounded by ``MAX_GROUND_SET`` so the exhaustive
    search over ``itertools.product`` stays a bounded combinatorial
    computation. When the ground set is empty there is exactly one coloring
    (the empty coloring) with discrepancy zero.
    """
    n = request.set_system.ground_set_size
    sets = request.set_system.sets

    if n == 0:
        return DiscrepancyOptimumResult(
            optimal_coloring=(),
            optimal_discrepancy=0,
            exhaustive=True,
        )

    best_coloring: tuple[int, ...] | None = None
    best_discrepancy: int | None = None
    for values in itertools.product((-1, 1), repeat=n):
        coloring = values
        max_imbalance = 0
        for subset in sets:
            signed_sum = sum(coloring[element] for element in subset)
            absolute = -signed_sum if signed_sum < 0 else signed_sum
            if absolute > max_imbalance:
                max_imbalance = absolute
                if best_discrepancy is not None and max_imbalance >= best_discrepancy:
                    break
        else:
            if best_discrepancy is None or max_imbalance < best_discrepancy:
                best_discrepancy = max_imbalance
                best_coloring = coloring
                if best_discrepancy == 0:
                    break

    assert best_coloring is not None
    assert best_discrepancy is not None
    return DiscrepancyOptimumResult(
        optimal_coloring=best_coloring,
        optimal_discrepancy=best_discrepancy,
        exhaustive=True,
    )
