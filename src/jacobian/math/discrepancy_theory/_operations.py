"""Domain-owned finite set-system discrepancy operations."""

from __future__ import annotations

import math
from fractions import Fraction

from sympy import ZZ
from sympy.polys.matrices import DomainMatrix

from jacobian._exact import CanonicalRational
from jacobian.math.discrepancy_theory._models import (
    MAX_OPTIMUM_SOLVER_MILLISECONDS,
    MAX_ROUNDING_INTERMEDIATE_DIGITS,
    DiscrepancyEvalRequest,
    DiscrepancyEvalResult,
    DiscrepancyOptimumRequest,
    DiscrepancyOptimumResult,
    HardConstraintRoundingRequest,
    HardConstraintRoundingResult,
    HardConstraintRowLedger,
    MonitoredColumnLedger,
    _budget_exceeded_result,
    _proven_optimal_result,
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
    """Minimize the maximum imbalance via an exact integer program.

    Variables ``x_u in {-1, +1}`` and a shared nonnegative integer ``D``
    with ``D >= |sum_{u in S} x_u|`` for every set ``S``.  A sat answer is
    only trusted as OPTIMAL when Z3's objective handle proves it: its
    lower and upper bounds must coincide with each other, with the model's
    objective value, and with an independent Python replay of the
    coloring's exact maximum imbalance.  An incumbent found after an
    exhausted budget has open bounds and is reported as
    ``BUDGET_EXCEEDED`` with no mathematical claim; likewise any
    non-sat outcome.  The empty ground set has the single empty coloring
    with discrepancy zero.
    """
    n = request.set_system.ground_set_size
    sets = request.set_system.sets

    if n == 0:
        return _proven_optimal_result(request.set_system, (), 0)

    import z3  # type: ignore[import-untyped]

    optimizer = z3.Optimize()
    optimizer.set("timeout", MAX_OPTIMUM_SOLVER_MILLISECONDS)
    variables = [z3.Int(f"x_{index}") for index in range(n)]
    optimizer.add(*(z3.Or(value == 1, value == -1) for value in variables))
    objective = z3.Int("D")
    optimizer.add(objective >= 0)
    for subset in sets:
        signed_sum = (
            z3.Sum([variables[element] for element in subset])
            if subset
            else z3.IntVal(0)
        )
        optimizer.add(objective >= signed_sum, objective >= -signed_sum)
    objective_handle = optimizer.minimize(objective)

    def _bound_digits(expression: object) -> int | None:
        """Return the integer value of a numeral bound, or None if open."""

        as_long = getattr(expression, "as_long", None)
        if callable(as_long):
            try:
                return int(as_long())
            except Exception:
                return None
        return None

    outcome = optimizer.check()
    if outcome != z3.sat:
        return _budget_exceeded_result(request.set_system)
    model = optimizer.model()
    coloring = tuple(int(model.evaluate(variable).as_long()) for variable in variables)
    model_objective = int(model.evaluate(objective).as_long())
    lower_bound = _bound_digits(optimizer.lower(objective_handle))
    upper_bound = _bound_digits(optimizer.upper(objective_handle))
    replayed_optimum = max(
        (abs(sum(coloring[element] for element in subset)) for subset in sets),
        default=0,
    )
    proven = (
        lower_bound is not None
        and upper_bound is not None
        and lower_bound == upper_bound == model_objective == replayed_optimum
    )
    if not proven:
        return _budget_exceeded_result(request.set_system)
    return _proven_optimal_result(
        request.set_system,
        coloring,
        model_objective,
    )
