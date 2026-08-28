"""Domain-owned finite set-system discrepancy operations."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

from sympy import ZZ
from sympy.polys.matrices import DomainMatrix

from jacobian._exact import CanonicalRational
from jacobian.math.discrepancy_theory import _models as discrepancy_models
from jacobian.math.discrepancy_theory._models import (
    MAX_OPTIMUM_SOLVER_MILLISECONDS,
    MAX_OPTIMUM_SOLVER_NODES,
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
    _execution_failed_result,
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
    return HardConstraintRoundingResult._from_kernel(
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
    """Minimize the maximum imbalance via a bounded incumbent search plus proof.

    Binary variables ``b_u in {0, 1}`` encode the coloring ``x_u = 2 b_u -1``
    and one continuous bound ``t`` satisfies, per set ``S`` of size ``k``::

        2 sum(b[u] for u in S) - t <= k
        -2 sum(b[u] for u in S) - t <= -k

    A time- and node-bounded HiGHS MILP through ``scipy.optimize.milp``
    produces the incumbent. The returned coloring is integrality-checked,
    checked against its binary domain before rounding, and its discrepancy is
    recomputed exactly with Python integers, so floating-point solver
    internals can never surface as a mathematical value. A positive optimum
    is carried by OPTIMAL only after the exact Z3 pseudo-boolean feasibility
    check proves no coloring attains one less; a witness for that smaller
    bound yields EXECUTION_FAILED and an exhausted or unavailable proof
    yields BUDGET_EXCEEDED. Zero is definitional. A solver limit produces
    BUDGET_EXCEEDED; any other nonzero status, a non-integral or out-of-domain
    assignment, an objective mismatch, or a failing backend call
    (failed NumPy/SciPy initialization included) produces the distinct
    non-mathematical EXECUTION_FAILED outcome. Neither carries a coloring or
    any claim. The empty ground set has the single empty coloring with
    discrepancy zero.
    """
    n = request.set_system.ground_set_size
    sets = request.set_system.sets

    if n == 0:
        return _proven_optimal_result(request.set_system, (), 0)

    variable_count = n + 1

    try:
        import numpy as np
        from scipy.optimize import (
            Bounds,
            LinearConstraint,
            milp,
        )

        objective = np.zeros(variable_count)
        objective[-1] = 1.0
        integrality = np.zeros(variable_count)
        integrality[:n] = 1
        lower = np.zeros(variable_count)
        upper = np.full(variable_count, np.inf)
        upper[:n] = 1.0

        rows: list[np.ndarray] = []
        bounds_upper: list[float] = []
        for subset in sets:
            size = len(subset)
            plus_row = np.zeros(variable_count)
            minus_row = np.zeros(variable_count)
            for element in subset:
                plus_row[element] = 2.0
                minus_row[element] = -2.0
            plus_row[-1] = -1.0
            minus_row[-1] = -1.0
            rows.append(plus_row)
            bounds_upper.append(float(size))
            rows.append(minus_row)
            bounds_upper.append(float(-size))

        constraints = None
        if rows:
            matrix = np.array(rows)
            constraints = LinearConstraint(
                matrix,
                np.full(matrix.shape[0], -np.inf),
                np.array(bounds_upper),
            )

        result = milp(
            c=objective,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            options={
                "mip_rel_gap": 0,
                "time_limit": MAX_OPTIMUM_SOLVER_MILLISECONDS / 1000,
                "node_limit": MAX_OPTIMUM_SOLVER_NODES,
            },
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        # Backend initialization, program construction, and the solve are one
        # bounded external call: an ABI/loader, native, or raised failure
        # there is transport, not mathematics, so report the typed claim-free
        # outcome instead of escaping the kernel.
        return _execution_failed_result(request.set_system)
    if result.status == 1:
        return _budget_exceeded_result(request.set_system)
    if result.status != 0 or result.x is None:
        return _execution_failed_result(request.set_system)
    return _incumbent_outcome(request, result, variable_count)


def _compute_optimal_discrepancy_isolated(
    request: DiscrepancyOptimumRequest,
) -> DiscrepancyOptimumResult:
    """Run the complete HiGHS/Z3 transaction in a bounded owner worker."""

    from jacobian.math.discrepancy_theory._optimum_process import (
        compute_optimal_discrepancy_isolated,
    )

    return compute_optimal_discrepancy_isolated(request)


def _incumbent_outcome(
    request: DiscrepancyOptimumRequest,
    result: Any,
    variable_count: int,
) -> DiscrepancyOptimumResult:
    """Validate one status-zero solve and gate OPTIMAL on the exact proof."""

    import numpy as np

    sets = request.set_system.sets
    n = request.set_system.ground_set_size
    try:
        raw_result = result.x
        if raw_result.shape != (variable_count,) or not bool(
            np.all(np.isfinite(raw_result))
        ):
            return _execution_failed_result(request.set_system)

        raw_assignment = raw_result[:n]
        if float(np.max(np.abs(raw_assignment - np.round(raw_assignment)))) > 1e-6:
            return _execution_failed_result(request.set_system)
        if bool(np.any(raw_assignment < -1e-6) or np.any(raw_assignment > 1 + 1e-6)):
            return _execution_failed_result(request.set_system)
        coloring = tuple(1 if value > 0.5 else -1 for value in raw_assignment)

        # Bind the claimed optimum to an exact integer recomputation so no
        # floating-point objective value reaches the public contract.
        recomputed = max(
            (abs(sum(coloring[element] for element in subset)) for subset in sets),
            default=0,
        )
        solved_bound = float(raw_result[-1])
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        # A status-zero result is only an incumbent candidate.  If its vector
        # cannot be inspected in the advertised finite floating-point domain,
        # no canonical coloring or mathematical conclusion may escape.
        return _execution_failed_result(request.set_system)
    if abs(solved_bound - recomputed) > 1e-6:
        return _execution_failed_result(request.set_system)
    if recomputed == 0:
        return _proven_optimal_result(request.set_system, coloring, recomputed)
    try:
        outcome = discrepancy_models._feasibility_outcome(
            request.set_system, recomputed - 1
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        # An unavailable exact proof cannot back the OPTIMAL claim; report
        # the claim-free outcome instead of escaping the kernel.
        return _budget_exceeded_result(request.set_system)
    if outcome == "unsat":
        return _proven_optimal_result(request.set_system, coloring, recomputed)
    if outcome == "sat":
        return _execution_failed_result(request.set_system)
    return _budget_exceeded_result(request.set_system)
