"""Complete exact fixed-node Lebesgue profiles."""

import time
from fractions import Fraction
from functools import cmp_to_key
from typing import Literal

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.math.analysis.approximation._models import (
    LagrangeBasisPolynomial,
    LagrangeBasisResult,
)
from jacobian.math.analysis.approximation.lebesgue._admission import admit
from jacobian.math.analysis.approximation.lebesgue._models import (
    LebesgueCriticalPoint,
    LebesgueIntervalProfile,
    LebesgueIntervalSource,
    LebesgueSignCell,
)
from jacobian.math.analysis.approximation.lebesgue._roots import (
    Root,
    compare,
    critical_points,
    evaluate,
    rational,
)
from jacobian.math.analysis.approximation.operations import (
    _divide_by_nodal_factor,
    _poly_multiply,
    _polynomial_from_coeffs,
)
from jacobian.math.analysis.intervals import ClosedRationalInterval


def _closed(lower: Fraction, upper: Fraction) -> ClosedRationalInterval:
    return ClosedRationalInterval(
        lower=CanonicalRational.from_fraction(lower),
        upper=CanonicalRational.from_fraction(upper),
    )


def _maximum(entries: tuple[tuple[Root, Root], ...]) -> tuple[Root, tuple[Root, ...]]:
    maximum = entries[0][1]
    points = []
    for point, value in entries:
        order = compare(value, maximum)
        if order > 0:
            maximum, points = value, [point]
        elif order == 0:
            points.append(point)
    unique = {
        (point.value.polynomial, point.value.real_root_index): point for point in points
    }
    return maximum, tuple(sorted(unique.values(), key=cmp_to_key(compare)))


def _critical_order(left: tuple[Root, int, Root], right: tuple[Root, int, Root]) -> int:
    return compare(left[0], right[0])


def lebesgue_interval_profile(
    source: LebesgueIntervalSource,
) -> LebesgueIntervalProfile:
    """Return all sign cells, critical points, and tied interval maximizers."""
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return lebesgue_interval_profile(source)
    deadline = execution.started_at + 180
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    plan = admit(source)
    nodal = [Fraction(1)]
    for node in plan.nodes:
        nodal = _poly_multiply(nodal, [-node, Fraction(1)])
    coefficients = tuple(
        tuple(weight * c for c in _divide_by_nodal_factor(nodal, node))
        for node, weight in zip(plan.nodes, plan.weights, strict=True)
    )
    basis = LagrangeBasisResult(
        nodes=source.nodes,
        node_count=len(plan.nodes),
        basis=tuple(
            LagrangeBasisPolynomial(
                index=i,
                polynomial=_polynomial_from_coeffs(row),
                barycentric_weight=CanonicalRational.from_fraction(plan.weights[i]),
            )
            for i, row in enumerate(coefficients)
        ),
    )
    if not plan.cells:
        point = source.interval.lower.as_fraction()
        value = rational(
            sum((abs(evaluate(row, point)) for row in coefficients), Fraction(0))
        )
        return LebesgueIntervalProfile(
            source=source,
            basis=basis,
            cells=(),
            maximum=value.value,
            maximum_isolating_interval=value.interval,
            maximizing_points=(),
            maximizing_intervals=(source.interval,),
        )
    cells = []
    cell_extrema = []
    for cell in plan.cells:
        request_checkpoint("before Lebesgue sign-cell expansion")
        midpoint = (cell.lower + cell.upper) / 2
        signs: tuple[Literal[-1, 1], ...] = tuple(
            1 if evaluate(row, midpoint) > 0 else -1 for row in coefficients
        )
        polynomial = tuple(
            sum(
                (sign * row[i] for sign, row in zip(signs, coefficients, strict=True)),
                Fraction(0),
            )
            for i in range(len(plan.nodes))
        )
        constant = not any(polynomial[1:])
        endpoint_values = (
            evaluate(polynomial, cell.lower),
            evaluate(polynomial, cell.upper),
        )
        critical = (
            critical_points(
                polynomial, cell.lower, cell.upper, plan.maximum_refinement_bits
            )
            if cell.kind == "INTERIOR" and not constant
            else ()
        )
        entries = (
            (rational(cell.lower), rational(endpoint_values[0])),
            (rational(cell.upper), rational(endpoint_values[1])),
            *((point, value) for point, _, value in critical),
        )
        maximum, points = _maximum(entries)
        closed = _closed(cell.lower, cell.upper)
        cells.append(
            LebesgueSignCell(
                interval=closed,
                basis_signs=signs,
                polynomial=_polynomial_from_coeffs(polynomial),
                endpoint_values=(
                    CanonicalRational.from_fraction(endpoint_values[0]),
                    CanonicalRational.from_fraction(endpoint_values[1]),
                ),
                constant_on_cell=constant,
                critical_points=tuple(
                    LebesgueCriticalPoint(
                        point=point.value,
                        point_isolating_interval=point.interval,
                        derivative_multiplicity=multiplicity,
                        value=value.value,
                        value_isolating_interval=value.interval,
                    )
                    for point, multiplicity, value in sorted(
                        critical, key=cmp_to_key(_critical_order)
                    )
                ),
                maximum=maximum.value,
                maximum_isolating_interval=maximum.interval,
                maximizing_points=()
                if constant
                else tuple(point.value for point in points),
            )
        )
        cell_extrema.append((maximum, points, closed if constant else None))
    global_maximum, _ = _maximum(
        tuple((points[0], maximum) for maximum, points, _ in cell_extrema)
    )
    plateaus: list[ClosedRationalInterval] = []
    candidates: list[Root] = []
    for maximum, points, plateau in cell_extrema:
        if compare(maximum, global_maximum):
            continue
        if plateau is not None:
            if plateaus and plateaus[-1].upper == plateau.lower:
                plateaus[-1] = ClosedRationalInterval(
                    lower=plateaus[-1].lower, upper=plateau.upper
                )
            else:
                plateaus.append(plateau)
        else:
            candidates.extend(points)
    isolated = tuple(
        (point, global_maximum)
        for point in candidates
        if not any(
            compare(point, rational(p.lower.as_fraction())) >= 0
            and compare(point, rational(p.upper.as_fraction())) <= 0
            for p in plateaus
        )
    )
    points = _maximum(isolated)[1] if isolated else ()
    return LebesgueIntervalProfile(
        source=source,
        basis=basis,
        cells=tuple(cells),
        maximum=global_maximum.value,
        maximum_isolating_interval=global_maximum.interval,
        maximizing_points=tuple(point.value for point in points),
        maximizing_intervals=tuple(plateaus),
    )
