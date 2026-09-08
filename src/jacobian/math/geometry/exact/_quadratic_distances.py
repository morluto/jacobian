"""Admitted exact pair ledgers in one selected real quadratic field."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from itertools import combinations
from math import lcm

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.exact._models import (
    DistanceGraphResult,
    DistanceMultiplicityEntry,
    DistanceProfileResult,
    QuadraticPointConfiguration,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.number_theory.algebraic_numbers.quadratic import (
    RealQuadraticValue,
    _sign,
    require_square_free_radicand,
)


@dataclass(frozen=True)
class _Pair:
    indices: tuple[int, int]
    differences: tuple[Fraction, Fraction, Fraction, Fraction]


def _reject() -> None:
    raise OperationResourceAdmissionError(
        location=("configuration",),
        code="geometry.quadratic_distance.coefficient_growth",
        message="complete squared distances can exceed the canonical 256-digit quadratic coefficients",
    )


@contextmanager
def _deadline() -> Iterator[None]:
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()), _deadline():
            yield
        return
    deadline = execution.started_at + 60
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    yield


def _admit(
    configuration: QuadraticPointConfiguration, *, retain_values: bool = True
) -> tuple[_Pair, ...]:
    # The shared value bounds each input coefficient to 256 digits and the
    # geometry owner bounds 64 points. Linear scalar cancellation is therefore
    # bounded before polynomial field multiplication, and is retained rather
    # than repeated. This accepts a large common translation of small points.
    request_checkpoint("before quadratic distance admission")
    require_square_free_radicand(configuration.radicand)
    rows = []
    output_bits = 0
    for i, j in combinations(range(len(configuration.points)), 2):
        request_checkpoint("during quadratic distance admission")
        left, right = configuration.points[i], configuration.points[j]
        differences = tuple(
            part.as_fraction() - other.as_fraction()
            for x, y in zip(left.coordinates, right.coordinates, strict=True)
            for part, other in (
                (x.rational_part, y.rational_part),
                (x.radical_coefficient, y.radical_coefficient),
            )
        )
        a, b, c, e = differences
        denominator = lcm(*(v.denominator for v in differences))
        # Bring all four differences to one positive denominator D. These
        # integer component bit bounds price the sum of squares and the two
        # cross terms before either is expanded.
        bits = tuple(
            0
            if not v
            else abs(v.numerator).bit_length()
            + (denominator // v.denominator).bit_length()
            for v in differences
        )
        terms = [
            2 * k + (configuration.radicand.bit_length() if index in (1, 3) else 0)
            for index, k in enumerate(bits)
            if k
        ]
        rational_bits = max(terms, default=0) + max(1, len(terms)).bit_length()
        products = [
            bits[i] + bits[j] + 1 for i, j in ((0, 1), (2, 3)) if bits[i] and bits[j]
        ]
        radical_bits = max(products, default=0) + max(1, len(products)).bit_length()
        denominator_bits = 2 * denominator.bit_length()
        if retain_values and any(
            (bound * 30103 + 99999) // 100000 > 256
            for bound in (rational_bits, radical_bits, denominator_bits)
        ):
            _reject()
        if retain_values:
            output_bits += rational_bits + radical_bits + 2 * denominator_bits
        rows.append(_Pair((i, j), (a, b, c, e)))
    if retain_values and output_bits > 16_777_216:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="geometry.quadratic_distance.output_allocation",
            message="complete quadratic distance coefficients exceed the admitted allocation",
        )
    return tuple(rows)


def _value(a: Fraction, b: Fraction, d: int) -> RealQuadraticValue:
    return RealQuadraticValue(
        rational_part=CanonicalRational.from_fraction(a),
        radical_coefficient=CanonicalRational.from_fraction(b),
        radicand=d,
    )


def quadratic_distance_profile(
    configuration: QuadraticPointConfiguration,
) -> DistanceProfileResult[QuadraticPointConfiguration, RealQuadraticValue]:
    with _deadline():
        return _profile(configuration, _admit(configuration))


def _profile(
    configuration: QuadraticPointConfiguration, plan: tuple[_Pair, ...]
) -> DistanceProfileResult[QuadraticPointConfiguration, RealQuadraticValue]:
    d = configuration.radicand
    classes: dict[tuple[Fraction, Fraction], list[tuple[int, int]]] = {}
    for row in plan:
        request_checkpoint("during exact quadratic distance expansion")
        a, b, c, e = row.differences
        key = (a * a + d * b * b + c * c + d * e * e, 2 * (a * b + c * e))
        classes.setdefault(key, []).append(row.indices)

    def order(left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]) -> int:
        return _sign(left[0] - right[0], left[1] - right[1], d)

    return DistanceProfileResult[QuadraticPointConfiguration, RealQuadraticValue](
        configuration=configuration,
        entries=tuple(
            DistanceMultiplicityEntry[RealQuadraticValue](
                squared_distance=_value(*key, d),
                pair_count=len(classes[key]),
                pairs=tuple(classes[key]),
            )
            for key in sorted(classes, key=cmp_to_key(order))
        ),
    )


def quadratic_distance_graph(
    configuration: QuadraticPointConfiguration, target: RealQuadraticValue
) -> DistanceGraphResult[QuadraticPointConfiguration, RealQuadraticValue]:
    if target.radicand != configuration.radicand:
        raise OperationDomainValidationError(
            location=("target_squared_distance", "radicand"),
            code="geometry.quadratic_distance.target_parent_mismatch",
            message="target and coordinates must use the same quadratic field",
        )
    with _deadline():
        plan = _admit(configuration, retain_values=False)
        wanted = (
            target.rational_part.as_fraction(),
            target.radical_coefficient.as_fraction(),
        )
        d = configuration.radicand
        if _sign(*wanted, d) < 0:
            raise OperationDomainValidationError(
                location=("target_squared_distance",),
                code="geometry.squared_distance_target_nonnegative",
                message="squared distance target must be nonnegative",
            )
        edges = []
        for row in plan:
            request_checkpoint("during exact quadratic distance selection")
            a, b, c, e = row.differences
            if (a * a + d * b * b + c * c + d * e * e, 2 * (a * b + c * e)) == wanted:
                edges.append(row.indices)
        return DistanceGraphResult[QuadraticPointConfiguration, RealQuadraticValue](
            configuration=configuration,
            target_squared_distance=target,
            graph=IndexedSimpleUndirectedGraph(
                vertex_count=len(configuration.points), edges=tuple(edges)
            ),
        )
