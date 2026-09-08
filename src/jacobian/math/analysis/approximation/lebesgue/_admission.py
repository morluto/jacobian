"""Source-derived bounds for complete fixed-node algebraic extrema."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from math import lcm, prod
from typing import Literal, NoReturn

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.analysis.approximation.lebesgue._models import LebesgueIntervalSource
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS,
    MAX_REAL_ALGEBRAIC_DEGREE,
)

type CellKind = Literal["CONSTANT", "LEFT", "INTERIOR", "RIGHT"]


@dataclass(frozen=True)
class CellPlan:
    lower: Fraction
    upper: Fraction
    kind: CellKind


@dataclass(frozen=True)
class LebesguePlan:
    nodes: tuple[Fraction, ...]
    weights: tuple[Fraction, ...]
    cells: tuple[CellPlan, ...]
    constant_query: bool
    maximum_refinement_bits: int
    coefficient_bits: int


def reject(reason: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("source",),
        code=f"approximation.lebesgue.{reason}",
        message=message,
    )


def _digits(bits: int) -> int:
    # log10(2) < 30103/100000, with integer rounding upwards.
    return max(1, (bits * 30103 + 99999) // 100000)


def _factor_bits(degree: int, height_bits: int) -> int:
    # H(F) <= 2^degree sqrt(degree+1) H(P) for an integral factor F of P.
    return height_bits + degree + (degree + 1).bit_length()


def _separation_bits(degree: int, height_bits: int) -> int:
    # A deliberately larger exponent than the Mignotte bound
    # sep(P) > sqrt(3) n^(-(n+2)/2) ||P||_2^(1-n), for square-free P.
    return max(8, degree * (height_bits + 2 * (degree + 1).bit_length() + 2) + 8)


def admit(source: LebesgueIntervalSource) -> LebesguePlan:
    """Reserve basis expansion, resultants, root refinement and complete values.

    The shared node carrier already bounds the preliminary scalar products:
    at most 32 nodes and 512 total rational-component digits. No polynomial
    is expanded in this phase. Exact barycentric weights are retained for the
    executor so the node-difference products are not computed a second time.
    """
    request_checkpoint("before Lebesgue source admission")
    nodes = tuple(node.as_fraction() for node in source.nodes.nodes)
    lower, upper = (
        source.interval.lower.as_fraction(),
        source.interval.upper.as_fraction(),
    )
    count = len(nodes)
    degree = count - 1
    constant_query = count == 1 or (
        count == 2 and nodes[0] <= lower <= upper <= nodes[-1]
    )
    boundaries = [lower]
    if count > 1:
        boundaries.extend(node for node in nodes if lower < node < upper)
    if upper != lower:
        boundaries.append(upper)
    cells = tuple(
        CellPlan(
            a,
            b,
            "CONSTANT"
            if count == 1 or (count == 2 and nodes[0] <= a < b <= nodes[-1])
            else "LEFT"
            if b <= nodes[0]
            else "RIGHT"
            if a >= nodes[-1]
            else "INTERIOR",
        )
        for a, b in pairwise(boundaries)
    )
    interior_count = sum(cell.kind == "INTERIOR" for cell in cells)
    if interior_count and degree - 1 > MAX_REAL_ALGEBRAIC_DEGREE:
        reject(
            "algebraic_degree",
            "interior critical values can exceed the canonical real-algebraic degree bound",
        )

    denominators = [Fraction(1) for _ in nodes]
    for i, node in enumerate(nodes):
        request_checkpoint("during Lebesgue node-difference admission")
        for j in range(i):
            difference = nodes[j] - node
            denominators[j] *= difference
            denominators[i] *= -difference
    weights = tuple(1 / denominator for denominator in denominators)
    # l_i = c_i * product_{j!=i}(q_j*x-p_j). These integer polynomials
    # have coefficient l1 norm at most product(|p_j|+q_j).
    scales = tuple(
        weight / prod(node.denominator for j, node in enumerate(nodes) if j != i)
        for i, weight in enumerate(weights)
    )
    common_denominator = lcm(*(scale.denominator for scale in scales))
    height = sum(
        abs(scale.numerator)
        * (common_denominator // scale.denominator)
        * prod(
            abs(node.numerator) + node.denominator
            for j, node in enumerate(nodes)
            if j != i
        )
        for i, scale in enumerate(scales)
    )
    height_bits = max(1, height.bit_length())
    denominator_bits = common_denominator.bit_length()
    # Every sign-cell polynomial is S(x)/D with integer coefficients of
    # magnitude at most height, regardless of its sign vector.
    endpoint_bits = 2
    if not constant_query:
        endpoint_bits = max(
            max(
                height_bits
                + count.bit_length()
                + degree
                * max(
                    abs(point.numerator).bit_length(), point.denominator.bit_length()
                ),
                denominator_bits + degree * point.denominator.bit_length(),
            )
            for point in boundaries
        )
    coefficient_bits = endpoint_bits
    refinement_bits = 8
    if interior_count:
        critical_degree = degree - 1
        derivative_bits = height_bits + max(1, degree).bit_length()
        # Hadamard on the Sylvester matrix of S' and D*y-S. On |y|=1,
        # their row l1 norms are bounded by d(d+1)H/2 and (d+1)H+D.
        # Cauchy's coefficient inequality gives the same bound coefficientwise.
        resultant_bits = (
            degree * (degree * (degree + 1) // 2 * height).bit_length()
            + critical_degree * (count * height + common_denominator).bit_length()
        )
        value_bits = _factor_bits(critical_degree, resultant_bits)
        point_bits = _factor_bits(critical_degree, derivative_bits)
        algebraic_bits = max(value_bits, point_bits)
        coefficient_bits = max(coefficient_bits, algebraic_bits)
        if _digits(algebraic_bits) > MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS:
            reject(
                "algebraic_height",
                "complete extrema can exceed canonical real-algebraic coefficient digits",
            )
        radius = max(
            1,
            abs(lower.numerator) // lower.denominator + 1,
            abs(upper.numerator) // upper.denominator + 1,
        )
        # Interval Horner has width(A*X) <= R*width(A) + max|A|*width(X).
        # With R >= 1 and |X| <= R, induction bounds the final width by
        # H*d(d+1)/2 * R^(d-1) * width(X); dependency does not require
        # replacing R by R+1. The two bit lengths below round upwards.
        lipschitz_bits = (
            height_bits
            + 2 * count.bit_length()
            + max(0, degree - 1) * radius.bit_length()
        )
        refinement_bits = (
            _separation_bits(critical_degree, value_bits) + lipschitz_bits + 8
        )
    if _digits(endpoint_bits) > MAX_CANONICAL_RATIONAL_DIGITS:
        reject(
            "endpoint_height",
            "endpoint values exceed the canonical rational digit envelope",
        )
    if refinement_bits > 65_536:
        reject(
            "refinement",
            "critical-value isolation exceeds the admitted rational refinement depth",
        )
    critical_rows = interior_count * max(0, degree - 1)
    # Each critical row has two algebraic polynomials and four interval
    # endpoints. The latter retain rational numerators and denominators.
    query_component_bits = max(
        max(abs(point.numerator).bit_length(), point.denominator.bit_length())
        for point in (lower, upper)
    )
    interval_component_bits = max(
        query_component_bits, 2 * refinement_bits + coefficient_bits + 32
    )
    output_bits = (
        (count * count + count) * (height_bits + denominator_bits)
        + (critical_rows + 2 * len(cells) + 4)
        * (4 * count * coefficient_bits + 8 * interval_component_bits)
        + 2
        * sum(
            abs(node.numerator).bit_length() + node.denominator.bit_length()
            for node in (*nodes, lower, upper)
        )
    )
    if output_bits > 268_435_456:
        reject(
            "output_allocation",
            "complete Lebesgue values exceed the exact coefficient allocation",
        )
    # Comparing values from different cells can isolate a product of two
    # distinct irreducible polynomials. These refined intervals are private;
    # reserve their degree, height, and rational arithmetic separately from
    # the serialized intervals retained above.
    comparison_degree = 2 * max(1, degree - 1)
    comparison_height = (
        2 * max(coefficient_bits, query_component_bits) + count.bit_length()
    )
    comparison_bits = _separation_bits(comparison_degree, comparison_height)
    intermediate_bits = max(1, count * count) * (
        comparison_bits + comparison_height
    ) + count * count * (interval_component_bits + height_bits + denominator_bits)
    if intermediate_bits > 268_435_456:
        reject(
            "intermediate_allocation",
            "exact isolation and comparison arithmetic exceed the admitted allocation",
        )
    work = (
        max(1, interior_count)
        * count**4
        * (coefficient_bits + refinement_bits + comparison_bits)
    )
    if work > 1_000_000_000_000:
        reject(
            "work",
            "complete Lebesgue root and comparison work exceeds its admitted envelope",
        )
    request_checkpoint("after complete Lebesgue admission")
    return LebesguePlan(
        nodes, weights, cells, constant_query, refinement_bits, coefficient_bits
    )
