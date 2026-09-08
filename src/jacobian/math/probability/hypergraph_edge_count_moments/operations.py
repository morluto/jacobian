"""Exact moments of the surviving-edge count under Bernoulli vertex retention."""
from collections import Counter
from fractions import Fraction
from math import comb

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

from ._models import EdgeOverlapMomentRow, HypergraphEdgeCountMomentsResult

MAX_PAIR_WORK = 20_000_000
MAX_PROFILE_ROWS = 65_536
MAX_RATIONAL_DIGITS = 32_768

def _reject(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(location=("hypergraph",), code=f"hypergraph_edge_count_moments.{code}", message=message)

def compute_hypergraph_edge_count_moments(hypergraph: FiniteHypergraph, retention_probability: CanonicalRational) -> HypergraphEdgeCountMomentsResult:
    if not isinstance(hypergraph, FiniteHypergraph):
        raise OperationDomainValidationError(location=("hypergraph",), code="hypergraph_edge_count_moments.invalid_hypergraph", message="hypergraph must be a FiniteHypergraph")
    if not isinstance(retention_probability, CanonicalRational):
        raise OperationDomainValidationError(location=("retention_probability",), code="hypergraph_edge_count_moments.invalid_probability", message="retention_probability must be a CanonicalRational")
    p = retention_probability.as_fraction()
    if not 0 <= p <= 1:
        raise OperationDomainValidationError(location=("retention_probability",), code="hypergraph_edge_count_moments.probability_out_of_range", message="retention_probability must be between 0 and 1")
    vertex_index = {vertex: index for index, vertex in enumerate(hypergraph.vertices)}
    masks = tuple(sum(1 << vertex_index[vertex] for vertex in members) for _, members in hypergraph.edges)
    multiplicities = Counter(masks)
    sizes = tuple(sorted({mask.bit_count() for mask in multiplicities}))
    possible_rows = sum(min(a, b) + 1 for a in sizes for b in sizes if a <= b)
    if possible_rows > MAX_PROFILE_ROWS:
        _reject("profile_size", "source size classes can produce too many overlap rows")
    pair_work = len(multiplicities) ** 2 + len(masks) + sum(mask.bit_count() for mask in masks)
    max_size = max(sizes, default=0)
    estimated_digits = len(str(p.denominator)) * (2 * max_size) + len(str(max(1, len(masks)))) + 2
    if pair_work > MAX_PAIR_WORK:
        _reject("pair_work", "edge overlap work exceeds the admitted bound")
    if estimated_digits > MAX_RATIONAL_DIGITS:
        _reject("rational_height", "exact moments exceed the rational height bound")
    powers: dict[int, Fraction] = {0: Fraction(1)}
    powers.update({exponent: p**exponent for exponent in range(1, 2 * max_size + 1)})
    expectation = sum((powers[mask.bit_count()] for mask in masks), Fraction(0))
    rows: Counter[tuple[int, int, int]] = Counter()
    for left, left_count in multiplicities.items():
        for right, right_count in multiplicities.items():
            if left > right:
                continue
            count = comb(left_count, 2) if left == right else left_count * right_count
            if count:
                a, b = sorted((left.bit_count(), right.bit_count()))
                rows[(a, b, (left & right).bit_count())] += count
    second = expectation + sum((2 * count * powers[a + b - q] for (a, b, q), count in rows.items()), Fraction(0))
    variance = second - expectation * expectation
    def canonical(value: Fraction) -> CanonicalRational:
        result = CanonicalRational.from_fraction(value)
        if max(len(str(abs(result.num))), len(str(result.den))) > MAX_RATIONAL_DIGITS:
            _reject("rational_height", "exact moment exceeds the rational height bound")
        return result
    profile = tuple(EdgeOverlapMomentRow(edge_size_min=a, edge_size_max=b, intersection_size=q, pair_count=count, union_size=a+b-q, covariance=canonical(powers[a+b-q] - powers[a] * powers[b])) for (a, b, q), count in sorted(rows.items()))
    output_bits = (len(profile) + 3) * estimated_digits * 4 + len(masks) * (len(hypergraph.vertices) + 1)
    if output_bits > 268_435_456:
        _reject("output_size", "exact moment profile exceeds its output bit bound")
    return HypergraphEdgeCountMomentsResult(hypergraph=hypergraph, retention_probability=retention_probability, edge_count_expectation=canonical(expectation), edge_count_second_moment=canonical(second), edge_count_variance=canonical(variance), overlap_profile=profile)
__all__ = ["compute_hypergraph_edge_count_moments"]
