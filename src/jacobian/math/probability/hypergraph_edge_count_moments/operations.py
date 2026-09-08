"""Exact moments of the surviving-edge count under Bernoulli vertex retention."""

import time
from collections import Counter
from fractions import Fraction
from itertools import combinations_with_replacement
from math import comb

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
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
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

from ._models import EdgeOverlapMomentRow, HypergraphEdgeCountMomentsResult

MAX_PAIR_WORK = 20_000_000
MAX_PROFILE_ROWS = 65_536
MAX_OUTPUT_BITS = 268_435_456
MAX_ARITHMETIC_WORK = 1 << 40


def _reject(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("hypergraph", "retention_probability"),
        code=f"hypergraph_edge_count_moments.{code}",
        message=message,
    )


def compute_hypergraph_edge_count_moments(
    hypergraph: FiniteHypergraph, retention_probability: CanonicalRational
) -> HypergraphEdgeCountMomentsResult:
    if not isinstance(hypergraph, FiniteHypergraph) or not isinstance(
        retention_probability, CanonicalRational
    ):
        raise OperationDomainValidationError(
            location=("hypergraph", "retention_probability"),
            code="hypergraph_edge_count_moments.source_type",
            message="expected a FiniteHypergraph and CanonicalRational",
        )
    if not 0 <= retention_probability.num <= retention_probability.den:
        raise OperationDomainValidationError(
            location=("retention_probability",),
            code="hypergraph_edge_count_moments.probability_out_of_range",
            message="retention_probability must be between 0 and 1",
        )
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return compute_hypergraph_edge_count_moments(
                hypergraph, retention_probability
            )
    deadline = execution.started_at + 60
    bind_request_deadline(
        min(deadline, execution.deadline)
        if execution.deadline is not None
        else deadline
    )
    request_checkpoint("before hypergraph moment admission")
    # Source canonicalization already bounds 256 vertices, 12000 indexed edges,
    # and 36000 incidences. Counting identical authored member tuples is linear
    # in that envelope; no edge-pair or probability expansion occurs here.
    memberships = Counter(members for _, members in hypergraph.edges)
    sizes = sorted({len(members) for members in memberships})
    edge_count = len(hypergraph.edges)
    max_size = max(sizes, default=0)
    possible_rows = min(
        comb(edge_count, 2),
        sum(a + 1 for a, b in combinations_with_replacement(sizes, 2)),
    )
    if possible_rows > MAX_PROFILE_ROWS:
        _reject(
            "profile_size", "source size classes exceed the compact overlap row bound"
        )
    words = max(1, (len(hypergraph.vertices) + 63) // 64)
    pair_work = words * (
        len(memberships) ** 2 + sum(len(members) for members in memberships)
    )
    if pair_work > MAX_PAIR_WORK:
        _reject("pair_work", "edge overlap work exceeds the admitted bound")
    # A common q^(2s) denominator contains every moment and covariance.
    # Z <= m bounds both second moment and variance numerator by m²q^(2s).
    # Bit bounds avoid decimal conversion of large authored integers.
    q_bits = retention_probability.den.bit_length()
    height = 2 * max_size * q_bits + 2 * max(1, edge_count.bit_length()) + 2
    if (height * 30103 + 99999) // 100000 > MAX_CANONICAL_RATIONAL_DIGITS:
        _reject(
            "rational_height",
            "exact moments exceed canonical rational coefficient bounds",
        )
    output_bits = (
        2 * (possible_rows + 3) * height
        + 2 * q_bits
        + 128
        * (
            edge_count
            + len(hypergraph.vertices)
            + sum(len(members) for _, members in hypergraph.edges)
        )
    )
    if output_bits > MAX_OUTPUT_BITS:
        _reject(
            "output_size", "exact moment profile exceeds its output allocation bound"
        )
    # Schoolbook integer arithmetic and Euclidean rational reduction are bounded
    # by a quadratic bit envelope per cached power, weighted sum and output.
    arithmetic_work = (
        (edge_count + 16 * possible_rows + 8 * max_size + 16) * height * height
    )
    if arithmetic_work > MAX_ARITHMETIC_WORK:
        _reject(
            "arithmetic_work",
            "exact rational arithmetic exceeds the admitted work bound",
        )
    request_checkpoint("after complete hypergraph moment admission")
    vertex_index = {vertex: index for index, vertex in enumerate(hypergraph.vertices)}
    multiplicities = Counter(
        {
            sum(1 << vertex_index[v] for v in members): count
            for members, count in memberships.items()
        }
    )
    rows: Counter[tuple[int, int, int]] = Counter()
    for (left, left_count), (right, right_count) in combinations_with_replacement(
        multiplicities.items(), 2
    ):
        count = comb(left_count, 2) if left == right else left_count * right_count
        if count:
            a, b = sorted((left.bit_count(), right.bit_count()))
            rows[(a, b, (left & right).bit_count())] += count
        request_checkpoint("during hypergraph overlap enumeration")
    p = retention_probability.as_fraction()
    powers = [Fraction(1)]
    for _exponent in range(1, 2 * max_size + 1):
        powers.append(powers[-1] * p)
    expectation = sum(
        (count * powers[mask.bit_count()] for mask, count in multiplicities.items()),
        Fraction(0),
    )
    second = expectation + sum(
        (2 * count * powers[a + b - q] for (a, b, q), count in rows.items()),
        Fraction(0),
    )
    variance = second - expectation * expectation
    profile = tuple(
        EdgeOverlapMomentRow(
            edge_size_min=a,
            edge_size_max=b,
            intersection_size=q,
            pair_count=count,
            union_size=a + b - q,
            covariance=CanonicalRational.from_fraction(
                powers[a + b - q] - powers[a] * powers[b]
            ),
        )
        for (a, b, q), count in sorted(rows.items())
    )
    request_checkpoint("after exact hypergraph moments")
    return HypergraphEdgeCountMomentsResult(
        hypergraph=hypergraph,
        retention_probability=retention_probability,
        edge_count_expectation=CanonicalRational.from_fraction(expectation),
        edge_count_second_moment=CanonicalRational.from_fraction(second),
        edge_count_variance=CanonicalRational.from_fraction(variance),
        overlap_profile=profile,
    )


__all__ = ["compute_hypergraph_edge_count_moments"]
