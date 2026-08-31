"""Maximum weight antichain kernel using min-cut reduction."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import lcm as _lcm
from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.posets.core._models import FinitePoset
from jacobian.math.combinatorics.posets.weighted_antichain._models import (
    MAX_ENUMERATION_WORK,
    MaximumWeightAntichainResult,
)

__all__ = ["compute_maximum_weight_antichain"]


@dataclass(frozen=True, slots=True)
class _MaximumWeightAntichainAdmission:
    weights: tuple[Fraction, ...]
    common_den: int
    int_weights: tuple[int, ...]
    total_int: int


def _int_decimal_digits(value: int) -> int:
    """Upper bound on the decimal digit count of a nonnegative integer.

    Uses ``int.bit_length`` (linear in the magnitude) instead of ``str(value)``
    so it stays below Python's integer-string conversion limit even for the
    32,768-digit canonical components the kernel legitimately admits.
    """
    if value <= 0:
        return 1
    # digits <= floor(bit_length * log10(2)) + 1; round up for an upper bound.
    return max(1, value.bit_length() * 30103 // 100000 + 1)


def _admit_maximum_weight_antichain(
    poset: FinitePoset, weights: tuple[CanonicalRational, ...]
) -> _MaximumWeightAntichainAdmission:
    if not isinstance(poset, FinitePoset):
        raise OperationDomainValidationError(
            location=("poset",),
            code="weighted_antichain.invalid_poset",
            message="poset must be a FinitePoset value",
        )
    if not isinstance(weights, tuple) or len(weights) != len(poset.elements):
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.weight_count_mismatch",
            message="weights must have exactly one entry per poset element",
        )
    if any(not isinstance(weight, CanonicalRational) for weight in weights):
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.invalid_weight",
            message="weights must be CanonicalRational values",
        )
    weight_fracs = tuple(weight.as_fraction() for weight in weights)
    if any(weight < 0 for weight in weight_fracs):
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.negative_weight",
            message="all weights must be nonnegative",
        )
    n = len(poset.elements)
    if n > MAX_ENUMERATION_WORK:
        raise OperationDomainValidationError(
            location=("poset", "elements"),
            code="weighted_antichain.work_bound_exceeded",
            message="the antichain work envelope is exceeded",
        )
    width = _poset_width(poset, list(poset.elements))
    numerators = [frac.numerator for frac in weight_fracs]
    denominators = [frac.denominator for frac in weight_fracs]
    num_digits_max = max(
        (_int_decimal_digits(num) for num in numerators), default=0
    )

    # The exact maximum antichain weight has denominator dividing the lcm of
    # the input denominators; the polynomial min-cut clears to exactly that
    # shared scale, so charge (and cap) that common-denominator expansion.
    # Building the common denominator is itself the expensive step we bound, so
    # halt as soon as its digit count would burst the canonical envelope.
    rational_digit_cap = 32_768
    common_den = 1
    for denominator in denominators:
        common_den = _lcm(common_den, denominator)
        if _int_decimal_digits(common_den) > rational_digit_cap:
            raise OperationDomainValidationError(
                location=("weights",),
                code="weighted_antichain.result_growth_exceeded",
                message=(
                    "maximum-weight rational growth exceeds the canonical "
                    "digit envelope"
                ),
            )
    common_den_digits = _int_decimal_digits(common_den)

    # Scaled integer capacities need about (common-den digits + numerator
    # digits) significant decimal digits during the min-cut.
    scaled_digits = max(1, common_den_digits + num_digits_max)
    num_nodes = 2 + 2 * n
    num_edges = len(poset.strict_order_pairs) + 2 * n
    arithmetic_work = max(1, num_nodes) * max(1, num_edges) * scaled_digits
    if arithmetic_work > MAX_ENUMERATION_WORK:
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.arithmetic_work_bound_exceeded",
            message="rational antichain summation exceeds the admitted work envelope",
        )

    # Denominator-clearing intermediates must fit the canonical envelope.
    int_weights = tuple(int(frac * common_den) for frac in weight_fracs)
    total_int = sum(int_weights)

    # Result growth: numerator and denominator tracked per weight. The maximum
    # antichain has at most `width` members, so its numerator gains at most
    # len(str(width)) carry digits over a scaled integer.
    result_den_digits = common_den_digits
    result_num_digits = scaled_digits + (len(str(width)) if width > 1 else 0)
    max_result_digits = max(result_den_digits, result_num_digits)
    if max_result_digits > rational_digit_cap:
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.result_growth_exceeded",
            message=(
                "maximum-weight rational growth exceeds the canonical "
                "digit envelope"
            ),
        )

    rational_size = strict_json_object_size(
        (
            ("num", len(encode_strict_json("9" * max_result_digits))),
            ("den", len(encode_strict_json("9" * max_result_digits))),
        )
    )
    labels_size = (
        2
        + max(n - 1, 0)
        + sum(len(encode_strict_json(element)) for element in poset.elements)
    )
    result_bytes = strict_json_object_size(
        (
            ("poset", len(encode_strict_json(poset.model_dump(mode="json")))),
            (
                "weights",
                len(
                    encode_strict_json(
                        [weight.model_dump(mode="json") for weight in weights]
                    )
                ),
            ),
            ("maximum_weight", rational_size),
            ("antichain", labels_size),
        )
    )
    if result_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("poset", "weights"),
            code="weighted_antichain.result_too_large",
            message="maximum-weight antichain result exceeds the canonical output envelope",
        )
    return _MaximumWeightAntichainAdmission(
        weights=weight_fracs,
        common_den=common_den,
        int_weights=int_weights,
        total_int=total_int,
    )


def compute_maximum_weight_antichain(
    poset: FinitePoset,
    weights: tuple[CanonicalRational, ...],
) -> MaximumWeightAntichainResult:
    """Return the exact maximum weight antichain and a deterministic witness.

    Uses a polynomial min-cut reduction via the bipartite vertex cover
    formulation: the maximum weight antichain on a poset equals the total
    weight minus the minimum weight vertex cover of the bipartite graph
    induced by the order relation.
    """
    import networkx as nx

    admission = _admit_maximum_weight_antichain(poset, weights)
    elements = list(poset.elements)
    n = len(elements)

    if n == 0:
        return MaximumWeightAntichainResult(
            poset=poset,
            weights=weights,
            maximum_weight=CanonicalRational.from_fraction(Fraction(0)),
            antichain=(),
        )

    common_den = admission.common_den
    int_weights = admission.int_weights
    total_weight_int = admission.total_int

    # Build flow network: source, sink, n left nodes, n right nodes
    idx = {element: i for i, element in enumerate(elements)}
    order_pairs: list[tuple[int, int]] = []
    for pair in poset.strict_order_pairs:
        order_pairs.append((idx[pair.lower], idx[pair.upper]))

    # Min vertex cover on the bipartite comparison graph is found via a
    # minimum cut (König's theorem):
    #   source -> v_L with capacity w_v
    #   v_R -> sink with capacity w_v
    #   v_L -> u_R with an uncuttable capacity for each order v < u.
    # The min cut value = min vertex cover weight, so
    # max antichain weight = total weight - min cut.
    #
    # The relation edges are genuinely uncuttable: their capacity sits strictly
    # above the sum of every finite vertex capacity, so any cut that removes a
    # relation edge is strictly more expensive than cutting all vertices.
    source = 0
    sink = 1

    def left(i: int) -> int:  # v_L
        return 2 + i

    def right(i: int) -> int:  # v_R
        return 2 + n + i

    inf_cap = total_weight_int + 1

    graph: nx.DiGraph[Any, Any] = nx.DiGraph()
    graph.add_nodes_from(range(2 + 2 * n))
    for i in range(n):
        graph.add_edge(source, left(i), weight=int_weights[i])
        graph.add_edge(right(i), sink, weight=int_weights[i])
    for v, u in order_pairs:
        graph.add_edge(left(v), right(u), weight=inf_cap)

    cut_value, (source_side, _sink_side) = nx.minimum_cut(
        graph, source, sink, capacity="weight"
    )
    min_vertex_cover = int(cut_value)
    max_antichain_weight_int = total_weight_int - min_vertex_cover

    # Recover the min vertex cover from the residual reachable set S (König):
    # left copies not in S plus right copies in S.
    in_vertex_cover = set()
    for i in range(n):
        if left(i) not in source_side:
            in_vertex_cover.add(i)
        if right(i) in source_side:
            in_vertex_cover.add(i)

    # Antichain = elements NOT in the vertex cover
    antichain_indices = [i for i in range(n) if i not in in_vertex_cover]
    antichain_indices.sort()

    best_weight = Fraction(max_antichain_weight_int, common_den)
    best_antichain = tuple(elements[i] for i in antichain_indices)

    return MaximumWeightAntichainResult(
        poset=poset,
        weights=weights,
        maximum_weight=CanonicalRational.from_fraction(best_weight),
        antichain=best_antichain,
    )


def _poset_width(poset: FinitePoset, elements: list[str]) -> int:
    """Return the maximum antichain size via the bipartite matching theorem."""
    index = {element: position for position, element in enumerate(elements)}
    adjacent: list[list[int]] = [[] for _ in elements]
    for pair in poset.strict_order_pairs:
        adjacent[index[pair.lower]].append(index[pair.upper])

    matched_upper = [-1] * len(elements)

    def augment(lower: int, seen: set[int]) -> bool:
        for upper in adjacent[lower]:
            if upper in seen:
                continue
            seen.add(upper)
            if matched_upper[upper] == -1 or augment(matched_upper[upper], seen):
                matched_upper[upper] = lower
                return True
        return False

    matching = sum(augment(lower, set()) for lower in range(len(elements)))
    return len(elements) - matching
