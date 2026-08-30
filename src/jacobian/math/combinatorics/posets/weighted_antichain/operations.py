"""Maximum weight antichain kernel using min-cut reduction."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from math import lcm as _lcm

from jacobian._exact import (
    CanonicalRational,
    canonical_rational_component_digits,
)
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
    max_digits = max(
        (canonical_rational_component_digits(weight) for weight in weights),
        default=1,
    )
    arithmetic_work = max(width, 1) * max_digits
    if arithmetic_work > MAX_ENUMERATION_WORK:
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.arithmetic_work_bound_exceeded",
            message="rational antichain summation exceeds the admitted work envelope",
        )
    # A singleton antichain returns the input weight unchanged; do not charge
    # an extra addition digit at that exact canonical boundary.
    denominators = {weight.as_fraction().denominator for weight in weights}
    if len(denominators) == 1:
        # Adding values with a common denominator does not multiply that
        # denominator at every summand; only the numerator can gain carry digits.
        max_sum_digits = max_digits + (len(str(width)) if width > 1 else 0)
    else:
        max_sum_digits = (
            max_digits if width <= 1 else width * max_digits + len(str(width))
        )
    if max_sum_digits > 32_768:
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.result_growth_exceeded",
            message="maximum-weight rational growth exceeds the canonical digit envelope",
        )
    rational_size = strict_json_object_size(
        (
            ("num", len(encode_strict_json("9" * max_sum_digits))),
            ("den", len(encode_strict_json("9" * max_sum_digits))),
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
    return _MaximumWeightAntichainAdmission(weights=weight_fracs)


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
    admission = _admit_maximum_weight_antichain(poset, weights)
    elements = list(poset.elements)
    n = len(elements)
    weight_fracs = admission.weights

    if n == 0:
        return MaximumWeightAntichainResult(
            poset=poset,
            weights=weights,
            maximum_weight=CanonicalRational.from_fraction(Fraction(0)),
            antichain=(),
        )

    # ------------------------------------------------------------------
    # Max weight antichain via min vertex cover on bipartite graph.
    #
    # For a poset P, the max weight antichain equals total weight minus
    # the min weight vertex cover of the bipartite graph G where:
    #   - Left vertices: copies of each poset element (v_L)
    #   - Right vertices: copies of each poset element (v_R)
    #   - For each order relation v < u: edge v_L -> u_R
    #
    # Min vertex cover on a bipartite graph is found via max-flow:
    #   source -> v_L with capacity w_v (for each element v)
    #   v_R -> sink with capacity w_v (for each element v)
    #   v_L -> u_R with infinite capacity (for each order v < u)
    #
    # The min cut value = min vertex cover weight.
    # Max antichain weight = total weight - min cut.
    # ------------------------------------------------------------------

    idx = {e: i for i, e in enumerate(elements)}
    order_pairs: list[tuple[int, int]] = []
    for pair in poset.strict_order_pairs:
        order_pairs.append((idx[pair.lower], idx[pair.upper]))

    # Convert fractions to common denominator for integer flow network.
    common_den = 1
    for w in weight_fracs:
        common_den = _lcm(common_den, w.denominator)
    int_weights = [int(w * common_den) for w in weight_fracs]
    total_weight_int = sum(int_weights)

    # Build flow network: source, sink, n left nodes, n right nodes
    SOURCE = 0
    SINK = 1
    left = lambda i: 2 + i        # v_L
    right = lambda i: 2 + n + i   # v_R
    num_nodes = 2 + 2 * n
    INF_CAP = 10**18

    adj: list[dict[int, int]] = [dict() for _ in range(num_nodes)]

    # source -> v_L with capacity w_v
    for i in range(n):
        w = int_weights[i]
        adj[SOURCE][left(i)] = w
        adj[left(i)].setdefault(SOURCE, 0)

    # v_R -> sink with capacity w_v
    for i in range(n):
        w = int_weights[i]
        adj[right(i)][SINK] = w
        adj[SINK].setdefault(right(i), 0)

    # v_L -> u_R with infinite capacity for each order v < u
    for v, u in order_pairs:
        old = adj[left(v)].get(right(u), 0)
        adj[left(v)][right(u)] = old + INF_CAP
        adj[right(u)].setdefault(left(v), 0)

    # Edmonds-Karp BFS max-flow
    total_flow = 0
    while True:
        parent = [-1] * num_nodes
        parent[SOURCE] = SOURCE
        queue = deque([SOURCE])
        found = False
        while queue:
            node = queue.popleft()
            for v, cap in adj[node].items():
                if parent[v] == -1 and cap > 0:
                    parent[v] = node
                    if v == SINK:
                        found = True
                        break
                    queue.append(v)
            if found:
                break
        if parent[SINK] == -1:
            break
        path_flow = float('inf')
        v = SINK
        while v != SOURCE:
            u = parent[v]
            path_flow = min(path_flow, adj[u][v])
            v = u
        v = SINK
        while v != SOURCE:
            u = parent[v]
            adj[u][v] -= path_flow
            adj[v][u] += path_flow
            v = u
        total_flow += path_flow

    min_vertex_cover = total_flow
    max_antichain_weight_int = total_weight_int - min_vertex_cover

    # Find min vertex cover set via König's theorem:
    # After max-flow, find nodes reachable from source in residual graph.
    source_reachable = {SOURCE}
    queue = deque([SOURCE])
    while queue:
        node = queue.popleft()
        for v, cap in adj[node].items():
            if v not in source_reachable and cap > 0:
                source_reachable.add(v)
                queue.append(v)

    # Min vertex cover: L nodes NOT reachable from source, R nodes reachable from source
    in_vertex_cover = set()
    for i in range(n):
        if left(i) not in source_reachable:
            in_vertex_cover.add(i)  # element i's left copy is in the cover
    for i in range(n):
        if right(i) in source_reachable:
            in_vertex_cover.add(i)  # element i's right copy is in the cover

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
