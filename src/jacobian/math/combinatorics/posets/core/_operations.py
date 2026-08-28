"""Exact finite-poset producers backed by maintained NetworkX primitives."""

from __future__ import annotations

import importlib
from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.posets.core._models import (
    AntichainProfileRequest,
    AntichainProfileResult,
    FinitePoset,
    FinitePosetMaterializationResult,
    FinitePosetRequest,
    IncidenceConvolutionRequest,
    IncidenceConvolutionResult,
    IncomparablePair,
    LinearExtensionCountResult,
    LinearExtensionRequest,
    MatchingEdge,
    MobiusContribution,
    MobiusFunctionRequest,
    MobiusFunctionResult,
    MobiusScope,
    MobiusValue,
    OrderedPair,
    PosetChain,
    PosetClosureRequest,
    PosetClosureResult,
    PosetDualRequest,
    PosetDualResult,
    PosetInterval,
    PosetRequest,
    PosetWidthResult,
    RelationInterpretation,
    ZetaTransformRequest,
    ZetaTransformResult,
    canonical_poset_ranks,
    finite_poset_digest,
)


def _networkx() -> Any:
    """Load the maintained graph backend only when a poset operation runs."""

    return importlib.import_module("networkx")


def _presentation_graph(request: FinitePosetRequest, nx: Any) -> Any:
    graph = nx.DiGraph()
    graph.add_nodes_from(request.elements)
    graph.add_edges_from(
        (pair.lower, pair.upper)
        for pair in request.relation
        if pair.lower != pair.upper
    )
    return graph


def _materialized_poset(request: FinitePosetRequest) -> FinitePoset:
    nx = _networkx()
    graph = _presentation_graph(request, nx)
    if request.interpretation is RelationInterpretation.COVER_EDGES:
        reduction_graph = graph
        closure_graph = nx.transitive_closure_dag(graph)
    else:
        closure_graph = graph
        reduction_graph = nx.transitive_reduction(graph)
    elements = tuple(sorted(request.elements))
    strict_pairs = tuple(sorted(closure_graph.edges()))
    covers = tuple(sorted(reduction_graph.edges()))
    strict_set = set(strict_pairs)
    incomparable = tuple(
        IncomparablePair(left=left, right=right)
        for index, left in enumerate(elements)
        for right in elements[index + 1 :]
        if (left, right) not in strict_set and (right, left) not in strict_set
    )
    minimal = tuple(
        element for element in elements if closure_graph.in_degree(element) == 0
    )
    maximal = tuple(
        element for element in elements if closure_graph.out_degree(element) == 0
    )
    ranks = canonical_poset_ranks(elements, set(covers))
    order_pairs = tuple(
        OrderedPair(lower=lower, upper=upper) for lower, upper in strict_pairs
    )
    cover_pairs = tuple(
        OrderedPair(lower=lower, upper=upper) for lower, upper in covers
    )
    digest = finite_poset_digest(
        elements=elements,
        strict_order_pairs=order_pairs,
        cover_relations=cover_pairs,
        incomparable_pairs=incomparable,
        minimal_elements=minimal,
        maximal_elements=maximal,
        graded=ranks is not None,
        ranks=ranks,
    )
    return FinitePoset(
        elements=elements,
        strict_order_pairs=order_pairs,
        cover_relations=cover_pairs,
        incomparable_pairs=incomparable,
        minimal_elements=minimal,
        maximal_elements=maximal,
        graded=ranks is not None,
        ranks=ranks,
        poset_digest=digest,
    )


def _materialize(
    request: FinitePosetRequest,
) -> FinitePosetMaterializationResult:
    return FinitePosetMaterializationResult(poset=_materialized_poset(request))


def _width(request: PosetRequest) -> PosetWidthResult:
    nx = _networkx()
    poset = request.poset
    elements = poset.elements
    left_nodes = tuple(("L", element) for element in elements)
    right_nodes = tuple(("R", element) for element in elements)
    graph = nx.Graph()
    graph.add_nodes_from(left_nodes, bipartite=0)
    graph.add_nodes_from(right_nodes, bipartite=1)
    graph.add_edges_from(
        (("L", pair.lower), ("R", pair.upper)) for pair in poset.strict_order_pairs
    )
    raw_matching: dict[tuple[str, str], tuple[str, str]] = (
        nx.algorithms.bipartite.maximum_matching(graph, top_nodes=left_nodes)
        if elements
        else {}
    )
    successor = {
        node[1]: raw_matching[node][1] for node in left_nodes if node in raw_matching
    }
    matched_right = set(successor.values())
    matching = tuple(
        MatchingEdge(left=lower, right=upper)
        for lower, upper in sorted(successor.items())
    )

    chains: list[PosetChain] = []
    for start in sorted(set(elements) - matched_right):
        chain = [start]
        while chain[-1] in successor:
            chain.append(successor[chain[-1]])
        chains.append(PosetChain(elements=tuple(chain)))

    reachable_left = {node for node in left_nodes if node not in raw_matching}
    reachable_right: set[tuple[str, str]] = set()
    frontier: list[tuple[str, str]] = sorted(reachable_left)
    while frontier:
        left = frontier.pop()
        for right in sorted(graph[left]):
            if raw_matching.get(left) == right or right in reachable_right:
                continue
            reachable_right.add(right)
            matched_left = raw_matching.get(right)
            if matched_left is not None and matched_left not in reachable_left:
                reachable_left.add(matched_left)
                frontier.append(matched_left)
    antichain = tuple(
        element
        for element in elements
        if ("L", element) in reachable_left and ("R", element) not in reachable_right
    )
    return PosetWidthResult(
        poset_digest=poset.poset_digest,
        width=len(chains),
        maximum_antichain=antichain,
        minimum_chain_cover=tuple(chains),
        matching=matching,
        matching_size=len(matching),
    )


def _linear_extensions(
    request: LinearExtensionRequest,
) -> LinearExtensionCountResult:
    poset = request.poset
    elements = poset.elements
    index = {element: position for position, element in enumerate(elements)}
    predecessor_masks = [0] * len(elements)
    successor_masks = [0] * len(elements)
    for pair in poset.strict_order_pairs:
        lower = index[pair.lower]
        upper = index[pair.upper]
        predecessor_masks[upper] |= 1 << lower
        successor_masks[lower] |= 1 << upper

    counts: dict[int, int] = {0: 1}
    subset_count = 1 << len(elements)
    for mask in range(1, subset_count):
        if any(
            mask & (1 << position)
            and predecessor_masks[position] & mask != predecessor_masks[position]
            for position in range(len(elements))
        ):
            continue
        removable = tuple(
            elements[position]
            for position in range(len(elements))
            if mask & (1 << position) and successor_masks[position] & mask == 0
        )
        count = sum(counts[mask ^ (1 << index[element])] for element in removable)
        counts[mask] = count
    return LinearExtensionCountResult(
        count=counts[subset_count - 1],
    )


def _mobius(
    request: MobiusFunctionRequest,
) -> MobiusFunctionResult:
    return _compute_mobius(request, include_recurrence=False)


def _compute_mobius(
    request: MobiusFunctionRequest,
    *,
    include_recurrence: bool,
) -> MobiusFunctionResult:
    nx = _networkx()
    poset = request.poset
    graph = nx.DiGraph()
    graph.add_nodes_from(poset.elements)
    graph.add_edges_from((pair.lower, pair.upper) for pair in poset.strict_order_pairs)
    topological = tuple(nx.lexicographical_topological_sort(graph, key=str))
    closure = {(pair.lower, pair.upper) for pair in poset.strict_order_pairs}
    mu: dict[tuple[str, str], int] = {}
    contributions: dict[tuple[str, str], tuple[tuple[str, int], ...]] = {}
    for lower_index, lower in enumerate(topological):
        mu[(lower, lower)] = 1
        contributions[(lower, lower)] = ()
        for upper in topological[lower_index + 1 :]:
            if (lower, upper) not in closure:
                continue
            terms = tuple(
                sorted(
                    (middle, mu[(lower, middle)])
                    for middle in topological[: topological.index(upper)]
                    if middle == lower
                    or ((lower, middle) in closure and (middle, upper) in closure)
                )
            )
            mu[(lower, upper)] = -sum(value for _, value in terms)
            contributions[(lower, upper)] = terms

    if request.scope is MobiusScope.COMPLETE_MATRIX:
        requested = tuple(
            sorted(
                (lower, upper)
                for lower in poset.elements
                for upper in poset.elements
                if lower == upper or (lower, upper) in closure
            )
        )
        intervals: tuple[PosetInterval, ...] = ()
    else:
        requested = tuple(
            sorted((interval.lower, interval.upper) for interval in request.intervals)
        )
        intervals = tuple(
            PosetInterval(lower=lower, upper=upper) for lower, upper in requested
        )
    values = tuple(
        MobiusValue(
            lower=lower,
            upper=upper,
            value=mu[(lower, upper)],
            recurrence_contributions=(
                tuple(
                    MobiusContribution(intermediate=middle, value=value)
                    for middle, value in contributions[(lower, upper)]
                )
                if include_recurrence
                else None
            ),
        )
        for lower, upper in requested
    )
    return MobiusFunctionResult(
        poset_digest=poset.poset_digest,
        element_order=poset.elements,
        scope=request.scope,
        intervals=intervals,
        values=values,
        completeness=(
            "COMPLETE_MATRIX"
            if request.scope is MobiusScope.COMPLETE_MATRIX
            else "SELECTED_INTERVALS"
        ),
    )


_DIAMOND: dict[str, Any] = {
    "elements": ["0", "a", "b", "1"],
    "relation": [
        {"lower": "0", "upper": "a"},
        {"lower": "0", "upper": "b"},
        {"lower": "a", "upper": "1"},
        {"lower": "b", "upper": "1"},
    ],
    "interpretation": "COVER_EDGES",
    "reflexive_pairs": "FORBIDDEN",
}

_MATERIALIZED_DIAMOND: dict[str, Any] = {
    "elements": ["0", "1", "a", "b"],
    "strict_order_pairs": [
        {"lower": "0", "upper": "1"},
        {"lower": "0", "upper": "a"},
        {"lower": "0", "upper": "b"},
        {"lower": "a", "upper": "1"},
        {"lower": "b", "upper": "1"},
    ],
    "cover_relations": [
        {"lower": "0", "upper": "a"},
        {"lower": "0", "upper": "b"},
        {"lower": "a", "upper": "1"},
        {"lower": "b", "upper": "1"},
    ],
    "incomparable_pairs": [{"left": "a", "right": "b"}],
    "minimal_elements": ["0"],
    "maximal_elements": ["1"],
    "graded": True,
    "ranks": [
        {"element": "0", "rank": 0},
        {"element": "1", "rank": 2},
        {"element": "a", "rank": 1},
        {"element": "b", "rank": 1},
    ],
    "poset_digest": "sha256:55e795bf7924508b0aa0efe1d0cf32371858ff8b122c890d3eba357dfe2a3374",
}


def _closure(request: PosetClosureRequest) -> PosetClosureResult:
    poset = request.poset
    elements = set(poset.elements)
    order_pairs = {(p.lower, p.upper) for p in poset.strict_order_pairs}
    order_set = order_pairs | {(e, e) for e in elements}
    if request.closure_type == "LOWER":
        result = set()
        for target in request.subset.elements:
            for lo, hi in order_set:
                if hi == target:
                    result.add(lo)
        result |= set(request.subset.elements)
    else:
        result = set()
        for target in request.subset.elements:
            for lo, hi in order_set:
                if lo == target:
                    result.add(hi)
        result |= set(request.subset.elements)
    return PosetClosureResult(
        poset_digest=poset.poset_digest,
        closure_type=request.closure_type,
        closure=tuple(sorted(result)),
        generated_element=tuple(sorted(result - set(request.subset.elements))),
    )


def _dual(request: PosetDualRequest) -> PosetDualResult:
    poset = request.poset
    elements = poset.elements
    reversed_pairs = tuple(
        OrderedPair(lower=p.upper, upper=p.lower) for p in poset.strict_order_pairs
    )
    reversed_covers = tuple(
        OrderedPair(lower=p.upper, upper=p.lower) for p in poset.cover_relations
    )
    sorted_pairs = tuple(sorted((p.lower, p.upper) for p in reversed_pairs))
    sorted_covers = tuple(sorted((p.lower, p.upper) for p in reversed_covers))
    order_pairs_obj = tuple(OrderedPair(lower=lo, upper=hi) for lo, hi in sorted_pairs)
    cover_pairs_obj = tuple(OrderedPair(lower=lo, upper=hi) for lo, hi in sorted_covers)
    if poset.graded and poset.ranks is not None:
        height = max(r.rank for r in poset.ranks)
        dual_ranks = tuple(
            type(poset.ranks[0])(element=r.element, rank=height - r.rank)
            for r in sorted(poset.ranks, key=lambda r: r.element)
        )
        dual_ranks_sorted = tuple(sorted(dual_ranks, key=lambda r: r.element))
    else:
        dual_ranks_sorted = None
    new_digest = finite_poset_digest(
        elements=elements,
        strict_order_pairs=order_pairs_obj,
        cover_relations=cover_pairs_obj,
        incomparable_pairs=poset.incomparable_pairs,
        minimal_elements=tuple(sorted(poset.maximal_elements)),
        maximal_elements=tuple(sorted(poset.minimal_elements)),
        graded=poset.graded,
        ranks=dual_ranks_sorted,
    )
    new_poset = FinitePoset(
        elements=elements,
        strict_order_pairs=order_pairs_obj,
        cover_relations=cover_pairs_obj,
        incomparable_pairs=poset.incomparable_pairs,
        minimal_elements=tuple(sorted(poset.maximal_elements)),
        maximal_elements=tuple(sorted(poset.minimal_elements)),
        graded=poset.graded,
        ranks=dual_ranks_sorted,
        poset_digest=new_digest,
    )
    return PosetDualResult(poset=new_poset)


def _zeta_transform(request: ZetaTransformRequest) -> ZetaTransformResult:
    poset = request.poset
    comparable = {(p.lower, p.upper) for p in poset.strict_order_pairs}
    func_lookup = {(v.lower, v.upper): v.value for v in request.function_values}
    all_intervals = []
    for a in poset.elements:
        for c in poset.elements:
            if a == c or (a, c) in comparable:
                all_intervals.append((a, c))
    results = []
    for a, c in all_intervals:
        total = 0
        for b in poset.elements:
            if (b == a or (a, b) in comparable) and (b == c or (b, c) in comparable):
                total += func_lookup.get((a, b), 0)
        results.append(MobiusValue(lower=a, upper=c, value=total))
    return ZetaTransformResult(
        poset_digest=poset.poset_digest,
        values=tuple(results),
    )


def _incidence_convolution(
    request: IncidenceConvolutionRequest,
) -> IncidenceConvolutionResult:
    poset = request.poset
    comparable = {(p.lower, p.upper) for p in poset.strict_order_pairs}
    first_lookup = {(v.lower, v.upper): v.value for v in request.first}
    second_lookup = {(v.lower, v.upper): v.value for v in request.second}
    all_intervals = []
    for a in poset.elements:
        for c in poset.elements:
            if a == c or (a, c) in comparable:
                all_intervals.append((a, c))
    results = []
    for a, c in all_intervals:
        total = 0
        for b in poset.elements:
            if (b == a or (a, b) in comparable) and (b == c or (b, c) in comparable):
                total += first_lookup.get((a, b), 0) * second_lookup.get((b, c), 0)
        results.append(MobiusValue(lower=a, upper=c, value=total))
    return IncidenceConvolutionResult(
        poset_digest=poset.poset_digest,
        values=tuple(results),
    )


def _antichain_profile(
    request: AntichainProfileRequest,
) -> AntichainProfileResult:
    poset = request.poset
    elements = poset.elements
    comparable = {(p.lower, p.upper) for p in poset.strict_order_pairs}

    def is_antichain(subset: tuple[str, ...]) -> bool:
        for i, a in enumerate(subset):
            for b in subset[i + 1 :]:
                if (a, b) in comparable or (b, a) in comparable:
                    return False
        return True

    n = len(elements)
    max_size = 0
    max_antichains: list[tuple[str, ...]] = []
    antichain_count = 1
    for mask in range(1, 1 << n):
        subset = tuple(sorted(elements[i] for i in range(n) if mask & (1 << i)))
        if is_antichain(subset):
            antichain_count += 1
            if len(subset) > max_size:
                max_size = len(subset)
                max_antichains = [subset]
            elif len(subset) == max_size:
                max_antichains.append(subset)
    return AntichainProfileResult(
        poset_digest=poset.poset_digest,
        maximum_antichain_size=max_size,
        antichain_count=antichain_count,
        maximum_antichains=tuple(max_antichains),
    )


FINITE_POSET_OPERATIONS: MathTools = (
    MathTool(
        operation_id="poset.finite.compute",
        title="Compute a canonical finite poset",
        description=(
            "Validate exact cover edges or a complete comparable relation and "
            "return canonical closure, Hasse reduction, incomparability, extrema, "
            "and ranks exactly when the poset is graded."
        ),
        request_type=FinitePosetRequest,
        result_type=FinitePosetMaterializationResult,
        run=_materialize,
        tags=(
            "poset",
            "partial-order",
            "partially-ordered-set",
            "hasse-diagram",
            "transitive-closure",
            "exact",
        ),
        examples=(
            example(
                "diamond",
                "Materialize the four-element diamond from its cover relation.",
                _DIAMOND,
            ),
            example(
                "three_element_chain",
                "Materialize the chain 0<1<2; the relation must be antisymmetric and acyclic.",
                {
                    "elements": ["0", "1", "2"],
                    "relation": [
                        {"lower": "0", "upper": "1"},
                        {"lower": "0", "upper": "2"},
                        {"lower": "1", "upper": "2"},
                    ],
                    "interpretation": "COMPARABLE_PAIRS",
                    "reflexive_pairs": "FORBIDDEN",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="poset.width.compute",
        title="Compute finite-poset width with dual witnesses",
        description=(
            "Return an exact maximum antichain and a same-size minimum chain "
            "partition, with the bipartite matching intermediate."
        ),
        request_type=PosetRequest,
        result_type=PosetWidthResult,
        run=_width,
        tags=(
            "poset",
            "partial-order",
            "partially-ordered-set",
            "width",
            "maximum-antichain",
            "minimum-chain-cover",
            "dilworth",
            "exact",
        ),
        examples=(
            example(
                "materialized_diamond",
                "Compute the width of the canonical four-element diamond.",
                {"poset": _MATERIALIZED_DIAMOND},
            ),
        ),
    ),
    MathTool(
        operation_id="poset.linear_extensions.count",
        title="Count linear extensions of a bounded finite poset",
        description=("Count every linear extension of a bounded finite poset exactly."),
        request_type=LinearExtensionRequest,
        result_type=LinearExtensionCountResult,
        run=_linear_extensions,
        tags=(
            "poset",
            "linear-extension",
            "exact-count",
            "order-ideal",
            "dynamic-programming",
        ),
        examples=(
            example(
                "materialized_diamond",
                "Count the linear extensions of the canonical diamond.",
                {"poset": _MATERIALIZED_DIAMOND},
            ),
            example(
                "diamond_complete_mobius_scope",
                "Count the diamond's linear extensions; the poset must have at most 14 elements.",
                {"poset": _MATERIALIZED_DIAMOND},
            ),
        ),
    ),
    MathTool(
        operation_id="poset.mobius_function.compute",
        title="Compute finite-poset Möbius values",
        description=(
            "Return exact incidence-algebra Möbius values for either every "
            "interval or an explicit selected interval scope."
        ),
        request_type=MobiusFunctionRequest,
        result_type=MobiusFunctionResult,
        run=_mobius,
        tags=(
            "poset",
            "mobius-function",
            "incidence-algebra",
            "interval",
            "exact",
        ),
        examples=(
            example(
                "materialized_diamond",
                "Compute every Möbius value of the canonical diamond.",
                {"poset": _MATERIALIZED_DIAMOND},
            ),
            example(
                "diamond_selected_interval",
                "Compute the selected Möbius interval [0,1]; selected endpoints must satisfy lower <= upper in the poset.",
                {
                    "poset": _MATERIALIZED_DIAMOND,
                    "scope": "SELECTED_INTERVALS",
                    "intervals": [
                        {"lower": "0", "upper": "1"},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="poset.closure.compute",
        title="Compute ideal or filter closure of a subset",
        description=(
            "Return the lower (ideal) or upper (filter) closure of a given "
            "subset in a finite poset."
        ),
        request_type=PosetClosureRequest,
        result_type=PosetClosureResult,
        run=_closure,
        tags=(
            "poset",
            "ideal",
            "filter",
            "closure",
            "exact",
        ),
        examples=(
            example(
                "diamond_lower_closure",
                "Compute the lower closure of {1} in the diamond poset.",
                {
                    "poset": _MATERIALIZED_DIAMOND,
                    "subset": {"elements": ["1"]},
                    "closure_type": "LOWER",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="poset.zeta_transform.compute",
        title="Compute the zeta transform of a function on a poset",
        description=(
            "Apply the incidence-algebra zeta transform to a function "
            "defined on intervals of a finite poset."
        ),
        request_type=ZetaTransformRequest,
        result_type=ZetaTransformResult,
        run=_zeta_transform,
        tags=(
            "poset",
            "zeta-transform",
            "incidence-algebra",
            "exact",
        ),
        examples=(
            example(
                "diamond_zeta",
                "Compute the zeta transform of a constant function on the diamond.",
                {
                    "poset": _MATERIALIZED_DIAMOND,
                    "function_values": [
                        {"lower": "0", "upper": "0", "value": 1},
                        {"lower": "a", "upper": "a", "value": 1},
                        {"lower": "b", "upper": "b", "value": 1},
                        {"lower": "1", "upper": "1", "value": 1},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="poset.incidence_convolution.compute",
        title="Convolve two incidence-algebra functions on a poset",
        description=(
            "Compute the incidence-algebra convolution of two functions "
            "defined on intervals of a finite poset."
        ),
        request_type=IncidenceConvolutionRequest,
        result_type=IncidenceConvolutionResult,
        run=_incidence_convolution,
        tags=(
            "poset",
            "incidence-algebra",
            "convolution",
            "exact",
        ),
        examples=(
            example(
                "diamond_convolution",
                "Convolve the zeta function with itself on the diamond.",
                {
                    "poset": _MATERIALIZED_DIAMOND,
                    "first": [
                        {"lower": "0", "upper": "0", "value": 1},
                        {"lower": "0", "upper": "a", "value": 1},
                        {"lower": "0", "upper": "b", "value": 1},
                        {"lower": "0", "upper": "1", "value": 1},
                        {"lower": "a", "upper": "a", "value": 1},
                        {"lower": "a", "upper": "1", "value": 1},
                        {"lower": "b", "upper": "b", "value": 1},
                        {"lower": "b", "upper": "1", "value": 1},
                        {"lower": "1", "upper": "1", "value": 1},
                    ],
                    "second": [
                        {"lower": "0", "upper": "0", "value": 1},
                        {"lower": "0", "upper": "a", "value": 1},
                        {"lower": "0", "upper": "b", "value": 1},
                        {"lower": "0", "upper": "1", "value": 1},
                        {"lower": "a", "upper": "a", "value": 1},
                        {"lower": "a", "upper": "1", "value": 1},
                        {"lower": "b", "upper": "b", "value": 1},
                        {"lower": "b", "upper": "1", "value": 1},
                        {"lower": "1", "upper": "1", "value": 1},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="poset.antichain_profile.compute",
        title="Compute the antichain profile of a finite poset",
        description=(
            "Return the maximum antichain size, total antichain count, and "
            "all maximum antichains by enumerating the at most 16,384 subsets "
            "of a poset with at most 14 elements."
        ),
        request_type=AntichainProfileRequest,
        result_type=AntichainProfileResult,
        run=_antichain_profile,
        tags=(
            "poset",
            "antichain",
            "profile",
            "exact",
        ),
        examples=(
            example(
                "materialized_diamond",
                "Compute the antichain profile of the canonical diamond.",
                {"poset": _MATERIALIZED_DIAMOND},
            ),
        ),
    ),
)

__all__ = ["FINITE_POSET_OPERATIONS"]
