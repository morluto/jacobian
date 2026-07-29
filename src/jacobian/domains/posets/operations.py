"""Exact finite-poset producers backed by maintained NetworkX primitives."""

from __future__ import annotations

from typing import Any

import networkx as nx

from jacobian.contracts.posets import (
    ElementRank,
    FinitePoset,
    FinitePosetMaterializationResult,
    FinitePosetRequest,
    IncomparablePair,
    LinearExtensionCountResult,
    LinearExtensionRequest,
    LinearExtensionState,
    MatchingEdge,
    MobiusContribution,
    MobiusFunctionRequest,
    MobiusFunctionResult,
    MobiusScope,
    MobiusValue,
    OrderedPair,
    PosetChain,
    PosetInterval,
    PosetRequest,
    PosetWidthResult,
    RelationInterpretation,
    finite_poset_digest,
    linear_extension_memo_digest,
)
from jacobian.domains._examples import example
from jacobian.operations import ComputedOperation, ComputedSuccess


def _presentation_graph(request: FinitePosetRequest) -> nx.DiGraph[str]:
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(request.elements)
    graph.add_edges_from(
        (pair.lower, pair.upper)
        for pair in request.relation
        if pair.lower != pair.upper
    )
    return graph


def _rank_entries(
    elements: tuple[str, ...],
    covers: tuple[tuple[str, str], ...],
) -> tuple[ElementRank, ...] | None:
    predecessors: dict[str, set[str]] = {element: set() for element in elements}
    successors: dict[str, set[str]] = {element: set() for element in elements}
    for lower, upper in covers:
        predecessors[upper].add(lower)
        successors[lower].add(upper)
    ranks: dict[str, int] = {}
    remaining = set(elements)
    while remaining:
        ready = sorted(
            element for element in remaining if predecessors[element].issubset(ranks)
        )
        if not ready:
            raise ValueError("poset cover relation is cyclic")
        for element in ready:
            parent_ranks = {ranks[parent] for parent in predecessors[element]}
            if len(parent_ranks) > 1:
                return None
            ranks[element] = 0 if not parent_ranks else next(iter(parent_ranks)) + 1
            remaining.remove(element)
    maximal_ranks = {ranks[element] for element in elements if not successors[element]}
    if len(maximal_ranks) > 1:
        return None
    return tuple(
        ElementRank(element=element, rank=ranks[element]) for element in elements
    )


def _materialized_poset(request: FinitePosetRequest) -> FinitePoset:
    graph = _presentation_graph(request)
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
    ranks = _rank_entries(elements, covers)
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
) -> ComputedSuccess[FinitePosetMaterializationResult]:
    return ComputedSuccess(
        FinitePosetMaterializationResult(poset=_materialized_poset(request))
    )


def _width(request: PosetRequest) -> ComputedSuccess[PosetWidthResult]:
    poset = request.poset
    elements = poset.elements
    left_nodes = tuple(("L", element) for element in elements)
    right_nodes = tuple(("R", element) for element in elements)
    graph: nx.Graph[tuple[str, str]] = nx.Graph()
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
    return ComputedSuccess(
        PosetWidthResult(
            poset_digest=poset.poset_digest,
            width=len(chains),
            maximum_antichain=antichain,
            minimum_chain_cover=tuple(chains),
            matching=matching,
            matching_size=len(matching),
        )
    )


def _linear_extensions(
    request: LinearExtensionRequest,
) -> ComputedSuccess[LinearExtensionCountResult]:
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
    states = [
        LinearExtensionState(
            ideal_mask=0,
            cardinality=0,
            removable_maximal_elements=(),
            count=1,
        )
    ]
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
        states.append(
            LinearExtensionState(
                ideal_mask=mask,
                cardinality=mask.bit_count(),
                removable_maximal_elements=removable,
                count=count,
            )
        )
    state_tuple = tuple(states)
    return ComputedSuccess(
        LinearExtensionCountResult(
            poset_digest=poset.poset_digest,
            element_order=elements,
            count=counts[subset_count - 1],
            states=state_tuple,
            state_count=len(state_tuple),
            explored_subset_count=subset_count,
            memo_digest=linear_extension_memo_digest(state_tuple),
        )
    )


def _mobius(
    request: MobiusFunctionRequest,
) -> ComputedSuccess[MobiusFunctionResult]:
    poset = request.poset
    graph: nx.DiGraph[str] = nx.DiGraph()
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
            recurrence_contributions=tuple(
                MobiusContribution(intermediate=middle, value=value)
                for middle, value in contributions[(lower, upper)]
            ),
        )
        for lower, upper in requested
    )
    return ComputedSuccess(
        MobiusFunctionResult(
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

FINITE_POSET_CAPABILITIES = (
    ComputedOperation(
        capability_id="poset.finite.materialize",
        title="Materialize a canonical finite poset",
        description=(
            "Validate exact cover edges or a complete comparable relation and "
            "return canonical closure, Hasse reduction, incomparability, extrema, "
            "and ranks exactly when the poset is graded."
        ),
        request_model=FinitePosetRequest,
        result_model=FinitePosetMaterializationResult,
        implementation=_materialize,
        relation_id="poset.finite.materialization.relation",
        tags=(
            "poset",
            "partial-order",
            "hasse-diagram",
            "transitive-closure",
            "exact",
        ),
        invocation_examples=(
            example(
                "diamond",
                "Materialize the four-element diamond from its cover relation.",
                _DIAMOND,
            ),
        ),
    ),
    ComputedOperation(
        capability_id="poset.width.compute",
        title="Compute finite-poset width with dual witnesses",
        description=(
            "Return an exact maximum antichain and a same-size minimum chain "
            "partition, with the bipartite matching intermediate."
        ),
        request_model=PosetRequest,
        result_model=PosetWidthResult,
        implementation=_width,
        relation_id="poset.width.dilworth.relation",
        tags=(
            "poset",
            "width",
            "maximum-antichain",
            "minimum-chain-cover",
            "dilworth",
            "exact",
        ),
    ),
    ComputedOperation(
        capability_id="poset.linear_extensions.count",
        title="Count linear extensions of a bounded finite poset",
        description=(
            "Count every linear extension exactly and expose the complete "
            "order-ideal subset recurrence table and its canonical digest."
        ),
        request_model=LinearExtensionRequest,
        result_model=LinearExtensionCountResult,
        implementation=_linear_extensions,
        relation_id="poset.linear_extensions.ideal_dp.relation",
        tags=(
            "poset",
            "linear-extension",
            "exact-count",
            "order-ideal",
            "dynamic-programming",
        ),
    ),
    ComputedOperation(
        capability_id="poset.mobius_function.compute",
        title="Compute finite-poset Möbius values",
        description=(
            "Return exact incidence-algebra Möbius values for either every "
            "interval or an explicit selected interval scope, with recurrence terms."
        ),
        request_model=MobiusFunctionRequest,
        result_model=MobiusFunctionResult,
        implementation=_mobius,
        relation_id="poset.mobius_function.recurrence.relation",
        tags=(
            "poset",
            "mobius-function",
            "incidence-algebra",
            "interval",
            "exact",
        ),
    ),
)

__all__ = ["FINITE_POSET_CAPABILITIES"]
