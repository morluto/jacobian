"""Exact finite-poset producers backed by maintained NetworkX primitives."""

from __future__ import annotations

import importlib
from typing import Any

from jacobian.contracts.posets import (
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
    canonical_poset_ranks,
    finite_poset_digest,
    linear_extension_memo_digest,
)
from jacobian.domains._examples import example
from jacobian.operation_bindings import (
    InstalledOperation,
    durable_operation,
    inline_operation,
)
from jacobian.operations import OperationSpec


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
    return LinearExtensionCountResult(
        poset_digest=poset.poset_digest,
        element_order=elements,
        count=counts[subset_count - 1],
        states=state_tuple,
        state_count=len(state_tuple),
        explored_subset_count=subset_count,
        memo_digest=linear_extension_memo_digest(state_tuple),
    )


def _mobius(
    request: MobiusFunctionRequest,
) -> MobiusFunctionResult:
    return _compute_mobius(request, include_recurrence=False)


def _materialize_mobius_recurrence(
    request: MobiusFunctionRequest,
) -> MobiusFunctionResult:
    return _compute_mobius(request, include_recurrence=True)


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

FINITE_POSET_CAPABILITIES: tuple[InstalledOperation[Any, Any], ...] = (
    inline_operation(
        OperationSpec(
            operation_id="poset.finite.compute",
            version="4",
            title="Compute a canonical finite poset",
            description=(
                "Validate exact cover edges or a complete comparable relation and "
                "return canonical closure, Hasse reduction, incomparability, extrema, "
                "and ranks exactly when the poset is graded."
            ),
            request_type=FinitePosetRequest,
            result_type=FinitePosetMaterializationResult,
            execute=_materialize,
            tags=(
                "poset",
                "partial-order",
                "partially-ordered-set",
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
        )
    ),
    inline_operation(
        OperationSpec(
            operation_id="poset.width.compute",
            version="4",
            title="Compute finite-poset width with dual witnesses",
            description=(
                "Return an exact maximum antichain and a same-size minimum chain "
                "partition, with the bipartite matching intermediate."
            ),
            request_type=PosetRequest,
            result_type=PosetWidthResult,
            execute=_width,
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
        )
    ),
    durable_operation(
        OperationSpec(
            operation_id="poset.linear_extensions.count",
            version="3",
            title="Count linear extensions of a bounded finite poset",
            description=(
                "Count every linear extension exactly and expose the complete "
                "order-ideal subset recurrence table and its canonical digest."
            ),
            request_type=LinearExtensionRequest,
            result_type=LinearExtensionCountResult,
            execute=_linear_extensions,
            tags=(
                "poset",
                "linear-extension",
                "exact-count",
                "order-ideal",
                "dynamic-programming",
            ),
        ),
        resource_reason=(
            "the full order-ideal recurrence table is retained for independent "
            "replay and exact count provenance"
        ),
    ),
    inline_operation(
        OperationSpec(
            operation_id="poset.mobius_function.compute",
            version="3",
            title="Compute finite-poset Möbius values",
            description=(
                "Return exact incidence-algebra Möbius values for either every "
                "interval or an explicit selected interval scope, with recurrence terms."
            ),
            request_type=MobiusFunctionRequest,
            result_type=MobiusFunctionResult,
            execute=_mobius,
            tags=(
                "poset",
                "mobius-function",
                "incidence-algebra",
                "interval",
                "exact",
            ),
        )
    ),
    durable_operation(
        OperationSpec(
            operation_id="poset.mobius_function.recurrence.materialize",
            version="3",
            title="Materialize the finite-poset Möbius recurrence table",
            description=(
                "Retain every interval-convolution recurrence contribution used by "
                "the bounded Möbius summary."
            ),
            request_type=MobiusFunctionRequest,
            result_type=MobiusFunctionResult,
            execute=_materialize_mobius_recurrence,
            tags=(
                "poset",
                "mobius-function",
                "recurrence",
                "ledger",
                "evidence",
            ),
        ),
        resource_reason=(
            "the full interval-convolution recurrence table is retained as "
            "explicit bulk evidence for independent replay"
        ),
    ),
)

__all__ = ["FINITE_POSET_CAPABILITIES"]
