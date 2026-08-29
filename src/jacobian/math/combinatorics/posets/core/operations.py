"""Native finite-poset operations backed by maintained NetworkX primitives."""

from __future__ import annotations

import importlib
from typing import Any, Literal

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.posets.core._closure_kernel import (
    dual_poset,
    induced_subposet,
    lower_closure,
    upper_closure,
)
from jacobian.math.combinatorics.posets.core._models import (
    MAX_ANTICHAIN_PROFILE_CANDIDATES,
    MAX_ANTICHAIN_PROFILE_ELEMENTS,
    MAX_LINEAR_EXTENSION_ELEMENTS,
    AntichainProfileResult,
    FinitePoset,
    IncidenceConvolutionResult,
    IncomparablePair,
    LinearExtensionCountResult,
    MatchingEdge,
    MobiusContribution,
    MobiusFunctionResult,
    MobiusScope,
    MobiusValue,
    OrderedPair,
    PosetChain,
    PosetClosureResult,
    PosetInterval,
    PosetSubset,
    PosetWidthResult,
    PresentationPair,
    ReflexivePairPolicy,
    RelationInterpretation,
    ZetaTransformResult,
    _validated_presentation,
    canonical_poset_ranks,
    finite_poset_digest,
)


def _run_admission(admission: Any) -> None:
    """Expose owner admission as a typed native-domain failure."""

    try:
        admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("poset",), code=exc.type, message=exc.message()
        ) from exc


def _admit_finite_poset(
    elements: tuple[str, ...],
    relation: tuple[PresentationPair, ...],
    interpretation: RelationInterpretation,
    reflexive_pairs: ReflexivePairPolicy,
) -> None:
    """Verify that a presentation describes a finite partial order."""

    _run_admission(
        lambda: _validated_presentation(
            elements, relation, interpretation, reflexive_pairs
        )
    )


def _admit_antichain_profile(poset: FinitePoset) -> None:
    """Admit the complete subset enumeration used by the profile kernel."""

    if len(poset.elements) > MAX_ANTICHAIN_PROFILE_ELEMENTS:
        raise OperationDomainValidationError(
            location=("poset", "elements"),
            code="poset.antichain_profile_elements",
            message=(
                "antichain profiles enumerate every subset and accept at most "
                f"{MAX_ANTICHAIN_PROFILE_ELEMENTS} elements "
                f"({MAX_ANTICHAIN_PROFILE_CANDIDATES} candidate subsets)"
            ),
        )


def _networkx() -> Any:
    """Load the maintained graph backend only when a poset operation runs."""

    return importlib.import_module("networkx")


def _presentation_graph(
    elements: tuple[str, ...],
    relation: tuple[PresentationPair, ...],
    nx: Any,
) -> Any:
    graph = nx.DiGraph()
    graph.add_nodes_from(elements)
    graph.add_edges_from(
        (pair.lower, pair.upper) for pair in relation if pair.lower != pair.upper
    )
    return graph


def materialize_finite_poset(
    elements: tuple[str, ...],
    relation: tuple[PresentationPair, ...],
    interpretation: RelationInterpretation,
    reflexive_pairs: ReflexivePairPolicy,
) -> FinitePoset:
    _admit_finite_poset(elements, relation, interpretation, reflexive_pairs)
    nx = _networkx()
    graph = _presentation_graph(elements, relation, nx)
    if interpretation is RelationInterpretation.COVER_EDGES:
        reduction_graph = graph
        closure_graph = nx.transitive_closure_dag(graph)
    else:
        closure_graph = graph
        reduction_graph = nx.transitive_reduction(graph)
    elements = tuple(sorted(elements))
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


def width(poset: FinitePoset) -> PosetWidthResult:
    nx = _networkx()
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


def linear_extension_count(poset: FinitePoset) -> LinearExtensionCountResult:
    elements = poset.elements
    if len(elements) > MAX_LINEAR_EXTENSION_ELEMENTS:
        raise OperationDomainValidationError(
            location=("poset", "elements"),
            code="poset.linear_extension_size_bound",
            message=(
                "linear-extension counting supports at most "
                f"{MAX_LINEAR_EXTENSION_ELEMENTS} elements"
            ),
        )
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


def mobius_function(
    poset: FinitePoset,
    scope: MobiusScope,
    intervals: tuple[PosetInterval, ...],
) -> MobiusFunctionResult:
    return _compute_mobius(poset, scope, intervals, include_recurrence=False)


def _compute_mobius(
    poset: FinitePoset,
    scope: MobiusScope,
    intervals: tuple[PosetInterval, ...],
    *,
    include_recurrence: bool,
) -> MobiusFunctionResult:
    nx = _networkx()
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

    if scope is MobiusScope.COMPLETE_MATRIX:
        requested = tuple(
            sorted(
                (lower, upper)
                for lower in poset.elements
                for upper in poset.elements
                if lower == upper or (lower, upper) in closure
            )
        )
        result_intervals: tuple[PosetInterval, ...] = ()
    else:
        requested = tuple(
            sorted((interval.lower, interval.upper) for interval in intervals)
        )
        result_intervals = tuple(
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
        scope=scope,
        intervals=result_intervals,
        values=values,
        completeness=(
            "COMPLETE_MATRIX"
            if scope is MobiusScope.COMPLETE_MATRIX
            else "SELECTED_INTERVALS"
        ),
    )


def closure(
    poset: FinitePoset,
    subset: PosetSubset,
    closure_type: Literal["LOWER", "UPPER"],
) -> PosetClosureResult:
    elements = set(poset.elements)
    order_pairs = {(p.lower, p.upper) for p in poset.strict_order_pairs}
    order_set = order_pairs | {(e, e) for e in elements}
    if closure_type == "LOWER":
        result = set()
        for target in subset.elements:
            for lo, hi in order_set:
                if hi == target:
                    result.add(lo)
        result |= set(subset.elements)
    else:
        result = set()
        for target in subset.elements:
            for lo, hi in order_set:
                if lo == target:
                    result.add(hi)
        result |= set(subset.elements)
    return PosetClosureResult(
        poset_digest=poset.poset_digest,
        closure_type=closure_type,
        closure=tuple(sorted(result)),
        generated_element=tuple(sorted(result - set(subset.elements))),
    )


def zeta_transform(
    poset: FinitePoset,
    function_values: tuple[MobiusValue, ...],
) -> ZetaTransformResult:
    comparable = {(p.lower, p.upper) for p in poset.strict_order_pairs}
    func_lookup = {(v.lower, v.upper): v.value for v in function_values}
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


def incidence_convolution(
    poset: FinitePoset,
    first: tuple[MobiusValue, ...],
    second: tuple[MobiusValue, ...],
) -> IncidenceConvolutionResult:
    comparable = {(p.lower, p.upper) for p in poset.strict_order_pairs}
    first_lookup = {(v.lower, v.upper): v.value for v in first}
    second_lookup = {(v.lower, v.upper): v.value for v in second}
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


def antichain_profile(poset: FinitePoset) -> AntichainProfileResult:
    _admit_antichain_profile(poset)
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
    max_antichains: list[tuple[str, ...]] = [()]
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


__all__ = [
    "antichain_profile",
    "closure",
    "dual_poset",
    "incidence_convolution",
    "induced_subposet",
    "linear_extension_count",
    "lower_closure",
    "materialize_finite_poset",
    "mobius_function",
    "upper_closure",
    "width",
    "zeta_transform",
]
