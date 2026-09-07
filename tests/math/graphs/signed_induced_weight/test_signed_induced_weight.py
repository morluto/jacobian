from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.optimization._models import (
    RationalWeightedEdge,
    RationalWeightedGraph,
)
from jacobian.math.graphs.signed_induced_weight._bounds import (
    MAX_SIGNED_WEIGHT_WORK_UNITS,
    admit_signed_induced_weight,
)
from jacobian.math.graphs.signed_induced_weight._models import (
    MAX_SIGNED_WEIGHT_EDGES,
    MAX_SIGNED_WEIGHT_VERTICES,
    SignedInducedWeightRequest,
)
from jacobian.math.graphs.signed_induced_weight.operations import (
    signed_induced_weight_extrema,
)


def _edge(a: str, b: str, w: int | Fraction) -> RationalWeightedEdge:
    frac = Fraction(w, 1) if isinstance(w, int) else w
    return RationalWeightedEdge(
        endpoints=(a, b) if a < b else (b, a),
        weight=CanonicalRational.from_fraction(frac),
    )


def _graph(
    vertices: Sequence[str], edges: Sequence[RationalWeightedEdge]
) -> RationalWeightedGraph:
    return RationalWeightedGraph(
        vertices=tuple(vertices),
        edges=tuple(edges),
    )


def _simple_graph(
    vertices: Sequence[str],
    edge_specs: Sequence[tuple[str, str, int | Fraction]],
) -> RationalWeightedGraph:
    return _graph(vertices, [_edge(a, b, w) for a, b, w in edge_specs])


def test_empty_graph() -> None:
    """Empty graph has min=max=0."""
    g = _simple_graph([], [])
    result = signed_induced_weight_extrema(g)
    assert result.minimum.value.as_fraction() == Fraction(0)
    assert result.maximum.value.as_fraction() == Fraction(0)
    assert result.minimum.witness_vertices == ()
    assert result.maximum.witness_vertices == ()


def test_single_edge_positive() -> None:
    """Graph with one positive edge: max picks both endpoints."""
    g = _simple_graph(["a", "b"], [("a", "b", 3)])
    result = signed_induced_weight_extrema(g)
    assert result.maximum.value.as_fraction() == Fraction(3)
    assert set(result.maximum.witness_vertices) == {"a", "b"}
    assert result.minimum.value.as_fraction() == Fraction(0)


def test_single_edge_negative() -> None:
    """Graph with one negative edge: min picks both endpoints, max stays at 0."""
    g = _simple_graph(["a", "b"], [("a", "b", -5)])
    result = signed_induced_weight_extrema(g)
    assert result.minimum.value.as_fraction() == Fraction(-5)
    assert set(result.minimum.witness_vertices) == {"a", "b"}
    assert result.maximum.value.as_fraction() == Fraction(0)


def test_mixed_weights_fixture() -> None:
    """Fixture from issue: w01=2, w02=-1, w12=-1 on three vertices."""
    g = _simple_graph(
        ["0", "1", "2"],
        [("0", "1", 2), ("0", "2", -1), ("1", "2", -1)],
    )
    result = signed_induced_weight_extrema(g)
    assert result.maximum.value.as_fraction() == Fraction(2)
    assert result.minimum.value.as_fraction() == Fraction(-1)

    attained = []
    for size in range(len(g.vertices) + 1):
        for selected in combinations(g.vertices, size):
            selected_set = set(selected)
            attained.append(
                sum(
                    (
                        edge.weight.as_fraction()
                        for edge in g.edges
                        if set(edge.endpoints) <= selected_set
                    ),
                    start=Fraction(0),
                )
            )
    assert result.minimum.value.as_fraction() == min(attained)
    assert result.maximum.value.as_fraction() == max(attained)


def test_edgeless_graph() -> None:
    """Edgeless graph: min=max=0 for all subsets."""
    g = _simple_graph(["a", "b", "c"], [])
    result = signed_induced_weight_extrema(g)
    assert result.minimum.value.as_fraction() == Fraction(0)
    assert result.maximum.value.as_fraction() == Fraction(0)


def test_rational_weights() -> None:
    """Test with non-integer rational weights."""
    g = _simple_graph(["a", "b"], [("a", "b", Fraction(1, 2))])
    result = signed_induced_weight_extrema(g)
    assert result.maximum.value.as_fraction() == Fraction(1, 2)


def test_witness_replay() -> None:
    """Replay the witness subset to verify the weight."""
    g = _simple_graph(
        ["a", "b", "c", "d"],
        [("a", "b", 1), ("b", "c", -2), ("c", "d", 3), ("a", "d", -1)],
    )
    result = signed_induced_weight_extrema(g)

    def replay(selected: Sequence[str]) -> Fraction:
        total = Fraction(0)
        sset = set(selected)
        for edge in g.edges:
            a, b = edge.endpoints
            if a in sset and b in sset:
                total += edge.weight.as_fraction()
        return total

    assert replay(result.minimum.witness_vertices) == result.minimum.value.as_fraction()
    assert replay(result.maximum.witness_vertices) == result.maximum.value.as_fraction()


def test_tie_breaking_lexicographic() -> None:
    """Ties use tuple lexicographic order, not cardinality or search order."""
    g = _simple_graph(["a", "b", "c"], [("b", "c", -1)])
    result = signed_induced_weight_extrema(g)
    assert result.minimum.value.as_fraction() == Fraction(-1)
    assert result.minimum.witness_vertices == ("a", "b", "c")


def test_rejects_too_many_vertices() -> None:
    """Vertex counts beyond the shared carrier are still rejected."""
    vertices = [str(i) for i in range(33)]
    with pytest.raises(ValidationError):
        SignedInducedWeightRequest.model_validate(
            {"graph": {"vertices": vertices, "edges": []}}
        )


def test_connected_twenty_one_vertex_component_still_rejected() -> None:
    """One support component beyond the per-component envelope is rejected."""
    vertices = [f"v{i:02}" for i in range(21)]
    g = _simple_graph(
        vertices,
        [(left, right, 1) for left, right in combinations(vertices, 2)],
    )
    with pytest.raises(
        OperationDomainValidationError,
        match="support component exceeds",
    ):
        signed_induced_weight_extrema(g)


def test_edgeless_twenty_one_vertex_graph_is_admitted() -> None:
    """Zero support needs no search: min=max=0 with empty witnesses."""
    vertices = [f"v{i:02}" for i in range(21)]
    result = signed_induced_weight_extrema(_simple_graph(vertices, []))
    assert result.minimum.value.as_fraction() == Fraction(0)
    assert result.maximum.value.as_fraction() == Fraction(0)
    assert result.minimum.witness_vertices == ()
    assert result.maximum.witness_vertices == ()


def test_disjoint_edge_components_compose() -> None:
    """11 disjoint signed edges: max 6, min -5, cross-checked by replay."""
    vertices = [f"v{i:02}" for i in range(22)]
    g = _simple_graph(
        vertices,
        [
            (f"v{2 * i:02}", f"v{2 * i + 1:02}", 1 if i % 2 == 0 else -1)
            for i in range(11)
        ],
    )
    result = signed_induced_weight_extrema(g)
    assert result.maximum.value.as_fraction() == Fraction(6)
    assert result.minimum.value.as_fraction() == Fraction(-5)

    def replay(selected: Sequence[str]) -> Fraction:
        total = Fraction(0)
        selected_set = set(selected)
        for edge in g.edges:
            left, right = edge.endpoints
            if left in selected_set and right in selected_set:
                total += edge.weight.as_fraction()
        return total

    assert replay(result.maximum.witness_vertices) == Fraction(6)
    assert replay(result.minimum.witness_vertices) == Fraction(-5)


def test_component_search_matches_brute_force_values_and_witnesses() -> None:
    """Exhaustive small-graph agreement on optima and lex-least witnesses."""

    from itertools import product

    vertices = ("a", "b", "c")
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]
    for weights in product((-1, 0, 1), repeat=len(pairs)):
        specs = [
            (left, right, weight)
            for (left, right), weight in zip(pairs, weights, strict=True)
        ]
        graph = _simple_graph(vertices, specs)
        result = signed_induced_weight_extrema(graph)
        attained: dict[tuple[str, ...], Fraction] = {}
        for size in range(len(vertices) + 1):
            for selected in combinations(vertices, size):
                selected_set = set(selected)
                attained[selected] = sum(
                    (
                        edge.weight.as_fraction()
                        for edge in graph.edges
                        if set(edge.endpoints) <= selected_set
                    ),
                    start=Fraction(0),
                )
        best_min = min(attained.values())
        best_max = max(attained.values())
        assert result.minimum.value.as_fraction() == best_min
        assert result.maximum.value.as_fraction() == best_max
        assert result.minimum.witness_vertices == min(
            selected for selected, value in attained.items() if value == best_min
        )
        assert result.maximum.witness_vertices == min(
            selected for selected, value in attained.items() if value == best_max
        )


def test_request_schema_publishes_the_operation_specific_graph_envelope() -> None:
    graph_schema = SignedInducedWeightRequest.model_json_schema()["properties"]["graph"]
    assert graph_schema["properties"]["vertices"]["maxItems"] == (
        MAX_SIGNED_WEIGHT_VERTICES
    )
    assert graph_schema["properties"]["edges"]["maxItems"] == MAX_SIGNED_WEIGHT_EDGES
    assert str(MAX_SIGNED_WEIGHT_VERTICES) in graph_schema["description"]


def test_complete_twenty_vertex_integer_graph_fits_the_derived_work_budget() -> None:
    vertices = tuple(f"v{index:02d}" for index in range(20))
    graph = _simple_graph(
        vertices,
        [(left, right, 1) for left, right in combinations(vertices, 2)],
    )
    admission = admit_signed_induced_weight(graph)
    assert len(admission.components) == 1
    assert admission.components[0].candidate_subsets == 1 << len(vertices)
    assert admission.work_units <= MAX_SIGNED_WEIGHT_WORK_UNITS


def _first_primes(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % prime for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def test_rejects_unrepresentable_rational_height_before_search() -> None:
    vertices = tuple(f"v{index:02d}" for index in range(17))
    endpoint_pairs = tuple(combinations(vertices, 2))[:129]
    edges = []
    for (left, right), prime in zip(
        endpoint_pairs, _first_primes(len(endpoint_pairs)), strict=True
    ):
        exponent = 1
        while len(str(prime ** (exponent + 1))) <= 256:
            exponent += 1
        edges.append(
            RationalWeightedEdge(
                endpoints=(left, right),
                weight=CanonicalRational(num=1, den=prime**exponent),
            )
        )
    graph = _graph(vertices, edges)

    with pytest.raises(
        OperationDomainValidationError,
        match=r"common denominator.*32,768-digit rational bound",
    ):
        signed_induced_weight_extrema(graph)
