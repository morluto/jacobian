"""Exact site reliability: /#1688 (undirected bond reliability already existed)."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.probability._site_reliability import (
    GraphSiteReliabilityResult,
    GraphSiteReliabilitySource,
    SiteReliabilityVertexProbability,
    compute_site_connection_probability,
)


def _probability(value: Fraction) -> CanonicalRational:
    return CanonicalRational(num=value.numerator, den=value.denominator)


def _source(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    probabilities: tuple[Fraction, ...],
    terminals: tuple[str, str],
) -> GraphSiteReliabilitySource:
    ordered = tuple(sorted(vertices))
    lookup = dict(zip(vertices, probabilities, strict=True))
    return GraphSiteReliabilitySource(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        vertex_probabilities=tuple(
            SiteReliabilityVertexProbability(
                vertex=vertex, open_probability=_probability(lookup[vertex])
            )
            for vertex in ordered
        ),
        terminals=terminals,
    )


def _brute_force(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    probabilities: tuple[Fraction, ...],
    terminals: tuple[str, str],
) -> Fraction:
    total = Fraction(0)
    for mask in range(1 << len(vertices)):
        open_set = {
            vertex for index, vertex in enumerate(vertices) if mask & (1 << index)
        }
        mass = Fraction(1)
        for index, vertex in enumerate(vertices):
            probability = probabilities[index]
            mass *= probability if vertex in open_set else 1 - probability
        if terminals[0] not in open_set or terminals[1] not in open_set:
            continue
        adjacency: dict[str, set[str]] = {vertex: set() for vertex in open_set}
        for left, right in edges:
            if left in adjacency and right in adjacency:
                adjacency[left].add(right)
                adjacency[right].add(left)
        seen = {terminals[0]}
        pending = [terminals[0]]
        while pending:
            vertex = pending.pop()
            for neighbor in adjacency[vertex] - seen:
                seen.add(neighbor)
                pending.append(neighbor)
        if terminals[1] in seen:
            total += mass
    return total


def test_two_vertex_path_requires_both_terminals_open() -> None:
    """A single edge needs both endpoints open, giving p^2."""
    result = compute_site_connection_probability(
        _source(("a", "b"), (("a", "b"),), (Fraction(1, 2), Fraction(1, 2)), ("a", "b"))
    )
    assert result.connection_probability.as_fraction() == Fraction(1, 4)
    assert result.visited_states == 4


def test_both_terminals_must_be_open() -> None:
    """A terminal with zero open probability disconnects the pair."""
    result = compute_site_connection_probability(
        _source(("a", "b"), (("a", "b"),), (Fraction(1), Fraction(0)), ("a", "b"))
    )
    assert result.connection_probability.as_fraction() == Fraction(0)


def test_isolation_vertex_cannot_connect() -> None:
    """An isolated third vertex never affects the terminal event."""
    result = compute_site_connection_probability(
        _source(
            ("a", "b", "z"),
            (("a", "b"),),
            (Fraction(1), Fraction(1), Fraction(1, 2)),
            ("a", "b"),
        )
    )
    assert result.connection_probability.as_fraction() == Fraction(1)
    assert result.visited_states == 8


def test_open_induced_subgraph_uses_only_open_vertices() -> None:
    """A path a-b-c connects only when b is open; otherwise it cannot."""
    result = compute_site_connection_probability(
        _source(
            ("a", "b", "c"),
            (("a", "b"), ("b", "c")),
            (Fraction(1), Fraction(1, 2), Fraction(1)),
            ("a", "c"),
        )
    )
    # a and c are certain to be open; success needs b open too.
    assert result.connection_probability.as_fraction() == Fraction(1, 2)


def test_states_partition_the_sample_space() -> None:
    """Every state probability is nonnegative and sums exactly to one."""
    result = compute_site_connection_probability(
        _source(
            ("a", "b", "c"),
            (("a", "b"), ("b", "c")),
            (Fraction(1, 3), Fraction(2, 5), Fraction(1, 7)),
            ("a", "c"),
        )
    )
    total = sum(
        (state.state_probability.as_fraction() for state in result.states),
        Fraction(0),
    )
    assert total == Fraction(1)


def test_high_height_inputs_are_admitted_by_the_complete_result_carrier() -> None:
    """Five 128-digit factors exceed the old 512-digit late validator cap."""
    denominator = 10**128 - 1
    result = compute_site_connection_probability(
        _source(
            tuple("v" + str(index) for index in range(5)),
            tuple(("v" + str(index), "v" + str(index + 1)) for index in range(4)),
            (Fraction(1, denominator),) * 5,
            ("v0", "v4"),
        )
    )
    assert result.visited_states == 32
    assert len(str(result.states[-1].state_probability.as_fraction().denominator)) > 512
    assert all(state.state_probability.as_fraction() >= 0 for state in result.states)


def test_successful_state_mass_equals_reported_total() -> None:
    """The reported scalar is the exact sum of the successful ledger rows."""
    result = compute_site_connection_probability(
        _source(
            ("a", "b", "c", "d"),
            (("a", "b"), ("b", "c"), ("c", "d")),
            (Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1, 5)),
            ("a", "d"),
        )
    )
    successful = sum(
        (
            state.state_probability.as_fraction()
            for state in result.states
            if state.terminals_connected
        ),
        Fraction(0),
    )
    assert successful == result.connection_probability.as_fraction()


def test_terminal_order_does_not_change_the_result() -> None:
    """The event is symmetric in its two terminals."""
    forward = compute_site_connection_probability(
        _source(("a", "b"), (("a", "b"),), (Fraction(1, 3), Fraction(1, 3)), ("a", "b"))
    )
    reverse = compute_site_connection_probability(
        _source(("a", "b"), (("a", "b"),), (Fraction(1, 3), Fraction(1, 3)), ("b", "a"))
    )
    assert forward.connection_probability == reverse.connection_probability


def test_declared_vertex_axis_is_authoritative() -> None:
    source = GraphSiteReliabilitySource(
        graph=SimpleUndirectedGraph(
            vertices=("b", "a"),
            edges=(("a", "b"),),
        ),
        vertex_probabilities=(
            SiteReliabilityVertexProbability(
                vertex="b", open_probability=_probability(Fraction(1, 2))
            ),
            SiteReliabilityVertexProbability(
                vertex="a", open_probability=_probability(Fraction(1, 3))
            ),
        ),
        terminals=("b", "a"),
    )
    result = compute_site_connection_probability(source)
    assert result.states[1].open_vertices == ("b",)


def test_forged_site_state_subset_is_rejected_after_json_round_trip() -> None:
    result = compute_site_connection_probability(
        _source(("a", "b"), (("a", "b"),), (Fraction(1, 2),) * 2, ("a", "b"))
    )
    payload = result.model_dump(mode="json")
    payload["states"][1]["open_vertices"] = ["b"]
    with pytest.raises(ValidationError, match="state vertices"):
        GraphSiteReliabilityResult.model_validate_json(
            encode_strict_json(payload), strict=True
        )


def test_site_resource_bound_uses_resource_admission_error() -> None:
    vertices = tuple(sorted(f"v{index}" for index in range(12)))
    edges = tuple(
        (left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    )
    with pytest.raises(OperationResourceAdmissionError):
        compute_site_connection_probability(
            _source(
                vertices,
                edges,
                (Fraction(1, 2),) * len(vertices),
                ("v0", "v1"),
            )
        )


def test_site_vertex_bound_is_owned_by_operation_admission() -> None:
    vertices = tuple(sorted(f"v{index}" for index in range(13)))
    source = _source(
        vertices,
        (),
        (Fraction(1, 2),) * len(vertices),
        ("v0", "v1"),
    )
    with pytest.raises(OperationResourceAdmissionError, match="vertex bound"):
        compute_site_connection_probability(source)


def test_matches_independent_brute_force() -> None:
    """Several small graphs agree with an independent vertex-subset enumeration."""
    cases = (
        (("a", "b", "c"), (("a", "b"), ("b", "c")), (Fraction(1, 2),) * 3, ("a", "c")),
        (
            ("a", "b", "c", "d"),
            (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")),
            (Fraction(1, 4), Fraction(3, 4), Fraction(1, 2), Fraction(2, 3)),
            ("a", "d"),
        ),
    )
    for vertices, edges, probabilities, terminals in cases:
        result = compute_site_connection_probability(
            _source(vertices, edges, probabilities, terminals)
        )
        assert result.connection_probability.as_fraction() == _brute_force(
            vertices, edges, probabilities, terminals
        )


def test_duplicate_or_unknown_terminals_are_rejected() -> None:
    """Terminals must be two distinct declared graph vertices."""
    with pytest.raises(ValidationError):
        _source(("a", "b"), (("a", "b"),), (Fraction(1, 2),) * 2, ("a", "a"))


def test_probability_outside_unit_interval_is_rejected() -> None:
    """Component probabilities must lie in [0, 1]."""
    with pytest.raises(OperationDomainValidationError):
        compute_site_connection_probability(
            _source(
                ("a", "b"), (("a", "b"),), (Fraction(3, 2), Fraction(1, 2)), ("a", "b")
            )
        )


def test_result_round_trips_through_strict_json() -> None:
    """The complete ledger survives strict JSON serialization unchanged."""
    result = compute_site_connection_probability(
        _source(("a", "b"), (("a", "b"),), (Fraction(1, 2), Fraction(1, 3)), ("a", "b"))
    )
    restored = GraphSiteReliabilityResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result
