"""Exact directed bond-reliability operation contracts."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.directed._models import DirectedGraph
from jacobian.math.probability._models import (
    MAX_DIRECTED_BOND_RELIABILITY_ARCS,
    MAX_DIRECTED_BOND_RELIABILITY_LOGICAL_WORK,
    MAX_DIRECTED_BOND_RELIABILITY_STATES,
    MAX_DIRECTED_BOND_RELIABILITY_VERTICES,
    DirectedBondConnectionProbabilityRequest,
    DirectedBondConnectionProbabilityResult,
    DirectedBondReliabilityArcProbability,
)
from jacobian.math.probability._operations import _directed_bond_connection_probability


def _rational(value: Fraction) -> dict[str, str]:
    return {"num": str(value.numerator), "den": str(value.denominator)}


def _request(
    *,
    vertex_count: int,
    arcs: tuple[tuple[int, int], ...],
    probabilities: tuple[Fraction, ...],
    source: int,
    target: int,
) -> DirectedBondConnectionProbabilityRequest:
    return DirectedBondConnectionProbabilityRequest(
        graph=DirectedGraph(vertex_count=vertex_count, edges=arcs),
        arc_probabilities=tuple(
            DirectedBondReliabilityArcProbability(
                arc=arc,
                open_probability=CanonicalRational.from_fraction(probability),
            )
            for arc, probability in zip(arcs, probabilities, strict=True)
        ),
        source=source,
        target=target,
    )


def _probability(result: DirectedBondConnectionProbabilityResult) -> Fraction:
    return result.connection_probability.as_fraction()


def test_single_directed_arc_has_its_open_probability() -> None:
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=2,
            arcs=((0, 1),),
            probabilities=(Fraction(2, 7),),
            source=0,
            target=1,
        )
    )

    assert _probability(result) == Fraction(2, 7)
    assert result.visited_states == 2
    assert tuple(state.open_arcs for state in result.states) == ((), ((0, 1),))
    assert tuple(state.source_reaches_target for state in result.states) == (
        False,
        True,
    )


def test_reverse_arc_does_not_create_forward_directed_connection() -> None:
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=2,
            arcs=((1, 0),),
            probabilities=(Fraction(1, 1),),
            source=0,
            target=1,
        )
    )

    assert _probability(result) == 0
    assert all(not state.source_reaches_target for state in result.states)


def test_two_arc_directed_series_multiplies_exact_probabilities() -> None:
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=3,
            arcs=((0, 1), (1, 2)),
            probabilities=(Fraction(2, 3), Fraction(3, 5)),
            source=0,
            target=2,
        )
    )

    assert _probability(result) == Fraction(2, 5)


def test_independent_directed_routes_use_exact_union_mass() -> None:
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=4,
            arcs=((0, 1), (1, 3), (0, 2), (2, 3)),
            probabilities=(
                Fraction(1, 2),
                Fraction(2, 3),
                Fraction(3, 4),
                Fraction(4, 5),
            ),
            source=0,
            target=3,
        )
    )

    assert _probability(result) == Fraction(11, 15)
    assert (
        sum(
            (state.state_probability.as_fraction() for state in result.states),
            Fraction(),
        )
        == 1
    )


def test_reversing_terminals_changes_the_directed_event() -> None:
    request = _request(
        vertex_count=2,
        arcs=((0, 1), (1, 0)),
        probabilities=(Fraction(2, 3), Fraction(3, 5)),
        source=0,
        target=1,
    )
    reverse = _request(
        vertex_count=2,
        arcs=((0, 1), (1, 0)),
        probabilities=(Fraction(2, 3), Fraction(3, 5)),
        source=1,
        target=0,
    )

    assert _probability(_directed_bond_connection_probability(request)) == Fraction(
        2, 3
    )
    assert _probability(_directed_bond_connection_probability(reverse)) == Fraction(
        3, 5
    )


def test_component_order_is_canonical_and_does_not_change_result() -> None:
    canonical = _request(
        vertex_count=3,
        arcs=((0, 1), (1, 2)),
        probabilities=(Fraction(2, 3), Fraction(3, 5)),
        source=0,
        target=2,
    )
    permuted = _request(
        vertex_count=3,
        arcs=((1, 2), (0, 1)),
        probabilities=(Fraction(3, 5), Fraction(2, 3)),
        source=0,
        target=2,
    )

    expected = _directed_bond_connection_probability(canonical)
    actual = _directed_bond_connection_probability(permuted)
    assert actual == expected
    assert actual.source.graph.edges == ((0, 1), (1, 2))


def test_sparse_graph_above_sixteen_vertices_is_admitted() -> None:
    """The derived work budget, not a fixed vertex ceiling, bounds vertices."""
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=17,
            arcs=((3, 16),),
            probabilities=(Fraction(2, 7),),
            source=3,
            target=16,
        )
    )

    assert _probability(result) == Fraction(2, 7)
    assert result.visited_states == 2


def test_edgeless_graph_above_the_shared_vertex_cap_is_admitted() -> None:
    """A 65-vertex edgeless source stays inside the derived work budget."""
    result = _directed_bond_connection_probability(
        _request(vertex_count=65, arcs=(), probabilities=(), source=0, target=1)
    )

    assert _probability(result) == 0
    assert result.arc_count == 0
    assert result.visited_states == 1


def test_single_arc_above_the_shared_vertex_cap_is_admitted() -> None:
    """Sparse sources above 64 vertices are not rejected by the shared graph."""
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=100,
            arcs=((63, 64),),
            probabilities=(Fraction(2, 7),),
            source=63,
            target=64,
        )
    )

    assert _probability(result) == Fraction(2, 7)
    assert result.visited_states == 2


def test_edgeless_graph_has_exact_zero_connection_probability() -> None:
    """Zero arcs keep the complete single-state ledger with probability zero."""
    result = _directed_bond_connection_probability(
        _request(vertex_count=2, arcs=(), probabilities=(), source=0, target=1)
    )

    assert _probability(result) == 0
    assert result.arc_count == 0
    assert result.visited_states == 1
    assert tuple(state.open_arcs for state in result.states) == ((),)
    assert tuple(state.source_reaches_target for state in result.states) == (False,)
    assert tuple(state.state_probability.as_fraction() for state in result.states) == (
        Fraction(1),
    )


def test_ledger_estimate_bounds_subset_numerator_growth_not_per_state_maxima() -> None:
    """A tall-numerator chain is charged by subset occurrence, not per state."""
    denominator = int("9" * 90)
    probability = Fraction(denominator - 1, denominator)
    arcs = tuple((index, index + 1) for index in range(12))
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=13,
            arcs=arcs,
            probabilities=(probability,) * len(arcs),
            source=0,
            target=12,
        )
    )

    assert _probability(result) == probability**12
    assert result.visited_states == 4096


def _mutate_open_arcs(payload: dict[str, Any]) -> None:
    payload["states"][1]["open_arcs"] = []


def _mutate_reachability(payload: dict[str, Any]) -> None:
    payload["states"][3]["source_reaches_target"] = False


def _mutate_connection_probability(payload: dict[str, Any]) -> None:
    payload["connection_probability"] = _rational(Fraction())


@pytest.mark.parametrize(
    "mutation",
    (
        _mutate_open_arcs,
        _mutate_reachability,
        _mutate_connection_probability,
    ),
    ids=("open-arcs", "reachability", "aggregate-probability"),
)
def test_result_replay_rejects_mutated_source_bound_conclusions(
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=3,
            arcs=((0, 1), (1, 2)),
            probabilities=(Fraction(1, 2), Fraction(1, 2)),
            source=0,
            target=2,
        )
    )
    payload = result.model_dump(mode="json")
    mutation(payload)

    with pytest.raises(ValidationError):
        DirectedBondConnectionProbabilityResult.model_validate(payload)


def test_probability_bound_rejects_thirteenth_arc_before_enumeration() -> None:
    arcs = tuple((index, index + 1) for index in range(13))
    with pytest.raises(ValidationError, match="at most 12 items"):
        _request(
            vertex_count=14,
            arcs=arcs,
            probabilities=(Fraction(1, 2),) * len(arcs),
            source=0,
            target=13,
        )


def test_logical_work_bound_covers_two_pass_reachability_and_state_records() -> None:
    """The 12-arc, 16-vertex envelope charges both complete passes."""
    per_state_kernel_work = (
        4 * MAX_DIRECTED_BOND_RELIABILITY_ARCS
        + 4 * MAX_DIRECTED_BOND_RELIABILITY_VERTICES
    )
    per_state_record_work = MAX_DIRECTED_BOND_RELIABILITY_ARCS + 3

    assert (
        2
        * MAX_DIRECTED_BOND_RELIABILITY_STATES
        * (per_state_kernel_work + per_state_record_work)
        + 8
    ) == MAX_DIRECTED_BOND_RELIABILITY_LOGICAL_WORK


def test_ledger_bound_rejects_large_exact_probability_products_before_enumeration() -> (
    None
):
    arcs = tuple((index, index + 1) for index in range(12))
    large_denominator = int("9" * 128)
    large_probability = Fraction(large_denominator - 1, large_denominator)
    with pytest.raises(ValidationError, match="complete ledger budget"):
        _request(
            vertex_count=13,
            arcs=arcs,
            probabilities=(large_probability,) * len(arcs),
            source=0,
            target=12,
        )


def test_zero_and_one_probabilities_keep_complete_state_convention() -> None:
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=3,
            arcs=((0, 1), (1, 2)),
            probabilities=(Fraction(1), Fraction()),
            source=0,
            target=2,
        )
    )

    assert _probability(result) == 0
    assert result.visited_states == 4
    assert tuple(state.state_index for state in result.states) == (0, 1, 2, 3)
