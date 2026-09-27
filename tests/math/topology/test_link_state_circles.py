"""Exact cyclic dart circles for complete link smoothing states."""

from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.math.topology.links import (
    LinkCrossing,
    LinkDiagramSmoothingState,
    OrientedDiagramArc,
    OrientedLinkDiagram,
    link_bracket,
    link_state_circles,
)
from jacobian.math.topology.links.extensions_tools import TOOLS


def _curl() -> OrientedLinkDiagram:
    return OrientedLinkDiagram(
        crossings=(
            LinkCrossing(
                crossing_id="curl",
                half_edges=("h0", "h1", "h2", "h3"),
                over_pair=(0, 2),
                under_pair=(1, 3),
                sign=-1,
            ),
        ),
        arcs=(
            OrientedDiagramArc(tail="h0", head="h3"),
            OrientedDiagramArc(tail="h1", head="h2"),
        ),
    )


def _hopf() -> OrientedLinkDiagram:
    return OrientedLinkDiagram(
        crossings=(
            LinkCrossing(
                crossing_id="c0",
                half_edges=("a0", "b0", "a1", "b1"),
                over_pair=(0, 2),
                under_pair=(1, 3),
                sign=-1,
            ),
            LinkCrossing(
                crossing_id="c1",
                half_edges=("a2", "b2", "a3", "b3"),
                over_pair=(0, 2),
                under_pair=(1, 3),
                sign=-1,
            ),
        ),
        arcs=(
            OrientedDiagramArc(tail="a1", head="b2"),
            OrientedDiagramArc(tail="a3", head="b0"),
            OrientedDiagramArc(tail="b1", head="a2"),
            OrientedDiagramArc(tail="b3", head="a0"),
        ),
    )


def _curl_with_odd_over_pair() -> OrientedLinkDiagram:
    return OrientedLinkDiagram(
        crossings=(
            LinkCrossing(
                crossing_id="curl",
                half_edges=("h0", "h1", "h2", "h3"),
                over_pair=(1, 3),
                under_pair=(0, 2),
                sign=1,
            ),
        ),
        arcs=(
            OrientedDiagramArc(tail="h0", head="h3"),
            OrientedDiagramArc(tail="h1", head="h2"),
        ),
    )


def _raw_circle_oracle(
    diagram: OrientedLinkDiagram, choices: tuple[str, ...]
) -> tuple[frozenset[str], ...]:
    """Enumerate connected components of a separately-built raw dart graph."""
    neighbors: dict[str, set[str]] = {
        dart: set() for crossing in diagram.crossings for dart in crossing.half_edges
    }
    for arc in diagram.arcs:
        neighbors[arc.tail].add(arc.head)
        neighbors[arc.head].add(arc.tail)
    for crossing, choice in zip(diagram.crossings, choices, strict=True):
        adjacent_pairs = ((0, 1), (2, 3)) if choice == "A" else ((1, 2), (3, 0))
        if set(crossing.over_pair) == {1, 3}:
            adjacent_pairs = ((1, 2), (3, 0)) if choice == "A" else ((0, 1), (2, 3))
        for left, right in adjacent_pairs:
            left_dart, right_dart = (
                crossing.half_edges[left],
                crossing.half_edges[right],
            )
            neighbors[left_dart].add(right_dart)
            neighbors[right_dart].add(left_dart)
    unseen = set(neighbors)
    components: list[frozenset[str]] = []
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[str] = set()
        while stack:
            dart = stack.pop()
            if dart in component:
                continue
            component.add(dart)
            stack.extend(neighbors[dart] - component)
        unseen.difference_update(component)
        components.append(frozenset(component))
    return tuple(sorted(components, key=lambda row: tuple(sorted(row))))


@pytest.mark.parametrize("diagram", (_curl(), _curl_with_odd_over_pair(), _hopf()))
def test_state_circles_match_independent_raw_dart_oracle_and_bracket(
    diagram: OrientedLinkDiagram,
) -> None:
    bracket_states = link_bracket(diagram).states
    for choices in product(("A", "B"), repeat=len(diagram.crossings)):
        result = link_state_circles(
            LinkDiagramSmoothingState(diagram=diagram, choices=choices)
        )
        expected_components = _raw_circle_oracle(diagram, choices)
        actual_components = tuple(
            sorted(
                (frozenset(circle.darts) for circle in result.circles),
                key=lambda row: tuple(sorted(row)),
            )
        )
        bracket_state = next(
            state
            for state in bracket_states
            if tuple("A" if choice == 0 else "B" for choice in state.choices) == choices
        )
        assert actual_components == expected_components
        assert (
            result.circle_count
            == len(expected_components)
            == bracket_state.circle_count
        )
        for circle in result.circles:
            for left, right in zip(
                circle.darts, (*circle.darts[1:], circle.darts[0]), strict=True
            ):
                assert right in _raw_neighbors(diagram, choices, left)


def _raw_neighbors(
    diagram: OrientedLinkDiagram, choices: tuple[str, ...], dart: str
) -> set[str]:
    """Neighbors from the raw arc and local smoothing incidence relation."""
    result = set()
    for arc in diagram.arcs:
        if dart == arc.tail:
            result.add(arc.head)
        elif dart == arc.head:
            result.add(arc.tail)
    for crossing, choice in zip(diagram.crossings, choices, strict=True):
        pairs = ((0, 1), (2, 3)) if choice == "A" else ((1, 2), (3, 0))
        if set(crossing.over_pair) == {1, 3}:
            pairs = ((1, 2), (3, 0)) if choice == "A" else ((0, 1), (2, 3))
        for left, right in pairs:
            if crossing.half_edges[left] == dart:
                result.add(crossing.half_edges[right])
            elif crossing.half_edges[right] == dart:
                result.add(crossing.half_edges[left])
    return result


def test_crossing_free_unlink_and_complete_state_shape() -> None:
    result = link_state_circles(
        LinkDiagramSmoothingState(diagram=OrientedLinkDiagram(free_loops=3), choices=())
    )
    assert result.circle_count == 3
    assert tuple(circle.darts for circle in result.circles) == ((), (), ())
    assert (
        LinkDiagramSmoothingState.model_validate_json(
            LinkDiagramSmoothingState(
                diagram=OrientedLinkDiagram(free_loops=1), choices=()
            ).model_dump_json()
        ).choices
        == ()
    )
    with pytest.raises(
        ValidationError, match="cover the complete source crossing axis"
    ):
        LinkDiagramSmoothingState(diagram=_curl(), choices=())


def test_state_circles_result_round_trip_and_catalog_publication() -> None:
    state = LinkDiagramSmoothingState(diagram=_curl(), choices=("B",))
    result = link_state_circles(state)
    assert type(result).model_validate_json(result.model_dump_json()) == result
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "link_diagram.state_circles.compute"
    )
    catalog_result = tool.run(tool.request_type.model_validate({"state": state}))
    assert catalog_result == result
