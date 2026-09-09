"""Complete fixed-length cycle enumeration tests."""

from jacobian.math.graphs.cycle_length_profile.operations import (
    enumerate_fixed_length_cycles,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _square_with_diagonal() -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=("0", "1", "2", "3"),
        edges=(("0", "1"), ("0", "2"), ("0", "3"), ("1", "2"), ("2", "3")),
    )


def test_simple_and_chordless_cycle_families_are_distinct() -> None:
    graph = _square_with_diagonal()
    simple = enumerate_fixed_length_cycles(graph, 4)
    chordless = enumerate_fixed_length_cycles(graph, 4, chordless=True)

    assert simple.cycles == (("0", "1", "2", "3"),)
    assert chordless.cycles == ()
    assert simple.vertex_incidence[0].cycle_indices == (0,)
    assert simple.edge_incidence[1].source == ("0", "2")
    assert simple.edge_incidence[1].cycle_indices == ()


def test_complete_graph_triangle_enumeration_matches_binomial_count() -> None:
    vertices = ("0", "1", "2", "3", "4")
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            (left, right) for left in vertices for right in vertices if left < right
        ),
    )
    result = enumerate_fixed_length_cycles(graph, 3, chordless=True)
    assert len(result.cycles) == 10
    assert all(
        cycle == min(cycle, (cycle[0], cycle[2], cycle[1])) for cycle in result.cycles
    )
