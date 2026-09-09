"""Maximum induced matching tests."""

from jacobian.math.graphs.induced_matching.operations import maximum_induced_matching
from jacobian.math.graphs.values import SimpleUndirectedGraph


def test_path_five_has_two_edge_induced_matching() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("0", "1", "2", "3", "4"),
        edges=(("0", "1"), ("1", "2"), ("2", "3"), ("3", "4")),
    )
    result = maximum_induced_matching(graph)
    assert result.status == "EXACT"
    assert result.cardinality == result.lower_bound == result.upper_bound == 2
    assert result.selected_edge_ids == ("e0", "e3")
    assert result.induced_endpoint_graph.edges == (("0", "1"), ("3", "4"))


def test_empty_graph_has_empty_induced_matching() -> None:
    graph = SimpleUndirectedGraph(vertices=("v",), edges=())
    result = maximum_induced_matching(graph)
    assert result.cardinality == 0
    assert result.selected_edge_ids == ()
    assert result.induced_endpoint_graph.vertices == ()
