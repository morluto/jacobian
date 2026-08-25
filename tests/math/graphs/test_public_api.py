import networkx as nx
import pytest

from jacobian.math import graphs
from jacobian.math.graphs.values import SimpleUndirectedGraph


def test_graph_algorithms_use_networkx_objects() -> None:
    graph = nx.cycle_graph(3)
    assert graphs.triangle_count(graph) == 1
    assert graphs.diameter(graph) == 1
    assert graphs.is_eulerian(graph)


def test_graph_input_errors_are_stable() -> None:
    with pytest.raises(TypeError):
        graphs.triangle_count({0: [1]})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        graphs.diameter(nx.Graph([(0, 1), (2, 3)]))
    with pytest.raises(ValueError):
        graphs.triangle_count(nx.DiGraph([(0, 1)]))  # type: ignore[arg-type]


def test_graph_construction_functions_use_immutable_graph_values() -> None:
    explicit = graphs.explicit_graph(
        ("c", "a", "b"),
        (("b", "a"), ("c", "b")),
    )

    assert explicit.vertices == ("a", "b", "c")
    assert explicit.edges == (("a", "b"), ("b", "c"))

    complement = graphs.compose_graphs(
        graphs.GraphCompositionInput(operation="COMPLEMENT", left=explicit)
    )

    assert complement.vertices == ("v0", "v1", "v2")
    assert complement.edges == (("v0", "v2"),)
    assert type(explicit) is SimpleUndirectedGraph


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the graphs public API."""
    expected = (
        "ColoredUndirectedGraph",
        "GraphCompositionInput",
        "IndependenceNumberBudget",
        "IndependenceNumberRequest",
        "IndependenceNumberResult",
        "IndexedSimpleUndirectedGraph",
        "SimpleUndirectedGraph",
        "biconnected_components",
        "complement",
        "compose_graphs",
        "diameter",
        "explicit_graph",
        "graph_power",
        "independence_number",
        "induced_subgraph",
        "is_eulerian",
        "line_graph",
        "radius",
        "strongly_connected_components",
        "triangle_count",
    )
    assert tuple(graphs.__all__) == expected
    assert len(graphs.__all__) == len(set(graphs.__all__))
    assert all(not name.startswith("_") for name in graphs.__all__)
    assert all(hasattr(graphs, name) for name in graphs.__all__)
