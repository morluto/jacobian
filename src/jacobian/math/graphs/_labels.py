"""Private helpers for accounting graph labels."""

from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph_label_characters(graph: SimpleUndirectedGraph) -> int:
    """Count the characters retained by vertex labels and edge endpoints."""

    return sum(len(vertex) for vertex in graph.vertices) + sum(
        len(left) + len(right) for left, right in graph.edges
    )
