"""Shared structural invariants for bounded simple undirected graphs."""


def require_simple_undirected_graph(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> None:
    """Reject duplicate vertices and non-simple or undeclared edges."""

    vertex_set = set(vertices)
    if len(vertex_set) != len(vertices):
        raise ValueError("graph vertices must be unique")
    normalized_edges = {tuple(sorted((left, right))) for left, right in edges}
    if any(left == right for left, right in edges):
        raise ValueError("graph edges must not contain self-loops")
    if any(left not in vertex_set or right not in vertex_set for left, right in edges):
        raise ValueError("graph edges must reference declared vertices")
    if len(normalized_edges) != len(edges):
        raise ValueError("graph edges must be unique ignoring orientation")


__all__ = ["require_simple_undirected_graph"]
