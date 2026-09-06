"""Independent linear-time checking of a retained Tutte-Berge relation."""

from jacobian.math.graphs.optimization._invariant_models import (
    GraphMaximumMatchingResult,
)


def verify_maximum_matching(claim: GraphMaximumMatchingResult) -> bool:
    """Check feasibility and the Tutte-Berge upper bound for the source graph.

    The canonical graph bounds this traversal by 256 vertices and 32640
    edges. No matching optimization or barrier search is repeated.
    """
    graph = claim.graph
    vertices = set(graph.vertices)
    barrier = set(claim.certificate.barrier_vertices)
    if not barrier <= vertices or not set(claim.witness_edges) <= set(graph.edges):
        return False
    adjacency: dict[str, set[str]] = {v: set() for v in vertices - barrier}
    for u, v in graph.edges:
        if u not in barrier and v not in barrier:
            adjacency[u].add(v)
            adjacency[v].add(u)
    remaining = set(adjacency)
    odd_count = 0
    while remaining:
        pending = [remaining.pop()]
        size = 0
        while pending:
            current = pending.pop()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
        odd_count += size % 2
    return (
        odd_count == claim.certificate.odd_component_count
        and 2 * claim.maximum_matching_cardinality
        == len(vertices) + len(barrier) - odd_count
    )
