"""Domain-owned graph coloring and independent set operations."""

from __future__ import annotations

from jacobian.math.graph_coloring_ops._models import (
    KColorabilityRequest,
    KColorabilityResult,
    MaximalIndependentSetRequest,
    MaximalIndependentSetResult,
    MaximumIndependentSetRequest,
    MaximumIndependentSetResult,
)


def compute_k_colorability(request: KColorabilityRequest) -> KColorabilityResult:
    import z3  # type: ignore[import-untyped]

    solver = z3.Solver()
    colors = [z3.Int(f"color_{vertex}") for vertex in range(request.graph.vertex_count)]
    solver.add(*(z3.And(color >= 0, color < request.colors) for color in colors))
    solver.add(*(colors[u] != colors[v] for u, v in request.graph.edges))
    if solver.check() == z3.sat:
        model = solver.model()
        coloring = tuple(model.eval(color).as_long() for color in colors)
        return KColorabilityResult(
            colorable=True,
            coloring=coloring,
            vertex_count=request.graph.vertex_count,
            colors=request.colors,
        )
    return KColorabilityResult(
        colorable=False,
        vertex_count=request.graph.vertex_count,
        colors=request.colors,
    )


def compute_maximum_independent_set(
    request: MaximumIndependentSetRequest,
) -> MaximumIndependentSetResult:
    import z3

    vertices = [
        z3.Bool(f"selected_{vertex}") for vertex in range(request.graph.vertex_count)
    ]
    solver = z3.Optimize()
    solver.add(
        *(z3.Not(z3.And(vertices[u], vertices[v])) for u, v in request.graph.edges)
    )
    solver.maximize(z3.Sum(*(z3.If(vertex, 1, 0) for vertex in vertices)))
    if solver.check() != z3.sat:
        raise RuntimeError("Z3 failed to optimize the bounded independent-set instance")
    model = solver.model()
    iset = tuple(
        index for index, vertex in enumerate(vertices) if z3.is_true(model.eval(vertex))
    )
    return MaximumIndependentSetResult(
        independent_set=iset,
        cardinality=len(iset),
    )


def compute_maximal_independent_set_decision(
    request: MaximalIndependentSetRequest,
) -> MaximalIndependentSetResult:
    """Decide maximal independence and return the first canonical obstruction."""
    candidate = frozenset(request.candidate_set)
    edges = tuple(sorted((min(u, v), max(u, v)) for u, v in request.graph.edges))
    for edge in edges:
        if edge[0] in candidate and edge[1] in candidate:
            return MaximalIndependentSetResult(
                decision="NOT_INDEPENDENT",
                blocking_edge=edge,
            )

    adjacency: list[set[int]] = [set() for _ in range(request.graph.vertex_count)]
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    for vertex in range(request.graph.vertex_count):
        if vertex not in candidate and adjacency[vertex].isdisjoint(candidate):
            return MaximalIndependentSetResult(
                decision="INDEPENDENT_NOT_MAXIMAL",
                addable_vertex=vertex,
            )
    return MaximalIndependentSetResult(decision="MAXIMAL")
