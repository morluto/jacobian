"""Domain adapter for graph coloring and independent set operations."""

from __future__ import annotations

from jacobian.contracts.graph_coloring_ops import (
    KColorabilityRequest,
    KColorabilityResult,
    MaximalIndependentSetRequest,
    MaximalIndependentSetResult,
    MaximumIndependentSetRequest,
    MaximumIndependentSetResult,
)


def compute_k_colorability(request: KColorabilityRequest) -> KColorabilityResult:
    """Decide k-colorability of a bounded simple graph.

    Encodes the proper-coloring constraint as a SAT instance and delegates to
    Z3. The result carries a coloring witness when one exists.
    """
    import z3  # type: ignore[import-untyped]

    solver = z3.Solver()
    solver.set("timeout", 10_000)
    colors = [z3.Int(f"color_{vertex}") for vertex in range(request.graph.vertex_count)]
    solver.add(*(z3.And(color >= 0, color < request.colors) for color in colors))
    solver.add(*(colors[u] != colors[v] for u, v in request.graph.edges))
    outcome = solver.check()
    if outcome == z3.sat:
        model = solver.model()
        coloring = tuple(model.eval(color).as_long() for color in colors)
        return KColorabilityResult(
            colorable=True,
            coloring=coloring,
            vertex_count=request.graph.vertex_count,
            colors=request.colors,
        )
    if outcome == z3.unsat:
        return KColorabilityResult(
            colorable=False,
            vertex_count=request.graph.vertex_count,
            colors=request.colors,
        )
    # outcome == z3.unknown - solver could not decide within the budget.
    raise RuntimeError(
        "k-colorability solver exceeded its budget without a decision"
    )


def compute_maximum_independent_set(
    request: MaximumIndependentSetRequest,
) -> MaximumIndependentSetResult:
    """Compute a maximum independent set of a bounded simple graph.

    Encodes the maximum-independent-set problem as a Z3 optimization instance
    and delegates to the solver.
    """
    import z3

    vertices = [
        z3.Bool(f"selected_{vertex}") for vertex in range(request.graph.vertex_count)
    ]
    solver = z3.Optimize()
    solver.set("timeout", 10_000)
    solver.add(
        *(z3.Not(z3.And(vertices[u], vertices[v])) for u, v in request.graph.edges)
    )
    solver.maximize(z3.Sum(*(z3.If(vertex, 1, 0) for vertex in vertices)))
    if solver.check() != z3.sat:
        raise RuntimeError(
            "Z3 failed to optimize the bounded independent-set instance"
        )
    model = solver.model()
    iset = tuple(
        index
        for index, vertex in enumerate(vertices)
        if z3.is_true(model.eval(vertex))
    )
    return MaximumIndependentSetResult(
        independent_set=iset,
        cardinality=len(iset),
    )


def compute_maximal_independent_set_decision(
    request: MaximalIndependentSetRequest,
) -> MaximalIndependentSetResult:
    """Decide whether a candidate vertex set is a maximal independent set.

    This is a direct combinatorial check against the explicit edge list - no
    solver budget is required.
    """
    vertex_count = request.graph.vertex_count
    candidate = set(request.candidate_set)

    # Adjacency lookup from the edge list.
    adjacency: list[set[int]] = [set() for _ in range(vertex_count)]
    for u, v in request.graph.edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    # Independence: no two candidate vertices share an edge.
    candidate_list = sorted(candidate)
    for i, u in enumerate(candidate_list):
        for v in candidate_list[i + 1:]:
            if v in adjacency[u]:
                return MaximalIndependentSetResult(
                    decision="NOT_INDEPENDENT",
                    candidate_set=request.candidate_set,
                    vertex_count=vertex_count,
                )

    # Maximality: every vertex outside the candidate must have a neighbor
    # inside the candidate, otherwise it could be added without breaking
    # independence.
    for vertex in range(vertex_count):
        if vertex in candidate:
            continue
        if not (adjacency[vertex] & candidate):
            return MaximalIndependentSetResult(
                decision="INDEPENDENT_NOT_MAXIMAL",
                candidate_set=request.candidate_set,
                vertex_count=vertex_count,
            )

    return MaximalIndependentSetResult(
        decision="MAXIMAL",
        candidate_set=request.candidate_set,
        vertex_count=vertex_count,
    )
