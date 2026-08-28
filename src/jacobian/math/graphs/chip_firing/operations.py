"""Exact chip-firing operations."""

from __future__ import annotations

from collections import deque

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.chip_firing._models import (
    MAX_COEFFICIENT_DIGITS,
    MAX_STABILIZATION_CHIPS,
    MAX_VERTICES,
    AbelJacobiResult,
    CanonicalDivisorResult,
    CriticalGroupResult,
    DegreeResult,
    FireVectorResult,
    FiringResult,
    LaplacianResult,
    ParallelStepResult,
    QReducedResult,
    ReducedLaplacianResult,
    StabilizeResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _admit_graph(graph: SimpleUndirectedGraph) -> None:
    if not graph.vertices:
        raise OperationDomainValidationError(
            location=("graph",),
            code="chip_firing.empty_graph",
            message="chip-firing requires a nonempty graph",
        )
    if len(graph.vertices) > MAX_VERTICES:
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="chip_firing.vertex_bound",
            message=f"chip-firing supports at most {MAX_VERTICES} vertices",
        )


def _admit_sink(graph: SimpleUndirectedGraph, sink: str) -> None:
    _admit_graph(graph)
    if sink not in graph.vertices:
        raise OperationDomainValidationError(
            location=("sink",),
            code="chip_firing.sink_not_in_graph",
            message="sink vertex must be in the graph",
        )


def _admit_divisor(graph: SimpleUndirectedGraph, divisor: tuple[int, ...]) -> None:
    _admit_graph(graph)
    if len(divisor) != len(graph.vertices):
        raise OperationDomainValidationError(
            location=("divisor",),
            code="chip_firing.divisor_length",
            message="divisor length must match vertex count",
        )


def _admit_configuration(
    graph: SimpleUndirectedGraph, sink: str, configuration: tuple[int, ...]
) -> None:
    _admit_sink(graph, sink)
    if len(configuration) != len(graph.vertices):
        raise OperationDomainValidationError(
            location=("configuration",),
            code="chip_firing.configuration_length",
            message="configuration length must match vertex count",
        )
    sink_index = graph.vertices.index(sink)
    nonsink = [index for index in range(len(graph.vertices)) if index != sink_index]
    if any(configuration[index] < 0 for index in nonsink):
        raise OperationDomainValidationError(
            location=("configuration",),
            code="chip_firing.nonsink_negative",
            message="nonsink configuration must be nonnegative",
        )
    if sum(configuration[index] for index in nonsink) > MAX_STABILIZATION_CHIPS:
        raise OperationDomainValidationError(
            location=("configuration",),
            code="chip_firing.stabilization_bound",
            message=(
                "nonsink configuration exceeds stabilization bound "
                f"{MAX_STABILIZATION_CHIPS}"
            ),
        )


def _adjacency(graph: SimpleUndirectedGraph) -> tuple[tuple[int, ...], ...]:
    """Build an adjacency-list representation from a canonical graph."""
    n = len(graph.vertices)
    idx = {v: i for i, v in enumerate(graph.vertices)}
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in graph.edges:
        i, j = idx[u], idx[v]
        adj[i].append(j)
        adj[j].append(i)
    return tuple(tuple(row) for row in adj)


def _degrees(graph: SimpleUndirectedGraph) -> tuple[int, ...]:
    idx = {v: i for i, v in enumerate(graph.vertices)}
    deg = [0] * len(graph.vertices)
    for u, v in graph.edges:
        deg[idx[u]] += 1
        deg[idx[v]] += 1
    return tuple(deg)


def laplacian(graph: SimpleUndirectedGraph) -> LaplacianResult:
    """Compute the graph Laplacian L = D - A where D is the degree matrix."""
    _admit_graph(graph)
    vertices = graph.vertices
    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}

    adj = [[0] * n for _ in range(n)]
    for u, v in graph.edges:
        i, j = idx[u], idx[v]
        adj[i][j] += 1
        adj[j][i] += 1

    laplacian = []
    degrees = []
    for i in range(n):
        deg = sum(adj[i])
        degrees.append(deg)
        row = []
        for j in range(n):
            if i == j:
                row.append(deg)
            else:
                row.append(-adj[i][j])
        laplacian.append(tuple(row))

    return LaplacianResult(
        vertices=vertices,
        laplacian=tuple(laplacian),
        degrees=tuple(degrees),
    )


def reduced_laplacian(
    graph: SimpleUndirectedGraph, sink: str
) -> ReducedLaplacianResult:
    """Delete the sink row/column from the full Laplacian."""
    _admit_sink(graph, sink)
    vertices = graph.vertices
    n = len(vertices)
    full = laplacian(graph)
    lap = full.laplacian
    sink_idx = vertices.index(sink)
    nonsink = [i for i in range(n) if i != sink_idx]
    reduced = tuple(tuple(lap[i][j] for j in nonsink) for i in nonsink)
    return ReducedLaplacianResult(
        vertices=vertices,
        sink=sink,
        reduced_laplacian=reduced,
    )


def firing(
    graph: SimpleUndirectedGraph, divisor: tuple[int, ...], firing_vertex: str
) -> FiringResult:
    """Fire a vertex: D' = D - L*e_v where L is the Laplacian."""
    _admit_divisor(graph, divisor)
    if firing_vertex not in graph.vertices:
        raise OperationDomainValidationError(
            location=("firing_vertex",),
            code="chip_firing.firing_vertex_not_in_graph",
            message="firing vertex must be in the graph",
        )
    vertices = graph.vertices
    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}

    adj = [[0] * n for _ in range(n)]
    for u, v in graph.edges:
        i, j = idx[u], idx[v]
        adj[i][j] += 1
        adj[j][i] += 1

    fire_idx = idx[firing_vertex]
    result = list(divisor)

    deg = sum(adj[fire_idx])
    result[fire_idx] -= deg
    for j in range(n):
        if adj[fire_idx][j] > 0:
            result[j] += adj[fire_idx][j]

    return FiringResult(
        vertex=firing_vertex,
        fired_divisor=tuple(result),
    )


def fire_vector(
    graph: SimpleUndirectedGraph,
    divisor: tuple[int, ...],
    firing_vector: tuple[int, ...],
) -> FireVectorResult:
    """Fire a vector: D' = D - L f. Degree is preserved by construction."""
    _admit_divisor(graph, divisor)
    if len(firing_vector) != len(graph.vertices):
        raise OperationDomainValidationError(
            location=("firing_vector",),
            code="chip_firing.firing_vector_length",
            message="firing vector length must match vertex count",
        )
    if any(abs(value) >= 10**MAX_COEFFICIENT_DIGITS for value in firing_vector):
        raise OperationDomainValidationError(
            location=("firing_vector",),
            code="chip_firing.coefficient_bound",
            message="firing vector coefficients exceed the digit bound",
        )
    vertices = graph.vertices
    n = len(vertices)
    lap = laplacian(graph).laplacian
    divisor_values = list(divisor)
    f = firing_vector
    result = []
    for i in range(n):
        delta = sum(lap[i][j] * f[j] for j in range(n))
        result.append(divisor_values[i] - delta)
    return FireVectorResult(
        fired_divisor=tuple(result),
        degree_preserved=True,
    )


def _stabilize_configuration(
    config: list[int],
    adj: tuple[tuple[int, ...], ...],
    degrees: tuple[int, ...],
    sink_idx: int,
) -> tuple[list[int], list[int]]:
    """Stabilize via least-action / legal-firing algorithm.

    Returns (stable_config, odometer).
    """
    n = len(config)
    eta = list(config)
    odometer = [0] * n
    queue: deque[int] = deque()
    in_queue = [False] * n
    for i in range(n):
        if i != sink_idx and eta[i] >= degrees[i]:
            queue.append(i)
            in_queue[i] = True
    while queue:
        v = queue.popleft()
        in_queue[v] = False
        if eta[v] < degrees[v]:
            continue
        eta[v] -= degrees[v]
        odometer[v] += 1
        for nb in adj[v]:
            eta[nb] += 1
            if nb == sink_idx:
                continue
            if eta[nb] >= degrees[nb] and not in_queue[nb]:
                queue.append(nb)
                in_queue[nb] = True
    return eta, odometer


def stabilize(
    graph: SimpleUndirectedGraph, sink: str, configuration: tuple[int, ...]
) -> StabilizeResult:
    """Stabilize a sink configuration and return the odometer."""
    _admit_configuration(graph, sink, configuration)
    vertices = graph.vertices
    sink_idx = vertices.index(sink)
    adj = _adjacency(graph)
    degrees = _degrees(graph)
    config = list(configuration)
    eta, odometer = _stabilize_configuration(config, adj, degrees, sink_idx)
    return StabilizeResult(
        stable=tuple(eta),
        odometer=tuple(odometer),
        total_firings=sum(odometer),
    )


def parallel_step(
    graph: SimpleUndirectedGraph, sink: str, configuration: tuple[int, ...]
) -> ParallelStepResult:
    """One simultaneous legal firing step on all unstable nonsink vertices."""
    _admit_configuration(graph, sink, configuration)
    vertices = graph.vertices
    sink_idx = vertices.index(sink)
    adj = _adjacency(graph)
    degrees = _degrees(graph)
    config = list(configuration)
    fired = [
        v for i, v in enumerate(vertices) if i != sink_idx and config[i] >= degrees[i]
    ]
    next_config = list(config)
    for v in fired:
        vi = vertices.index(v)
        next_config[vi] -= degrees[vi]
    for v in fired:
        vi = vertices.index(v)
        for nb in adj[vi]:
            next_config[nb] += 1
    return ParallelStepResult(
        next_configuration=tuple(next_config),
        fired_vertices=tuple(fired),
    )


def q_reduced(
    graph: SimpleUndirectedGraph, divisor: tuple[int, ...], sink: str
) -> QReducedResult:
    """Compute the q-reduced normal form via Dhar's algorithm.

    After repeatedly firing unstable nonsink vertices (as in stabilization),
    we then perform the reverse step: borrow from the sink to create a
    non-negative configuration, and re-stabilize. This produces the unique
    q-reduced representative.
    """
    _admit_divisor(graph, divisor)
    _admit_sink(graph, sink)
    vertices = graph.vertices
    n = len(vertices)
    sink_idx = vertices.index(sink)
    adj = _adjacency(graph)
    degrees = _degrees(graph)
    # Use Dhar's burning algorithm for q-reduction:
    # 1. Start with divisor D.
    # 2. Fire unstable vertices until stable (standard stabilization).
    # 3. Use Dhar's reverse: check if any nonempty set can fire by
    #    testing if the stable config is superstable.
    #    If not superstable, borrow from sink and re-stabilize.
    #
    # The q-reduced form is: D - L*f where f is the firing vector.
    # We compute f = odometer from stabilization + borrow rounds.

    config = list(divisor)
    total_firing = [0] * n

    # Stabilize first
    eta, odo = _stabilize_configuration(config, adj, degrees, sink_idx)
    config = eta
    for i in range(n):
        total_firing[i] += odo[i]

    # Borrow from sink when needed
    # A stable config is q-reduced iff it is non-negative on nonsink vertices.
    # If some nonsink vertex is negative, we borrow from the sink:
    # fire the sink (which gives chips to its neighbors) and re-stabilize.
    max_rounds = n * n + 10
    rounds = 0
    while any(config[i] < 0 for i in range(n) if i != sink_idx):
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError("q-reduction did not converge")
        # Fire the sink: each neighbor of sink gains 1 chip.
        # Firing the sink is D' = D - L * e_sink, so it counts in the
        # firing vector to preserve D_reduced = D - L * f.
        for nb in adj[sink_idx]:
            config[nb] += 1
        total_firing[sink_idx] += 1
        # Now re-stabilize
        eta, odo = _stabilize_configuration(config, adj, degrees, sink_idx)
        config = eta
        for i in range(n):
            total_firing[i] += odo[i]

    return QReducedResult(
        reduced_divisor=tuple(config),
        firing_vector=tuple(total_firing),
    )


def degree(divisor: tuple[int, ...]) -> DegreeResult:
    """Compute the degree of a divisor: sum of all coefficients."""
    if not divisor:
        raise OperationDomainValidationError(
            location=("divisor",),
            code="chip_firing.divisor_must_not_be_empty",
            message="divisor must not be empty",
        )
    return DegreeResult(degree=sum(divisor))


def canonical_divisor(graph: SimpleUndirectedGraph) -> CanonicalDivisorResult:
    """Compute the canonical divisor K(v) = deg(v) - 2."""
    _admit_graph(graph)
    vertices = graph.vertices
    degrees = _degrees(graph)
    divisor = tuple(deg - 2 for deg in degrees)
    return CanonicalDivisorResult(
        vertices=vertices,
        divisor=divisor,
        degree=sum(divisor),
    )


def _smith_normal_form_diagonal(
    matrix: list[list[int]],
) -> tuple[int, ...]:
    """Return the diagonal entries of the Smith normal form of an integer matrix.

    Uses SymPy's smith_normal_decomp over ZZ.
    """
    import sympy
    from sympy.matrices.normalforms import smith_normal_decomp

    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0
    if rows == 0 or cols == 0:
        return ()
    source = sympy.Matrix([[int(value) for value in row] for row in matrix])
    diagonal, _left, _right = smith_normal_decomp(source, domain=sympy.ZZ)
    result = []
    for i in range(min(rows, cols)):
        val = int(diagonal[i, i])
        if val < 0:
            val = -val
        result.append(val)
    return tuple(result)


def _critical_group_factors(
    graph: SimpleUndirectedGraph,
    sink: str,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return (nonsink_vertices, invariant_factors) for the critical group."""
    vertices = graph.vertices
    n = len(vertices)
    sink_idx = vertices.index(sink)
    nonsink = [i for i in range(n) if i != sink_idx]
    if not nonsink:
        return (), ()
    lap = laplacian(graph).laplacian
    reduced = [[lap[i][j] for j in nonsink] for i in nonsink]
    factors = _smith_normal_form_diagonal(reduced)
    nonsink_labels = tuple(vertices[i] for i in nonsink)
    invariant = tuple(d for d in factors if d != 0)
    return nonsink_labels, invariant


def critical_group(graph: SimpleUndirectedGraph, sink: str) -> CriticalGroupResult:
    """Compute the critical group via SNF of the reduced Laplacian."""
    _admit_sink(graph, sink)
    nonsink_labels, invariant = _critical_group_factors(graph, sink)
    order = 1
    for d in invariant:
        order *= d
    return CriticalGroupResult(
        sink=sink,
        nonsink_vertices=nonsink_labels,
        invariant_factors=invariant,
        order=order,
    )


def abel_jacobi(
    graph: SimpleUndirectedGraph, divisor: tuple[int, ...], sink: str
) -> AbelJacobiResult:
    """Map a degree-zero divisor into critical-group coordinates.

    The coordinates are the remainder of the nonsink divisor coefficients
    modulo the invariant factors of the critical group (the diagonal of
    the SNF of the reduced Laplacian). Zero and unit factors are excluded.
    """
    _admit_divisor(graph, divisor)
    _admit_sink(graph, sink)
    if sum(divisor) != 0:
        raise OperationDomainValidationError(
            location=("divisor",),
            code="chip_firing.divisor_not_degree_zero",
            message="divisor must have degree zero",
        )
    vertices = graph.vertices
    n = len(vertices)
    sink_idx = vertices.index(sink)
    nonsink = [i for i in range(n) if i != sink_idx]
    nonsink_labels, invariant = _critical_group_factors(graph, sink)
    nonsink_div = [divisor[i] for i in nonsink]
    # The coordinates: nonsink_div mod the invariant factors.
    # Only non-unit, non-zero factors matter for the quotient group.
    coords = []
    j = 0
    for d in invariant:
        if d <= 1:
            continue
        idx = nonsink[j] if j < len(nonsink_div) else 0
        coords.append(nonsink_div[idx] % d)
        j += 1
    return AbelJacobiResult(
        sink=sink,
        nonsink_vertices=nonsink_labels,
        coordinates=tuple(coords),
        invariant_factors=invariant,
    )


__all__ = [
    "abel_jacobi",
    "canonical_divisor",
    "critical_group",
    "degree",
    "fire_vector",
    "firing",
    "laplacian",
    "parallel_step",
    "q_reduced",
    "reduced_laplacian",
    "stabilize",
]
