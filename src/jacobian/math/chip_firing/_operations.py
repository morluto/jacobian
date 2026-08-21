"""Exact chip-firing operations."""

from __future__ import annotations

from collections import deque

from jacobian.math.chip_firing._models import (
    AbelJacobiRequest,
    AbelJacobiResult,
    CanonicalDivisorRequest,
    CanonicalDivisorResult,
    CriticalGroupRequest,
    CriticalGroupResult,
    DegreeRequest,
    DegreeResult,
    FireVectorRequest,
    FireVectorResult,
    FiringRequest,
    FiringResult,
    LaplacianRequest,
    LaplacianResult,
    ParallelStepRequest,
    ParallelStepResult,
    QReducedRequest,
    QReducedResult,
    ReducedLaplacianRequest,
    ReducedLaplacianResult,
    StabilizeRequest,
    StabilizeResult,
)


def _adjacency(graph) -> tuple[tuple[int, ...], ...]:  # type: ignore[no-untyped-def]
    """Build an adjacency-list representation from a LabelledGraph."""
    n = len(graph.vertices)
    idx = {v: i for i, v in enumerate(graph.vertices)}
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in graph.edges:
        i, j = idx[u], idx[v]
        adj[i].append(j)
        adj[j].append(i)
    return tuple(tuple(row) for row in adj)


def _degrees(graph) -> tuple[int, ...]:  # type: ignore[no-untyped-def]
    idx = {v: i for i, v in enumerate(graph.vertices)}
    deg = [0] * len(graph.vertices)
    for u, v in graph.edges:
        deg[idx[u]] += 1
        deg[idx[v]] += 1
    return tuple(deg)


def compute_laplacian(request: LaplacianRequest) -> LaplacianResult:
    """Compute the graph Laplacian L = D - A where D is the degree matrix."""
    vertices = request.graph.vertices
    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}

    adj = [[0] * n for _ in range(n)]
    for u, v in request.graph.edges:
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


def compute_reduced_laplacian(
    request: ReducedLaplacianRequest,
) -> ReducedLaplacianResult:
    """Delete the sink row/column from the full Laplacian."""
    vertices = request.graph.vertices
    n = len(vertices)
    full = compute_laplacian(LaplacianRequest(graph=request.graph))
    lap = full.laplacian
    sink_idx = vertices.index(request.sink)
    nonsink = [i for i in range(n) if i != sink_idx]
    reduced = tuple(tuple(lap[i][j] for j in nonsink) for i in nonsink)
    return ReducedLaplacianResult(
        vertices=vertices,
        sink=request.sink,
        reduced_laplacian=reduced,
    )


def compute_firing(request: FiringRequest) -> FiringResult:
    """Fire a vertex: D' = D - L*e_v where L is the Laplacian."""
    vertices = request.graph.vertices
    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}

    adj = [[0] * n for _ in range(n)]
    for u, v in request.graph.edges:
        i, j = idx[u], idx[v]
        adj[i][j] += 1
        adj[j][i] += 1

    fire_idx = idx[request.firing_vertex]
    result = list(request.divisor)

    deg = sum(adj[fire_idx])
    result[fire_idx] -= deg
    for j in range(n):
        if adj[fire_idx][j] > 0:
            result[j] += adj[fire_idx][j]

    return FiringResult(
        vertex=request.firing_vertex,
        fired_divisor=tuple(result),
    )


def compute_fire_vector(request: FireVectorRequest) -> FireVectorResult:
    """Fire a vector: D' = D - L f. Degree is preserved by construction."""
    vertices = request.graph.vertices
    n = len(vertices)
    lap = compute_laplacian(LaplacianRequest(graph=request.graph)).laplacian
    divisor = list(request.divisor)
    f = request.firing_vector
    result = []
    for i in range(n):
        delta = sum(lap[i][j] * f[j] for j in range(n))
        result.append(divisor[i] - delta)
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


def compute_stabilize(request: StabilizeRequest) -> StabilizeResult:
    """Stabilize a sink configuration and return the odometer."""
    sc = request.configuration
    vertices = sc.graph.vertices
    sink_idx = vertices.index(sc.sink)
    adj = _adjacency(sc.graph)
    degrees = _degrees(sc.graph)
    config = list(sc.configuration)
    eta, odometer = _stabilize_configuration(config, adj, degrees, sink_idx)
    return StabilizeResult(
        stable=tuple(eta),
        odometer=tuple(odometer),
        total_firings=sum(odometer),
    )


def compute_parallel_step(request: ParallelStepRequest) -> ParallelStepResult:
    """One simultaneous legal firing step on all unstable nonsink vertices."""
    sc = request.configuration
    vertices = sc.graph.vertices
    sink_idx = vertices.index(sc.sink)
    adj = _adjacency(sc.graph)
    degrees = _degrees(sc.graph)
    config = list(sc.configuration)
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


def compute_q_reduced(request: QReducedRequest) -> QReducedResult:
    """Compute the q-reduced normal form via Dhar's algorithm.

    After repeatedly firing unstable nonsink vertices (as in stabilization),
    we then perform the reverse step: borrow from the sink to create a
    non-negative configuration, and re-stabilize. This produces the unique
    q-reduced representative.
    """
    graph = request.graph
    vertices = graph.vertices
    n = len(vertices)
    sink_idx = vertices.index(request.sink)
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

    config = list(request.divisor)
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


def compute_degree(request: DegreeRequest) -> DegreeResult:
    """Compute the degree of a divisor: sum of all coefficients."""
    return DegreeResult(degree=sum(request.divisor))


def compute_canonical_divisor(
    request: CanonicalDivisorRequest,
) -> CanonicalDivisorResult:
    """Compute the canonical divisor K(v) = deg(v) - 2."""
    vertices = request.graph.vertices
    degrees = _degrees(request.graph)
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


def _critical_group_factors(  # type: ignore[no-untyped-def]
    graph,
    sink: str,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return (nonsink_vertices, invariant_factors) for the critical group."""
    vertices = graph.vertices
    n = len(vertices)
    sink_idx = vertices.index(sink)
    nonsink = [i for i in range(n) if i != sink_idx]
    if not nonsink:
        return (), ()
    lap = compute_laplacian(LaplacianRequest(graph=graph)).laplacian
    reduced = [[lap[i][j] for j in nonsink] for i in nonsink]
    factors = _smith_normal_form_diagonal(reduced)
    nonsink_labels = tuple(vertices[i] for i in nonsink)
    invariant = tuple(d for d in factors if d != 0)
    return nonsink_labels, invariant


def compute_critical_group(request: CriticalGroupRequest) -> CriticalGroupResult:
    """Compute the critical group via SNF of the reduced Laplacian."""
    nonsink_labels, invariant = _critical_group_factors(request.graph, request.sink)
    order = 1
    for d in invariant:
        order *= d
    return CriticalGroupResult(
        sink=request.sink,
        nonsink_vertices=nonsink_labels,
        invariant_factors=invariant,
        order=order,
    )


def compute_abel_jacobi(request: AbelJacobiRequest) -> AbelJacobiResult:
    """Map a degree-zero divisor into critical-group coordinates.

    The coordinates are the remainder of the nonsink divisor coefficients
    modulo the invariant factors of the critical group (the diagonal of
    the SNF of the reduced Laplacian). Zero and unit factors are excluded.
    """
    vertices = request.graph.vertices
    n = len(vertices)
    sink_idx = vertices.index(request.sink)
    nonsink = [i for i in range(n) if i != sink_idx]
    nonsink_labels, invariant = _critical_group_factors(request.graph, request.sink)
    nonsink_div = [request.divisor[i] for i in nonsink]
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
        sink=request.sink,
        nonsink_vertices=nonsink_labels,
        coordinates=tuple(coords),
        invariant_factors=invariant,
    )
