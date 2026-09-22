"""Exact chip-firing operations."""

from __future__ import annotations

from collections import deque

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.chip_firing._models import (
    MAX_COEFFICIENT_DIGITS,
    MAX_CRITICAL_GROUP_VERTICES,
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
from jacobian.math.matrices.values import IntegerMatrix


def _admit_connected(
    graph: SimpleUndirectedGraph,
    adjacency: tuple[tuple[int, ...], ...] | None = None,
) -> None:
    if not _is_connected(graph, adjacency):
        raise OperationDomainValidationError(
            location=("graph",),
            code="chip_firing.requires_connected_graph",
            message="sink chip-firing requires a connected graph",
        )


def _admit_coefficient_height(divisor: tuple[int, ...]) -> None:
    if any(abs(value) >= 10**MAX_COEFFICIENT_DIGITS for value in divisor):
        raise OperationDomainValidationError(
            location=("divisor",),
            code="chip_firing.coefficient_bound",
            message="divisor coefficients exceed the digit bound",
        )


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


def _graph_structure(
    graph: SimpleUndirectedGraph,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Build request-local adjacency and degrees in one source scan."""

    rows: list[list[int]] = [[] for _ in graph.vertices]
    indices = {vertex: index for index, vertex in enumerate(graph.vertices)}
    for left, right in graph.edges:
        left_index, right_index = indices[left], indices[right]
        rows[left_index].append(right_index)
        rows[right_index].append(left_index)
    adjacency = tuple(tuple(row) for row in rows)
    return adjacency, tuple(len(row) for row in adjacency)


def _laplacian_entries(
    adjacency: tuple[tuple[int, ...], ...], degrees: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    dimension = len(adjacency)
    rows = []
    for row in range(dimension):
        neighbors = set(adjacency[row])
        rows.append(
            tuple(
                degrees[row] if row == column else -int(column in neighbors)
                for column in range(dimension)
            )
        )
    return tuple(rows)


def _reduced_laplacian_entries(
    adjacency: tuple[tuple[int, ...], ...],
    degrees: tuple[int, ...],
    sink_index: int,
) -> tuple[tuple[int, ...], ...]:
    nonsink = tuple(index for index in range(len(adjacency)) if index != sink_index)
    rows = []
    for row in nonsink:
        neighbors = set(adjacency[row])
        rows.append(
            tuple(
                degrees[row] if row == column else -int(column in neighbors)
                for column in nonsink
            )
        )
    return tuple(rows)


def laplacian(graph: SimpleUndirectedGraph) -> LaplacianResult:
    """Compute the graph Laplacian L = D - A where D is the degree matrix."""
    _admit_graph(graph)
    vertices = graph.vertices
    n = len(vertices)
    adjacency, degrees = _graph_structure(graph)
    entries = _laplacian_entries(adjacency, degrees)

    return LaplacianResult(
        graph=graph,
        vertices=vertices,
        laplacian=IntegerMatrix(
            row_count=n,
            column_count=n,
            entries=entries,
        ),
        degrees=degrees,
    )


def reduced_laplacian(
    graph: SimpleUndirectedGraph, sink: str
) -> ReducedLaplacianResult:
    """Delete the sink row/column from the full Laplacian."""
    _admit_sink(graph, sink)
    vertices = graph.vertices
    sink_idx = vertices.index(sink)
    nonsink = tuple(i for i in range(len(vertices)) if i != sink_idx)
    adjacency, degrees = _graph_structure(graph)
    reduced = _reduced_laplacian_entries(adjacency, degrees, sink_idx)
    return ReducedLaplacianResult(
        graph=graph,
        vertices=tuple(vertices[i] for i in nonsink),
        sink=sink,
        reduced_laplacian=IntegerMatrix(
            row_count=len(reduced),
            column_count=len(reduced[0]) if reduced else 0,
            entries=tuple(tuple(int(value) for value in row) for row in reduced),
        ),
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
    adjacency, degrees = _graph_structure(graph)
    fire_idx = vertices.index(firing_vertex)
    result = list(divisor)

    result[fire_idx] -= degrees[fire_idx]
    for neighbor in adjacency[fire_idx]:
        result[neighbor] += 1

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
    adjacency, degrees = _graph_structure(graph)
    result = []
    for index, value in enumerate(divisor):
        delta = degrees[index] * firing_vector[index] - sum(
            firing_vector[neighbor] for neighbor in adjacency[index]
        )
        result.append(value - delta)
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

    Returns (stable_config, odometer). The owner admits a connected graph
    and an effective nonsink configuration. If H=Lq^-1, the maximum
    principle and effective resistance give 0 <= H_ij <= n-1: column j
    is the voltage of a unit j-to-sink current, maximized at j, whose
    resistance is at most a simple path length. Thus h=H*1 <= (n-1)^2.
    The nonnegative potential h.T*eta drops by one per legal firing,
    bounding all firings by sum(eta_nonsink)*(n-1)^2. A batch is a
    sequence of legal firings at the same vertex, not a parallel firing.
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
        request_checkpoint("during chip-firing stabilization")
        v = queue.popleft()
        in_queue[v] = False
        if eta[v] < degrees[v]:
            continue
        count = eta[v] // degrees[v]
        eta[v] -= count * degrees[v]
        odometer[v] += count
        for nb in adj[v]:
            eta[nb] += count
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
    adjacency, degrees = _graph_structure(graph)
    _admit_connected(graph, adjacency)
    request_checkpoint("before chip-firing stabilization")
    vertices = graph.vertices
    sink_idx = vertices.index(sink)
    config = list(configuration)
    eta, odometer = _stabilize_configuration(config, adjacency, degrees, sink_idx)
    request_checkpoint("after chip-firing stabilization")
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
    adjacency, degrees = _graph_structure(graph)
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
        for neighbor in adjacency[vi]:
            next_config[neighbor] += 1
    return ParallelStepResult(
        next_configuration=tuple(next_config),
        fired_vertices=tuple(fired),
    )


def q_reduced(
    graph: SimpleUndirectedGraph, divisor: tuple[int, ...], sink: str
) -> QReducedResult:
    """Compute the q-reduced normal form via Dhar's algorithm.

    First make the nonsink divisor effective by an exact rational solve,
    then fire Dhar's unburned sets. Every update uses D' = D - L*f with
    f(sink)=0. All subsets burning is exactly the q-reduced criterion.
    """
    _admit_divisor(graph, divisor)
    _admit_sink(graph, sink)
    adjacency, degrees = _graph_structure(graph)
    _admit_connected(graph, adjacency)
    _admit_coefficient_height(divisor)
    request_checkpoint("before q-reduction")
    vertices = graph.vertices
    n = len(vertices)
    sink_idx = vertices.index(sink)
    config = list(divisor)
    total_firing = [0] * n
    nonsink = [i for i in range(n) if i != sink_idx]
    if nonsink:
        from flint import fmpq_mat

        # FLINT solves a nonsingular integer reduced Laplacian over Q.
        # At n<=50 and 1000-digit coefficients, fraction-free elimination
        # has O(n^3) arithmetic steps; its minors have at most
        # 1000 + O(n log n) digits by Hadamard (only the RHS is large).
        # Put x=Lq^-1*(D-deg), f=floor(x), r=x-f in [0,1)^n.
        # Then D-Lq*f=deg+Lq*r is >=0 and <2*deg coordinatewise.
        reduced = [
            [degrees[i] if i == j else -int(j in adjacency[i]) for j in nonsink]
            for i in nonsink
        ]
        solution = fmpq_mat(reduced).solve(  # type: ignore[call-arg]  # python-flint 0.9 stubs omit documented algorithm.
            fmpq_mat([[divisor[i] - degrees[i]] for i in nonsink]),
            algorithm="fflu",
        )
        request_checkpoint("after q-reduction exact solve")
        for j, i in enumerate(nonsink):
            value = solution[j, 0]
            total_firing[i] = int(value.numerator) // int(value.denominator)
        for i in range(n):
            config[i] -= degrees[i] * total_firing[i] - sum(
                total_firing[j] for j in adjacency[i]
            )

    # Initial nonsink mass < 2*sum(deg) <= 4m. The Green-function
    # potential from stabilization bounds total vertex firings by
    # 4m*(n-1)^2, including |S| for each Dhar set S. Thus this loop is
    # independent of divisor magnitude. Each burn pass costs O(n+m).
    while True:
        config, odo = _stabilize_configuration(config, adjacency, degrees, sink_idx)
        for i in nonsink:
            total_firing[i] += odo[i]
        request_checkpoint("during q-reduction Dhar burning")
        burned = {sink_idx}
        pending = deque([sink_idx])
        outgoing = [0] * n
        while pending:
            v = pending.popleft()
            for j in adjacency[v]:
                outgoing[j] += 1
                if j not in burned and config[j] < outgoing[j]:
                    burned.add(j)
                    pending.append(j)
        unburned = [i for i in nonsink if i not in burned]
        if not unburned:
            break
        # Only cut edges change a simultaneous subset firing. Vertices
        # inside S have at least outdeg_S chips, so effectivity persists.
        for i in unburned:
            total_firing[i] += 1
            for j in adjacency[i]:
                if j in burned:
                    config[i] -= 1
                    config[j] += 1

    request_checkpoint("after q-reduction")
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
    _adjacency, degrees = _graph_structure(graph)
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

    Uses FLINT's diagonal-only exact integer SNF kernel in a deadline-bound
    killable worker so expanded critical-group requests stay inside the
    request-scoped execution envelope.
    """
    from jacobian.math.graphs.chip_firing._snf_process import (
        smith_normal_form_diagonal,
    )

    return smith_normal_form_diagonal(matrix)


def _critical_group_factors(
    graph: SimpleUndirectedGraph,
    sink: str,
    adjacency: tuple[tuple[int, ...], ...],
    degrees: tuple[int, ...],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return (nonsink_vertices, invariant_factors) for the critical group."""
    vertices = graph.vertices
    sink_idx = vertices.index(sink)
    nonsink = [i for i in range(len(vertices)) if i != sink_idx]
    if not nonsink:
        return (), ()
    reduced = [
        list(row) for row in _reduced_laplacian_entries(adjacency, degrees, sink_idx)
    ]
    factors = _smith_normal_form_diagonal(reduced)
    nonsink_labels = tuple(vertices[i] for i in nonsink)
    invariant = tuple(d for d in factors if d != 0)
    return nonsink_labels, invariant


def _is_connected(
    graph: SimpleUndirectedGraph,
    adjacency: tuple[tuple[int, ...], ...] | None = None,
) -> bool:
    if not graph.vertices:
        return False
    if adjacency is None:
        adjacency, _degrees = _graph_structure(graph)
    reached = {0}
    pending = [0]
    while pending:
        vertex = pending.pop()
        for neighbor in adjacency[vertex]:
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return len(reached) == len(graph.vertices)


def critical_group(graph: SimpleUndirectedGraph, sink: str) -> CriticalGroupResult:
    """Compute the critical group via SNF of the reduced Laplacian."""
    if not 1 <= len(graph.vertices) <= MAX_CRITICAL_GROUP_VERTICES:
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="chip_firing.critical_group_vertex_bound",
            message="critical-group computation supports at most "
            f"{MAX_CRITICAL_GROUP_VERTICES} vertices",
        )
    if sink not in graph.vertices:
        raise OperationDomainValidationError(
            location=("sink",),
            code="chip_firing.sink_not_in_graph",
            message="sink vertex must be in the graph",
        )
    adjacency, degrees = _graph_structure(graph)
    if not _is_connected(graph, adjacency):
        raise OperationDomainValidationError(
            location=("graph",),
            code="chip_firing.critical_group_requires_connected_graph",
            message="critical-group computation requires a connected graph",
        )
    dimension = len(graph.vertices) - 1
    # A simple connected graph has at most n^(n-2) spanning trees, so the
    # result size is bounded independently of the cubic SNF work estimate.
    if dimension**3 > 1_500_000:
        raise OperationDomainValidationError(
            location=("graph",),
            code="chip_firing.critical_group_work_bound",
            message="reduced-Laplacian SNF exceeds the exact work bound",
        )
    nonsink_labels, invariant = _critical_group_factors(graph, sink, adjacency, degrees)
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

    Successive column and row Hermite forms eliminate trivial quotient
    directions while transporting the divisor by each exact row map.
    For pinned SymPy U*C*V=S on the remaining quotient, return the reduced
    image transported by U modulo S, omitting unit factors. Relative axis
    order is retained throughout. Admission uses the reduced problem.
    """
    _admit_divisor(graph, divisor)
    _admit_sink(graph, sink)
    adjacency, degrees = _graph_structure(graph)
    _admit_connected(graph, adjacency)
    _admit_coefficient_height(divisor)
    request_checkpoint("before Abel-Jacobi coordinates")
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
    from jacobian.math.graphs.chip_firing._snf_process import smith_coordinates

    matrix = [
        list(row) for row in _reduced_laplacian_entries(adjacency, degrees, sink_idx)
    ]
    invariant, coords = smith_coordinates(matrix, [divisor[i] for i in nonsink])
    request_checkpoint("after Abel-Jacobi coordinates")
    return AbelJacobiResult(
        sink=sink,
        nonsink_vertices=tuple(vertices[i] for i in nonsink),
        coordinates=tuple(coords),
        invariant_factors=invariant,
    )


def verify_laplacian(claim: LaplacianResult) -> bool:
    """Verify a labelled Laplacian claim against its retained graph."""
    try:
        return laplacian(claim.graph) == claim
    except (TypeError, ValueError, OperationDomainValidationError):
        return False


def verify_reduced_laplacian(claim: ReducedLaplacianResult) -> bool:
    """Verify a reduced Laplacian claim against its graph and sink."""
    try:
        return reduced_laplacian(claim.graph, claim.sink) == claim
    except (TypeError, ValueError, OperationDomainValidationError):
        return False


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
    "verify_laplacian",
    "verify_reduced_laplacian",
]
