"""Bounded exact chromatic bipartition feasibility."""

from __future__ import annotations

import time
from typing import Literal, NoReturn, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._execution import OperationExecutionTimeoutError
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    MathTool,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.optimization._budget import remaining_ms
from jacobian.math.graphs.optimization._chromatic_kernel import (
    build_simple_graph,
    solve_chromatic_number,
)
from jacobian.math.graphs.optimization._coloring_models import ChromaticNumberBudget
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_CHROMATIC_BIPARTITION_WORK = 2_000_000
MAX_CHROMATIC_BACKEND_ORDER = 32
MAX_CHROMATIC_BIPARTITION_VERTICES = 256
MAX_CHROMATIC_BIPARTITION_PARTITIONS = MAX_CHROMATIC_BIPARTITION_WORK
MAX_CHROMATIC_BIPARTITION_LABEL_CHARACTERS = 1_000_000


class ChromaticBipartitionRequest(StrictModel):
    """Decide whether both sides of a vertex bipartition meet chromatic thresholds."""

    graph: SimpleUndirectedGraph
    s: StrictInt = Field(ge=1, le=32)
    t: StrictInt = Field(ge=1, le=32)
    resource_budget: ChromaticNumberBudget = Field(
        default_factory=ChromaticNumberBudget
    )


class ChromaticBipartitionResult(StrictModel):
    """Exact split or exact negative result after a complete admitted search."""

    graph: SimpleUndirectedGraph
    s: StrictInt = Field(ge=1, le=32)
    t: StrictInt = Field(ge=1, le=32)
    status: Literal["SPLIT", "NO_SPLIT"]
    side_a: tuple[str, ...] | None = Field(default=None, max_length=256)
    side_b: tuple[str, ...] | None = Field(default=None, max_length=256)
    chromatic_a: StrictInt | None = Field(default=None, ge=0, le=32)
    chromatic_b: StrictInt | None = Field(default=None, ge=0, le=32)
    checked_partitions: StrictInt = Field(ge=0, le=MAX_CHROMATIC_BIPARTITION_PARTITIONS)

    @model_validator(mode="after")
    def bind_result_to_source(self) -> Self:
        if self.status == "SPLIT":
            if self.side_a is None or self.side_b is None:
                raise PydanticCustomError(
                    "graph.chromatic_bipartition_witness_required",
                    "a split result requires both sides",
                )
            if self.chromatic_a is None or self.chromatic_b is None:
                raise PydanticCustomError(
                    "graph.chromatic_bipartition_values_required",
                    "a split result requires both chromatic values",
                )
            if not self.side_a or not self.side_b:
                raise PydanticCustomError(
                    "graph.chromatic_bipartition_sides_nonempty",
                    "a split must have two nonempty sides",
                )
            if (
                len(set(self.side_a)) != len(self.side_a)
                or len(set(self.side_b)) != len(self.side_b)
                or len(self.side_a) + len(self.side_b) != len(self.graph.vertices)
                or set(self.side_a) & set(self.side_b)
                or set(self.side_a) | set(self.side_b) != set(self.graph.vertices)
                or any(
                    vertex not in self.graph.vertices
                    for vertex in self.side_a + self.side_b
                )
            ):
                raise PydanticCustomError(
                    "graph.chromatic_bipartition_must_partition_vertices",
                    "split sides must partition graph vertices exactly once",
                )
            if (
                tuple(vertex for vertex in self.graph.vertices if vertex in self.side_a)
                != self.side_a
                or tuple(
                    vertex for vertex in self.graph.vertices if vertex in self.side_b
                )
                != self.side_b
            ):
                raise PydanticCustomError(
                    "graph.chromatic_bipartition_sides_not_canonical",
                    "split sides must preserve the source vertex axis order",
                )
            if self.chromatic_a < self.s or self.chromatic_b < self.t:
                raise PydanticCustomError(
                    "graph.chromatic_bipartition_thresholds_not_met",
                    "reported chromatic values must meet the requested thresholds",
                )
        elif (
            self.side_a is not None
            or self.side_b is not None
            or self.chromatic_a is not None
            or self.chromatic_b is not None
        ):
            raise PydanticCustomError(
                "graph.chromatic_bipartition_non_split_must_not_claim_witness",
                "non-split results cannot carry a witness or chromatic values",
            )
        if self.checked_partitions > _unordered_partition_count(
            len(self.graph.vertices)
        ):
            raise PydanticCustomError(
                "graph.chromatic_bipartition_checked_axis_exceeded",
                "checked_partitions cannot exceed the unordered partition axis",
            )
        return self


def _induced_graph(
    graph: SimpleUndirectedGraph, vertices: tuple[str, ...]
) -> SimpleUndirectedGraph:
    selected = set(vertices)
    return SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            (left, right)
            for left, right in graph.edges
            if left in selected and right in selected
        ),
    )


def _unordered_partition_count(order: int) -> int:
    """Count nonempty proper subsets with a strictly smaller complement mask."""

    if order < 2:
        return 0
    return (1 << (order - 1)) - 1


def _edgeless_chromatic_bipartition(
    request: ChromaticBipartitionRequest,
) -> ChromaticBipartitionResult:
    """Decide an edgeless instance without enumerating 2^{n-1} partitions."""

    vertices = request.graph.vertices
    if len(vertices) < 2 or request.s > 1 or request.t > 1:
        return ChromaticBipartitionResult(
            graph=request.graph,
            s=request.s,
            t=request.t,
            status="NO_SPLIT",
            checked_partitions=0,
        )
    return ChromaticBipartitionResult(
        graph=request.graph,
        s=request.s,
        t=request.t,
        status="SPLIT",
        side_a=(vertices[0],),
        side_b=vertices[1:],
        chromatic_a=1,
        chromatic_b=1,
        checked_partitions=0,
    )


def _threshold_sum_exceeds_order(request: ChromaticBipartitionRequest) -> bool:
    """chi(G[A]) <= |A| and chi(G[B]) <= |B|, so s + t > n cannot split."""

    return request.s + request.t > len(request.graph.vertices)


def _impossible_threshold_bipartition(
    request: ChromaticBipartitionRequest,
) -> ChromaticBipartitionResult:
    return ChromaticBipartitionResult(
        graph=request.graph,
        s=request.s,
        t=request.t,
        status="NO_SPLIT",
        checked_partitions=0,
    )


def _is_bipartite(graph: SimpleUndirectedGraph) -> bool:
    color: dict[str, int] = {}
    adjacency: dict[str, list[str]] = {vertex: [] for vertex in graph.vertices}
    for left, right in graph.edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    for start in graph.vertices:
        if start in color:
            continue
        color[start] = 0
        queue = [start]
        for vertex in queue:
            for neighbor in adjacency[vertex]:
                assigned = color.get(neighbor)
                if assigned is None:
                    color[neighbor] = 1 - color[vertex]
                    queue.append(neighbor)
                elif assigned == color[vertex]:
                    return False
    return True


def _chromatic_search_work(order: int, edge_count: int) -> int:
    """Charge one bounded k-colorability sweep of an induced core."""

    encoding = order * order + edge_count * order * order
    return order * (encoding + order + edge_count + 1)


def _chromatic_bipartition_can_return_split(
    request: ChromaticBipartitionRequest,
) -> bool:
    """True when an admitted search can still return a SPLIT witness."""

    if len(request.graph.vertices) < 2:
        return False
    if _threshold_sum_exceeds_order(request):
        return False
    return not (not request.graph.edges and (request.s > 1 or request.t > 1))


def _retained_label_characters(
    graph: SimpleUndirectedGraph, *, charge_witness_axes: bool
) -> int:
    """Charge the source graph, and witness axes only when a SPLIT is possible."""

    source = sum(map(len, graph.vertices)) + sum(
        len(left) + len(right) for left, right in graph.edges
    )
    if not charge_witness_axes:
        return source
    return source + sum(map(len, graph.vertices))


def _chromatic_bipartition_reconstruction_work(
    order: int, edge_count: int, partitions: int
) -> int:
    """Charge the two induced-graph scans performed for each partition."""

    return partitions * 2 * (order + edge_count)


def _chromatic_deadline_expired(request: ChromaticBipartitionRequest) -> NoReturn:
    raise OperationExecutionTimeoutError(
        "chromatic bipartition deadline expired during the kernel search",
        configured_seconds=request.resource_budget.wall_seconds,
        adjustable_field_path=("resource_budget", "wall_seconds"),
    )


def _induced_edge_core(
    graph: SimpleUndirectedGraph, vertices: tuple[str, ...]
) -> SimpleUndirectedGraph:
    """Drop isolates; they do not change chromatic number of a nonempty side."""

    induced = _induced_graph(graph, vertices)
    if not induced.edges:
        return induced
    support = {endpoint for edge in induced.edges for endpoint in edge}
    return _induced_graph(
        induced, tuple(vertex for vertex in vertices if vertex in support)
    )


def _exact_induced_chromatic(
    graph: SimpleUndirectedGraph,
    vertices: tuple[str, ...],
    request: ChromaticBipartitionRequest,
    started: float,
) -> int | None:
    core = _induced_edge_core(graph, vertices)
    if len(core.vertices) <= 1 or not core.edges:
        return 1
    if _is_bipartite(core):
        return 2
    if len(core.vertices) > MAX_CHROMATIC_BACKEND_ORDER:
        return None
    return _chromatic_number(core, request, started)


def _unit_threshold_remainder(
    vertices: tuple[str, ...], index: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    side_a = (vertices[index],)
    side_b = vertices[:index] + vertices[index + 1 :]
    return side_a, side_b


def _unit_threshold_core_is_admitted(core: SimpleUndirectedGraph) -> bool:
    if not core.edges or _is_bipartite(core):
        return True
    if len(core.vertices) > MAX_CHROMATIC_BACKEND_ORDER:
        return False
    return _chromatic_search_work(len(core.vertices), len(core.edges)) <= (
        MAX_CHROMATIC_BIPARTITION_WORK
    )


def _unit_threshold_coloring_work(core: SimpleUndirectedGraph) -> int:
    """Charge the exact branch used for one singleton remainder."""

    if not core.edges or _is_bipartite(core):
        return len(core.vertices) + len(core.edges) + 1
    return _chromatic_search_work(len(core.vertices), len(core.edges))


def _unit_threshold_bipartition(
    request: ChromaticBipartitionRequest,
) -> ChromaticBipartitionResult:
    """Split off a singleton whose remainder has an exact cheap chromatic number."""

    vertices = request.graph.vertices
    started = time.monotonic()
    checked = 0
    for index in range(len(vertices)):
        side_a, side_b = _unit_threshold_remainder(vertices, index)
        core = _induced_edge_core(request.graph, side_b)
        if not _unit_threshold_core_is_admitted(core):
            continue
        chromatic_b = _exact_induced_chromatic(request.graph, side_b, request, started)
        if chromatic_b is None:
            _chromatic_deadline_expired(request)
        checked += 1
        return ChromaticBipartitionResult(
            graph=request.graph,
            s=request.s,
            t=request.t,
            status="SPLIT",
            side_a=side_a,
            side_b=side_b,
            chromatic_a=1,
            chromatic_b=chromatic_b,
            checked_partitions=checked,
        )
    return ChromaticBipartitionResult(
        graph=request.graph,
        s=request.s,
        t=request.t,
        status="NO_SPLIT",
        checked_partitions=0,
    )


def _refuse_chromatic_bipartition_work() -> None:
    raise OperationResourceAdmissionError(
        location=("graph",),
        code="graph.chromatic_bipartition_exact_work_exceeds",
        message=(
            "chromatic bipartition search exceeds the admitted complete-search "
            "work bound"
        ),
    )


def _admit_unit_threshold_chromatic(request: ChromaticBipartitionRequest) -> None:
    """Admit singleton traversal, including skipped cores, for both phases."""

    vertices = request.graph.vertices
    order = len(vertices)
    edge_count = len(request.graph.edges)
    work = 0
    has_admitted_candidate = False
    for index in range(order):
        _, side_b = _unit_threshold_remainder(vertices, index)
        reconstruction = 2 * (order + edge_count)
        core = _induced_edge_core(request.graph, side_b)
        classification = len(core.vertices) + len(core.edges) + 1
        # Admission reconstructs and classifies each remainder; the worker
        # repeats that work before coloring the first usable core.
        work += 2 * (reconstruction + classification)
        if not _unit_threshold_core_is_admitted(core):
            continue
        has_admitted_candidate = True
        coloring = _unit_threshold_coloring_work(core)
        if coloring > classification:
            work += coloring
        break
    if has_admitted_candidate and work <= MAX_CHROMATIC_BIPARTITION_WORK:
        return
    _refuse_chromatic_bipartition_work()


def _admit_chromatic_bipartition(request: ChromaticBipartitionRequest) -> None:
    """Charge every unordered partition and its inner k-colorability encodings."""

    order = len(request.graph.vertices)
    cheap_no_witness = not _chromatic_bipartition_can_return_split(request)
    if order > MAX_CHROMATIC_BIPARTITION_VERTICES and not cheap_no_witness:
        raise OperationResourceAdmissionError(
            location=("graph", "vertices"),
            code="graph.chromatic_bipartition.vertex_axis_bound",
            message=(
                "chromatic bipartition results support at most "
                f"{MAX_CHROMATIC_BIPARTITION_VERTICES} vertices"
            ),
        )
    if _retained_label_characters(
        request.graph,
        charge_witness_axes=_chromatic_bipartition_can_return_split(request),
    ) > (
        MAX_CHROMATIC_BIPARTITION_LABEL_CHARACTERS
    ):
        raise OperationResourceAdmissionError(
            location=("graph",),
            code="graph.chromatic_bipartition.retained_labels_exceed_bound",
            message=(
                "chromatic bipartition source and witness labels exceed the "
                "admitted character bound"
            ),
        )
    if not request.graph.edges or _threshold_sum_exceeds_order(request):
        return
    if request.s == 1 and request.t == 1 and order >= 2:
        _admit_unit_threshold_chromatic(request)
        return
    n = len(request.graph.vertices)
    m = len(request.graph.edges)
    partitions = _unordered_partition_count(n)
    # Two induced graphs, each trying up to n color counts. One encoding of
    # order n and k<=n uses n*k vertex literals plus m*k^2 edge separations.
    work = partitions * 2 * _chromatic_search_work(n, m)
    work += _chromatic_bipartition_reconstruction_work(n, m, partitions)
    if work > MAX_CHROMATIC_BIPARTITION_WORK:
        _refuse_chromatic_bipartition_work()


def _find_chromatic_bipartition_kernel(
    request: ChromaticBipartitionRequest,
) -> ChromaticBipartitionResult:
    """Search every canonical unordered vertex bipartition under one deadline."""
    graph = request.graph
    if _threshold_sum_exceeds_order(request) or len(graph.vertices) < 2:
        return _impossible_threshold_bipartition(request)
    if request.s == 1 and request.t == 1:
        return _unit_threshold_bipartition(request)
    if not graph.edges:
        return _edgeless_chromatic_bipartition(request)
    source_vertices = graph.vertices
    started = time.monotonic()
    checked = 0
    order = len(source_vertices)
    for mask in range(1, 1 << order):
        complement = ((1 << order) - 1) ^ mask
        if complement == 0 or mask > complement:
            continue
        side_a = tuple(
            vertex
            for index, vertex in enumerate(source_vertices)
            if mask & (1 << index)
        )
        side_b = tuple(
            vertex
            for index, vertex in enumerate(source_vertices)
            if not mask & (1 << index)
        )
        if remaining_ms(started, request.resource_budget.wall_seconds) <= 0:
            _chromatic_deadline_expired(request)
        chromatic_a = _chromatic_number(_induced_graph(graph, side_a), request, started)
        if chromatic_a is None:
            _chromatic_deadline_expired(request)
        chromatic_b = _chromatic_number(_induced_graph(graph, side_b), request, started)
        if chromatic_b is None:
            _chromatic_deadline_expired(request)
        checked += 1
        if chromatic_a >= request.s and chromatic_b >= request.t:
            return ChromaticBipartitionResult(
                graph=graph,
                s=request.s,
                t=request.t,
                status="SPLIT",
                side_a=side_a,
                side_b=side_b,
                chromatic_a=chromatic_a,
                chromatic_b=chromatic_b,
                checked_partitions=checked,
            )
        if (
            request.s != request.t
            and chromatic_a >= request.t
            and chromatic_b >= request.s
        ):
            return ChromaticBipartitionResult(
                graph=graph,
                s=request.s,
                t=request.t,
                status="SPLIT",
                side_a=side_b,
                side_b=side_a,
                chromatic_a=chromatic_b,
                chromatic_b=chromatic_a,
                checked_partitions=checked,
            )
    return ChromaticBipartitionResult(
        graph=graph,
        s=request.s,
        t=request.t,
        status="NO_SPLIT",
        checked_partitions=checked,
    )


def find_chromatic_bipartition(
    request: ChromaticBipartitionRequest,
) -> ChromaticBipartitionResult:
    """Run the aggregate search in a killable worker with one request deadline."""

    _admit_chromatic_bipartition(request)
    # Circular: the process owner imports this module's request and result types.
    from jacobian.math.graphs.optimization._chromatic_bipartition_process import (
        find_chromatic_bipartition as run_worker,
    )

    return run_worker(request)


def _chromatic_number(
    graph: SimpleUndirectedGraph, request: ChromaticBipartitionRequest, started: float
) -> int | None:
    output = solve_chromatic_number(
        build_simple_graph(graph),
        graph=graph,
        vertices=graph.vertices,
        wall_seconds=request.resource_budget.wall_seconds,
        started=started,
    )
    return output.chromatic_number if output.status == "EXACT" else None


CHROMATIC_BIPARTITION_OPERATION = MathTool(
    operation_id="graph.chromatic_bipartition.find",
    title="Find a chromatic bipartition",
    description=(
        "Search a bounded simple graph for a canonical vertex bipartition whose "
        "two induced subgraphs have chromatic numbers at least s and t. "
        "Return a witness, an exact NO_SPLIT after complete search, or raise an "
        "execution timeout when the admitted kernel deadline expires."
    ),
    request_type=ChromaticBipartitionRequest,
    result_type=ChromaticBipartitionResult,
    run=find_chromatic_bipartition,
    tags=("graph", "chromatic", "bipartition", "exact", "bounded"),
    examples=(
        OperationExample(
            name="k4_two_two_split",
            description="K4 splits into two induced K2 graphs, each of chromatic number 2.",
            input={
                "graph": {
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [
                        ["a", "b"],
                        ["a", "c"],
                        ["a", "d"],
                        ["b", "c"],
                        ["b", "d"],
                        ["c", "d"],
                    ],
                },
                "s": 2,
                "t": 2,
            },
        ),
    ),
)

__all__ = [
    "CHROMATIC_BIPARTITION_OPERATION",
    "MAX_CHROMATIC_BIPARTITION_LABEL_CHARACTERS",
    "MAX_CHROMATIC_BIPARTITION_PARTITIONS",
    "MAX_CHROMATIC_BIPARTITION_VERTICES",
    "MAX_CHROMATIC_BIPARTITION_WORK",
    "ChromaticBipartitionRequest",
    "ChromaticBipartitionResult",
    "find_chromatic_bipartition",
]
