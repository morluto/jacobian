"""Bounded exact chromatic bipartition feasibility."""

from __future__ import annotations

import time
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

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


class ChromaticBipartitionRequest(StrictModel):
    """Decide whether both sides of a vertex bipartition meet chromatic thresholds."""

    graph: SimpleUndirectedGraph
    s: StrictInt = Field(ge=1, le=32)
    t: StrictInt = Field(ge=1, le=32)
    resource_budget: ChromaticNumberBudget = Field(
        default_factory=ChromaticNumberBudget
    )


class ChromaticBipartitionResult(StrictModel):
    """Exact split, exact negative result, or operationally unresolved search."""

    graph: SimpleUndirectedGraph
    s: StrictInt = Field(ge=1, le=32)
    t: StrictInt = Field(ge=1, le=32)
    status: Literal["SPLIT", "NO_SPLIT", "UNKNOWN"]
    side_a: tuple[str, ...] | None = None
    side_b: tuple[str, ...] | None = None
    chromatic_a: StrictInt | None = Field(default=None, ge=0, le=32)
    chromatic_b: StrictInt | None = Field(default=None, ge=0, le=32)
    checked_partitions: StrictInt = Field(ge=0)

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


def _admit_chromatic_bipartition(request: ChromaticBipartitionRequest) -> None:
    """Charge every unordered partition and its inner k-colorability encodings."""

    if not request.graph.edges:
        return
    n = len(request.graph.vertices)
    m = len(request.graph.edges)
    partitions = _unordered_partition_count(n)
    # Two induced graphs, each trying up to n color counts. One encoding of
    # order n and k<=n uses n*k vertex literals plus m*k^2 edge separations.
    encoding = n * n + m * n * n
    work = partitions * 2 * n * (encoding + n + m + 1)
    if work > MAX_CHROMATIC_BIPARTITION_WORK:
        raise OperationResourceAdmissionError(
            location=("graph",),
            code="graph.chromatic_bipartition_exact_work_exceeds",
            message=(
                "chromatic bipartition search exceeds the admitted complete-search "
                "work bound"
            ),
        )


def _find_chromatic_bipartition_kernel(
    request: ChromaticBipartitionRequest,
) -> ChromaticBipartitionResult:
    """Search every canonical unordered vertex bipartition under one deadline."""
    graph = request.graph
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
        checked += 1
        if remaining_ms(started, request.resource_budget.wall_seconds) <= 0:
            return ChromaticBipartitionResult(
                graph=graph,
                s=request.s,
                t=request.t,
                status="UNKNOWN",
                checked_partitions=checked,
            )
        chromatic_a = _chromatic_number(_induced_graph(graph, side_a), request, started)
        if chromatic_a is None:
            return ChromaticBipartitionResult(
                graph=graph,
                s=request.s,
                t=request.t,
                status="UNKNOWN",
                checked_partitions=checked,
            )
        chromatic_b = _chromatic_number(_induced_graph(graph, side_b), request, started)
        if chromatic_b is None:
            return ChromaticBipartitionResult(
                graph=graph,
                s=request.s,
                t=request.t,
                status="UNKNOWN",
                checked_partitions=checked,
            )
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


def _unknown_result(request: ChromaticBipartitionRequest) -> ChromaticBipartitionResult:
    return ChromaticBipartitionResult(
        graph=request.graph,
        s=request.s,
        t=request.t,
        status="UNKNOWN",
        checked_partitions=0,
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
        "Return a witness, an exact NO_SPLIT after complete search, or UNKNOWN "
        "when the admitted shared exact-search budget is unresolved."
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
    "MAX_CHROMATIC_BIPARTITION_WORK",
    "ChromaticBipartitionRequest",
    "ChromaticBipartitionResult",
    "find_chromatic_bipartition",
]
