"""Typed wire contracts for chip-firing operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_VERTICES = 50
MAX_DEGREE = 100
MAX_COEFFICIENT_DIGITS = 1_000
MAX_STABILIZATION_CHIPS = 1_000_000


def _validate_divisor(
    vertices: tuple[str, ...],
    divisor: tuple[int, ...],
    *,
    label: str = "divisor",
) -> None:
    if len(divisor) != len(vertices):
        raise ValueError(f"{label} length must match vertex count")
    if any(abs(c) >= 10**MAX_COEFFICIENT_DIGITS for c in divisor):
        raise ValueError(f"{label} coefficients exceed the digit bound")


def _validate_sink(vertices: tuple[str, ...], sink: str) -> None:
    if sink not in set(vertices):
        raise ValueError("sink vertex must be in the graph")


class LabelledGraph(StrictModel):
    """A finite undirected simple graph with labelled vertices."""

    vertices: tuple[str, ...] = Field(min_length=1, max_length=MAX_VERTICES)
    edges: tuple[tuple[str, str], ...] = Field(default=())

    @model_validator(mode="after")
    def require_valid_graph(self) -> Self:
        labels = set(self.vertices)
        if len(labels) != len(self.vertices):
            raise ValueError("vertex labels must be distinct")
        seen_edges: set[tuple[str, str]] = set()
        for edge in self.edges:
            u, v = edge
            if u not in labels or v not in labels:
                raise ValueError("every edge endpoint must be a declared vertex")
            if u == v:
                raise ValueError("self-loops are not allowed")
            canonical = (min(u, v), max(u, v))
            if canonical in seen_edges:
                raise ValueError("duplicate edges are not allowed in a simple graph")
            seen_edges.add(canonical)
        degree: dict[str, int] = dict.fromkeys(self.vertices, 0)
        for u, v in self.edges:
            degree[u] += 1
            degree[v] += 1
            if degree[u] > MAX_DEGREE or degree[v] > MAX_DEGREE:
                raise ValueError(f"vertex degree exceeds maximum {MAX_DEGREE}")
        if len(self.edges) > MAX_VERTICES * MAX_DEGREE // 2:
            raise ValueError("too many edges")
        return self


class LaplacianRequest(StrictModel):
    graph: LabelledGraph


class LaplacianResult(StrictModel):
    """The graph Laplacian matrix with degree vector."""

    vertices: tuple[str, ...]
    laplacian: tuple[tuple[int, ...], ...]
    degrees: tuple[int, ...]


class ReducedLaplacianRequest(StrictModel):
    """Request the reduced Laplacian (sink row/column deleted)."""

    graph: LabelledGraph
    sink: str

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        _validate_sink(self.graph.vertices, self.sink)
        return self


class ReducedLaplacianResult(StrictModel):
    """The reduced Laplacian with nonsink vertex labels."""

    vertices: tuple[str, ...]
    sink: str
    reduced_laplacian: tuple[tuple[int, ...], ...]


class FiringRequest(StrictModel):
    """Fire a vertex: transfer one chip to each neighbor."""

    graph: LabelledGraph
    divisor: tuple[int, ...] = Field(min_length=1)
    firing_vertex: str

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        if len(self.divisor) != len(self.graph.vertices):
            raise ValueError("divisor length must match vertex count")
        if self.firing_vertex not in set(self.graph.vertices):
            raise ValueError("firing vertex must be in the graph")
        return self


class FiringResult(StrictModel):
    """Result of firing a vertex."""

    vertex: str
    fired_divisor: tuple[int, ...]


class FireVectorRequest(StrictModel):
    """Fire a vector: D' = D - L f."""

    graph: LabelledGraph
    divisor: tuple[int, ...] = Field(min_length=1)
    firing_vector: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        n = len(self.graph.vertices)
        if len(self.divisor) != n:
            raise ValueError("divisor length must match vertex count")
        if len(self.firing_vector) != n:
            raise ValueError("firing vector length must match vertex count")
        if any(abs(c) >= 10**MAX_COEFFICIENT_DIGITS for c in self.firing_vector):
            raise ValueError("firing vector coefficients exceed the digit bound")
        return self


class FireVectorResult(StrictModel):
    """Result of firing a vector."""

    fired_divisor: tuple[int, ...]
    degree_preserved: bool


class SinkConfiguration(StrictModel):
    """A sink configuration for chip-firing stabilization."""

    graph: LabelledGraph
    sink: str
    configuration: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        vertices = self.graph.vertices
        _validate_sink(vertices, self.sink)
        if len(self.configuration) != len(vertices):
            raise ValueError("configuration length must match vertex count")
        nonsink = [i for i, v in enumerate(vertices) if v != self.sink]
        if any(self.configuration[i] < 0 for i in nonsink):
            raise ValueError("nonsink configuration must be nonnegative")
        if sum(self.configuration[i] for i in nonsink) > MAX_STABILIZATION_CHIPS:
            raise ValueError(
                f"nonsink configuration exceeds stabilization bound "
                f"{MAX_STABILIZATION_CHIPS}"
            )
        return self


class StabilizeRequest(StrictModel):
    """Stabilize a sink configuration."""

    configuration: SinkConfiguration


class StabilizeResult(StrictModel):
    """The stable configuration and odometer vector."""

    stable: tuple[int, ...]
    odometer: tuple[int, ...]
    total_firings: int


class ParallelStepRequest(StrictModel):
    """One parallel firing step."""

    configuration: SinkConfiguration


class ParallelStepResult(StrictModel):
    """The next configuration and the set of vertices that fired."""

    next_configuration: tuple[int, ...]
    fired_vertices: tuple[str, ...]


class QReducedRequest(StrictModel):
    """Compute the q-reduced normal form of a divisor."""

    graph: LabelledGraph
    divisor: tuple[int, ...] = Field(min_length=1)
    sink: str

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        n = len(self.graph.vertices)
        _validate_sink(self.graph.vertices, self.sink)
        if len(self.divisor) != n:
            raise ValueError("divisor length must match vertex count")
        return self


class QReducedResult(StrictModel):
    """The q-reduced divisor and the exact firing vector."""

    reduced_divisor: tuple[int, ...]
    firing_vector: tuple[int, ...]


class DegreeRequest(StrictModel):
    """Compute the degree of a graph divisor."""

    divisor: tuple[int, ...] = Field(min_length=1)


class DegreeResult(StrictModel):
    """The degree of the divisor."""

    degree: int


class CanonicalDivisorRequest(StrictModel):
    """Compute the graph canonical divisor K(v) = deg(v) - 2."""

    graph: LabelledGraph


class CanonicalDivisorResult(StrictModel):
    """The canonical divisor and its degree."""

    vertices: tuple[str, ...]
    divisor: tuple[int, ...]
    degree: int


class CriticalGroupRequest(StrictModel):
    """Request the critical group (sandpile group) of a graph."""

    graph: LabelledGraph
    sink: str

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        _validate_sink(self.graph.vertices, self.sink)
        return self


class CriticalGroupResult(StrictModel):
    """The critical group invariant factors and order."""

    sink: str
    nonsink_vertices: tuple[str, ...]
    invariant_factors: tuple[int, ...]
    order: int


class AbelJacobiRequest(StrictModel):
    """Map a degree-zero divisor into the critical group."""

    graph: LabelledGraph
    divisor: tuple[int, ...] = Field(min_length=1)
    sink: str

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        n = len(self.graph.vertices)
        _validate_sink(self.graph.vertices, self.sink)
        if len(self.divisor) != n:
            raise ValueError("divisor length must match vertex count")
        if sum(self.divisor) != 0:
            raise ValueError("divisor must have degree zero")
        return self


class AbelJacobiResult(StrictModel):
    """The critical-group coordinates of a degree-zero divisor."""

    sink: str
    nonsink_vertices: tuple[str, ...]
    coordinates: tuple[int, ...]
    invariant_factors: tuple[int, ...]
