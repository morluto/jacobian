"""Typed wire contracts for chip-firing operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import AfterValidator, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_VERTICES = 50
MAX_COEFFICIENT_DIGITS = 1_000
MAX_STABILIZATION_CHIPS = 1_000_000


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


def _validate_divisor(
    vertices: tuple[str, ...],
    divisor: tuple[int, ...],
    *,
    label: str = "divisor",
) -> None:
    if len(divisor) != len(vertices):
        raise _validation_error(
            "chip_firing.divisor_length", f"{label} length must match vertex count"
        )
    if any(abs(c) >= 10**MAX_COEFFICIENT_DIGITS for c in divisor):
        raise _validation_error(
            "chip_firing.coefficient_bound",
            f"{label} coefficients exceed the digit bound",
        )


def _validate_sink(vertices: tuple[str, ...], sink: str) -> None:
    if sink not in set(vertices):
        raise _validation_error(
            "chip_firing.sink_not_in_graph", "sink vertex must be in the graph"
        )


def _require_chip_firing_graph(
    graph: SimpleUndirectedGraph,
) -> SimpleUndirectedGraph:
    if not graph.vertices:
        raise _validation_error(
            "chip_firing.empty_graph", "chip-firing requires a nonempty graph"
        )
    if len(graph.vertices) > MAX_VERTICES:
        raise _validation_error(
            "chip_firing.vertex_bound",
            f"chip-firing supports at most {MAX_VERTICES} vertices",
        )
    return graph


_ChipFiringGraph = Annotated[
    SimpleUndirectedGraph,
    AfterValidator(_require_chip_firing_graph),
]


class LaplacianRequest(StrictModel):
    graph: _ChipFiringGraph


class LaplacianResult(StrictModel):
    """The graph Laplacian matrix with degree vector."""

    vertices: tuple[str, ...]
    laplacian: tuple[tuple[int, ...], ...]
    degrees: tuple[int, ...]


class ReducedLaplacianRequest(StrictModel):
    """Request the reduced Laplacian (sink row/column deleted)."""

    graph: _ChipFiringGraph
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

    graph: _ChipFiringGraph
    divisor: tuple[int, ...] = Field(min_length=1)
    firing_vertex: str

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        if len(self.divisor) != len(self.graph.vertices):
            raise _validation_error(
                "chip_firing.divisor_length", "divisor length must match vertex count"
            )
        if self.firing_vertex not in set(self.graph.vertices):
            raise _validation_error(
                "chip_firing.firing_vertex_not_in_graph",
                "firing vertex must be in the graph",
            )
        return self


class FiringResult(StrictModel):
    """Result of firing a vertex."""

    vertex: str
    fired_divisor: tuple[int, ...]


class FireVectorRequest(StrictModel):
    """Fire a vector: D' = D - L f."""

    graph: _ChipFiringGraph
    divisor: tuple[int, ...] = Field(min_length=1)
    firing_vector: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        n = len(self.graph.vertices)
        if len(self.divisor) != n:
            raise _validation_error(
                "chip_firing.divisor_length", "divisor length must match vertex count"
            )
        if len(self.firing_vector) != n:
            raise _validation_error(
                "chip_firing.firing_vector_length",
                "firing vector length must match vertex count",
            )
        if any(abs(c) >= 10**MAX_COEFFICIENT_DIGITS for c in self.firing_vector):
            raise _validation_error(
                "chip_firing.coefficient_bound",
                "firing vector coefficients exceed the digit bound",
            )
        return self


class FireVectorResult(StrictModel):
    """Result of firing a vector."""

    fired_divisor: tuple[int, ...]
    degree_preserved: bool


class SinkConfiguration(StrictModel):
    """A sink configuration for chip-firing stabilization."""

    graph: _ChipFiringGraph
    sink: str
    configuration: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        vertices = self.graph.vertices
        _validate_sink(vertices, self.sink)
        if len(self.configuration) != len(vertices):
            raise _validation_error(
                "chip_firing.configuration_length",
                "configuration length must match vertex count",
            )
        nonsink = [i for i, v in enumerate(vertices) if v != self.sink]
        if any(self.configuration[i] < 0 for i in nonsink):
            raise _validation_error(
                "chip_firing.nonsink_negative",
                "nonsink configuration must be nonnegative",
            )
        if sum(self.configuration[i] for i in nonsink) > MAX_STABILIZATION_CHIPS:
            raise _validation_error(
                "chip_firing.stabilization_bound",
                f"nonsink configuration exceeds stabilization bound "
                f"{MAX_STABILIZATION_CHIPS}",
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

    graph: _ChipFiringGraph
    divisor: tuple[int, ...] = Field(min_length=1)
    sink: str

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        n = len(self.graph.vertices)
        _validate_sink(self.graph.vertices, self.sink)
        if len(self.divisor) != n:
            raise _validation_error(
                "chip_firing.divisor_length", "divisor length must match vertex count"
            )
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

    graph: _ChipFiringGraph


class CanonicalDivisorResult(StrictModel):
    """The canonical divisor and its degree."""

    vertices: tuple[str, ...]
    divisor: tuple[int, ...]
    degree: int


class CriticalGroupRequest(StrictModel):
    """Request the critical group (sandpile group) of a graph."""

    graph: _ChipFiringGraph
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

    graph: _ChipFiringGraph
    divisor: tuple[int, ...] = Field(min_length=1)
    sink: str

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        n = len(self.graph.vertices)
        _validate_sink(self.graph.vertices, self.sink)
        if len(self.divisor) != n:
            raise _validation_error(
                "chip_firing.divisor_length", "divisor length must match vertex count"
            )
        if sum(self.divisor) != 0:
            raise _validation_error(
                "chip_firing.divisor_not_degree_zero", "divisor must have degree zero"
            )
        return self


class AbelJacobiResult(StrictModel):
    """The critical-group coordinates of a degree-zero divisor."""

    sink: str
    nonsink_vertices: tuple[str, ...]
    coordinates: tuple[int, ...]
    invariant_factors: tuple[int, ...]
