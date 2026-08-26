"""Typed wire contracts for directed graph operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, WithJsonSchema, field_validator, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits

# These are the execution envelope for the four direct NetworkX operations
# below.  ``DirectedGraph`` itself remains reusable by consumers whose work is
# bounded more sharply from their own request data.
MAX_DIRECTED_OPERATION_VERTICES = 256
MAX_DIRECTED_OPERATION_EDGES = 512

# Parse-safety envelope for the shared carrier's edge list: a transport guard,
# not admission.  The minimal canonical JSON encoding of one arc inside
# ``edges`` is ``[0,0],`` -- five content bytes plus one separator byte --
# so a maximal 10 MiB request document could still minimally encode roughly
# 1.75 million arcs.  This envelope deliberately sits far below that: it is
# one sixty-fourth of ``CanonicalLimits().max_input_bytes`` (163,840 arcs),
# so even a worst-case admitted list consumes under a tenth of the transport
# input envelope while remaining 320x the loosest current admission (the
# direct operations' 512 arcs above; directed bond reliability admits 12).  No schema-admitted request can reach it, and
# payloads beyond it reject here in O(1) instead of paying the full
# duplicate-detecting scan first.
MAX_DIRECTED_GRAPH_PARSE_EDGES = CanonicalLimits().max_input_bytes // 64


class DirectedGraph(StrictModel):
    """A structurally valid finite simple directed graph."""

    vertex_count: int = Field(ge=2)
    edges: tuple[tuple[int, int], ...] = Field()

    @field_validator("edges", mode="before")
    @classmethod
    def require_edge_parse_envelope(cls, edges: object) -> object:
        """Bound the raw edge sequence before any nested row is materialized.

        Pydantic coerces and validates each nested tuple only after this
        before-validator returns, so a payload beyond the parse-safety
        envelope rejects on its raw sequence length without building the
        edge tuples or entering structural validation at all. Admitted
        lists are canonicalized to tuples here because strict JSON parsing
        treats values returned by a before-validator as runtime data, which
        no longer coerces lists to tuples (the container-canonicalization
        rule shared with strict-JSON preflight models).
        """

        if isinstance(edges, list):
            if len(edges) > MAX_DIRECTED_GRAPH_PARSE_EDGES:
                raise PydanticCustomError(
                    "graph.edge_list_parse_envelope_exceeded",
                    "directed graph edge list exceeds the "
                    f"{MAX_DIRECTED_GRAPH_PARSE_EDGES}-edge parse-safety envelope",
                )
            return tuple(tuple(row) if isinstance(row, list) else row for row in edges)
        if isinstance(edges, tuple) and len(edges) > MAX_DIRECTED_GRAPH_PARSE_EDGES:
            raise PydanticCustomError(
                "graph.edge_list_parse_envelope_exceeded",
                "directed graph edge list exceeds the "
                f"{MAX_DIRECTED_GRAPH_PARSE_EDGES}-edge parse-safety envelope",
            )
        return edges

    @model_validator(mode="after")
    def require_valid_edges(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for source, target in self.edges:
            if not (
                0 <= source < self.vertex_count and 0 <= target < self.vertex_count
            ):
                raise PydanticCustomError(
                    "graph.edge_vertices_must_be_in_0_vertex_count_1",
                    "edge vertices must be in 0..vertex_count-1",
                )
            if source == target:
                raise PydanticCustomError(
                    "graph.self_loops_are_not_allowed", "self-loops are not allowed"
                )
            endpoint_pair = (source, target)
            if endpoint_pair in seen:
                raise PydanticCustomError(
                    "graph.directed_edges_must_be_unique",
                    "directed edges must be unique",
                )
            seen.add(endpoint_pair)
        return self


def _require_directed_operation_admission(graph: DirectedGraph) -> None:
    """Bound one direct traversal operation before constructing NetworkX state."""

    if graph.vertex_count > MAX_DIRECTED_OPERATION_VERTICES:
        raise PydanticCustomError(
            "graph.directed_vertex_budget_exceeded",
            "directed graph operation supports at most "
            f"{MAX_DIRECTED_OPERATION_VERTICES} vertices",
        )
    if len(graph.edges) > MAX_DIRECTED_OPERATION_EDGES:
        raise PydanticCustomError(
            "graph.directed_edge_budget_exceeded",
            "directed graph operation supports at most "
            f"{MAX_DIRECTED_OPERATION_EDGES} edges",
        )


def _directed_operation_graph_schema() -> JsonSchemaValue:
    """Project the direct-operation envelope onto the shared carrier schema."""

    schema = DirectedGraph.model_json_schema()
    schema["description"] = (
        "A structurally valid finite simple directed graph accepted by the "
        "direct traversal operations: at most "
        f"{MAX_DIRECTED_OPERATION_VERTICES} vertices and at most "
        f"{MAX_DIRECTED_OPERATION_EDGES} edges."
    )
    schema["properties"]["vertex_count"].update(
        maximum=MAX_DIRECTED_OPERATION_VERTICES,
    )
    schema["properties"]["edges"].update(maxItems=MAX_DIRECTED_OPERATION_EDGES)
    return schema


DirectedOperationGraph = Annotated[
    DirectedGraph,
    WithJsonSchema(_directed_operation_graph_schema()),
]


class ReachabilityRequest(StrictModel):
    graph: DirectedOperationGraph
    source: int = Field(ge=0, le=MAX_DIRECTED_OPERATION_VERTICES - 1)

    @model_validator(mode="after")
    def require_valid_source(self) -> Self:
        _require_directed_operation_admission(self.graph)
        if not (0 <= self.source < self.graph.vertex_count):
            raise PydanticCustomError(
                "graph.source_must_be_in_0_graph_vertex_count_1",
                "source must be in 0..graph.vertex_count-1",
            )
        return self


class ReachabilityResult(StrictModel):
    source: int = Field(ge=0, le=255)
    reachable: tuple[int, ...]
    unreachable: tuple[int, ...]
    convention: Literal["NETWORKX_DESCENDANTS"] = "NETWORKX_DESCENDANTS"


class StronglyConnectedComponentsRequest(StrictModel):
    graph: DirectedOperationGraph

    @model_validator(mode="after")
    def require_operation_admission(self) -> Self:
        _require_directed_operation_admission(self.graph)
        return self


class StronglyConnectedComponentsResult(StrictModel):
    component_count: int = Field(ge=0, strict=True)
    components: tuple[tuple[int, ...], ...]
    convention: Literal["NETWORKX_STRONGLY_CONNECTED_COMPONENTS"] = (
        "NETWORKX_STRONGLY_CONNECTED_COMPONENTS"
    )


class CondensationRequest(StrictModel):
    graph: DirectedOperationGraph

    @model_validator(mode="after")
    def require_operation_admission(self) -> Self:
        _require_directed_operation_admission(self.graph)
        return self


class CondensationEdge(StrictModel):
    source: int = Field(ge=0)
    target: int = Field(ge=0)


class CondensationResult(StrictModel):
    vertex_count: int = Field(ge=0, strict=True)
    components: tuple[tuple[int, ...], ...]
    edges: tuple[CondensationEdge, ...] = Field(default=())
    convention: Literal["NETWORKX_CONDENSATION"] = "NETWORKX_CONDENSATION"


class AcyclicOrderRequest(StrictModel):
    graph: DirectedOperationGraph

    @model_validator(mode="after")
    def require_operation_admission(self) -> Self:
        _require_directed_operation_admission(self.graph)
        return self


class AcyclicOrderResult(StrictModel):
    acyclic: bool
    order: tuple[int, ...]
    convention: Literal["NETWORKX_TOPOLOGICAL_SORT"] = "NETWORKX_TOPOLOGICAL_SORT"

    @model_validator(mode="after")
    def require_order_matches_acyclicity(self) -> Self:
        if self.acyclic:
            if not self.order:
                raise PydanticCustomError(
                    "graph.acyclic_order_must_list_every_vertex",
                    "acyclic order must list every vertex",
                )
        elif self.order:
            raise PydanticCustomError(
                "graph.cyclic_graph_must_not_report_a_topological_order",
                "cyclic graph must not report a topological order",
            )
        return self
