"""Typed wire contracts for directed graph operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, WithJsonSchema, field_validator, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

# These are the execution envelope for the four direct NetworkX operations
# below.  ``DirectedGraph`` itself remains reusable by consumers whose work is
# bounded more sharply from their own request data.
MAX_DIRECTED_OPERATION_VERTICES = 256
MAX_DIRECTED_OPERATION_EDGES = 512

# A cheap raw-item guard rejects pathological carrier lists before the
# duplicate-detecting structural scan. Operation admission remains much tighter.
MAX_DIRECTED_GRAPH_PARSE_EDGES = 163_840


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


class ReachabilityResult(StrictModel):
    graph: DirectedOperationGraph
    source: int = Field(ge=0, le=255)
    reachable: tuple[int, ...]
    unreachable: tuple[int, ...]

    @model_validator(mode="after")
    def require_vertex_partition(self) -> Self:
        universe = set(range(self.graph.vertex_count))
        if self.source not in universe:
            raise PydanticCustomError(
                "graph.reachability_source_must_belong_to_graph",
                "source must be a vertex of graph",
            )
        if self.reachable != tuple(
            sorted(set(self.reachable))
        ) or self.unreachable != tuple(sorted(set(self.unreachable))):
            raise PydanticCustomError(
                "graph.reachability_axes_must_be_sorted_unique",
                "reachable and unreachable axes must be sorted and unique",
            )
        if set(self.reachable) | set(self.unreachable) != universe or set(
            self.reachable
        ) & set(self.unreachable):
            raise PydanticCustomError(
                "graph.reachability_axes_must_partition_graph",
                "reachable and unreachable axes must partition graph vertices",
            )
        if self.source not in self.reachable:
            raise PydanticCustomError(
                "graph.reachability_source_must_be_reachable",
                "source must be included in reachable vertices",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        graph: DirectedOperationGraph,
        source: int,
        reachable: tuple[int, ...],
        unreachable: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            graph=graph,
            source=source,
            reachable=reachable,
            unreachable=unreachable,
        )


class StronglyConnectedComponentsRequest(StrictModel):
    graph: DirectedOperationGraph


class StronglyConnectedComponentsResult(StrictModel):
    graph: DirectedOperationGraph
    component_count: int = Field(ge=0, strict=True)
    components: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def require_vertex_partition(self) -> Self:
        universe = set(range(self.graph.vertex_count))
        normalized = tuple(
            tuple(sorted(set(component))) for component in self.components
        )
        if normalized != self.components or any(
            not component for component in self.components
        ):
            raise PydanticCustomError(
                "graph.scc_components_must_be_sorted_nonempty",
                "SCC components must be sorted and nonempty",
            )
        if (
            self.component_count != len(self.components)
            or set().union(*(set(component) for component in self.components))
            != universe
            or sum(map(len, self.components)) != len(universe)
        ):
            raise PydanticCustomError(
                "graph.scc_components_must_partition_graph",
                "SCC components must partition graph vertices",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        graph: DirectedOperationGraph,
        component_count: int,
        components: tuple[tuple[int, ...], ...],
    ) -> Self:
        return cls.model_construct(
            graph=graph,
            component_count=component_count,
            components=components,
        )


class CondensationRequest(StrictModel):
    graph: DirectedOperationGraph


class CondensationEdge(StrictModel):
    source: int = Field(ge=0)
    target: int = Field(ge=0)


class CondensationResult(StrictModel):
    graph: DirectedOperationGraph
    vertex_count: int = Field(ge=0, strict=True)
    components: tuple[tuple[int, ...], ...]
    edges: tuple[CondensationEdge, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_condensation(self) -> Self:
        universe = set(range(self.graph.vertex_count))
        normalized = tuple(
            tuple(sorted(set(component))) for component in self.components
        )
        if normalized != self.components or any(
            not component for component in self.components
        ):
            raise PydanticCustomError(
                "graph.condensation_components_must_be_sorted_nonempty",
                "condensation components must be sorted and nonempty",
            )
        if (
            self.vertex_count != len(self.components)
            or set().union(*(set(component) for component in self.components))
            != universe
            or sum(map(len, self.components)) != len(universe)
        ):
            raise PydanticCustomError(
                "graph.condensation_components_must_partition_graph",
                "condensation components must partition graph vertices",
            )
        edge_keys = tuple((edge.source, edge.target) for edge in self.edges)
        if edge_keys != tuple(sorted(set(edge_keys))) or any(
            source == target
            or not (0 <= source < self.vertex_count and 0 <= target < self.vertex_count)
            for source, target in edge_keys
        ):
            raise PydanticCustomError(
                "graph.condensation_edges_must_be_canonical",
                "condensation edges must be sorted, unique, and loop-free",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        graph: DirectedOperationGraph,
        vertex_count: int,
        components: tuple[tuple[int, ...], ...],
        edges: tuple[CondensationEdge, ...],
    ) -> Self:
        return cls.model_construct(
            graph=graph,
            vertex_count=vertex_count,
            components=components,
            edges=edges,
        )


class AcyclicOrderRequest(StrictModel):
    graph: DirectedOperationGraph


class AcyclicOrderResult(StrictModel):
    graph: DirectedOperationGraph
    acyclic: bool
    order: tuple[int, ...]

    @model_validator(mode="after")
    def require_order_matches_acyclicity(self) -> Self:
        universe = set(range(self.graph.vertex_count))
        if (
            self.order != tuple(dict.fromkeys(self.order))
            or set(self.order) != universe
        ) and (self.acyclic or self.order):
            raise PydanticCustomError(
                "graph.acyclic_order_must_be_a_vertex_permutation",
                "a topological order must contain each graph vertex exactly once",
            )
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

    @classmethod
    def _from_kernel(
        cls, graph: DirectedOperationGraph, acyclic: bool, order: tuple[int, ...]
    ) -> Self:
        return cls.model_construct(graph=graph, acyclic=acyclic, order=order)


class DagLongestPathRequest(StrictModel):
    graph: DirectedOperationGraph


class DagLongestPathResult(StrictModel):
    status: Literal["ACYCLIC", "NOT_APPLICABLE"]
    maximum_edge_count: int = Field(default=0, ge=0, strict=True)
    path: tuple[int, ...] = Field(default=())
    source: DirectedOperationGraph
    convention: Literal["JACOBIAN_DAG_LONGEST_PATH"] = "JACOBIAN_DAG_LONGEST_PATH"

    @model_validator(mode="after")
    def require_consistent_fields(self) -> Self:
        if self.status == "NOT_APPLICABLE":
            if self.maximum_edge_count != 0:
                raise PydanticCustomError(
                    "graph.dag_longest_path_not_applicable_has_no_edge_count",
                    "NOT_APPLICABLE status must report zero edge count",
                )
            if self.path:
                raise PydanticCustomError(
                    "graph.dag_longest_path_not_applicable_has_no_path",
                    "NOT_APPLICABLE status must not report a path",
                )
        else:
            if not self.path:
                raise PydanticCustomError(
                    "graph.dag_longest_path_acyclic_must_report_a_path",
                    "ACYCLIC status must report a path witness",
                )
            if self.maximum_edge_count != len(self.path) - 1:
                raise PydanticCustomError(
                    "graph.dag_longest_path_edge_count_must_match_path",
                    "maximum_edge_count must equal len(path) - 1",
                )
            vertices = set(range(self.source.vertex_count))
            if (
                self.path != tuple(dict.fromkeys(self.path))
                or not set(self.path) <= vertices
            ):
                raise PydanticCustomError(
                    "graph.dag_longest_path_must_use_distinct_graph_vertices",
                    "DAG path must use distinct graph vertices",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        status: Literal["ACYCLIC", "NOT_APPLICABLE"],
        maximum_edge_count: int,
        path: tuple[int, ...],
        source: DirectedOperationGraph,
        convention: Literal["JACOBIAN_DAG_LONGEST_PATH"] = "JACOBIAN_DAG_LONGEST_PATH",
    ) -> Self:
        return cls.model_construct(
            status=status,
            maximum_edge_count=maximum_edge_count,
            path=path,
            source=source,
            convention=convention,
        )
