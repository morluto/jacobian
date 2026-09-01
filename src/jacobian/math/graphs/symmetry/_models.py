"""Typed contracts for exact declared graph-symmetry actions."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.symmetry._edges import canonical_edge
from jacobian.math.graphs.values import ColoredUndirectedGraph

MAX_GRAPH_SYMMETRY_VERTICES = 256
MAX_GRAPH_SYMMETRY_EDGES = 4_096
MAX_GRAPH_SYMMETRY_GENERATORS = 64
_UNCOLORED = "__UNCOLORED__"

GraphSymmetryLabel = Annotated[str, Field(min_length=1, max_length=64)]
GraphSymmetryEdge = tuple[GraphSymmetryLabel, GraphSymmetryLabel]


class GraphAutomorphismGenerator(StrictModel):
    generator_id: GraphSymmetryLabel = Field(
        description=(
            "Unique identifier of this declared generator; must already be "
            "normalized to Unicode NFC."
        )
    )
    mapping: tuple[tuple[GraphSymmetryLabel, GraphSymmetryLabel], ...] = Field(
        max_length=MAX_GRAPH_SYMMETRY_VERTICES,
        description=(
            "Total vertex permutation as (vertex, image) pairs covering every "
            "declared vertex exactly once in the graph's declared vertex "
            "order; labels must already be normalized to Unicode NFC."
        ),
    )


def _validate_automorphism_generator(
    generator: GraphAutomorphismGenerator,
    vertices: tuple[GraphSymmetryLabel, ...],
    edges: tuple[GraphSymmetryEdge, ...],
    vertex_set: set[GraphSymmetryLabel],
    edge_set: set[GraphSymmetryEdge],
    vertex_colors: dict[GraphSymmetryLabel, str],
    edge_colors: dict[GraphSymmetryEdge, str],
) -> None:
    mapping = dict(generator.mapping)
    if (
        tuple(vertex for vertex, _ in generator.mapping) != vertices
        or set(mapping.values()) != vertex_set
    ):
        raise PydanticCustomError(
            "graph.every_symmetry_generator_total_vertex_permutation_declared",
            "every graph symmetry generator must be a total vertex permutation "
            "declared as one (vertex, image) pair per declared vertex in the "
            "graph's declared vertex order",
        )
    if any(
        vertex_colors[vertex] != vertex_colors[mapping[vertex]] for vertex in vertices
    ):
        raise PydanticCustomError(
            "graph.symmetry_generators_preserve_declared_vertex_colors",
            "graph symmetry generators must preserve declared vertex colors",
        )
    mapped_edges = {
        canonical_edge(mapping[left], mapping[right]) for left, right in edges
    }
    if mapped_edges != edge_set:
        raise PydanticCustomError(
            "graph.symmetry_generators_preserve_complete_edge_set",
            "graph symmetry generators must preserve the complete edge set",
        )
    if any(
        edge_colors[edge]
        != edge_colors[canonical_edge(mapping[edge[0]], mapping[edge[1]])]
        for edge in edges
    ):
        raise PydanticCustomError(
            "graph.graph_symmetry_generators_must_preserve_declared",
            "graph symmetry generators must preserve declared edge colors",
        )


class GraphSymmetryOrbitSource(StrictModel):
    """Canonical source action for a declared graph-symmetry computation."""

    graph: ColoredUndirectedGraph
    generators: tuple[GraphAutomorphismGenerator, ...] = Field(
        max_length=MAX_GRAPH_SYMMETRY_GENERATORS
    )
    action: Literal["DECLARED_AUTOMORPHISM_GENERATORS"] = (
        "DECLARED_AUTOMORPHISM_GENERATORS"
    )


class GraphSymmetryOrbitRequest(GraphSymmetryOrbitSource):
    """Declared color-preserving generators of one bounded graph's subgroup."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Color-preserving automorphism generators declared for one "
                "bounded colored graph. Each generator is a total vertex "
                "permutation as (vertex, image) pairs covering every "
                "declared vertex exactly once in the graph's declared vertex "
                "order; generator identifiers and color names must already "
                "be normalized to Unicode NFC. The computed result retains "
                "this complete request as its source plus the derived "
                "vertex and edge orbits."
            )
        }
    )


class GraphVertexOrbit(StrictModel):
    orbit_index: StrictInt = Field(ge=0, le=MAX_GRAPH_SYMMETRY_VERTICES - 1)
    representative: GraphSymmetryLabel
    members: tuple[GraphSymmetryLabel, ...] = Field(
        min_length=1,
        max_length=MAX_GRAPH_SYMMETRY_VERTICES,
    )

    @model_validator(mode="after")
    def require_canonical_orbit(self) -> Self:
        if (
            tuple(sorted(self.members)) != self.members
            or len(set(self.members)) != len(self.members)
            or self.representative != self.members[0]
        ):
            raise PydanticCustomError(
                "graph.vertex_orbit_members_must_be_unique_and_canonica",
                "vertex orbit members must be unique and canonical",
            )
        return self


class GraphEdgeOrbit(StrictModel):
    orbit_index: StrictInt = Field(ge=0, le=MAX_GRAPH_SYMMETRY_EDGES - 1)
    representative: GraphSymmetryEdge
    members: tuple[GraphSymmetryEdge, ...] = Field(
        min_length=1,
        max_length=MAX_GRAPH_SYMMETRY_EDGES,
    )

    @model_validator(mode="after")
    def require_canonical_orbit(self) -> Self:
        if (
            any(left >= right for left, right in self.members)
            or tuple(sorted(self.members)) != self.members
            or len(set(self.members)) != len(self.members)
            or self.representative != self.members[0]
        ):
            raise PydanticCustomError(
                "graph.edge_orbit_members_must_be_unique_and_canonical",
                "edge orbit members must be unique and canonical",
            )
        return self


class GraphSymmetryOrbitResult(StrictModel):
    """Complete vertex and edge orbits of one declared generated subgroup.

    The result retains its complete declared source action - the canonical
    graph, the generator mappings, and the declared colors - through the
    domain-owned request value. Validation checks its canonical structure and
    source binding. The trusted kernel constructs exact partitions.
    """

    source: GraphSymmetryOrbitSource
    vertices: tuple[GraphSymmetryLabel, ...] = Field(
        max_length=MAX_GRAPH_SYMMETRY_VERTICES
    )
    edges: tuple[GraphSymmetryEdge, ...] = Field(max_length=MAX_GRAPH_SYMMETRY_EDGES)
    generator_ids: tuple[GraphSymmetryLabel, ...] = Field(
        max_length=MAX_GRAPH_SYMMETRY_GENERATORS
    )
    generator_count: StrictInt = Field(ge=0, le=MAX_GRAPH_SYMMETRY_GENERATORS)
    vertex_orbits: tuple[GraphVertexOrbit, ...] = Field(
        max_length=MAX_GRAPH_SYMMETRY_VERTICES
    )
    edge_orbits: tuple[GraphEdgeOrbit, ...] = Field(max_length=MAX_GRAPH_SYMMETRY_EDGES)
    vertex_orbit_count: StrictInt = Field(ge=0, le=MAX_GRAPH_SYMMETRY_VERTICES)
    edge_orbit_count: StrictInt = Field(ge=0, le=MAX_GRAPH_SYMMETRY_EDGES)
    vertex_color_mode: Literal["UNCOLORED", "DECLARED"]
    edge_color_mode: Literal["UNCOLORED", "DECLARED"]
    action: Literal["DECLARED_GENERATED_SUBGROUP"] = "DECLARED_GENERATED_SUBGROUP"
    generator_validation: Literal[
        "ALL_DECLARED_GENERATORS_PRESERVE_GRAPH_AND_COLORS"
    ] = "ALL_DECLARED_GENERATORS_PRESERVE_GRAPH_AND_COLORS"
    orbit_completeness: Literal["COMPLETE_FOR_DECLARED_GENERATORS"] = (
        "COMPLETE_FOR_DECLARED_GENERATORS"
    )
    automorphism_group_completeness: Literal["FULL_AUTOMORPHISM_GROUP_NOT_CLAIMED"] = (
        "FULL_AUTOMORPHISM_GROUP_NOT_CLAIMED"
    )

    @model_validator(mode="after")
    def require_canonical_source_bound_partitions(self) -> Self:
        if tuple(sorted(self.vertices)) != self.vertices or len(
            set(self.vertices)
        ) != len(self.vertices):
            raise PydanticCustomError(
                "graph.result_vertices_must_be_unique_and_canonical",
                "result vertices must be unique and canonical",
            )
        if (
            any(left >= right for left, right in self.edges)
            or tuple(sorted(self.edges)) != self.edges
            or len(set(self.edges)) != len(self.edges)
        ):
            raise PydanticCustomError(
                "graph.result_edges_must_be_unique_and_canonical",
                "result edges must be unique and canonical",
            )
        if (
            tuple(sorted(self.generator_ids)) != self.generator_ids
            or len(set(self.generator_ids)) != len(self.generator_ids)
            or self.generator_count != len(self.generator_ids)
        ):
            raise PydanticCustomError(
                "graph.result_generator_identifiers_unique_canonical",
                "result generator identifiers must be unique and canonical",
            )
        declared_generator_ids = tuple(
            sorted(generator.generator_id for generator in self.source.generators)
        )
        if (
            self.vertices != tuple(sorted(self.source.graph.graph.vertices))
            or self.edges != tuple(sorted(self.source.graph.graph.edges))
            or self.generator_ids != declared_generator_ids
        ):
            raise PydanticCustomError(
                "graph.result_generators_equal_retained_source_action",
                "result graph and generators must equal the retained source action",
            )
        if self.vertex_color_mode != (
            "DECLARED" if self.source.graph.vertex_colors else "UNCOLORED"
        ):
            raise PydanticCustomError(
                "graph.vertex_color_mode_match_retained_source_vertex",
                "vertex color mode must match the retained source vertex colors",
            )
        if self.edge_color_mode != (
            "DECLARED" if self.source.graph.edge_colors else "UNCOLORED"
        ):
            raise PydanticCustomError(
                "graph.edge_color_mode_match_retained_source_edge",
                "edge color mode must match the retained source edge colors",
            )
        vertex_members = tuple(
            member for orbit in self.vertex_orbits for member in orbit.members
        )
        if (
            self.vertex_orbit_count != len(self.vertex_orbits)
            or tuple(orbit.orbit_index for orbit in self.vertex_orbits)
            != tuple(range(len(self.vertex_orbits)))
            or tuple(orbit.representative for orbit in self.vertex_orbits)
            != tuple(sorted(orbit.representative for orbit in self.vertex_orbits))
            or len(vertex_members) != len(self.vertices)
            or set(vertex_members) != set(self.vertices)
        ):
            raise PydanticCustomError(
                "graph.vertex_orbits_complete_canonical_vertex_partition",
                "vertex orbits must be a complete canonical vertex partition",
            )
        edge_members = tuple(
            member for orbit in self.edge_orbits for member in orbit.members
        )
        if (
            self.edge_orbit_count != len(self.edge_orbits)
            or tuple(orbit.orbit_index for orbit in self.edge_orbits)
            != tuple(range(len(self.edge_orbits)))
            or tuple(orbit.representative for orbit in self.edge_orbits)
            != tuple(sorted(orbit.representative for orbit in self.edge_orbits))
            or len(edge_members) != len(self.edges)
            or set(edge_members) != set(self.edges)
        ):
            raise PydanticCustomError(
                "graph.edge_orbits_must_be_a_complete_canonical_edge_pa",
                "edge orbits must be a complete canonical edge partition",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: ColoredUndirectedGraph,
        generators: tuple[GraphAutomorphismGenerator, ...],
        vertices: tuple[GraphSymmetryLabel, ...],
        edges: tuple[GraphSymmetryEdge, ...],
        generator_ids: tuple[GraphSymmetryLabel, ...],
        vertex_orbits: tuple[GraphVertexOrbit, ...],
        edge_orbits: tuple[GraphEdgeOrbit, ...],
        vertex_color_mode: Literal["UNCOLORED", "DECLARED"],
        edge_color_mode: Literal["UNCOLORED", "DECLARED"],
    ) -> Self:
        """Construct an exact source-bound result from the trusted kernel."""

        return cls.model_construct(
            source=GraphSymmetryOrbitSource(graph=graph, generators=generators),
            vertices=vertices,
            edges=edges,
            generator_ids=generator_ids,
            generator_count=len(generator_ids),
            vertex_orbits=vertex_orbits,
            edge_orbits=edge_orbits,
            vertex_orbit_count=len(vertex_orbits),
            edge_orbit_count=len(edge_orbits),
            vertex_color_mode=vertex_color_mode,
            edge_color_mode=edge_color_mode,
        )


__all__ = [
    "MAX_GRAPH_SYMMETRY_EDGES",
    "MAX_GRAPH_SYMMETRY_GENERATORS",
    "MAX_GRAPH_SYMMETRY_VERTICES",
    "GraphAutomorphismGenerator",
    "GraphEdgeOrbit",
    "GraphSymmetryOrbitRequest",
    "GraphSymmetryOrbitResult",
    "GraphSymmetryOrbitSource",
    "GraphVertexOrbit",
]
