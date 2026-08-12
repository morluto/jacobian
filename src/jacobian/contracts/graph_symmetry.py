"""Typed contracts for exact declared graph-symmetry actions."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.graph_isomorphism import SimpleUndirectedGraph
from jacobian.contracts.results import ContractModel

MAX_GRAPH_SYMMETRY_VERTICES = 256
MAX_GRAPH_SYMMETRY_EDGES = 4_096
MAX_GRAPH_SYMMETRY_GENERATORS = 64

GraphSymmetryLabel = Annotated[str, Field(min_length=1, max_length=64)]
GraphSymmetryColor = Annotated[str, Field(min_length=1, max_length=128)]
GraphSymmetryEdge = tuple[GraphSymmetryLabel, GraphSymmetryLabel]


def _canonical_edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


class GraphAutomorphismGenerator(ContractModel):
    generator_id: GraphSymmetryLabel
    mapping: dict[GraphSymmetryLabel, GraphSymmetryLabel] = Field(
        max_length=MAX_GRAPH_SYMMETRY_VERTICES
    )


class GraphVertexColor(ContractModel):
    vertex: GraphSymmetryLabel
    color: GraphSymmetryColor


class GraphEdgeColor(ContractModel):
    edge: GraphSymmetryEdge
    color: GraphSymmetryColor

    @model_validator(mode="after")
    def require_canonical_edge(self) -> Self:
        if self.edge[0] >= self.edge[1]:
            raise ValueError("colored graph edges must use canonical endpoint order")
        return self


def _validate_graph_symmetry_bounds_and_colors(
    vertices: tuple[GraphSymmetryLabel, ...],
    edges: tuple[GraphSymmetryEdge, ...],
    generators: tuple[GraphAutomorphismGenerator, ...],
    vertex_colors: tuple[GraphVertexColor, ...],
    edge_colors: tuple[GraphEdgeColor, ...],
) -> None:
    if len(vertices) > MAX_GRAPH_SYMMETRY_VERTICES:
        raise ValueError(
            f"graph symmetry exceeds the {MAX_GRAPH_SYMMETRY_VERTICES}-vertex bound"
        )
    if len(edges) > MAX_GRAPH_SYMMETRY_EDGES:
        raise ValueError(
            f"graph symmetry exceeds the {MAX_GRAPH_SYMMETRY_EDGES}-edge bound"
        )
    if any(not vertex or len(vertex) > 64 for vertex in vertices):
        raise ValueError("graph symmetry vertex labels must contain 1-64 characters")
    generator_ids = tuple(generator.generator_id for generator in generators)
    if len(set(generator_ids)) != len(generator_ids):
        raise ValueError("graph symmetry generator identifiers must be unique")

    if vertex_colors and tuple(item.vertex for item in vertex_colors) != vertices:
        raise ValueError(
            "vertex colors must cover vertices in the graph's declared order"
        )
    if edge_colors and tuple(item.edge for item in edge_colors) != edges:
        raise ValueError("edge colors must cover edges in the graph's declared order")


def _validate_automorphism_generator(
    generator: GraphAutomorphismGenerator,
    vertices: tuple[GraphSymmetryLabel, ...],
    edges: tuple[GraphSymmetryEdge, ...],
    vertex_set: set[GraphSymmetryLabel],
    edge_set: set[GraphSymmetryEdge],
    vertex_colors: dict[GraphSymmetryLabel, GraphSymmetryColor],
    edge_colors: dict[GraphSymmetryEdge, GraphSymmetryColor],
) -> None:
    mapping = generator.mapping
    if set(mapping) != vertex_set or set(mapping.values()) != vertex_set:
        raise ValueError(
            "every graph symmetry generator must be a total vertex permutation"
        )
    if any(
        vertex_colors[vertex] != vertex_colors[mapping[vertex]] for vertex in vertices
    ):
        raise ValueError(
            "graph symmetry generators must preserve declared vertex colors"
        )
    mapped_edges = {
        _canonical_edge(mapping[left], mapping[right]) for left, right in edges
    }
    if mapped_edges != edge_set:
        raise ValueError(
            "graph symmetry generators must preserve the complete edge set"
        )
    if any(
        edge_colors[edge]
        != edge_colors[_canonical_edge(mapping[edge[0]], mapping[edge[1]])]
        for edge in edges
    ):
        raise ValueError("graph symmetry generators must preserve declared edge colors")


class GraphSymmetryOrbitRequest(ContractModel):
    graph: SimpleUndirectedGraph
    generators: tuple[GraphAutomorphismGenerator, ...] = Field(
        max_length=MAX_GRAPH_SYMMETRY_GENERATORS
    )
    vertex_colors: tuple[GraphVertexColor, ...] = Field(
        default=(),
        max_length=MAX_GRAPH_SYMMETRY_VERTICES,
    )
    edge_colors: tuple[GraphEdgeColor, ...] = Field(
        default=(),
        max_length=MAX_GRAPH_SYMMETRY_EDGES,
    )
    action: Literal["DECLARED_AUTOMORPHISM_GENERATORS"] = (
        "DECLARED_AUTOMORPHISM_GENERATORS"
    )

    @model_validator(mode="after")
    def require_bounded_color_preserving_automorphisms(self) -> Self:
        vertices = self.graph.vertices
        edges = self.graph.edges
        _validate_graph_symmetry_bounds_and_colors(
            vertices,
            edges,
            self.generators,
            self.vertex_colors,
            self.edge_colors,
        )

        vertex_set = set(vertices)
        edge_set = set(edges)
        vertex_colors = (
            {item.vertex: item.color for item in self.vertex_colors}
            if self.vertex_colors
            else dict.fromkeys(vertices, "__UNCOLORED__")
        )
        edge_colors = (
            {item.edge: item.color for item in self.edge_colors}
            if self.edge_colors
            else dict.fromkeys(edges, "__UNCOLORED__")
        )
        for generator in self.generators:
            _validate_automorphism_generator(
                generator,
                vertices,
                edges,
                vertex_set,
                edge_set,
                vertex_colors,
                edge_colors,
            )
        return self


class GraphVertexOrbit(ContractModel):
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
            raise ValueError("vertex orbit members must be unique and canonical")
        return self


class GraphEdgeOrbit(ContractModel):
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
            raise ValueError("edge orbit members must be unique and canonical")
        return self


class GraphSymmetryOrbitResult(ContractModel):
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
    exactness: Literal["EXACT_COMBINATORIAL"] = "EXACT_COMBINATORIAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    backend: Literal["networkx"] = "networkx"
    backend_version: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def bind_complete_canonical_partitions(self) -> Self:
        if tuple(sorted(self.vertices)) != self.vertices or len(
            set(self.vertices)
        ) != len(self.vertices):
            raise ValueError("result vertices must be unique and canonical")
        if (
            any(left >= right for left, right in self.edges)
            or tuple(sorted(self.edges)) != self.edges
            or len(set(self.edges)) != len(self.edges)
        ):
            raise ValueError("result edges must be unique and canonical")
        if (
            tuple(sorted(self.generator_ids)) != self.generator_ids
            or len(set(self.generator_ids)) != len(self.generator_ids)
            or self.generator_count != len(self.generator_ids)
        ):
            raise ValueError(
                "result generator identifiers must be unique and canonical"
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
            raise ValueError(
                "vertex orbits must be a complete canonical vertex partition"
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
            raise ValueError("edge orbits must be a complete canonical edge partition")
        return self


__all__ = [
    "MAX_GRAPH_SYMMETRY_EDGES",
    "MAX_GRAPH_SYMMETRY_GENERATORS",
    "MAX_GRAPH_SYMMETRY_VERTICES",
    "GraphAutomorphismGenerator",
    "GraphEdgeColor",
    "GraphEdgeOrbit",
    "GraphSymmetryOrbitRequest",
    "GraphSymmetryOrbitResult",
    "GraphVertexColor",
    "GraphVertexOrbit",
]
