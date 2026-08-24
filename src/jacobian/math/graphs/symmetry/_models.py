"""Typed contracts for exact declared graph-symmetry actions."""

from __future__ import annotations

import unicodedata
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.graphs.values import ColoredUndirectedGraph

MAX_GRAPH_SYMMETRY_VERTICES = 256
MAX_GRAPH_SYMMETRY_EDGES = 4_096
MAX_GRAPH_SYMMETRY_GENERATORS = 64
_UNCOLORED = "__UNCOLORED__"

GraphSymmetryLabel = Annotated[str, Field(min_length=1, max_length=64)]
GraphSymmetryEdge = tuple[GraphSymmetryLabel, GraphSymmetryLabel]


def _canonical_edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


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
        raise ValueError(
            "every graph symmetry generator must be a total vertex permutation "
            "declared as one (vertex, image) pair per declared vertex in the "
            "graph's declared vertex order"
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


_RESULT_ENVELOPE_RESERVE_BYTES = 2_048
_ORBIT_OBJECT_FIXED_WIRE_BYTES = 47


def _estimate_orbit_result_wire_bytes(request: GraphSymmetryOrbitRequest) -> int:
    """Upper-bound the canonical wire size of this request's orbit result.

    The result echoes its complete declared source action and repeats every
    vertex label once (the ``vertices`` field plus the vertex-orbit members)
    and every canonical edge once (the ``edges`` field plus the edge-orbit
    members) beyond that echo. Orbit representatives are drawn from those
    same declared labels and pairs, and the declared action fixes exactly
    which elements can be representatives: the union components of the
    declared generator mappings are the generated subgroup's orbits, so
    admission prices one representative per computed orbit instead of
    re-charging the complete graph. Every retained string is Unicode NFC -
    vertices through the canonical graph value, generator identifiers and
    colors through request admission - so these strict-JSON measurements
    equal their canonicalized sizes. Each orbit object contributes its exact
    fixed wire structure - the ``orbit_index``, ``representative``, and
    ``members`` keys with their punctuation, that orbit index's digit width,
    and one separating comma - so singleton-orbit results carry no per-orbit
    padding beyond the representatives they must retain. The bound also
    charges the exact list separators implied by the computed partition -
    one comma between consecutive members of each repetition beyond the
    echo, so singleton orbit blocks contribute no phantom separator bytes -
    together with each retained generator identifier and the envelope
    reserve, so every accepted request serializes inside the canonical
    output limit.
    """
    from jacobian.math.graphs.symmetry._operations import _declared_orbit_partitions

    vertex_label_bytes = [
        len(encode_strict_json(vertex)) for vertex in request.graph.graph.vertices
    ]
    edge_pair_bytes = [
        len(encode_strict_json(left)) + len(encode_strict_json(right)) + 3
        for left, right in request.graph.graph.edges
    ]
    declared_vertex_members, declared_edge_members = _declared_orbit_partitions(request)
    representative_bytes = sum(
        len(encode_strict_json(representative))
        for representative in (members[0] for members in declared_vertex_members)
    ) + sum(
        len(encode_strict_json(left)) + len(encode_strict_json(right)) + 3
        for left, right in (members[0] for members in declared_edge_members)
    )
    vertex_orbit_count = len(declared_vertex_members)
    edge_orbit_count = len(declared_edge_members)
    vertex_separator_bytes = (
        max(len(vertex_label_bytes) - 1, 0)
        + len(vertex_label_bytes)
        - vertex_orbit_count
    )
    edge_separator_bytes = (
        max(len(edge_pair_bytes) - 1, 0) + len(edge_pair_bytes) - edge_orbit_count
    )
    return (
        len(encode_strict_json(request.model_dump(mode="json")))
        + 2 * sum(vertex_label_bytes)
        + 2 * sum(edge_pair_bytes)
        + vertex_separator_bytes
        + edge_separator_bytes
        + (vertex_orbit_count + edge_orbit_count) * (_ORBIT_OBJECT_FIXED_WIRE_BYTES + 1)
        + sum(len(str(index)) for index in range(vertex_orbit_count))
        + sum(len(str(index)) for index in range(edge_orbit_count))
        + representative_bytes
        + sum(
            len(encode_strict_json(generator.generator_id)) + 4
            for generator in request.generators
        )
        + _RESULT_ENVELOPE_RESERVE_BYTES
    )


def _require_result_output_headroom(request: GraphSymmetryOrbitRequest) -> None:
    output_limit = CanonicalLimits().max_output_bytes
    if _estimate_orbit_result_wire_bytes(request) > output_limit:
        raise ValueError(
            "the graph symmetry orbit result retains its declared source and "
            f"would exceed the {output_limit}-byte canonical output limit; "
            "shorten vertex labels or shrink the graph"
        )


class GraphSymmetryOrbitRequest(StrictModel):
    graph: ColoredUndirectedGraph
    generators: tuple[GraphAutomorphismGenerator, ...] = Field(
        max_length=MAX_GRAPH_SYMMETRY_GENERATORS
    )
    action: Literal["DECLARED_AUTOMORPHISM_GENERATORS"] = (
        "DECLARED_AUTOMORPHISM_GENERATORS"
    )

    @model_validator(mode="after")
    def require_bounded_color_preserving_automorphisms(self) -> Self:
        vertices = self.graph.graph.vertices
        edges = self.graph.graph.edges
        if len(vertices) > MAX_GRAPH_SYMMETRY_VERTICES:
            raise ValueError(
                f"graph symmetry exceeds the {MAX_GRAPH_SYMMETRY_VERTICES}-vertex bound"
            )
        if len(edges) > MAX_GRAPH_SYMMETRY_EDGES:
            raise ValueError(
                f"graph symmetry exceeds the {MAX_GRAPH_SYMMETRY_EDGES}-edge bound"
            )

        generator_ids = tuple(generator.generator_id for generator in self.generators)
        if len(set(generator_ids)) != len(generator_ids):
            raise ValueError("graph symmetry generator identifiers must be unique")
        if any(
            not unicodedata.is_normalized("NFC", generator_id)
            for generator_id in generator_ids
        ):
            raise ValueError(
                "graph symmetry generator identifiers must use Unicode NFC"
            )

        vertex_set = set(vertices)
        edge_set = set(edges)
        vertex_colors = (
            dict(zip(vertices, self.graph.vertex_colors, strict=True))
            if self.graph.vertex_colors
            else dict.fromkeys(vertices, _UNCOLORED)
        )
        edge_colors = (
            dict(zip(edges, self.graph.edge_colors, strict=True))
            if self.graph.edge_colors
            else dict.fromkeys(edges, _UNCOLORED)
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

    @model_validator(mode="after")
    def require_result_within_canonical_output_limit(self) -> Self:
        _require_result_output_headroom(self)
        return self


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
            raise ValueError("vertex orbit members must be unique and canonical")
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
            raise ValueError("edge orbit members must be unique and canonical")
        return self


class GraphSymmetryOrbitResult(StrictModel):
    """Complete vertex and edge orbits of one declared generated subgroup.

    The result retains its complete declared source action - the canonical
    graph, the generator mappings, and the declared colors - through the
    domain-owned request value. Validation binds every claim to that source:
    the retained graph and generators must equal it, the color modes must
    match its declared colors, and both returned partitions must equal a
    replay of the exact orbits of the declared generators.
    """

    source: GraphSymmetryOrbitRequest
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
        declared_generator_ids = tuple(
            sorted(generator.generator_id for generator in self.source.generators)
        )
        if (
            self.vertices != tuple(sorted(self.source.graph.graph.vertices))
            or self.edges != tuple(sorted(self.source.graph.graph.edges))
            or self.generator_ids != declared_generator_ids
        ):
            raise ValueError(
                "result graph and generators must equal the retained source action"
            )
        if self.vertex_color_mode != (
            "DECLARED" if self.source.graph.vertex_colors else "UNCOLORED"
        ):
            raise ValueError(
                "vertex color mode must match the retained source vertex colors"
            )
        if self.edge_color_mode != (
            "DECLARED" if self.source.graph.edge_colors else "UNCOLORED"
        ):
            raise ValueError(
                "edge color mode must match the retained source edge colors"
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
        from jacobian.math.graphs.symmetry._operations import _declared_orbit_partitions

        expected_vertex_members, expected_edge_members = _declared_orbit_partitions(
            self.source
        )
        if tuple(orbit.members for orbit in self.vertex_orbits) != (
            expected_vertex_members
        ):
            raise ValueError(
                "vertex orbits must be the exact orbits of the declared generators"
            )
        if tuple(orbit.members for orbit in self.edge_orbits) != (
            expected_edge_members
        ):
            raise ValueError(
                "edge orbits must be the exact orbits of the declared generators"
            )
        return self


__all__ = [
    "MAX_GRAPH_SYMMETRY_EDGES",
    "MAX_GRAPH_SYMMETRY_GENERATORS",
    "MAX_GRAPH_SYMMETRY_VERTICES",
    "GraphAutomorphismGenerator",
    "GraphEdgeOrbit",
    "GraphSymmetryOrbitRequest",
    "GraphSymmetryOrbitResult",
    "GraphVertexOrbit",
]
