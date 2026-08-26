"""Typed contracts for exact declared graph-symmetry actions."""

from __future__ import annotations

import unicodedata
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.graphs.symmetry._edges import canonical_edge
from jacobian.math.graphs.symmetry._orbits import declared_orbit_partitions
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


_ORBIT_OBJECT_FIXED_WIRE_BYTES = 47


def _quoted_wire_size(value: str) -> int:
    """Strict-JSON wire size of one retained string, including its quotes.

    Every retained graph-symmetry string is Unicode NFC, so this strict
    measurement equals the string's canonicalized wire size.
    """

    return len(encode_strict_json(value))


def _json_array_wire_bytes(element_sizes: list[int] | tuple[int, ...]) -> int:
    """Exact wire size of a JSON array from exact element sizes."""

    return 2 + sum(element_sizes) + max(len(element_sizes) - 1, 0)


def _orbit_result_fixed_frame_bytes() -> int:
    """Exact key-name, punctuation, and constant-literal cost of the result.

    Derived from ``GraphSymmetryOrbitResult`` itself - the field-name tuple
    and the defaulted constant literals - so this admission bound cannot
    drift from the published wire contract. Covers the object braces, the
    sixteen commas between the seventeen top-level fields, every quoted key
    name with its colon, and the six fixed literal values.
    """

    fields = GraphSymmetryOrbitResult.model_fields
    names = tuple(fields)
    frame_bytes = (
        2
        + sum(len(encode_strict_json(name)) + 1 for name in names)
        + max(len(names) - 1, 0)
    )
    constants = (
        fields["action"].default,
        fields["generator_validation"].default,
        fields["orbit_completeness"].default,
        fields["automorphism_group_completeness"].default,
        fields["exactness"].default,
        fields["determinism"].default,
    )
    return frame_bytes + sum(_quoted_wire_size(value) for value in constants)


def _orbit_result_canonical_wire_bytes(request: GraphSymmetryOrbitRequest) -> int:
    """Exact canonical wire size of this request's orbit result.

    Defining invariant: for every admitted request, canonicalizing the
    produced ``GraphSymmetryOrbitResult`` yields exactly this many bytes, so
    output-headroom admission rejects precisely those requests whose typed
    result cannot fit the canonical output envelope - with no reserve and
    no per-orbit padding. Exactness holds because every term measures the
    same encoder: all retained strings are Unicode NFC (vertices through the
    canonical graph value, generator identifiers through request admission),
    so strict-JSON measurements equal their canonicalized sizes; the fixed
    frame derives from the result model itself; and every variable component
    is charged exactly - the echoed source, both label axes with the array
    and intra-partition separators implied by the computed orbits, one
    representative per computed orbit, each orbit list's enclosing brackets,
    each orbit object's fixed structure plus its index's digit width and its
    single separating comma, the retained generator identifiers, the three
    count values' digit widths, and the two color-mode literals selected by
    the declared colors.
    """
    vertices = tuple(sorted(request.graph.graph.vertices))
    edges = tuple(sorted(request.graph.graph.edges))
    vertex_members, edge_members = declared_orbit_partitions(
        vertices,
        edges,
        tuple(dict(generator.mapping) for generator in request.generators),
    )

    vertex_label_sizes = [_quoted_wire_size(vertex) for vertex in vertices]
    edge_pair_sizes = [
        _quoted_wire_size(left) + _quoted_wire_size(right) + 3 for left, right in edges
    ]
    generator_id_sizes = [
        _quoted_wire_size(generator.generator_id) for generator in request.generators
    ]

    vertex_orbit_count = len(vertex_members)
    edge_orbit_count = len(edge_members)

    vertex_member_bytes = sum(vertex_label_sizes) + max(
        len(vertices) - vertex_orbit_count, 0
    )
    edge_member_bytes = sum(edge_pair_sizes) + max(len(edges) - edge_orbit_count, 0)
    representative_bytes = sum(
        _quoted_wire_size(members[0]) for members in vertex_members
    ) + sum(
        _quoted_wire_size(left) + _quoted_wire_size(right) + 3
        for left, right in (members[0] for members in edge_members)
    )
    orbit_object_bytes = (
        4
        + (vertex_orbit_count + edge_orbit_count) * _ORBIT_OBJECT_FIXED_WIRE_BYTES
        + sum(len(str(index)) for index in range(vertex_orbit_count))
        + sum(len(str(index)) for index in range(edge_orbit_count))
        + max(vertex_orbit_count - 1, 0)
        + max(edge_orbit_count - 1, 0)
        + representative_bytes
    )
    color_mode_bytes = sum(
        _quoted_wire_size("DECLARED" if declared else "UNCOLORED")
        for declared in (
            bool(request.graph.vertex_colors),
            bool(request.graph.edge_colors),
        )
    )

    return (
        _orbit_result_fixed_frame_bytes()
        + len(encode_strict_json(request.model_dump(mode="json")))
        + _json_array_wire_bytes(vertex_label_sizes)
        + _json_array_wire_bytes(edge_pair_sizes)
        + vertex_member_bytes
        + edge_member_bytes
        + _json_array_wire_bytes(generator_id_sizes)
        + orbit_object_bytes
        + len(str(len(request.generators)))
        + len(str(vertex_orbit_count))
        + len(str(edge_orbit_count))
        + color_mode_bytes
    )


def _require_result_output_headroom(request: GraphSymmetryOrbitRequest) -> None:
    output_limit = CanonicalLimits().max_output_bytes
    if _orbit_result_canonical_wire_bytes(request) > output_limit:
        raise PydanticCustomError(
            "graph.symmetry_orbit_result_retains_its_declared_source",
            "the graph symmetry orbit result retains its declared source and "
            "its complete canonical serialization would exceed the "
            f"{output_limit}-byte canonical output limit; shorten vertex "
            "labels or shrink the graph",
        )


class GraphSymmetryOrbitRequest(StrictModel):
    """Declared color-preserving generators of one bounded graph's subgroup.

    Admission is aggregate as well as field-level: the result retains this
    complete request as its source and adds the derived orbit partitions,
    so a request whose complete canonical result would exceed Jacobian's
    canonical output limit is rejected here, before execution, even when
    every field-level bound is satisfied.
    """

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
                "vertex and edge orbits, so request validation also applies "
                "an aggregate retained-result bound: a request whose "
                "complete canonical result would exceed Jacobian's "
                f"{CanonicalLimits().max_output_bytes}-byte canonical "
                "output limit is rejected at admission even when every "
                "field-level bound is satisfied."
            )
        }
    )

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
            raise PydanticCustomError(
                "graph.symmetry_exceeds_max_symmetry_vertices_vertex_bound",
                f"graph symmetry exceeds the {MAX_GRAPH_SYMMETRY_VERTICES}-vertex bound",
            )
        if len(edges) > MAX_GRAPH_SYMMETRY_EDGES:
            raise PydanticCustomError(
                "graph.symmetry_exceeds_max_symmetry_edges_edge_bound",
                f"graph symmetry exceeds the {MAX_GRAPH_SYMMETRY_EDGES}-edge bound",
            )

        generator_ids = tuple(generator.generator_id for generator in self.generators)
        if len(set(generator_ids)) != len(generator_ids):
            raise PydanticCustomError(
                "graph.graph_symmetry_generator_identifiers_must_be_uni",
                "graph symmetry generator identifiers must be unique",
            )
        if any(
            not unicodedata.is_normalized("NFC", generator_id)
            for generator_id in generator_ids
        ):
            raise PydanticCustomError(
                "graph.symmetry_generator_identifiers_use_unicode_nfc",
                "graph symmetry generator identifiers must use Unicode NFC",
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
    source binding. The trusted kernel constructs exact partitions; an
    independently supplied partition can be checked by the explicit bounded
    verifier in ``_operations``.
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
        source: GraphSymmetryOrbitRequest,
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
            source=source,
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
    "GraphVertexOrbit",
]
