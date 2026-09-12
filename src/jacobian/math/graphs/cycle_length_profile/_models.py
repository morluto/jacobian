"""Typed contracts for the cycle-length profile operation."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import (
    MAX_SIMPLE_GRAPH_VERTICES,
    SimpleUndirectedGraph,
)

MAX_VERTICES = MAX_SIMPLE_GRAPH_VERTICES


def dihedral_canonical_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    """Rotate to the unique minimum vertex and keep the lex-smaller orientation."""

    min_index = min(range(len(cycle)), key=cycle.__getitem__)
    rotated = cycle[min_index:] + cycle[:min_index]
    reversed_orientation = (rotated[0], *reversed(rotated[1:]))
    if reversed_orientation < rotated:
        return reversed_orientation
    return rotated


def is_dihedral_canonical_cycle(cycle: tuple[str, ...]) -> bool:
    """Return whether ``cycle`` is already in dihedral-canonical form."""

    return cycle == dihedral_canonical_cycle(cycle)


class CycleLengthProfileRequest(StrictModel):
    """Request for the simple-cycle length profile of a graph."""

    graph: SimpleUndirectedGraph = Field(
        description=(
            "Canonical simple graph. Admission also requires the complete first-"
            "witness search to fit the 10,000,000-unit work bound and the complete "
            "profile to fit its retained row and witness bounds."
        )
    )


class CycleLengthRow(StrictModel):
    """One cycle length with a canonical witness cycle."""

    cycle_length: StrictInt = Field(ge=3, le=MAX_VERTICES)
    witness: tuple[str, ...] = Field(min_length=3, max_length=MAX_VERTICES)

    @model_validator(mode="after")
    def require_canonical_witness(self) -> Self:
        if self.cycle_length != len(self.witness):
            raise PydanticCustomError(
                "cycle_profile.witness_length_mismatch",
                "cycle witness length must match cycle_length",
            )
        if len(set(self.witness)) != len(self.witness):
            raise PydanticCustomError(
                "cycle_profile.witness_vertices_must_be_distinct",
                "cycle witnesses must have distinct vertices",
            )
        if not is_dihedral_canonical_cycle(self.witness):
            raise PydanticCustomError(
                "cycle_profile.witness_must_be_canonical",
                "cycle witnesses must use canonical rotation and orientation",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, cycle_length: int, witness: tuple[str, ...]
    ) -> CycleLengthRow:
        """Construct a row after the owner kernel established its invariants."""

        return cls.model_construct(cycle_length=cycle_length, witness=witness)


class CycleLengthProfileResult(StrictModel):
    """The complete cycle-length profile of a graph."""

    graph: SimpleUndirectedGraph
    rows: tuple[CycleLengthRow, ...] = Field(max_length=MAX_VERTICES - 2)

    @model_validator(mode="after")
    def require_structural_profile(self) -> Self:
        lengths = tuple(row.cycle_length for row in self.rows)
        if lengths != tuple(sorted(lengths)) or len(set(lengths)) != len(lengths):
            raise PydanticCustomError(
                "cycle_profile.rows_must_be_sorted_unique",
                "cycle profile rows must be sorted and unique",
            )
        vertices = set(self.graph.vertices)
        if any(not set(row.witness) <= vertices for row in self.rows):
            raise PydanticCustomError(
                "cycle_profile.witness_vertices_must_belong_to_graph",
                "cycle witnesses must use graph vertices",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        graph: SimpleUndirectedGraph,
        rows: tuple[CycleLengthRow, ...],
    ) -> CycleLengthProfileResult:
        """Construct a result after the owner admission and kernel checks."""

        return cls.model_construct(graph=graph, rows=rows)


class FixedLengthCycleEnumerationRequest(StrictModel):
    """Enumerate every simple cycle of one fixed length.

    The graph is the canonical finite simple undirected carrier.  A length
    above the graph order is a valid empty family; the owner can answer that
    case without search.
    """

    graph: SimpleUndirectedGraph = Field(
        description="Canonical finite simple undirected graph; directed and multigraph values are not in this operation's domain."
    )
    cycle_length: StrictInt = Field(
        ge=3,
        le=MAX_VERTICES,
        description="The exact number k of distinct vertices in each returned cycle; k greater than the graph order returns an empty family.",
    )


class CycleIncidenceRow(StrictModel):
    """Cycles incident with one source vertex or edge."""

    source: tuple[str, ...] = Field(min_length=1, max_length=2)
    cycle_indices: tuple[StrictInt, ...] = Field(
        max_length=20_000,
        description="Zero-based indices into the complete, lexicographically sorted cycle family.",
    )


class CycleFamilyKind(StrEnum):
    """The completeness interpretation carried by a cycle family."""

    SIMPLE = "SIMPLE"
    CHORDLESS = "CHORDLESS"


class FixedLengthCycleEnumerationResult(StrictModel):
    """Complete dihedrally canonical fixed-length cycle family.

    The incidence rows are structural indexes over the retained source axes.
    They bind every source vertex and edge to the cycle indices. Deserialization
    checks only the local cycle-edge relation and these indexes. ``family_kind``
    records whether completeness is over all simple or only chordless cycles.
    """

    graph: SimpleUndirectedGraph
    cycle_length: StrictInt = Field(ge=3, le=MAX_VERTICES)
    family_kind: CycleFamilyKind = Field(
        description="Whether the complete family contains simple or chordless cycles."
    )
    cycle_count: StrictInt = Field(ge=0, le=20_000)
    cycles: tuple[tuple[str, ...], ...] = Field(max_length=20_000)
    vertex_incidence: tuple[CycleIncidenceRow, ...]
    edge_incidence: tuple[CycleIncidenceRow, ...]

    @model_validator(mode="after")
    def require_structural_family(self) -> Self:  # noqa: C901
        if self.cycle_count != len(self.cycles):
            raise PydanticCustomError(
                "cycle_enumeration.cycle_count_mismatch",
                "cycle_count must equal the number of returned cycles",
            )
        if tuple(self.cycles) != tuple(sorted(self.cycles)):
            raise PydanticCustomError(
                "cycle_enumeration.cycles_must_be_sorted",
                "cycles must use lexicographic canonical order",
            )
        if len(set(self.cycles)) != len(self.cycles):
            raise PydanticCustomError(
                "cycle_enumeration.cycles_must_be_unique",
                "cycles must be unique after dihedral canonicalization",
            )

        vertices = set(self.graph.vertices)
        graph_edges = {frozenset(edge) for edge in self.graph.edges}
        expected_vertex_incidence: dict[str, list[int]] = {
            vertex: [] for vertex in self.graph.vertices
        }
        expected_edge_incidence: dict[frozenset[str], list[int]] = {
            frozenset(edge): [] for edge in self.graph.edges
        }
        for index, cycle in enumerate(self.cycles):
            if len(cycle) != self.cycle_length:
                raise PydanticCustomError(
                    "cycle_enumeration.cycle_length_mismatch",
                    "every cycle must contain exactly cycle_length vertices",
                )
            if len(set(cycle)) != len(cycle) or not set(cycle) <= vertices:
                raise PydanticCustomError(
                    "cycle_enumeration.cycle_vertices_invalid",
                    "cycles must contain distinct declared graph vertices",
                )
            cycle_edges = tuple(
                frozenset((cycle[position], cycle[(position + 1) % self.cycle_length]))
                for position in range(self.cycle_length)
            )
            if any(edge not in graph_edges for edge in cycle_edges):
                raise PydanticCustomError(
                    "cycle_enumeration.cycle_edges_invalid",
                    "every cycle must close through declared graph edges",
                )
            if not is_dihedral_canonical_cycle(cycle):
                raise PydanticCustomError(
                    "cycle_enumeration.cycle_must_be_canonical",
                    "cycles must use canonical rotation and orientation",
                )
            for vertex in cycle:
                expected_vertex_incidence[vertex].append(index)
            for edge in cycle_edges:
                expected_edge_incidence[edge].append(index)

        expected_vertex_sources = tuple((vertex,) for vertex in self.graph.vertices)
        if (
            tuple(row.source for row in self.vertex_incidence)
            != expected_vertex_sources
        ):
            raise PydanticCustomError(
                "cycle_enumeration.vertex_axis_mismatch",
                "vertex incidence must cover the source vertex axis in source order",
            )
        expected_edge_sources = tuple(self.graph.edges)
        if tuple(row.source for row in self.edge_incidence) != expected_edge_sources:
            raise PydanticCustomError(
                "cycle_enumeration.edge_axis_mismatch",
                "edge incidence must cover the source edge axis in source order",
            )

        valid_indices = range(self.cycle_count)
        for row in (*self.vertex_incidence, *self.edge_incidence):
            if tuple(row.cycle_indices) != tuple(sorted(set(row.cycle_indices))):
                raise PydanticCustomError(
                    "cycle_enumeration.incidence_indices_must_be_sorted_unique",
                    "incidence indices must be sorted and unique",
                )
            if any(index not in valid_indices for index in row.cycle_indices):
                raise PydanticCustomError(
                    "cycle_enumeration.incidence_index_out_of_range",
                    "incidence indices must refer to returned cycles",
                )

        expected_vertex_incidence_tuple = tuple(
            tuple(expected_vertex_incidence[vertex]) for vertex in self.graph.vertices
        )
        if (
            tuple(row.cycle_indices for row in self.vertex_incidence)
            != expected_vertex_incidence_tuple
        ):
            raise PydanticCustomError(
                "cycle_enumeration.vertex_incidence_mismatch",
                "vertex incidence does not bind to the returned cycle family",
            )

        expected_edge_incidence_tuple = tuple(
            tuple(expected_edge_incidence[frozenset(edge)]) for edge in self.graph.edges
        )
        if (
            tuple(row.cycle_indices for row in self.edge_incidence)
            != expected_edge_incidence_tuple
        ):
            raise PydanticCustomError(
                "cycle_enumeration.edge_incidence_mismatch",
                "edge incidence does not bind to the returned cycle family",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: SimpleUndirectedGraph,
        cycle_length: int,
        family_kind: CycleFamilyKind,
        cycles: tuple[tuple[str, ...], ...],
        vertex_incidence: tuple[CycleIncidenceRow, ...],
        edge_incidence: tuple[CycleIncidenceRow, ...],
    ) -> Self:
        """Construct a complete family after owner admission and enumeration."""

        return cls.model_construct(
            graph=graph,
            cycle_length=cycle_length,
            family_kind=family_kind,
            cycle_count=len(cycles),
            cycles=cycles,
            vertex_incidence=vertex_incidence,
            edge_incidence=edge_incidence,
        )


__all__ = [
    "MAX_VERTICES",
    "CycleFamilyKind",
    "CycleIncidenceRow",
    "CycleLengthProfileRequest",
    "CycleLengthProfileResult",
    "CycleLengthRow",
    "FixedLengthCycleEnumerationRequest",
    "FixedLengthCycleEnumerationResult",
    "dihedral_canonical_cycle",
    "is_dihedral_canonical_cycle",
]
