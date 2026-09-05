"""Typed contracts for the finite-Abelian zero-sum atom hypergraph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    MAX_VERTICES,
    FiniteHypergraph,
)
from jacobian.math.groups.finite_abelian import (
    BoundedGroupElement,
    FiniteAbelianProductGroup,
)

MAX_ATOM_SOURCE_ELEMENTS = 24
MAX_ATOM_GROUP_ORDER = 4_096
MAX_ATOM_SUBSET_CHECKS = 20_000_000
MAX_ATOM_MINIMALITY_CHECKS = 20_000_000
MAX_ATOM_EDGES = MAX_EDGES
MAX_ATOM_INCIDENCES = MAX_TOTAL_INCIDENCES
MAX_ATOM_LABEL_LENGTH = 64
MAX_ATOM_RETAINED_AXES = 32_768


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"additive_combinatorics.{reason}", message)


class ZeroSumAtomSource(StrictModel):
    """A distinct finite subset of one explicit product of cyclic groups.

    Coordinates are reduced on their declared axes and rows are sorted
    lexicographically. Distinctness is checked after reduction, so every
    serialized vertex has one canonical group-element value and one stable
    source index.
    """

    group: FiniteAbelianProductGroup
    elements: tuple[BoundedGroupElement, ...] = Field(
        default=(),
        max_length=MAX_ATOM_SOURCE_ELEMENTS,
        description=(
            "Distinct group elements as integer coordinate tuples. Each row "
            "is reduced modulo its cyclic factor; reduced rows must be "
            "distinct and sorted lexicographically."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_source(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        prepared: dict[str, object] = dict(value)
        raw_group = prepared.get("group")
        if isinstance(raw_group, Mapping):
            group: dict[str, object] = dict(raw_group)
            raw_moduli = group.get("moduli")
            if isinstance(raw_moduli, list):
                if len(raw_moduli) > MAX_ATOM_RETAINED_AXES:
                    raise _validation_error(
                        "zero_sum_atom_source_rank",
                        "group axes exceed the raw retained-coordinate envelope",
                    )
                group["moduli"] = tuple(raw_moduli)
            prepared["group"] = group
        raw_elements = prepared.get("elements")
        if isinstance(raw_elements, list):
            if len(raw_elements) > MAX_ATOM_SOURCE_ELEMENTS:
                raise _validation_error(
                    "zero_sum_atom_source_cardinality",
                    "zero-sum atom source permits at most 24 items",
                )
            if any(
                not isinstance(element, list) or len(element) > MAX_ATOM_RETAINED_AXES
                for element in raw_elements
            ):
                raise _validation_error(
                    "zero_sum_atom_source_rank",
                    "source coordinates exceed the raw retained-coordinate envelope",
                )
            prepared["elements"] = tuple(raw_elements)
        return canonicalize_json_containers(prepared)

    @model_validator(mode="after")
    def require_canonical_source(self) -> Self:
        rank = len(self.group.moduli)
        if len(self.elements) > MAX_ATOM_SOURCE_ELEMENTS:
            raise _validation_error(
                "zero_sum_atom_source_cardinality",
                "zero-sum atom source exceeds the "
                f"{MAX_ATOM_SOURCE_ELEMENTS}-element bound",
            )
        if any(len(element) != rank for element in self.elements):
            raise _validation_error(
                "zero_sum_atom_source_rank",
                "every zero-sum atom source element must match the group rank",
            )
        normalized = [
            tuple(
                coordinate % modulus
                for coordinate, modulus in zip(element, self.group.moduli, strict=True)
            )
            for element in self.elements
        ]
        if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
            raise _validation_error(
                "zero_sum_atom_source_canonical",
                "reduced zero-sum atom source elements must be distinct and sorted",
            )
        if normalized:
            object.__setattr__(self, "elements", tuple(normalized))
        return self


class ZeroSumAtomHypergraphRequest(StrictModel):
    """Construct the complete inclusion-minimal zero-sum subset hypergraph.

    Every returned hyperedge is nonempty, sums to the group identity, and has
    no nonempty proper zero-sum subset. The empty source has no hyperedges;
    if the identity element belongs to the source, the singleton containing it
    is an atom.
    """

    source: ZeroSumAtomSource


class ZeroSumAtomHypergraphResult(StrictModel):
    """The complete source-bound atom hypergraph and indexed projection.

    ``vertex_source_indices`` maps the decimal vertex label to its index in
    the retained source.  Every edge ID is the comma-separated decimal source
    indices of its members, in increasing order; group-element identity is
    therefore reconstructible from the retained source without operation-local
    coordinate strings.
    """

    source: ZeroSumAtomSource
    hypergraph: FiniteHypergraph
    vertex_source_indices: tuple[int, ...] = Field(
        max_length=MAX_VERTICES,
        description="Source index of each vertex, in hypergraph vertex order.",
    )
    atom_count: int = Field(ge=0, le=MAX_ATOM_EDGES)
    total_incidences: int = Field(ge=0, le=MAX_ATOM_INCIDENCES)
    subset_checks: int = Field(ge=0, le=MAX_ATOM_SUBSET_CHECKS)
    minimality_checks: int = Field(ge=0, le=MAX_ATOM_MINIMALITY_CHECKS)

    @model_validator(mode="after")
    def require_consistent_projection(self) -> Self:
        if self.vertex_source_indices != tuple(range(len(self.source.elements))):
            raise _validation_error(
                "zero_sum_atom_vertex_indices",
                "vertex_source_indices must enumerate the retained source",
            )
        label_width = len(str(max(0, len(self.source.elements) - 1)))
        vertices = tuple(
            str(index).zfill(label_width) for index in self.vertex_source_indices
        )
        if self.hypergraph.vertices != vertices:
            raise _validation_error(
                "zero_sum_atom_vertices",
                "hypergraph vertices must be decimal source indices in order",
            )
        for edge_id, members in self.hypergraph.edges:
            expected = tuple(
                str(index).zfill(label_width) for index in sorted(map(int, members))
            )
            if edge_id != ",".join(expected) or members != expected:
                raise _validation_error(
                    "zero_sum_atom_edge_encoding",
                    "each edge ID and member tuple must encode increasing source indices",
                )
        if self.atom_count != len(self.hypergraph.edges):
            raise _validation_error(
                "zero_sum_atom_count",
                "atom_count must equal the number of returned hyperedges",
            )
        incidences = sum(len(members) for _, members in self.hypergraph.edges)
        if self.total_incidences != incidences:
            raise _validation_error(
                "zero_sum_atom_incidence_count",
                "total_incidences must equal the sum of edge sizes",
            )
        return self


__all__ = [
    "MAX_ATOM_EDGES",
    "MAX_ATOM_GROUP_ORDER",
    "MAX_ATOM_INCIDENCES",
    "MAX_ATOM_MINIMALITY_CHECKS",
    "MAX_ATOM_SOURCE_ELEMENTS",
    "MAX_ATOM_SUBSET_CHECKS",
    "ZeroSumAtomHypergraphRequest",
    "ZeroSumAtomHypergraphResult",
    "ZeroSumAtomSource",
]
