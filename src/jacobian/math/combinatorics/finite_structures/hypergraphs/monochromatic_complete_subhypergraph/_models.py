"""Typed contracts for complete monochromatic uniform-subhypergraph profiles."""

from __future__ import annotations

from math import comb
from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_VERTICES,
    FiniteHypergraph,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.colorings import (
    IndexedHyperedgeColoring,
)

ColorIndex = Annotated[int, Field(ge=0, lt=MAX_EDGES, strict=True)]
SourceEdgeWitness = Annotated[
    tuple[Annotated[str, Field(max_length=64)], ...],
    Field(max_length=MAX_EDGES),
]


class MonochromaticCompleteSubhypergraphRequest(StrictModel):
    """Request all complete monochromatic target subsets of one source."""

    coloring: IndexedHyperedgeColoring
    source_uniformity: StrictInt = Field(ge=1, le=MAX_VERTICES)
    target_uniformity: StrictInt = Field(ge=1, le=MAX_VERTICES)


class MonochromaticCompleteSubhypergraphProfile(StrictModel):
    """The complete target profile and source-edge witnesses.

    ``hypergraph.edges`` and ``candidate_colors`` are aligned in target-edge
    order.  Each row in ``source_edge_witnesses`` lists the source IDs for the
    target's ``binom(target_uniformity, source_uniformity)`` source subsets in
    lexicographic subset order.  The producer establishes the complete
    monochromatic predicate; this model checks axes and source references.
    """

    coloring: IndexedHyperedgeColoring
    source_uniformity: StrictInt = Field(ge=1, le=MAX_VERTICES)
    target_uniformity: StrictInt = Field(ge=1, le=MAX_VERTICES)
    hypergraph: FiniteHypergraph
    candidate_colors: tuple[ColorIndex, ...] = Field(max_length=MAX_EDGES)
    source_edge_witnesses: tuple[SourceEdgeWitness, ...] = Field(max_length=MAX_EDGES)

    @model_validator(mode="after")
    def require_source_bound_axes(self) -> Self:
        source = self.coloring.hypergraph
        if self.target_uniformity < self.source_uniformity:
            raise ValueError("target uniformity must be at least source uniformity")
        if self.target_uniformity > len(source.vertices):
            raise ValueError("target uniformity cannot exceed the source vertex count")
        if self.hypergraph.vertices != source.vertices:
            raise ValueError("profile must retain the source vertex axis")
        edges = self.hypergraph.edges
        if len(self.candidate_colors) != len(edges):
            raise ValueError("candidate colors must align with target edges")
        if len(self.source_edge_witnesses) != len(edges):
            raise ValueError("source-edge witnesses must align with target edges")
        source_ids = {edge_id for edge_id, _ in source.edges}
        seen_members: set[frozenset[str]] = set()
        required_witness_size = comb(self.target_uniformity, self.source_uniformity)
        for (_, members), color, witness in zip(
            edges, self.candidate_colors, self.source_edge_witnesses, strict=True
        ):
            if len(members) != self.target_uniformity:
                raise ValueError("profile edges must have the target uniformity")
            if tuple(sorted(set(members))) != members:
                raise ValueError("profile edge members must be sorted and distinct")
            member_set = frozenset(members)
            if member_set in seen_members:
                raise ValueError("profile target vertex sets must occur once")
            seen_members.add(member_set)
            if color >= self.coloring.color_count:
                raise ValueError("candidate color must belong to the source palette")
            if len(witness) != required_witness_size:
                raise ValueError("witnesses must cover every source subset")
            if len(set(witness)) != len(witness) or any(
                edge_id not in source_ids for edge_id in witness
            ):
                raise ValueError("witnesses must reference distinct source edge IDs")
        return self


__all__ = [
    "ColorIndex",
    "MonochromaticCompleteSubhypergraphProfile",
    "MonochromaticCompleteSubhypergraphRequest",
    "SourceEdgeWitness",
]
