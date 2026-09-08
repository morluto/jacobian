"""Complete same-colour union conflicts and their source-edge provenance."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.colorings import (
    IndexedHyperedgeColoring,
)

MAX_CONFLICT_PAIRS = 65_536


class SameColorConflictsRequest(StrictModel):
    coloring: IndexedHyperedgeColoring


class SameColorConflictProvenance(StrictModel):
    """One unordered source-edge pair producing the named conflict union.

    Source IDs occur in their retained source-edge-axis order; the colour index
    identifies their common colour. Different source IDs remain distinct even
    if they have equal vertex sets.
    """

    conflict_edge_id: str = Field(max_length=64)
    source_edge_ids: tuple[
        Annotated[str, Field(max_length=64)], Annotated[str, Field(max_length=64)]
    ]
    color_index: int = Field(ge=0, lt=12_000)


class SameColorConflictsResult(StrictModel):
    """One edge per distinct union, with complete flat source-pair provenance.

    Vertices retain the source order. Conflict edges follow increasing bitmask
    of source vertex positions, with IDs c0,c1,...; this transports unchanged
    under coherent relabelling.     Provenance follows lexicographic source-edge
    position pairs, across all colours. Deserialization rewrites reversed pair
    IDs, shuffles, and identical duplicate rows onto that axis order; conflicting
    duplicate pairs are rejected. Empty unions are retained as empty
    hyperedges: then no vertex subset, including the empty one, is independent.
    The existing independence-number consumer admits only nonempty hyperedges.
    """

    coloring: IndexedHyperedgeColoring
    hypergraph: FiniteHypergraph
    provenance: tuple[SameColorConflictProvenance, ...] = Field(
        max_length=MAX_CONFLICT_PAIRS
    )

    @model_validator(mode="after")
    def require_source_references(self) -> Self:
        if self.hypergraph.vertices != self.coloring.hypergraph.vertices:
            raise ValueError("conflict vertices must retain the source vertex axis")
        source_ids = tuple(edge_id for edge_id, _ in self.coloring.hypergraph.edges)
        positions = {edge_id: index for index, edge_id in enumerate(source_ids)}
        result_ids = {edge_id for edge_id, _ in self.hypergraph.edges}
        ranked: list[tuple[int, int, SameColorConflictProvenance]] = []
        for row in self.provenance:
            left, right = row.source_edge_ids
            if left == right or left not in positions or right not in positions:
                raise ValueError("provenance must reference two distinct source IDs")
            if row.conflict_edge_id not in result_ids:
                raise ValueError("provenance must reference a declared conflict edge")
            if row.color_index >= self.coloring.color_count:
                raise ValueError("provenance colour must belong to the source palette")
            left_pos = positions[left]
            right_pos = positions[right]
            if left_pos > right_pos:
                left, right = right, left
                left_pos, right_pos = right_pos, left_pos
            if row.source_edge_ids != (left, right):
                row = row.model_copy(update={"source_edge_ids": (left, right)})
            ranked.append((left_pos, right_pos, row))
        ranked.sort(key=lambda item: (item[0], item[1]))
        canonical: list[SameColorConflictProvenance] = []
        for _, _, row in ranked:
            if canonical and canonical[-1].source_edge_ids == row.source_edge_ids:
                if canonical[-1] != row:
                    raise ValueError(
                        "provenance must not contain conflicting duplicate pairs"
                    )
                continue
            canonical.append(row)
        object.__setattr__(self, "provenance", tuple(canonical))
        return self
