"""Complete same-colour union conflicts and their source-edge provenance."""

from collections import Counter
from math import comb
from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator

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
    color_index: StrictInt = Field(ge=0, lt=12_000)


class SameColorConflictsResult(StrictModel):
    """One edge per distinct union, with complete flat source-pair provenance.

    Vertices retain the source order. Conflict edges follow increasing bitmask
    of source vertex positions, with IDs c0,c1,...; this transports unchanged
    under coherent relabelling.     Provenance follows lexicographic source-edge
    position pairs, across all colours. Deserialization rewrites reversed pair
    IDs, shuffles, and identical duplicate rows onto that axis order; conflicting
    duplicate pairs are rejected. The accepted row count equals the number of
    same-colour source pairs. Each row names the conflict edge whose members are
    the union of its two source member sets. Empty unions are retained as empty
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
        source_members = {
            edge_id: set(members) for edge_id, members in self.coloring.hypergraph.edges
        }
        positions = {edge_id: index for index, edge_id in enumerate(source_ids)}
        result_members = {
            edge_id: set(members) for edge_id, members in self.hypergraph.edges
        }
        ranked: list[tuple[int, int, SameColorConflictProvenance]] = []
        for row in self.provenance:
            left, right = row.source_edge_ids
            if left == right or left not in positions or right not in positions:
                raise ValueError("provenance must reference two distinct source IDs")
            if row.conflict_edge_id not in result_members:
                raise ValueError("provenance must reference a declared conflict edge")
            left_pos = positions[left]
            right_pos = positions[right]
            if (
                self.coloring.assignments[left_pos].color_index != row.color_index
                or self.coloring.assignments[right_pos].color_index != row.color_index
            ):
                raise ValueError(
                    "provenance colour must match both referenced source edges"
                )
            if result_members[row.conflict_edge_id] != (
                source_members[left] | source_members[right]
            ):
                raise ValueError(
                    "provenance must reference the union of the two source edges"
                )
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
        expected_pairs = sum(
            comb(count, 2)
            for count in Counter(
                assignment.color_index for assignment in self.coloring.assignments
            ).values()
        )
        if len(canonical) != expected_pairs:
            raise ValueError("provenance must include every same-colour source pair")
        referenced = {row.conflict_edge_id for row in canonical}
        ledger_ids = tuple(edge_id for edge_id, _ in self.hypergraph.edges)
        expected_ids = tuple(f"c{index}" for index in range(len(ledger_ids)))
        if ledger_ids != expected_ids or set(ledger_ids) != referenced:
            raise ValueError(
                "conflict hypergraph must be the distinct referenced unions "
                "with canonical c0,c1,... identifiers"
            )
        object.__setattr__(self, "provenance", tuple(canonical))
        return self
