"""Source-bound indexed colour partitions of finite hyperedges."""

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    FiniteHypergraph,
)

__all__ = ["HyperedgeColorAssignment", "IndexedHyperedgeColoring"]


class HyperedgeColorAssignment(StrictModel):
    """A colour index attached to an explicit source hyperedge identity."""

    edge_id: str = Field(max_length=64)
    color_index: StrictInt = Field(ge=0, lt=MAX_EDGES)


class IndexedHyperedgeColoring(StrictModel):
    """A total partition of named source edges into nonempty indexed colours.

    Equal vertex sets with different edge IDs remain different edges. Entries
    are canonicalized to the retained source edge order; their explicit IDs
    preserve the binding when a caller serializes or reorders assignments.
    Colour indices occupy exactly ``0..color_count-1``. An empty source has
    zero colours. Colour meanings, when present, belong to a producer's palette.
    """

    hypergraph: FiniteHypergraph
    color_count: StrictInt = Field(ge=0, le=MAX_EDGES)
    assignments: tuple[HyperedgeColorAssignment, ...] = Field(max_length=MAX_EDGES)

    @model_validator(mode="after")
    def require_total_partition(self) -> Self:
        by_id = {entry.edge_id: entry for entry in self.assignments}
        source_ids = tuple(edge_id for edge_id, _ in self.hypergraph.edges)
        if len(by_id) != len(self.assignments) or set(by_id) != set(source_ids):
            raise ValueError("colour assignments must cover every source edge ID once")
        if {entry.color_index for entry in self.assignments} != set(
            range(self.color_count)
        ):
            raise ValueError("colour indices must occupy exactly 0..color_count-1")
        object.__setattr__(
            self, "assignments", tuple(by_id[edge_id] for edge_id in source_ids)
        )
        return self
