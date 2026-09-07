"""Dulmage--Mendelsohn decomposition contracts."""

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.bipartite.values import (
    BipartiteVertexRegion,
    FixedBipartiteGraph,
    Vertex,
)


class DulmageMendelsohnRequest(StrictModel):
    graph: FixedBipartiteGraph


class DulmageMendelsohnDecomposition(StrictModel):
    """Source-bound deficient regions and balanced elementary blocks.

    Orient all edges left→right and matching edges also right→left.
    left_excess is reachable from unmatched left vertices; right_excess
    reaches unmatched right vertices. Balanced blocks are SCCs on the
    remainder, labelled by earliest input-left position. Vertex tuples
    retain their input-side order. condensation_edges contains every direct
    inter-block arc, not transitive closure or a chosen topological order.
    These structural sets and arcs are independent of the maximum matching.
    """

    graph: FixedBipartiteGraph
    structural_rank: int = Field(ge=0, le=512)
    left_excess: BipartiteVertexRegion
    right_excess: BipartiteVertexRegion
    balanced_blocks: tuple[BipartiteVertexRegion, ...] = Field(max_length=512)
    condensation_edges: tuple[tuple[Vertex, Vertex], ...] = Field(max_length=65536)

    @model_validator(mode="after")
    def require_partition_shape(self) -> Self:
        left = self.graph.left_vertices
        right = self.graph.right_vertices
        regions = (self.left_excess, self.right_excess, *self.balanced_blocks)
        for source, field in ((left, "left_vertices"), (right, "right_vertices")):
            positions = {v: i for i, v in enumerate(source)}
            flat = tuple(v for region in regions for v in getattr(region, field))
            if len(flat) != len(source) or set(flat) != set(source):
                raise ValueError("regions must partition each original side")
            for region in regions:
                members = getattr(region, field)
                if members != tuple(sorted(members, key=positions.__getitem__)):
                    raise ValueError("region vertices must retain input-side order")
        if any(
            not block.left_vertices
            or len(block.left_vertices) != len(block.right_vertices)
            for block in self.balanced_blocks
        ):
            raise ValueError("balanced blocks must be nonempty with equal side sizes")
        left_position = {v: i for i, v in enumerate(left)}
        first = [
            left_position[block.left_vertices[0]] for block in self.balanced_blocks
        ]
        if first != sorted(first):
            raise ValueError("balanced blocks must be ordered by first left position")
        if self.condensation_edges != tuple(sorted(set(self.condensation_edges))):
            raise ValueError("condensation edges must be distinct and sorted")
        k = len(self.balanced_blocks)
        if any(a == b or a >= k or b >= k for a, b in self.condensation_edges):
            raise ValueError("condensation edges must join distinct balanced blocks")
        if self.structural_rank > min(len(left), len(right)):
            raise ValueError("structural rank cannot exceed either side")
        return self
