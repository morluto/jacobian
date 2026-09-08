"""Source-bound values for deletion-partition compatibility closure."""

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.graphs.coloring.deletion_partition_closure.values import (
    DeletionPartitionTemplate,
    Label,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


class DeletionPartitionClosureRequest(StrictModel):
    template: DeletionPartitionTemplate


class DeletionPartitionBlocker(StrictModel):
    """First source row co-classifying a nonedge, with its canonical block."""

    pair_index: int = Field(ge=0, le=32639)
    row_index: int = Field(ge=0, le=255)
    block_index: int = Field(ge=0, le=254)


class DeletionPartitionClosure(StrictModel):
    """The largest graph making every source deletion partition proper.

    The pair axis follows combinations of carrier positions; each pair's
    endpoints are lexically oriented as required by SimpleUndirectedGraph.
    Blockers follow pair-axis order. Each references the first source row
    co-classifying its pair and the unique block in that row. These mathematical
    relations are established by construct, not replayed during decoding.
    """

    template: DeletionPartitionTemplate
    source_pair_axis: tuple[tuple[Label, Label], ...] = Field(max_length=32640)
    graph: SimpleUndirectedGraph
    nonedge_blockers: tuple[DeletionPartitionBlocker, ...] = Field(max_length=32640)
