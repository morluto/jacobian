"""Typed contracts for the induced edge deletion profile operation."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.coloring._models import (
    MAX_COLORING_COLORS,
    MAX_SOLVER_CONFLICT_BUDGET,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_INDUCED_DELETION_VERTICES = 8
"""Conservative vertex envelope for the complete 2^n profile (8 => 256 rows)."""

MAX_INDUCED_DELETION_R = MAX_COLORING_COLORS
DEFAULT_INDUCED_SOLVER_CONFLICTS = 100_000
MAX_INDUCED_RETAINED_LABEL_CHARACTERS = 1_000_000
MAX_INDUCED_EDGE_MATERIALIZATION = 200_000
MAX_INDUCED_SOLVER_CALLS = 6000
MAX_INDUCED_LEDGER_CONFLICTS = 600_000_000


def _induced_graph_schema() -> JsonSchemaValue:
    schema = SimpleUndirectedGraph.model_json_schema()
    schema["description"] = (
        "A finite simple graph on Jacobian's canonical axis. The induced deletion "
        "profile admits requests with at most 8 vertices (256 subsets) and checks "
        "aggregate solver-call, materialization, and retained-witness budgets."
    )
    return schema


InducedDeletionGraph = Annotated[
    SimpleUndirectedGraph,
    WithJsonSchema(_induced_graph_schema()),
]


class InducedEdgeDeletionProfileRequest(StrictModel):
    """Request the complete induced-subgraph distance to r-colourability."""

    graph: InducedDeletionGraph
    r: StrictInt = Field(
        ge=1,
        le=MAX_INDUCED_DELETION_R,
        description="Target colour count r >=1; D(S) is min deletions to make G[S] r-colourable.",
    )
    solver_conflicts: StrictInt = Field(
        default=DEFAULT_INDUCED_SOLVER_CONFLICTS,
        ge=1,
        le=MAX_SOLVER_CONFLICT_BUDGET,
        description=(
            "Per-decision SAT conflict budget for the bounded Z3 colourability kernel; "
            "exhaustion is an operational timeout and establishes no deletion claim."
        ),
    )


class PerSizeMaximum(StrictModel):
    """Derived maximum minimum deletion cost at one subset size."""

    subset_size: StrictInt = Field(ge=0, le=MAX_INDUCED_DELETION_VERTICES)
    maximum_min_deletions: StrictInt = Field(ge=0)
    attaining_subset_count: StrictInt = Field(ge=1)


class InducedDeletionRow(StrictModel):
    """One vertex subset S, its minimum deletions, and one canonical attaining F."""

    vertex_subset: tuple[str, ...]
    min_deletions: StrictInt = Field(ge=0)
    deleted_edges: tuple[tuple[str, str], ...]
    induced_edge_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_canonical_row(self) -> Self:
        if tuple(sorted(self.vertex_subset)) != self.vertex_subset:
            raise PydanticCustomError(
                "graph.vertex_subset_must_be_canonically_sorted",
                "vertex_subset must be canonically sorted",
            )
        if len(set(self.vertex_subset)) != len(self.vertex_subset):
            raise PydanticCustomError(
                "graph.vertex_subset_must_have_unique_vertices",
                "vertex_subset must have unique vertices",
            )
        if self.deleted_edges != tuple(sorted(self.deleted_edges)):
            raise PydanticCustomError(
                "graph.deleted_edges_must_be_canonically_sorted",
                "deleted_edges must be canonically sorted",
            )
        if len(set(self.deleted_edges)) != len(self.deleted_edges):
            raise PydanticCustomError(
                "graph.deleted_edges_must_be_unique",
                "deleted_edges must be unique",
            )
        for left, right in self.deleted_edges:
            if left >= right:
                raise PydanticCustomError(
                    "graph.deleted_edges_must_be_canonical_pairs",
                    "deleted edges must be canonical pairs with left < right",
                )
        if self.min_deletions != len(self.deleted_edges):
            raise PydanticCustomError(
                "graph.min_deletions_must_match_deleted_edges",
                "min_deletions must equal len(deleted_edges)",
            )
        if self.min_deletions > self.induced_edge_count:
            raise PydanticCustomError(
                "graph.min_deletions_cannot_exceed_induced_edges",
                "min_deletions cannot exceed induced_edge_count",
            )
        return self


class InducedEdgeDeletionProfileResult(StrictModel):
    """Complete profile D_{G,r}(S) over all 2^n vertex subsets with per-size maxima."""

    graph: SimpleUndirectedGraph
    r: StrictInt = Field(ge=1, le=MAX_INDUCED_DELETION_R)
    solver_conflicts: StrictInt = Field(ge=1, le=MAX_SOLVER_CONFLICT_BUDGET)
    rows: tuple[InducedDeletionRow, ...]
    max_deletions_by_size: tuple[PerSizeMaximum, ...]

    @model_validator(mode="after")
    def require_profile_consistency(self) -> Self:  # noqa: C901
        n = len(self.graph.vertices)
        expected_rows = 1 << n if n <= MAX_INDUCED_DELETION_VERTICES else None
        if expected_rows is not None and len(self.rows) != expected_rows:
            raise PydanticCustomError(
                "graph.rows_must_cover_all_vertex_subsets",
                "rows must cover all 2^n vertex subsets",
            )
        if len(self.max_deletions_by_size) != n + 1:
            raise PydanticCustomError(
                "graph.max_by_size_must_cover_0_n",
                "max_deletions_by_size must have n+1 entries for sizes 0..n",
            )
        for entry in self.max_deletions_by_size:
            if entry.subset_size < 0 or entry.subset_size > n:
                raise PydanticCustomError(
                    "graph.per_size_subset_size_out_of_range",
                    "per-size subset_size must be in 0..n",
                )
        sizes = tuple(e.subset_size for e in self.max_deletions_by_size)
        if sizes != tuple(sorted(sizes)) or len(set(sizes)) != len(sizes):
            raise PydanticCustomError(
                "graph.per_size_must_be_sorted_unique",
                "per-size entries must be sorted and unique by subset_size",
            )
        # Verify rows are canonically ordered by (size, lexicographic)
        expected_order = tuple(
            sorted(
                self.rows, key=lambda row: (len(row.vertex_subset), row.vertex_subset)
            )
        )
        if self.rows != expected_order:
            raise PydanticCustomError(
                "graph.rows_must_be_canonically_ordered",
                "rows must be ordered by (subset_size, lexicographic vertex_subset)",
            )
        # Verify max projection matches rows
        from collections import defaultdict

        grouped: dict[int, list[int]] = defaultdict(list)
        for row in self.rows:
            grouped[len(row.vertex_subset)].append(row.min_deletions)
        for entry in self.max_deletions_by_size:
            vals = grouped.get(entry.subset_size, [])
            if not vals:
                # empty size class impossible when n limited, but handle
                if (
                    entry.maximum_min_deletions != 0
                    or entry.attaining_subset_count != 0
                ):
                    raise PydanticCustomError(
                        "graph.max_must_match_rows",
                        "per-size maximum must match rows",
                    )
                continue
            max_val = max(vals)
            count = sum(1 for v in vals if v == max_val)
            if (
                entry.maximum_min_deletions != max_val
                or entry.attaining_subset_count != count
            ):
                raise PydanticCustomError(
                    "graph.max_must_be_derived_from_rows",
                    "per-size maximum must be derived from rows",
                )
        # Deleted edges must be subset of graph edges and induced
        graph_edge_set = set(self.graph.edges)
        vertex_set = set(self.graph.vertices)
        for row in self.rows:
            subset_set = set(row.vertex_subset)
            if not subset_set <= vertex_set:
                raise PydanticCustomError(
                    "graph.vertex_subset_must_be_subset_of_graph",
                    "vertex_subset must be subset of graph vertices",
                )
            induced = {
                e for e in graph_edge_set if e[0] in subset_set and e[1] in subset_set
            }
            if not set(row.deleted_edges) <= induced:
                raise PydanticCustomError(
                    "graph.deleted_edges_must_be_subset_of_induced",
                    "deleted_edges must be subset of induced edges E(G[S])",
                )
            if row.induced_edge_count != len(induced):
                raise PydanticCustomError(
                    "graph.induced_edge_count_must_match_graph",
                    "induced_edge_count must equal |E(G[S])|",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: SimpleUndirectedGraph,
        r: int,
        solver_conflicts: int,
        rows: tuple[InducedDeletionRow, ...],
        max_deletions_by_size: tuple[PerSizeMaximum, ...],
    ) -> Self:
        return cls.model_construct(
            graph=graph,
            r=r,
            solver_conflicts=solver_conflicts,
            rows=rows,
            max_deletions_by_size=max_deletions_by_size,
        )


__all__ = [
    "DEFAULT_INDUCED_SOLVER_CONFLICTS",
    "MAX_INDUCED_DELETION_R",
    "MAX_INDUCED_DELETION_VERTICES",
    "MAX_INDUCED_EDGE_MATERIALIZATION",
    "MAX_INDUCED_LEDGER_CONFLICTS",
    "MAX_INDUCED_RETAINED_LABEL_CHARACTERS",
    "MAX_INDUCED_SOLVER_CALLS",
    "InducedDeletionGraph",
    "InducedDeletionRow",
    "InducedEdgeDeletionProfileRequest",
    "InducedEdgeDeletionProfileResult",
    "PerSizeMaximum",
]
