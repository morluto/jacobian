"""Typed contracts for edge-clique partition checking."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import (
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    SimpleUndirectedGraph,
)

# Candidate parts are vertex subsets; pair work per part is quadratic in its
# size. Both the part count and the aggregate pair checks are bounded before
# any adjacency expansion.
MAX_PARTITION_PARTS = 4_096
MAX_PARTITION_PAIR_WORK = 1_000_000


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


class EdgeCliquePartitionRequest(StrictModel):
    """Check a supplied family of vertex subsets as an edge-clique partition.

    Every part must hold at least two distinct declared vertices; empty,
    singleton, duplicate-member, and foreign-vertex parts are malformed
    requests. Duplicate parts (identical member sets) are not malformed:
    they cover their edges twice and fail as an invalid partition.
    """

    graph: SimpleUndirectedGraph
    parts: tuple[tuple[str, ...], ...] = Field(
        max_length=MAX_PARTITION_PARTS,
        description=(
            "Candidate parts as vertex subsets in any order. Each part must "
            "hold at least two distinct declared vertices."
        ),
    )

    @model_validator(mode="after")
    def require_well_formed_parts(self) -> Self:
        carrier = set(self.graph.vertices)
        if len(self.parts) > MAX_PARTITION_PARTS:
            raise _validation_error(
                "graph.clique_partition.too_many_parts",
                f"edge-clique partitions admit at most {MAX_PARTITION_PARTS} parts",
            )
        pair_work = 0
        for part in self.parts:
            if len(part) < 2:
                raise _validation_error(
                    "graph.clique_partition.part_too_small",
                    "every partition part must hold at least two vertices",
                )
            if len(set(part)) != len(part):
                raise _validation_error(
                    "graph.clique_partition.part_members_not_unique",
                    "every partition part must hold distinct vertices",
                )
            unknown = set(part) - carrier
            if unknown:
                raise _validation_error(
                    "graph.clique_partition.part_vertex_unknown",
                    "every partition part must use declared graph vertices",
                )
            if len(part) > MAX_INDEXED_SIMPLE_GRAPH_VERTICES:
                raise _validation_error(
                    "graph.clique_partition.part_too_large",
                    "partition parts cannot exceed the graph vertex bound",
                )
            pair_work += len(part) * (len(part) - 1) // 2
            if pair_work > MAX_PARTITION_PAIR_WORK:
                raise _validation_error(
                    "graph.clique_partition.pair_work_bound",
                    "partition parts exceed the "
                    f"{MAX_PARTITION_PAIR_WORK:,}-pair checking bound",
                )
        return self


PartitionVerdict = Literal["VALID", "INVALID"]


class EdgeCliquePartitionResult(StrictModel):
    """The verdict on a supplied edge-clique partition with one failure.

    A valid partition has every part a clique of order at least two and
    every graph edge in exactly one part. An invalid result carries exactly
    one failure class, prioritized as non-clique part, uncovered edge, then
    overcovered edge; endpoints use lexicographic label order.
    """

    graph: SimpleUndirectedGraph
    parts: tuple[tuple[str, ...], ...] = Field(max_length=MAX_PARTITION_PARTS)
    is_partition: bool
    verdict: PartitionVerdict
    failing_part: StrictInt | None = Field(default=None, ge=0)
    failing_nonedge: tuple[str, str] | None = None
    uncovered_edge: tuple[str, str] | None = None
    overcovered_edge: tuple[str, str] | None = None
    overcovering_parts: tuple[StrictInt, ...] = ()

    @model_validator(mode="after")
    def require_coherent_verdict(self) -> Self:
        if self.is_partition != (self.verdict == "VALID"):
            raise _validation_error(
                "graph.clique_partition.verdict_mismatch",
                "is_partition must agree with the VALID verdict",
            )
        failures = [
            self.failing_part is not None,
            self.uncovered_edge is not None,
            self.overcovered_edge is not None,
        ]
        if self.verdict == "VALID":
            if any(failures) or self.overcovering_parts:
                raise _validation_error(
                    "graph.clique_partition.valid_must_carry_no_failure",
                    "a valid partition carries no failure detail",
                )
        elif sum(failures) != 1:
            raise _validation_error(
                "graph.clique_partition.invalid_needs_one_failure",
                "an invalid partition carries exactly one failure class",
            )
        if self.failing_part is not None and not (
            0 <= self.failing_part < len(self.parts)
        ):
            raise _validation_error(
                "graph.clique_partition.failing_part_out_of_range",
                "the failing part index must identify a supplied part",
            )
        if self.failing_nonedge is not None and (
            self.failing_nonedge[0] >= self.failing_nonedge[1]
        ):
            raise _validation_error(
                "graph.clique_partition.nonedge_order",
                "the failing nonedge must use lexicographic label order",
            )
        for edge in (self.uncovered_edge, self.overcovered_edge):
            if edge is not None and (
                edge[0] >= edge[1] or edge not in set(self.graph.edges)
            ):
                raise _validation_error(
                    "graph.clique_partition.coverage_edge_invalid",
                    "a coverage failure must name a canonical graph edge",
                )
        if self.overcovered_edge is None and self.overcovering_parts:
            raise _validation_error(
                "graph.clique_partition.overcovering_parts_orphaned",
                "overcovering parts require their overcovered edge",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: SimpleUndirectedGraph,
        parts: tuple[tuple[str, ...], ...],
        is_partition: bool,
        failing_part: int | None = None,
        failing_nonedge: tuple[str, str] | None = None,
        uncovered_edge: tuple[str, str] | None = None,
        overcovered_edge: tuple[str, str] | None = None,
        overcovering_parts: tuple[int, ...] = (),
    ) -> Self:
        """Construct a verdict established by the owner-local kernel."""

        return cls.model_construct(
            graph=graph,
            parts=parts,
            is_partition=is_partition,
            verdict="VALID" if is_partition else "INVALID",
            failing_part=failing_part,
            failing_nonedge=failing_nonedge,
            uncovered_edge=uncovered_edge,
            overcovered_edge=overcovered_edge,
            overcovering_parts=overcovering_parts,
        )


__all__ = [
    "MAX_PARTITION_PAIR_WORK",
    "MAX_PARTITION_PARTS",
    "EdgeCliquePartitionRequest",
    "EdgeCliquePartitionResult",
]
