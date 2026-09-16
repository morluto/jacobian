"""Typed wire contracts for the deterministic minor-model checker."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import (
    MAX_SIMPLE_GRAPH_EDGES,
    MAX_SIMPLE_GRAPH_VERTICES,
    SimpleUndirectedGraph,
)

MAX_MINOR_BRANCH_MEMBERSHIPS = 4096

MinorModelStatus = Literal["VALID_MINOR_MODEL", "INVALID_MINOR_MODEL"]

MinorModelObstructionCode = Literal[
    "DUPLICATE_BRANCH_SET",
    "MISSING_BRANCH_SET",
    "UNKNOWN_BRANCH_TARGET",
    "EMPTY_BRANCH_SET",
    "UNDECLARED_SOURCE_VERTEX",
    "OVERLAPPING_BRANCH_SETS",
    "DISCONNECTED_BRANCH_SET",
    "MISSING_WITNESS",
    "SUPERFLUOUS_WITNESS",
    "WITNESS_EDGE_ABSENT",
    "WITNESS_ENDPOINT_MISMATCH",
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"graph.{reason}", message)


class BranchSet(StrictModel):
    """One candidate branch set bound to a target vertex."""

    target: str = Field(min_length=1)
    members: tuple[str, ...] = Field(
        default=(),
        description="Candidate source vertices; empty is an INVALID model, not wire-invalid.",
    )

    @model_validator(mode="after")
    def require_canonical_members(self) -> Self:
        if self.members != tuple(sorted(set(self.members))):
            raise _validation_error(
                "branch_members_not_canonical",
                "branch members must be unique and sorted",
            )
        return self


class EdgeWitness(StrictModel):
    """One candidate source edge witnessing a target edge."""

    targets: tuple[str, str] = Field(
        description="Target-edge endpoints in lexicographic label order."
    )
    source_edge: tuple[str, str] = Field(
        description="Source-edge endpoints in lexicographic label order."
    )

    @model_validator(mode="after")
    def require_canonical_witness(self) -> Self:
        if (
            tuple(sorted(self.targets)) != self.targets
            or self.targets[0] == self.targets[1]
        ):
            raise _validation_error(
                "witness_targets_not_canonical",
                "witness targets must be two distinct labels in order",
            )
        if (
            tuple(sorted(self.source_edge)) != self.source_edge
            or self.source_edge[0] == self.source_edge[1]
        ):
            raise _validation_error(
                "witness_edge_not_canonical",
                "witness source edges must be two distinct labels in order",
            )
        return self


class BranchConnectivityLedger(StrictModel):
    """Connectivity evidence for one checked branch set."""

    target: str = Field(min_length=1)
    member_count: int = Field(ge=1)
    internal_edge_count: int = Field(ge=0)
    root: str = Field(min_length=1)


class MinorModelCheckRequest(StrictModel):
    """Check one candidate H-minor model in G."""

    source: SimpleUndirectedGraph = Field(
        description="Source graph G holding the candidate branch sets."
    )
    target: SimpleUndirectedGraph = Field(
        description="Target graph H whose vertices and edges the model covers."
    )
    branch_sets: tuple[BranchSet, ...] = Field(
        max_length=MAX_SIMPLE_GRAPH_VERTICES,
        description="One candidate branch set per target vertex.",
    )
    witnesses: tuple[EdgeWitness, ...] = Field(
        max_length=MAX_SIMPLE_GRAPH_EDGES,
        description="One candidate source edge per target edge.",
    )

    @model_validator(mode="after")
    def require_canonical_candidate(self) -> Self:
        if len({branch.target for branch in self.branch_sets}) != len(self.branch_sets):
            raise _validation_error(
                "branch_targets_not_unique", "branch targets must be unique"
            )
        if len({witness.targets for witness in self.witnesses}) != len(self.witnesses):
            raise _validation_error(
                "witness_targets_not_unique", "witness targets must be unique"
            )
        return self


class MinorModelCheckResult(StrictModel):
    """Deterministic minor-model verdict with ledgers or first obstruction."""

    source: SimpleUndirectedGraph
    target: SimpleUndirectedGraph
    status: MinorModelStatus
    branch_sets: tuple[BranchSet, ...]
    witnesses: tuple[EdgeWitness, ...]
    connectivity: tuple[BranchConnectivityLedger, ...] = ()
    used_source_vertices: tuple[str, ...] = ()
    deleted_source_vertices: tuple[str, ...] = ()
    used_source_edges: tuple[tuple[str, str], ...] = ()
    deleted_source_edges: tuple[tuple[str, str], ...] = ()
    obstruction_code: MinorModelObstructionCode | None = None
    obstruction_detail: str | None = None

    @model_validator(mode="after")
    def require_branch_consistency(self) -> Self:
        if (self.obstruction_code is None) != (self.obstruction_detail is None):
            raise _validation_error(
                "obstruction_payload",
                "obstruction code and detail must agree",
            )
        if self.status == "VALID_MINOR_MODEL":
            if self.obstruction_code is not None:
                raise _validation_error(
                    "valid_payload", "a valid model carries no obstruction"
                )
            if {ledger.target for ledger in self.connectivity} != {
                branch.target for branch in self.branch_sets
            }:
                raise _validation_error(
                    "connectivity_coverage",
                    "connectivity must cover every branch set",
                )
        elif self.obstruction_code is None:
            raise _validation_error(
                "invalid_payload", "an invalid model carries its first obstruction"
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = [
    "MAX_MINOR_BRANCH_MEMBERSHIPS",
    "BranchConnectivityLedger",
    "BranchSet",
    "EdgeWitness",
    "MinorModelCheckRequest",
    "MinorModelCheckResult",
    "MinorModelObstructionCode",
    "MinorModelStatus",
]
