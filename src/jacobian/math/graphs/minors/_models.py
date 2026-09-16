"""Typed wire contracts for the deterministic minor-model checker."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, StrictInt, model_validator
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
    "MAX_MINOR_FIND_CANDIDATES",
    "MAX_MINOR_FIND_SOURCE_VERTICES",
    "MAX_MINOR_FIND_TARGET_VERTICES",
    "MAX_TOPO_FIND_CANDIDATES",
    "MAX_TOPO_FIND_SOURCE_VERTICES",
    "MAX_TOPO_FIND_TARGET_VERTICES",
    "BranchConnectivityLedger",
    "BranchSet",
    "BranchVertex",
    "EdgeWitness",
    "MinorModelCheckRequest",
    "MinorModelCheckResult",
    "MinorModelFindBudget",
    "MinorModelFindRequest",
    "MinorModelFindResult",
    "MinorModelFindStatus",
    "MinorModelFindTermination",
    "MinorModelObstructionCode",
    "MinorModelStatus",
    "SubdivisionPath",
    "TopologicalMinorCheckRequest",
    "TopologicalMinorCheckResult",
    "TopologicalMinorFindBudget",
    "TopologicalMinorFindRequest",
    "TopologicalMinorFindResult",
    "TopologicalMinorObstructionCode",
    "TopologicalMinorStatus",
]


MAX_MINOR_FIND_SOURCE_VERTICES = 8
"""Source-order envelope for the bounded H-minor-model search.

Pure-Python backtracking ranges over ``(target_order + 1) ** source_order``
branch assignments, so the source order owns the dominant work factor.
"""

MAX_MINOR_FIND_TARGET_VERTICES = 5
"""Target-order envelope for the bounded H-minor-model search."""

MAX_MINOR_FIND_CANDIDATES = 200_000
"""Ceiling for branch-assignment candidates one minor-model search may test."""

DEFAULT_MINOR_FIND_CANDIDATES = 50_000
"""Default branch-assignment budget for the H-minor-model search."""

MinorModelFindStatus = Literal["FOUND", "EXHAUSTED", "UNKNOWN"]

MinorModelFindTermination = Literal[
    "WITNESS_FOUND",
    "SEARCH_EXHAUSTED",
    "CANDIDATE_BUDGET_EXCEEDED",
]


class MinorModelFindBudget(StrictModel):
    """Explicit public limits for one bounded H-minor-model search."""

    max_candidates: StrictInt = Field(
        default=DEFAULT_MINOR_FIND_CANDIDATES,
        ge=1,
        le=MAX_MINOR_FIND_CANDIDATES,
        description=(
            "Maximum number of branch assignments the search may test before "
            "declaring UNKNOWN. A truncated search never reports EXHAUSTED."
        ),
    )


class MinorModelFindRequest(StrictModel):
    """Search for an H-minor model of target H in source G.

    The bounded backtracking search ranges over branch assignments of the
    source vertices to the target vertices (plus deletion) and returns
    FOUND with a checked witness, EXHAUSTED when the complete assignment
    space was covered and no model exists, or UNKNOWN when the candidate
    budget was exceeded first.
    """

    source: SimpleUndirectedGraph = Field(
        description="Source graph G holding the searched branch sets."
    )
    target: SimpleUndirectedGraph = Field(
        description="Target graph H whose minor model is searched."
    )
    resource_budget: MinorModelFindBudget = Field(default_factory=MinorModelFindBudget)


class MinorModelFindResult(StrictModel):
    """Outcome of a bounded H-minor-model search, bound to its request."""

    source: SimpleUndirectedGraph
    target: SimpleUndirectedGraph
    resource_budget: MinorModelFindBudget
    status: MinorModelFindStatus
    branch_sets: tuple[BranchSet, ...] = ()
    witnesses: tuple[EdgeWitness, ...] = ()
    candidates_enumerated: StrictInt = Field(
        default=0,
        ge=0,
        description=(
            "Branch assignments tested. EXHAUSTED covers the complete "
            "assignment space; UNKNOWN stopped at the candidate budget."
        ),
    )
    termination_reason: MinorModelFindTermination = "SEARCH_EXHAUSTED"

    @model_validator(mode="after")
    def require_consistent_find_status(self) -> Self:
        if self.status == "FOUND":
            if not self.branch_sets:
                raise _validation_error(
                    "found_requires_branch_sets",
                    "FOUND status requires branch sets",
                )
            if self.termination_reason != "WITNESS_FOUND":
                raise _validation_error(
                    "found_requires_witness_found",
                    "FOUND status requires WITNESS_FOUND reason",
                )
        else:
            if self.branch_sets or self.witnesses:
                raise _validation_error(
                    "unresolved_requires_no_witness",
                    "EXHAUSTED and UNKNOWN statuses carry no witness",
                )
            if self.status == "EXHAUSTED":
                if self.termination_reason != "SEARCH_EXHAUSTED":
                    raise _validation_error(
                        "exhausted_requires_search_exhausted",
                        "EXHAUSTED status requires SEARCH_EXHAUSTED reason",
                    )
            elif self.termination_reason != "CANDIDATE_BUDGET_EXCEEDED":
                raise _validation_error(
                    "unknown_requires_budget_exceeded",
                    "UNKNOWN status requires CANDIDATE_BUDGET_EXCEEDED reason",
                )
        if self.candidates_enumerated > self.resource_budget.max_candidates:
            raise _validation_error(
                "candidates_exceed_budget",
                "candidates enumerated exceeds the resource budget",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


MAX_TOPO_FIND_SOURCE_VERTICES = 8
"""Source-order envelope for the bounded subdivision search."""

MAX_TOPO_FIND_TARGET_VERTICES = 5
"""Target-order envelope for the bounded subdivision search."""

MAX_TOPO_FIND_CANDIDATES = 200_000
"""Ceiling for search probes one subdivision search may spend."""

DEFAULT_TOPO_FIND_CANDIDATES = 50_000
"""Default probe budget for the subdivision search."""

TopologicalMinorStatus = Literal["VALID_SUBDIVISION", "INVALID_SUBDIVISION"]

TopologicalMinorObstructionCode = Literal[
    "DUPLICATE_BRANCH_TARGET",
    "MISSING_BRANCH_VERTEX",
    "UNKNOWN_BRANCH_TARGET",
    "UNDECLARED_SOURCE_VERTEX",
    "DUPLICATE_BRANCH_VERTEX",
    "MISSING_PATH_WITNESS",
    "SUPERFLUOUS_PATH_WITNESS",
    "NON_SIMPLE_PATH",
    "UNDECLARED_PATH_VERTEX",
    "PATH_ENDPOINT_MISMATCH",
    "PATH_EDGE_ABSENT",
    "PATH_INTERNAL_INTERSECTION",
]


class BranchVertex(StrictModel):
    """One branch vertex binding a target vertex to a source vertex."""

    target: str = Field(min_length=1)
    source: str = Field(min_length=1)


class SubdivisionPath(StrictModel):
    """One candidate source path witnessing a target edge.

    ``vertices`` runs from the branch vertex of ``targets[0]`` to the
    branch vertex of ``targets[1]``; a reversed row is a
    PATH_ENDPOINT_MISMATCH, not a canonical reorientation.
    """

    targets: tuple[str, str] = Field(
        description="Target-edge endpoints in lexicographic label order."
    )
    vertices: tuple[str, ...] = Field(
        min_length=2,
        description=(
            "Source-vertex path from the branch vertex of targets[0] to the "
            "branch vertex of targets[1]. Repeated vertices are an INVALID "
            "subdivision, not wire-invalid."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_path_targets(self) -> Self:
        if (
            tuple(sorted(self.targets)) != self.targets
            or self.targets[0] == self.targets[1]
        ):
            raise _validation_error(
                "path_targets_not_canonical",
                "path targets must be two distinct labels in order",
            )
        return self


class TopologicalMinorCheckRequest(StrictModel):
    """Check one candidate subdivision model of target H in source G."""

    source: SimpleUndirectedGraph = Field(
        description="Source graph G holding the candidate subdivision."
    )
    target: SimpleUndirectedGraph = Field(
        description="Target graph H whose subdivision is checked."
    )
    branch_vertices: tuple[BranchVertex, ...] = Field(
        max_length=MAX_SIMPLE_GRAPH_VERTICES,
        description="One candidate branch vertex per target vertex.",
    )
    paths: tuple[SubdivisionPath, ...] = Field(
        max_length=MAX_SIMPLE_GRAPH_EDGES,
        description="One candidate source path per target edge.",
    )

    @model_validator(mode="after")
    def require_canonical_subdivision_candidate(self) -> Self:
        if len({row.target for row in self.branch_vertices}) != len(
            self.branch_vertices
        ):
            raise _validation_error(
                "branch_targets_not_unique", "branch targets must be unique"
            )
        if len({path.targets for path in self.paths}) != len(self.paths):
            raise _validation_error(
                "path_targets_not_unique", "path targets must be unique"
            )
        return self


class TopologicalMinorCheckResult(StrictModel):
    """Deterministic subdivision verdict with ledgers or first obstruction."""

    source: SimpleUndirectedGraph
    target: SimpleUndirectedGraph
    status: TopologicalMinorStatus
    branch_vertices: tuple[BranchVertex, ...]
    paths: tuple[SubdivisionPath, ...]
    used_source_vertices: tuple[str, ...] = ()
    deleted_source_vertices: tuple[str, ...] = ()
    used_source_edges: tuple[tuple[str, str], ...] = ()
    deleted_source_edges: tuple[tuple[str, str], ...] = ()
    obstruction_code: TopologicalMinorObstructionCode | None = None
    obstruction_detail: str | None = None

    @model_validator(mode="after")
    def require_subdivision_consistency(self) -> Self:
        if (self.obstruction_code is None) != (self.obstruction_detail is None):
            raise _validation_error(
                "obstruction_payload",
                "obstruction code and detail must agree",
            )
        if self.status == "VALID_SUBDIVISION":
            if self.obstruction_code is not None:
                raise _validation_error(
                    "valid_payload", "a valid subdivision carries no obstruction"
                )
        elif self.obstruction_code is None:
            raise _validation_error(
                "invalid_payload",
                "an invalid subdivision carries its first obstruction",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class TopologicalMinorFindBudget(StrictModel):
    """Explicit public limits for one bounded subdivision search."""

    max_candidates: StrictInt = Field(
        default=DEFAULT_TOPO_FIND_CANDIDATES,
        ge=1,
        le=MAX_TOPO_FIND_CANDIDATES,
        description=(
            "Maximum number of search probes (branch injections plus candidate "
            "path probes) the search may spend before declaring UNKNOWN. A "
            "truncated search never reports EXHAUSTED."
        ),
    )


class TopologicalMinorFindRequest(StrictModel):
    """Search for a subdivision of target H in source G.

    The bounded search ranges over injective branch-vertex maps and routes
    internally vertex-disjoint source paths for the target edges. It returns
    FOUND with a checked witness, EXHAUSTED when the complete search space
    was covered and no subdivision exists, or UNKNOWN when the probe budget
    was exceeded first.
    """

    source: SimpleUndirectedGraph = Field(
        description="Source graph G searched for the subdivision."
    )
    target: SimpleUndirectedGraph = Field(
        description="Target graph H whose subdivision is searched."
    )
    resource_budget: TopologicalMinorFindBudget = Field(
        default_factory=TopologicalMinorFindBudget
    )


class TopologicalMinorFindResult(StrictModel):
    """Outcome of a bounded subdivision search, bound to its request."""

    source: SimpleUndirectedGraph
    target: SimpleUndirectedGraph
    resource_budget: TopologicalMinorFindBudget
    status: MinorModelFindStatus
    branch_vertices: tuple[BranchVertex, ...] = ()
    paths: tuple[SubdivisionPath, ...] = ()
    candidates_enumerated: StrictInt = Field(
        default=0,
        ge=0,
        description=(
            "Search probes spent (branch injections plus candidate path "
            "probes). EXHAUSTED covers the complete search space; UNKNOWN "
            "stopped at the probe budget."
        ),
    )
    termination_reason: MinorModelFindTermination = "SEARCH_EXHAUSTED"

    @model_validator(mode="after")
    def require_consistent_topo_find_status(self) -> Self:
        if self.status == "FOUND":
            if not self.branch_vertices:
                raise _validation_error(
                    "found_requires_branch_vertices",
                    "FOUND status requires branch vertices",
                )
            if self.termination_reason != "WITNESS_FOUND":
                raise _validation_error(
                    "found_requires_witness_found",
                    "FOUND status requires WITNESS_FOUND reason",
                )
        else:
            if self.branch_vertices or self.paths:
                raise _validation_error(
                    "unresolved_requires_no_witness",
                    "EXHAUSTED and UNKNOWN statuses carry no witness",
                )
            if self.status == "EXHAUSTED":
                if self.termination_reason != "SEARCH_EXHAUSTED":
                    raise _validation_error(
                        "exhausted_requires_search_exhausted",
                        "EXHAUSTED status requires SEARCH_EXHAUSTED reason",
                    )
            elif self.termination_reason != "CANDIDATE_BUDGET_EXCEEDED":
                raise _validation_error(
                    "unknown_requires_budget_exceeded",
                    "UNKNOWN status requires CANDIDATE_BUDGET_EXCEEDED reason",
                )
        if self.candidates_enumerated > self.resource_budget.max_candidates:
            raise _validation_error(
                "candidates_exceed_budget",
                "probes spent exceeds the resource budget",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)
