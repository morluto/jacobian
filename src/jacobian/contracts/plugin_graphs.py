"""Domain-owned request contracts for graph reference plugins."""

from __future__ import annotations

from itertools import pairwise
from typing import Any, Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from jacobian.contracts.claims import flatten_claim_spec
from jacobian.contracts.plugin_protocol import PluginRequestContext
from jacobian.contracts.results import ContractModel


class GraphPathClaim(ContractModel):
    predicate: Literal["intended_paths_complete", "is_bipartite"]
    simple: StrictBool | None = None
    max_path_length: StrictInt | None = Field(default=None, ge=1)

    @model_validator(mode="before")
    @classmethod
    def flatten_generic_claim(cls, value: object) -> object:
        return flatten_claim_spec(value)

    @model_validator(mode="after")
    def validate_predicate_fields(self) -> Self:
        if self.predicate == "intended_paths_complete" and self.simple is not True:
            raise ValueError("intended_paths_complete requires simple=True")
        if self.predicate == "is_bipartite" and (
            self.simple is not None or self.max_path_length is not None
        ):
            raise ValueError("is_bipartite does not accept path-bound fields")
        return self


class GraphPathCandidate(ContractModel):
    vertices: tuple[str, ...] = Field(min_length=1, max_length=256)
    arcs: tuple[tuple[str, str], ...] = Field(max_length=65_536)
    source: str | None = None
    terminals: tuple[str, ...] | None = Field(default=None, min_length=1)
    intended_paths: tuple[tuple[str, ...], ...] | None = None

    @model_validator(mode="after")
    def validate_graph_shape(self) -> Self:
        vertices = set(self.vertices)
        if len(vertices) != len(self.vertices):
            raise ValueError("vertices must be unique strings")
        if any(
            left not in vertices or right not in vertices for left, right in self.arcs
        ):
            raise ValueError("arcs must reference declared vertices")
        if len(set(self.arcs)) != len(self.arcs):
            raise ValueError("arcs must be unique")
        if self.source is not None and self.source not in vertices:
            raise ValueError("source is not a graph vertex")
        if self.terminals is not None and (
            not set(self.terminals) <= vertices
            or len(set(self.terminals)) != len(self.terminals)
        ):
            raise ValueError("terminals must be unique graph vertices")
        arc_set = set(self.arcs)
        terminals = self.terminals or ()
        for path in self.intended_paths or ():
            if (
                len(path) < 2
                or len(set(path)) != len(path)
                or any(vertex not in vertices for vertex in path)
                or self.source is None
                or path[0] != self.source
                or path[-1] not in terminals
                or any(pair not in arc_set for pair in pairwise(path))
            ):
                raise ValueError(
                    "intended_paths must contain legal source-terminal paths"
                )
        return self


class GraphPathCapabilityRequest(PluginRequestContext):
    claim: GraphPathClaim
    candidate: GraphPathCandidate
    witness_role: Literal["DEFEATS_CANDIDATE", "SUPPORTS_CLAIM"] = "DEFEATS_CANDIDATE"

    @model_validator(mode="after")
    def require_path_roles(self) -> Self:
        if self.claim.predicate == "intended_paths_complete":
            if self.candidate.source is None or not self.candidate.terminals:
                raise ValueError(
                    "intended_paths_complete requires source and terminals"
                )
            if self.candidate.intended_paths is None:
                raise ValueError(
                    "intended_paths_complete requires an explicit intended_paths set"
                )
            if self.candidate.source in self.candidate.terminals:
                raise ValueError("source cannot also be a terminal")
        return self


class GraphPathReductionRequest(PluginRequestContext):
    target_kind: Literal["candidate", "witness"] = "candidate"
    target: GraphPathCandidate
    claim: GraphPathClaim
    reducers: tuple[Literal["delete_vertex", "delete_edge"], ...] = ()
    objectives: tuple[Literal["vertices", "edges"], ...] = ()

    @model_validator(mode="after")
    def require_path_target_for_candidate(self) -> Self:
        if (
            self.target_kind == "candidate"
            and self.claim.predicate == "intended_paths_complete"
            and (self.target.source is None or not self.target.terminals)
        ):
            raise ValueError("path claims require source and terminals")
        if (
            self.target_kind == "candidate"
            and self.claim.predicate == "intended_paths_complete"
            and self.target.intended_paths is None
        ):
            raise ValueError("path claims require an explicit intended_paths set")
        if (
            self.target_kind == "candidate"
            and self.claim.predicate == "intended_paths_complete"
            and self.target.terminals is not None
            and self.target.source in self.target.terminals
        ):
            raise ValueError("source cannot also be a terminal")
        return self


class GraphCursor(ContractModel):
    offset: StrictInt = Field(ge=0)


class GraphEnumerationRequest(PluginRequestContext):
    bounds: dict[str, StrictInt]
    page_size: StrictInt = Field(ge=1, le=256)
    cursor: GraphCursor | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if set(self.bounds) != {"vertices"} or not 1 <= self.bounds["vertices"] <= 8:
            raise ValueError("graph bounds require vertices from one through eight")
        return self


class GraphCanonicalizeRequest(PluginRequestContext):
    structure: GraphPathCandidate


class GraphShrinkTarget(ContractModel):
    graph_schema_version: Literal["1"] = "1"
    vertices: tuple[str, ...] = Field(min_length=1, max_length=256)
    edges: tuple[tuple[str, str], ...] = Field(max_length=32_640)

    @model_validator(mode="after")
    def validate_simple_graph(self) -> Self:
        vertices = set(self.vertices)
        if len(vertices) != len(self.vertices):
            raise ValueError("graph vertices must be unique")
        if any(left == right for left, right in self.edges):
            raise ValueError("graph edges must not contain self-loops")
        if any(
            left not in vertices or right not in vertices for left, right in self.edges
        ):
            raise ValueError("graph edges must reference declared vertices")
        normalized = {tuple(sorted(edge)) for edge in self.edges}
        if len(normalized) != len(self.edges):
            raise ValueError("graph edges must be unique ignoring orientation")
        return self


class GraphShrinkRequest(PluginRequestContext):
    target_kind: Literal["candidate"] = "candidate"
    target: GraphShrinkTarget
    reducers: tuple[Literal["delete_vertex", "delete_edge"], ...]
    objectives: tuple[Literal["vertices", "edges"], ...]
    claim: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "GraphCanonicalizeRequest",
    "GraphCursor",
    "GraphEnumerationRequest",
    "GraphPathCandidate",
    "GraphPathCapabilityRequest",
    "GraphPathClaim",
    "GraphPathReductionRequest",
    "GraphShrinkRequest",
    "GraphShrinkTarget",
]
