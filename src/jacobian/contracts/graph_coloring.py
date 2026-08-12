"""Contracts for bounded exact graph-coloring exploration."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator

from jacobian.contracts.common import ArtifactUri, Sha256Digest
from jacobian.contracts.results import ContractModel
from jacobian.contracts.sat import CanonicalCnf

GraphVertex = Annotated[
    str,
    StringConstraints(min_length=1, max_length=256, strict=True),
]


class ChromaticGraph(ContractModel):
    """A bounded simple undirected graph, accepting either edge orientation."""

    graph_schema_version: Literal["1"] = "1"
    vertices: tuple[GraphVertex, ...] = Field(max_length=32)
    edges: tuple[tuple[GraphVertex, GraphVertex], ...] = Field(max_length=496)

    @model_validator(mode="after")
    def require_simple_graph(self) -> Self:
        vertex_set = set(self.vertices)
        if len(vertex_set) != len(self.vertices):
            raise ValueError("graph vertices must be unique")
        normalized_edges = {tuple(sorted((left, right))) for left, right in self.edges}
        if any(left == right for left, right in self.edges):
            raise ValueError("graph edges must not contain self-loops")
        if any(
            left not in vertex_set or right not in vertex_set
            for left, right in self.edges
        ):
            raise ValueError("graph edges must reference declared vertices")
        if len(normalized_edges) != len(self.edges):
            raise ValueError("graph edges must be unique ignoring orientation")
        return self


class GraphColoringEncodingRequest(ContractModel):
    """Materialize the exact CNF semantics of k-colorability."""

    graph: ChromaticGraph
    colors: StrictInt = Field(ge=1, le=32)


class GraphColoringEncodingClaim(ContractModel):
    """Claim that one graph/color-count pair has a canonical encoding."""

    claim_schema_version: Literal["1"] = "1"
    predicate: Literal["GRAPH_K_COLORABILITY_ENCODING"] = (
        "GRAPH_K_COLORABILITY_ENCODING"
    )
    graph: ChromaticGraph
    colors: StrictInt = Field(ge=1, le=32)


class GraphColoringEncodingScope(ContractModel):
    """Graph-owned scope binding the encoding to the SAT CNF artifact."""

    scope_schema_version: Literal["1"] = "1"
    graph: ChromaticGraph
    colors: StrictInt = Field(ge=1, le=32)
    cnf_uri: ArtifactUri
    cnf_object_digest: Sha256Digest
    cnf: CanonicalCnf


class GraphColoringEncodingCandidate(ContractModel):
    """Pointer candidate checked against the graph-owned encoding scope."""

    candidate_schema_version: Literal["1"] = "1"
    cnf_uri: ArtifactUri
    scope_uri: ArtifactUri


class GraphColoringEncodingReplay(ContractModel):
    """Certificate payload for independent graph-to-CNF replay."""

    method: Literal["INDEPENDENT_GRAPH_COLORING_CNF_REPLAY"] = (
        "INDEPENDENT_GRAPH_COLORING_CNF_REPLAY"
    )
    claim_uri: ArtifactUri
    candidate_uri: ArtifactUri
    scope_uri: ArtifactUri


class GraphColoringEncodingOutput(ContractModel):
    """Materialized encoding and its replay certificate."""

    graph: ChromaticGraph
    colors: StrictInt = Field(ge=1, le=32)
    cnf_uri: ArtifactUri
    scope_uri: ArtifactUri
    claim_uri: ArtifactUri
    candidate_uri: ArtifactUri
    certificate_uri: ArtifactUri
    variable_count: StrictInt = Field(ge=0, le=1_000_000)
    clause_count: StrictInt = Field(ge=0, le=1_000_000)
    encoding_version: Literal["exactly-one-and-edge-separation/v1"] = (
        "exactly-one-and-edge-separation/v1"
    )


class ChromaticNumberBudget(ContractModel):
    """Total wall-clock budget for the bounded coloring search."""

    wall_seconds: StrictInt = Field(default=5, ge=1, le=120)


class GraphChromaticNumberRequest(ContractModel):
    """Request one bounded exact chromatic-number exploration."""

    graph: ChromaticGraph
    resource_budget: ChromaticNumberBudget = Field(
        default_factory=ChromaticNumberBudget
    )


class ChromaticSearchStep(ContractModel):
    """One k-colorability decision made by the solver."""

    colors: StrictInt = Field(ge=1, le=32)
    status: Literal["SATISFIABLE", "UNSATISFIABLE", "UNKNOWN"]


class GraphChromaticNumberOutput(ContractModel):
    """Exact result or bounded non-conclusion with inspectable evidence."""

    status: Literal["EXACT", "UNKNOWN"]
    vertices: tuple[GraphVertex, ...]
    order: StrictInt = Field(ge=0, le=32)
    chromatic_number: StrictInt | None = Field(default=None, ge=0, le=32)
    lower_bound: StrictInt = Field(ge=0, le=32)
    upper_bound: StrictInt = Field(ge=0, le=32)
    coloring: dict[GraphVertex, StrictInt] | None = None
    solver_status: Literal["SATISFIABLE", "UNKNOWN", "SPECIAL_CASE"]
    tested: tuple[ChromaticSearchStep, ...]
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_result_status(self) -> Self:
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("result vertices must be unique")
        if self.order != len(self.vertices):
            raise ValueError("result order must match the vertex list")
        if self.lower_bound > self.upper_bound:
            raise ValueError("chromatic bounds must be ordered")
        if self.coloring is not None and set(self.coloring) != set(self.vertices):
            raise ValueError("coloring must assign every result vertex")
        if self.coloring is not None and any(
            color < 0 or color >= self.upper_bound for color in self.coloring.values()
        ):
            raise ValueError("coloring values must lie below the upper bound")
        if self.status == "EXACT":
            if (
                self.chromatic_number is None
                or self.lower_bound != self.chromatic_number
                or self.upper_bound != self.chromatic_number
                or self.coloring is None
                or self.solver_status not in {"SATISFIABLE", "SPECIAL_CASE"}
            ):
                raise ValueError("exact result evidence is incomplete")
        elif self.chromatic_number is not None:
            raise ValueError("unknown result cannot carry a chromatic number")
        return self
