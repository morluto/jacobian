"""Typed contracts for exact finite-graph invariant capabilities."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian.contracts.common import ArtifactUri
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.graph_isomorphism import SimpleUndirectedGraph
from jacobian.contracts.results import ContractModel

GraphVertex = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, strict=True),
]
GraphInvariantName = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
        min_length=1,
        max_length=64,
        strict=True,
    ),
]


class GraphAtlasConstraints(ContractModel):
    """Exact predicates accepted by the bounded Graph Atlas search."""

    connected: bool | None = None
    bipartite: bool | None = None
    tree: bool | None = None
    triangle_free: bool | None = None
    minimum_edges: int | None = Field(default=None, ge=0)
    maximum_edges: int | None = Field(default=None, ge=0)
    minimum_degree: int | None = Field(default=None, ge=0)
    maximum_degree: int | None = Field(default=None, ge=0)
    independence_number: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_ordered_ranges(self) -> Self:
        for name, lower, upper in (
            ("edges", self.minimum_edges, self.maximum_edges),
            ("degree", self.minimum_degree, self.maximum_degree),
        ):
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"minimum_{name} cannot exceed maximum_{name}")
        return self


class GraphAtlasSearchRequest(ContractModel):
    order: int = Field(ge=0, le=7)
    constraints: GraphAtlasConstraints
    limit: int = Field(default=10, ge=1, le=100)


class GraphAtlasProperties(ContractModel):
    order: int = Field(ge=0, le=7)
    size: int = Field(ge=0)
    connected: bool
    bipartite: bool
    tree: bool
    degree_sequence: tuple[int, ...]
    minimum_degree: int | None
    maximum_degree: int | None
    triangle_count: int = Field(ge=0)
    independence_number: int = Field(ge=0)


class GraphAtlasCandidate(ContractModel):
    graph_uri: ArtifactUri
    graph: SimpleUndirectedGraph
    properties: GraphAtlasProperties


class GraphAtlasSearchOutput(ContractModel):
    candidates: tuple[GraphAtlasCandidate, ...] = Field(max_length=100)
    match_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    truncated: bool
    scope_uri: ArtifactUri
    backend: Literal["networkx.graph_atlas_g"] = "networkx.graph_atlas_g"
    backend_version: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_consistent_counts(self) -> Self:
        if self.returned_count != len(self.candidates):
            raise ValueError("returned count must match the candidate window")
        if self.returned_count > self.match_count:
            raise ValueError("returned count cannot exceed the total match count")
        if self.truncated != (self.returned_count < self.match_count):
            raise ValueError("truncation must reflect omitted matches")
        return self


class GraphInvariantBatchRequest(ContractModel):
    graph_uri: ArtifactUri
    properties: tuple[GraphInvariantName, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def require_unique_properties(self) -> Self:
        if len(set(self.properties)) != len(self.properties):
            raise ValueError("requested graph invariants must be unique")
        return self


class GraphInvariantResult(ContractModel):
    invariant: GraphInvariantName
    status: Literal["COMPUTED", "NOT_COMPUTED", "NOT_APPLICABLE", "UNSUPPORTED"]
    value: Any = None
    exactness: Literal["EXACT", "NOT_APPLICABLE"]
    backend: str | None = Field(default=None, min_length=1, max_length=128)
    detail: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def require_status_consistency(self) -> Self:
        if self.status == "COMPUTED":
            if self.exactness != "EXACT" or self.backend is None:
                raise ValueError("computed invariants require an exact backend")
            if self.detail is not None:
                raise ValueError("computed invariants cannot carry failure detail")
        else:
            if self.exactness != "NOT_APPLICABLE" or self.detail is None:
                raise ValueError(
                    "unsupported or inapplicable invariants require explicit detail"
                )
            if self.value is not None:
                raise ValueError(
                    "unsupported or inapplicable invariants cannot carry a value"
                )
        if self.status == "UNSUPPORTED" and self.backend is not None:
            raise ValueError("unsupported invariants cannot name a backend")
        return self


class GraphInvariantResultArtifact(ContractModel):
    invariant_result_version: Literal["1"] = "1"
    graph_uri: ArtifactUri
    registry_version: Literal["1"] = "1"
    backend_version: str = Field(min_length=1, max_length=64)
    result: GraphInvariantResult


class GraphInvariantBinding(ContractModel):
    invariant: GraphInvariantName
    artifact_uri: ArtifactUri
    result: GraphInvariantResult

    @model_validator(mode="after")
    def bind_invariant_name(self) -> Self:
        if self.invariant != self.result.invariant:
            raise ValueError("invariant binding must match its result")
        return self


class GraphInvariantBatchArtifact(ContractModel):
    invariant_batch_version: Literal["2"] = "2"
    graph_uri: ArtifactUri
    registry_version: Literal["1"] = "1"
    supported_invariants: tuple[GraphInvariantName, ...]
    requested_invariants: tuple[GraphInvariantName, ...]
    backend_version: str = Field(min_length=1, max_length=64)
    results: tuple[GraphInvariantBinding, ...]
    properties: dict[str, Any]

    @model_validator(mode="after")
    def require_complete_ordered_bindings(self) -> Self:
        if self.supported_invariants != tuple(sorted(set(self.supported_invariants))):
            raise ValueError("supported invariant registry must be unique and sorted")
        if self.requested_invariants != tuple(sorted(set(self.requested_invariants))):
            raise ValueError("requested invariants must be unique and sorted")
        if tuple(binding.invariant for binding in self.results) != (
            self.requested_invariants
        ):
            raise ValueError("batch results must cover requested invariants in order")
        expected_properties = {
            binding.invariant: {
                "value": binding.result.value,
                "exactness": binding.result.exactness,
                "backend": binding.result.backend,
            }
            for binding in self.results
            if binding.result.status == "COMPUTED"
        }
        if self.properties != expected_properties:
            raise ValueError(
                "properties must be the exact compatibility projection of "
                "computed results"
            )
        return self


class GraphInvariantBatchOutput(GraphInvariantBatchArtifact):
    property_artifact_uri: ArtifactUri


class GraphNeighborhoodIndependenceRequest(ContractModel):
    graph_uri: ArtifactUri


class GraphNeighborhoodIndependenceRecord(ContractModel):
    vertex: GraphVertex
    neighborhood: tuple[GraphVertex, ...] = Field(max_length=24)
    independent_set: tuple[GraphVertex, ...] = Field(max_length=24)
    independence_number: int = Field(ge=0, le=24)

    @model_validator(mode="after")
    def require_canonical_witness(self) -> Self:
        if self.neighborhood != tuple(sorted(set(self.neighborhood))):
            raise ValueError("neighborhood labels must be unique and sorted")
        if self.independent_set != tuple(sorted(set(self.independent_set))):
            raise ValueError("independent-set labels must be unique and sorted")
        if not set(self.independent_set) <= set(self.neighborhood):
            raise ValueError("independent-set witness must lie in the neighborhood")
        if len(self.independent_set) != self.independence_number:
            raise ValueError("independent-set witness size must match the optimum")
        return self


class GraphNeighborhoodIndependenceArtifact(ContractModel):
    invariant_schema_version: Literal["1"] = "1"
    graph_uri: ArtifactUri
    records: tuple[GraphNeighborhoodIndependenceRecord, ...] = Field(max_length=256)
    total: int = Field(ge=0)
    average: CanonicalRational
    maximum_neighborhood_order: Literal[24] = 24
    backend: Literal["networkx"] = "networkx"
    backend_version: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_complete_consistent_profile(self) -> Self:
        vertices = tuple(record.vertex for record in self.records)
        if vertices != tuple(sorted(set(vertices))):
            raise ValueError("profile vertices must be unique and sorted")
        if self.total != sum(record.independence_number for record in self.records):
            raise ValueError("profile total must equal the sum of local optima")
        expected_average = (
            Fraction(self.total, len(self.records)) if self.records else Fraction(0)
        )
        if self.average.as_fraction() != expected_average:
            raise ValueError("profile average must equal total divided by graph order")
        return self


class GraphNeighborhoodIndependenceClaim(ContractModel):
    claim_schema_version: Literal["1"] = "1"
    predicate: Literal["EXACT_NEIGHBORHOOD_INDEPENDENCE_PROFILE"] = (
        "EXACT_NEIGHBORHOOD_INDEPENDENCE_PROFILE"
    )
    source_graph_uri: ArtifactUri


class GraphNeighborhoodIndependenceReplayPayload(ContractModel):
    method: Literal["EXACT_STDLIB_BRANCH_AND_BOUND"] = "EXACT_STDLIB_BRANCH_AND_BOUND"
    source_graph_uri: ArtifactUri
    invariant_uri: ArtifactUri


class GraphNeighborhoodIndependenceOutput(ContractModel):
    graph_uri: ArtifactUri
    invariant_uri: ArtifactUri
    claim_uri: ArtifactUri
    certificate_uri: ArtifactUri
    records: tuple[GraphNeighborhoodIndependenceRecord, ...]
    total: int
    average: CanonicalRational
    completeness: Literal["COMPLETE"] = "COMPLETE"

    @model_validator(mode="after")
    def require_consistent_summary(self) -> Self:
        if self.total != sum(record.independence_number for record in self.records):
            raise ValueError("output total must equal the sum of local optima")
        expected_average = (
            Fraction(self.total, len(self.records)) if self.records else Fraction(0)
        )
        if self.average.as_fraction() != expected_average:
            raise ValueError("output average must equal total divided by graph order")
        return self
