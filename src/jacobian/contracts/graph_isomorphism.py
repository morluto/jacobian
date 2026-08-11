"""Direct finite-graph isomorphism verification contracts."""

from __future__ import annotations

import unicodedata
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest
from jacobian.contracts.results import ContractModel


class SimpleUndirectedGraph(ContractModel):
    graph_schema_version: Literal["1"] = "1"
    vertices: tuple[str, ...] = Field(max_length=256)
    edges: tuple[tuple[str, str], ...] = Field(max_length=32640)

    @model_validator(mode="after")
    def require_canonical_simple_graph(self) -> Self:
        if any(
            not unicodedata.is_normalized("NFC", vertex) for vertex in self.vertices
        ):
            raise ValueError("graph vertices must use Unicode NFC")
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("graph vertices must be unique")
        if any(
            left >= right or left not in self.vertices or right not in self.vertices
            for left, right in self.edges
        ):
            raise ValueError("edges must contain two declared vertices in order")
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("graph edges must be unique")
        return self


class GraphPair(ContractModel):
    pair_schema_version: Literal["1"] = "1"
    left_graph_uri: ArtifactUri
    right_graph_uri: ArtifactUri
    left_graph_digest: Sha256Digest
    right_graph_digest: Sha256Digest
    graph_schema_uri: ArtifactUri
    graph_semantics_uri: ArtifactUri
    left: SimpleUndirectedGraph
    right: SimpleUndirectedGraph


class GraphVertexMapping(ContractModel):
    mapping_schema_version: Literal["1"] = "1"
    mapping: dict[str, str] = Field(max_length=256)


class GraphIsomorphismVerifyRequest(ContractModel):
    left_graph_uri: ArtifactUri
    right_graph_uri: ArtifactUri
    mapping: dict[str, str] = Field(max_length=256)


class GraphIsomorphismClaim(ContractModel):
    claim_schema_version: Literal["1"] = "1"
    predicate: Literal["MAPPING_IS_GRAPH_ISOMORPHISM"] = "MAPPING_IS_GRAPH_ISOMORPHISM"
    graph_pair_uri: ArtifactUri
    mapping_uri: ArtifactUri


class GraphIsomorphismReplay(ContractModel):
    method: Literal["DIRECT_ADJACENCY_REPLAY"] = "DIRECT_ADJACENCY_REPLAY"
    graph_pair_uri: ArtifactUri
    mapping_uri: ArtifactUri
    left_graph_uri: ArtifactUri
    right_graph_uri: ArtifactUri
    left_graph_digest: Sha256Digest
    right_graph_digest: Sha256Digest
    graph_schema_uri: ArtifactUri
    graph_semantics_uri: ArtifactUri


class GraphIsomorphismVerifyOutput(ContractModel):
    is_isomorphism: bool | None
    conclusion: Literal["TRUE", "FALSE", "UNKNOWN"]
    left_graph_uri: ArtifactUri
    right_graph_uri: ArtifactUri
    graph_pair_uri: ArtifactUri
    mapping_uri: ArtifactUri
    claim_uri: ArtifactUri
    certificate_uri: ArtifactUri
    verification_record_uri: ArtifactUri | None = None
    checker_id: CheckerUri | None = None
    coverage: Literal["EXHAUSTIVE", "UNKNOWN"]

    @model_validator(mode="after")
    def preserve_truth_and_checker_assurance(self) -> Self:
        expected = {
            "TRUE": True,
            "FALSE": False,
            "UNKNOWN": None,
        }[self.conclusion]
        if self.is_isomorphism is not expected:
            raise ValueError("is_isomorphism must preserve TRUE, FALSE, and UNKNOWN")
        if self.coverage == "EXHAUSTIVE":
            if (
                self.conclusion == "UNKNOWN"
                or self.verification_record_uri is None
                or self.checker_id is None
            ):
                raise ValueError(
                    "exhaustive isomorphism verification requires a decisive "
                    "checker-backed record"
                )
        elif self.conclusion != "UNKNOWN" or self.verification_record_uri is not None:
            raise ValueError(
                "unknown coverage cannot carry a conclusion or verification record"
            )
        return self
