"""Typed contracts for simple-graph degree-sequence realization."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.common import ArtifactUri
from jacobian.contracts.results import ContractModel


class GraphDegreeSequenceRequest(ContractModel):
    degree_sequence: tuple[int, ...] = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def require_nonnegative_degrees(self) -> Self:
        if any(degree < 0 for degree in self.degree_sequence):
            raise ValueError("degree sequence entries must be nonnegative")
        return self


class GraphDegreeSequenceObstruction(ContractModel):
    kind: Literal["ODD_SUM", "MAX_DEGREE", "ERDOS_GALLAI"]
    k: int | None = Field(default=None, ge=1, le=512)
    lhs: int | None = Field(default=None, ge=0)
    rhs: int | None = Field(default=None, ge=0)
    index: int | None = Field(default=None, ge=0, le=511)
    degree: int | None = Field(default=None, ge=0)
    order: int | None = Field(default=None, ge=1, le=512)
    degree_sum: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_kind_fields(self) -> Self:
        populated = {
            "k": self.k,
            "lhs": self.lhs,
            "rhs": self.rhs,
            "index": self.index,
            "degree": self.degree,
            "order": self.order,
            "degree_sum": self.degree_sum,
        }
        expected = {
            "ODD_SUM": {"degree_sum"},
            "MAX_DEGREE": {"index", "degree", "order"},
            "ERDOS_GALLAI": {"k", "lhs", "rhs"},
        }[self.kind]
        if {name for name, value in populated.items() if value is not None} != expected:
            raise ValueError("obstruction fields do not match its kind")
        if (
            self.kind == "ODD_SUM"
            and self.degree_sum is not None
            and self.degree_sum % 2 != 1
        ):
            raise ValueError("odd-sum obstruction must contain an odd sum")
        if (
            self.kind == "MAX_DEGREE"
            and self.degree is not None
            and self.order is not None
            and self.degree < self.order
        ):
            raise ValueError("maximum-degree obstruction does not violate order")
        if (
            self.kind == "ERDOS_GALLAI"
            and self.lhs is not None
            and self.rhs is not None
            and self.lhs <= self.rhs
        ):
            raise ValueError("Erdos-Gallai obstruction must violate the inequality")
        return self


class GraphDegreeSequenceClaim(ContractModel):
    claim_schema_version: Literal["1"] = "1"
    predicate: Literal["SIMPLE_GRAPH_DEGREE_SEQUENCE"] = "SIMPLE_GRAPH_DEGREE_SEQUENCE"
    degree_sequence: tuple[int, ...] = Field(min_length=1, max_length=512)


class GraphDegreeSequenceResultArtifact(ContractModel):
    result_schema_version: Literal["1"] = "1"
    degree_sequence: tuple[int, ...] = Field(min_length=1, max_length=512)
    conclusion: Literal["GRAPHICAL", "NON_GRAPHICAL"]
    graph_uri: ArtifactUri | None = None
    graph: dict[str, Any] | None = None
    obstruction: GraphDegreeSequenceObstruction | None = None

    @model_validator(mode="after")
    def require_conclusion_evidence(self) -> Self:
        if self.conclusion == "GRAPHICAL":
            if (
                self.graph_uri is None
                or self.graph is None
                or self.obstruction is not None
            ):
                raise ValueError("graphical result requires only a graph witness")
        elif self.graph_uri is not None or self.graph is not None:
            raise ValueError("non-graphical result cannot carry a graph")
        elif self.obstruction is None:
            raise ValueError("non-graphical result requires an obstruction")
        return self


class GraphDegreeSequenceReplayPayload(ContractModel):
    method: Literal[
        "EXACT_DEGREE_REPLAY",
        "ODD_SUM_OBSTRUCTION",
        "MAX_DEGREE_OBSTRUCTION",
        "ERDOS_GALLAI_OBSTRUCTION",
    ]
    degree_sequence: tuple[int, ...] = Field(min_length=1, max_length=512)
    conclusion: Literal["GRAPHICAL", "NON_GRAPHICAL"]
    graph_uri: ArtifactUri | None = None
    obstruction: GraphDegreeSequenceObstruction | None = None


class GraphDegreeSequenceOutput(ContractModel):
    degree_sequence: tuple[int, ...]
    conclusion: Literal["GRAPHICAL", "NON_GRAPHICAL"]
    graph_uri: ArtifactUri | None = None
    graph: dict[str, Any] | None = None
    obstruction: GraphDegreeSequenceObstruction | None = None
    result_uri: ArtifactUri
    claim_uri: ArtifactUri
    certificate_uri: ArtifactUri
    exactness: Literal["EXACT"] = "EXACT"
    backend: Literal["networkx"] = "networkx"
    backend_version: str
