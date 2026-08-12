"""Exact finite rational-polytope contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.common import ArtifactUri
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.results import ContractModel, Execution, InputValidation


class RationalVector(ContractModel):
    values: tuple[CanonicalRational, ...] = Field(min_length=1)


class RationalPoint(ContractModel):
    point_schema_version: Literal["1"] = "1"
    coordinates: tuple[CanonicalRational, ...] = Field(min_length=1)


class FiniteGeneratorSet(ContractModel):
    generator_set_schema_version: Literal["1"] = "1"
    dimension: int = Field(ge=1, le=256)
    generators: tuple[RationalVector, ...] = Field(min_length=1, max_length=100_000)

    @model_validator(mode="after")
    def dimensions_match(self) -> Self:
        if any(
            len(generator.values) != self.dimension for generator in self.generators
        ):
            raise ValueError("every generator must match the declared dimension")
        return self


class PolytopePredicate(StrEnum):
    INSIDE_CONVEX_HULL = "INSIDE_CONVEX_HULL"
    OUTSIDE_CONVEX_HULL = "OUTSIDE_CONVEX_HULL"


class PolytopeClaim(ContractModel):
    claim_schema_version: Literal["1"] = "1"
    predicate: PolytopePredicate
    dimension: int = Field(ge=1, le=256)
    point_uri: ArtifactUri
    generator_set_uri: ArtifactUri


class PolytopeStatus(StrEnum):
    MEMBER = "MEMBER"
    SEPARATED = "SEPARATED"
    UNKNOWN = "UNKNOWN"


class PolytopeSeparateRequest(ContractModel):
    point_uri: ArtifactUri
    generator_set_uri: ArtifactUri
    projection: tuple[int, ...] | None = None
    wall_seconds: int = Field(default=30, ge=1, le=86_400)

    @model_validator(mode="after")
    def projection_has_unique_nonnegative_indices(self) -> Self:
        if self.projection is not None:
            if not self.projection:
                raise ValueError("projection cannot be empty")
            if any(index < 0 for index in self.projection):
                raise ValueError("projection indices must be nonnegative")
            if len(set(self.projection)) != len(self.projection):
                raise ValueError("projection indices must be unique")
        return self


class PolytopeSeparateResult(ContractModel):
    result_schema_version: Literal["1"] = "1"
    status: PolytopeStatus
    point_uri: ArtifactUri
    generator_set_uri: ArtifactUri
    effective_point_uri: ArtifactUri | None = None
    effective_generator_set_uri: ArtifactUri | None = None
    claim_uri: ArtifactUri | None = None
    witness_uri: ArtifactUri | None = None
    certificate_uri: ArtifactUri | None = None
    execution: Execution
    input: InputValidation

    @model_validator(mode="after")
    def evidence_matches_status(self) -> Self:
        if self.status == PolytopeStatus.MEMBER:
            if self.witness_uri is None or self.certificate_uri is not None:
                raise ValueError("membership requires a witness and no separator")
        elif self.status == PolytopeStatus.SEPARATED:
            if self.certificate_uri is None or self.witness_uri is not None:
                raise ValueError("separation requires a certificate and no witness")
        elif self.witness_uri is not None or self.certificate_uri is not None:
            raise ValueError("unknown results cannot carry decisive evidence")
        if self.status is not PolytopeStatus.UNKNOWN and any(
            value is None
            for value in (
                self.effective_point_uri,
                self.effective_generator_set_uri,
                self.claim_uri,
            )
        ):
            raise ValueError("decisive results require replay bindings")
        return self
