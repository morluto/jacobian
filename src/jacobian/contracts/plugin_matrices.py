"""Domain-owned request contracts for the integer-matrix reference plugin."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, StrictStr, model_validator

from jacobian.contracts.claims import flatten_claim_spec
from jacobian.contracts.exact import CanonicalInteger
from jacobian.contracts.plugin_protocol import PluginRequestContext
from jacobian.contracts.results import ContractModel


class MatrixScope(ContractModel):
    rows: StrictInt = Field(ge=1, le=32)
    cols: StrictInt = Field(ge=1, le=32)
    entries: tuple[StrictInt | CanonicalInteger, ...] = Field(min_length=1)


class MatrixClaim(ContractModel):
    predicate: Literal["is_nonsingular", "maximize_absolute_determinant"]
    scope: MatrixScope | None = None

    @model_validator(mode="before")
    @classmethod
    def flatten_generic_claim(cls, value: object) -> object:
        return flatten_claim_spec(value)

    @model_validator(mode="after")
    def validate_scope_for_predicate(self) -> Self:
        if self.predicate == "maximize_absolute_determinant":
            if self.scope is None:
                raise ValueError("maximize_absolute_determinant requires a scope")
            if self.scope.rows != self.scope.cols:
                raise ValueError("determinant scope must be square")
        return self


class MatrixCandidate(ContractModel):
    rows: StrictInt = Field(ge=1, le=32)
    cols: StrictInt = Field(ge=1, le=32)
    entries: tuple[tuple[StrictInt | CanonicalInteger, ...], ...]

    @model_validator(mode="after")
    def validate_matrix_shape(self) -> Self:
        if len(self.entries) != self.rows or any(
            len(row) != self.cols for row in self.entries
        ):
            raise ValueError("matrix entries must match rows and cols")
        return self

    def validate_for_claim(self, claim: MatrixClaim) -> None:
        if (
            claim.predicate in {"is_nonsingular", "maximize_absolute_determinant"}
            and self.rows != self.cols
        ):
            raise ValueError("determinant predicates require a square matrix")
        if claim.predicate == "maximize_absolute_determinant":
            assert claim.scope is not None
            if self.rows != claim.scope.rows or self.cols != claim.scope.cols:
                raise ValueError("candidate dimensions do not match claim scope")
            allowed = {int(value) for value in claim.scope.entries}
            if any(int(entry) not in allowed for row in self.entries for entry in row):
                raise ValueError("candidate entry is outside claim scope")


class MatrixCapabilityRequest(PluginRequestContext):
    claim: MatrixClaim
    candidate: MatrixCandidate | None = None
    witness_role: Literal["DEFEATS_CANDIDATE", "SUPPORTS_CLAIM"] = "DEFEATS_CANDIDATE"

    @model_validator(mode="after")
    def validate_candidate_for_claim(self) -> Self:
        if self.claim.predicate == "is_nonsingular" and self.candidate is None:
            raise ValueError("is_nonsingular capability requires a candidate")
        if self.candidate is not None:
            self.candidate.validate_for_claim(self.claim)
        return self


class MatrixReductionRequest(PluginRequestContext):
    target_kind: Literal["candidate"] = "candidate"
    target: MatrixCandidate
    claim: MatrixClaim
    reducers: tuple[Literal["delete_row_column", "zero_entry"], ...] = ()
    objectives: tuple[Literal["elements", "max_abs_entry"], ...] = ()

    @model_validator(mode="after")
    def validate_target_for_claim(self) -> Self:
        self.target.validate_for_claim(self.claim)
        return self


class MatrixCursor(ContractModel):
    offset: StrictInt = Field(ge=0)


class MatrixEnumerationRequest(PluginRequestContext):
    bounds: MatrixScope
    page_size: StrictInt = Field(ge=1, le=256)
    cursor: MatrixCursor | None = None


class MatrixTransformRequest(PluginRequestContext):
    requested_relation: Literal[
        "EQUIVALENT", "OVER_APPROXIMATION", "UNDER_APPROXIMATION", "HEURISTIC"
    ] = "EQUIVALENT"
    target_schema_uri: StrictStr | None = None
    target_semantics_uri: StrictStr | None = None
    source: MatrixCandidate


class MatrixMaterializeRequest(PluginRequestContext):
    claim: MatrixClaim


__all__ = [
    "MatrixCandidate",
    "MatrixCapabilityRequest",
    "MatrixClaim",
    "MatrixCursor",
    "MatrixEnumerationRequest",
    "MatrixMaterializeRequest",
    "MatrixReductionRequest",
    "MatrixScope",
    "MatrixTransformRequest",
]
