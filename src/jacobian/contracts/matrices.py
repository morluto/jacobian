"""Exact rational matrix capability contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.common import ArtifactUri, CheckerUri
from jacobian.contracts.exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalInteger,
    CanonicalRational,
)
from jacobian.contracts.results import ContractModel

MAX_MATRIX_DIMENSION = 32
MAX_MATRIX_SCALAR_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS


def require_matrix_scalar_digits(
    entries: tuple[tuple[str | CanonicalRational, ...], ...],
    *,
    maximum: int,
    label: str,
) -> None:
    """Apply an operation-owned scalar budget to an authoritative matrix value."""

    for row in entries:
        for value in row:
            components = (value,) if isinstance(value, str) else (value.num, value.den)
            if any(len(component.lstrip("-")) > maximum for component in components):
                raise ValueError(
                    f"{label} scalars are limited to {maximum} decimal digits"
                )


class RationalMatrix(ContractModel):
    """One nonempty rectangular matrix over canonical rationals."""

    matrix_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    entries: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1,
        max_length=MAX_MATRIX_DIMENSION,
    )

    @model_validator(mode="after")
    def require_rectangular_nonempty_rows(self) -> Self:
        column_count = len(self.entries[0])
        if column_count == 0 or column_count > MAX_MATRIX_DIMENSION:
            raise ValueError("matrix rows must contain between 1 and 32 entries")
        if any(len(row) != column_count for row in self.entries):
            raise ValueError("matrix rows must all have the same length")
        require_matrix_scalar_digits(
            self.entries,
            maximum=MAX_MATRIX_SCALAR_DIGITS,
            label="matrix",
        )
        return self


class IntegerMatrix(ContractModel):
    """One nonempty rectangular matrix over exact canonical integers."""

    matrix_schema_version: Literal["1"] = "1"
    domain: Literal["ZZ"] = "ZZ"
    entries: tuple[tuple[CanonicalInteger, ...], ...] = Field(
        min_length=1,
        max_length=MAX_MATRIX_DIMENSION,
    )

    @model_validator(mode="after")
    def require_rectangular_nonempty_rows(self) -> Self:
        column_count = len(self.entries[0])
        if column_count == 0 or column_count > MAX_MATRIX_DIMENSION:
            raise ValueError("matrix rows must contain between 1 and 32 entries")
        if any(len(row) != column_count for row in self.entries):
            raise ValueError("matrix rows must all have the same length")
        require_matrix_scalar_digits(
            self.entries,
            maximum=MAX_MATRIX_SCALAR_DIGITS,
            label="matrix",
        )
        return self


class MatrixDeterminantArtifact(ContractModel):
    result_schema_version: Literal["1"] = "1"
    matrix_uri: ArtifactUri
    determinant: CanonicalRational
    method: Literal["FRACTION_FREE_BAREISS"] = "FRACTION_FREE_BAREISS"
    backend: Literal["sympy"] = "sympy"
    backend_version: str


class MatrixRankArtifact(ContractModel):
    result_schema_version: Literal["1"] = "1"
    matrix_uri: ArtifactUri
    rank: int = Field(ge=0, le=32)
    pivot_columns: tuple[int, ...] = Field(max_length=32)
    method: Literal["EXACT_RATIONAL_ROW_REDUCTION"] = "EXACT_RATIONAL_ROW_REDUCTION"
    backend: Literal["sympy"] = "sympy"
    backend_version: str


class MatrixDeterminantVerificationRequest(ContractModel):
    determinant_uri: ArtifactUri


class MatrixDeterminantVerificationOutput(ContractModel):
    """Projection of an independent exact determinant recomputation."""

    status: Literal[
        "VERIFIED_DETERMINANT",
        "REJECTED",
        "TIMEOUT",
        "CANCELLED",
        "ERROR",
    ]
    conclusion: Literal["TRUE", "UNKNOWN"]
    matrix_uri: ArtifactUri
    determinant_uri: ArtifactUri
    witness_uri: ArtifactUri
    checker_id: CheckerUri
    verification_record_uri: ArtifactUri | None = None
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_verified_projection(self) -> Self:
        if self.status == "VERIFIED_DETERMINANT":
            if self.conclusion != "TRUE" or self.verification_record_uri is None:
                raise ValueError(
                    "verified determinant output requires TRUE and a verification record"
                )
        elif self.conclusion != "UNKNOWN" or self.verification_record_uri is not None:
            raise ValueError(
                "non-verified determinant output cannot carry a conclusion or record"
            )
        return self


class MatrixRankVerificationRequest(ContractModel):
    rank_uri: ArtifactUri


class MatrixRankVerificationOutput(ContractModel):
    status: Literal["VERIFIED_RANK", "REJECTED", "TIMEOUT", "CANCELLED", "ERROR"]
    conclusion: Literal["TRUE", "UNKNOWN"]
    matrix_uri: ArtifactUri
    rank_uri: ArtifactUri
    witness_uri: ArtifactUri
    checker_id: CheckerUri
    verification_record_uri: ArtifactUri | None = None
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_verified_projection(self) -> Self:
        if self.status == "VERIFIED_RANK":
            if self.conclusion != "TRUE" or self.verification_record_uri is None:
                raise ValueError(
                    "verified rank requires TRUE and a verification record"
                )
        elif self.conclusion != "UNKNOWN" or self.verification_record_uri is not None:
            raise ValueError("non-verified rank cannot carry a conclusion or record")
        return self
