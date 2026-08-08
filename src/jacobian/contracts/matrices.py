"""Exact rational matrix capability contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.capabilities import (
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
)
from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest
from jacobian.contracts.exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalInteger,
    CanonicalRational,
)
from jacobian.contracts.results import ContractModel

MAX_MATRIX_DIMENSION = 32
MAX_MATRIX_SCALAR_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS
PYTHON_FLINT_HNF_CONFIGURATION = {
    "distribution": "python-flint",
    "domain": "ZZ",
    "operation": "fmpz_mat.hnf(transform=True)",
    "flint_library_version": "3.6.0",
    "maximum_rows": 32,
    "maximum_columns": 32,
    "normal_form_convention": "FLINT_ROW_HNF",
    "relation": "H=U*A",
}


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


class MatrixHermiteResourceBudget(ContractModel):
    """Wall-clock bound for one isolated Python-FLINT HNF attempt."""

    budget_version: Literal["1"] = "1"
    wall_seconds: StrictInt = Field(default=10, ge=1, le=60)


class MatrixBinding(ContractModel):
    """Exact stored identity and dimensions of one integer matrix."""

    binding_version: Literal["1"] = "1"
    matrix_artifact_uri: ArtifactUri
    matrix_object_digest: Sha256Digest
    matrix_payload_digest: Sha256Digest
    row_count: StrictInt = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    column_count: StrictInt = Field(ge=1, le=MAX_MATRIX_DIMENSION)


class MatrixHermiteNormalFormArtifact(ContractModel):
    """One HNF candidate plus its proposed left transformation."""

    normal_form_schema_version: Literal["1"] = "1"
    source: MatrixBinding
    declared_scope: Literal["FULL_MATRIX"] = "FULL_MATRIX"
    normal_form: IntegerMatrix
    transformation: IntegerMatrix
    producer: CapabilityProviderRuntime
    resource_budget: MatrixHermiteResourceBudget
    method: Literal["ROW_HNF_LEFT_UNIMODULAR_TRANSFORM"] = (
        "ROW_HNF_LEFT_UNIMODULAR_TRANSFORM"
    )

    @model_validator(mode="after")
    def require_bound_shapes_and_pinned_producer(self) -> Self:
        normal_rows = len(self.normal_form.entries)
        normal_columns = len(self.normal_form.entries[0])
        transform_rows = len(self.transformation.entries)
        transform_columns = len(self.transformation.entries[0])
        if (normal_rows, normal_columns) != (
            self.source.row_count,
            self.source.column_count,
        ):
            raise ValueError("normal form dimensions must match the source matrix")
        if (transform_rows, transform_columns) != (
            self.source.row_count,
            self.source.row_count,
        ):
            raise ValueError("left transformation must be square by source row count")
        if (
            self.producer.provider != "python-flint"
            or self.producer.availability
            is not CapabilityProviderAvailability.AVAILABLE
            or self.producer.version != "0.9.0"
            or self.producer.configuration != PYTHON_FLINT_HNF_CONFIGURATION
        ):
            raise ValueError(
                "normal-form producer must be the pinned Python-FLINT HNF profile"
            )
        return self


class MatrixHermiteNormalFormRequest(ContractModel):
    matrix: IntegerMatrix
    resource_budget: MatrixHermiteResourceBudget = Field(
        default_factory=MatrixHermiteResourceBudget
    )

    @model_validator(mode="after")
    def require_hnf_input_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.matrix.entries,
            maximum=256,
            label="Hermite normal form input",
        )
        return self


class MatrixHermiteNormalFormOutput(ContractModel):
    """Unverified outcome of one bounded row-HNF computation."""

    status: Literal["NORMAL_FORM_PRODUCED", "NO_NORMAL_FORM_PRODUCED"]
    conclusion: Literal["UNKNOWN"] = "UNKNOWN"
    matrix_uri: ArtifactUri
    normal_form_uri: ArtifactUri | None = None
    normal_form: IntegerMatrix | None = None
    transformation: IntegerMatrix | None = None
    exactness: Literal["EXACT_INTEGER"] = "EXACT_INTEGER"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"
    certificate_available: bool
    method: Literal["ROW_HNF_LEFT_UNIMODULAR_TRANSFORM"] = (
        "ROW_HNF_LEFT_UNIMODULAR_TRANSFORM"
    )
    backend: Literal["python-flint"] = "python-flint"
    backend_version: Literal["0.9.0"] = "0.9.0"
    flint_library_version: Literal["3.6.0"] = "3.6.0"
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_candidate_projection(self) -> Self:
        produced = self.status == "NORMAL_FORM_PRODUCED"
        if produced != (
            self.normal_form_uri is not None
            and self.normal_form is not None
            and self.transformation is not None
            and self.certificate_available
        ):
            raise ValueError(
                "produced output requires one durable normal form and transformation"
            )
        if not produced and (
            self.normal_form_uri is not None
            or self.normal_form is not None
            or self.transformation is not None
            or self.certificate_available
        ):
            raise ValueError("failed output cannot carry normal-form evidence")
        return self


class MatrixHermiteNormalFormVerificationRequest(ContractModel):
    normal_form_uri: ArtifactUri


class MatrixHermiteNormalFormVerificationOutput(ContractModel):
    """Projection of independent HNF and row-equivalence replay."""

    status: Literal[
        "VERIFIED_HERMITE_NORMAL_FORM",
        "REJECTED",
        "TIMEOUT",
        "CANCELLED",
        "ERROR",
    ]
    conclusion: Literal["TRUE", "UNKNOWN"]
    matrix_uri: ArtifactUri
    normal_form_uri: ArtifactUri
    witness_uri: ArtifactUri
    checker_id: CheckerUri
    verification_record_uri: ArtifactUri | None = None
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_verified_projection(self) -> Self:
        if self.status == "VERIFIED_HERMITE_NORMAL_FORM":
            if self.conclusion != "TRUE" or self.verification_record_uri is None:
                raise ValueError(
                    "verified HNF output requires TRUE and a verification record"
                )
        elif self.conclusion != "UNKNOWN" or self.verification_record_uri is not None:
            raise ValueError(
                "non-verified HNF output cannot carry a conclusion or record"
            )
        return self
