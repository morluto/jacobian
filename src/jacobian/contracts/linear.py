"""Exact rational linear-system and solution-witness contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator

from jacobian.canonical import canonicalize_json, sha256_digest
from jacobian.contracts.capabilities import (
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
)
from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest
from jacobian.contracts.exact import CanonicalRational, require_bounded_rational
from jacobian.contracts.matrices import RationalMatrix
from jacobian.contracts.results import ContractModel

MAX_LINEAR_DIMENSION = 32
MAX_RATIONAL_DIGITS = 256

LinearVariableName = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$",
        strict=True,
    ),
]


def linear_variable_order_digest(variables: tuple[str, ...]) -> str:
    """Bind the declared column order without inventing a generic object schema."""

    return sha256_digest(canonicalize_json({"variables": list(variables)}))


def _require_bounded_rationals(values: tuple[CanonicalRational, ...]) -> None:
    for value in values:
        require_bounded_rational(
            value,
            max_digits=MAX_RATIONAL_DIGITS,
            label="linear-system rational",
        )


class LinearRationalSystem(ContractModel):
    """One declared finite system ``A x = b`` over exact rationals."""

    system_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    relation: Literal["AX_EQUALS_B"] = "AX_EQUALS_B"
    variables: tuple[LinearVariableName, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )
    coefficients: RationalMatrix
    rhs: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )

    @model_validator(mode="after")
    def require_matching_canonical_dimensions(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("linear-system variable names must be unique")
        if len(self.coefficients.entries[0]) != len(self.variables):
            raise ValueError(
                "the coefficient column count must equal the declared variable count"
            )
        if len(self.coefficients.entries) != len(self.rhs):
            raise ValueError(
                "the right-hand side length must equal the coefficient row count"
            )
        _require_bounded_rationals(
            tuple(value for row in self.coefficients.entries for value in row)
            + self.rhs
        )
        return self


class LinearSystemBinding(ContractModel):
    """Exact stored identity and dimensions of one rational system."""

    binding_version: Literal["1"] = "1"
    system_artifact_uri: ArtifactUri
    system_object_digest: Sha256Digest
    system_payload_digest: Sha256Digest
    variable_order_digest: Sha256Digest
    row_count: StrictInt = Field(ge=1, le=MAX_LINEAR_DIMENSION)
    column_count: StrictInt = Field(ge=1, le=MAX_LINEAR_DIMENSION)


class LinearRationalResourceBudget(ContractModel):
    """Wall-clock bound enforced around one isolated Python-FLINT attempt."""

    budget_version: Literal["1"] = "1"
    wall_seconds: StrictInt = Field(default=10, ge=1, le=60)


class LinearRationalSolutionArtifact(ContractModel):
    """One exact candidate vector, not a self-verifying solver report."""

    solution_schema_version: Literal["1"] = "1"
    system: LinearSystemBinding
    declared_scope: Literal["FULL_SYSTEM"] = "FULL_SYSTEM"
    values: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )
    producer: CapabilityProviderRuntime
    resource_budget: LinearRationalResourceBudget
    method: Literal["RREF_FREE_VARIABLES_ZERO"] = "RREF_FREE_VARIABLES_ZERO"

    @model_validator(mode="after")
    def require_total_bound_candidate(self) -> Self:
        if len(self.values) != self.system.column_count:
            raise ValueError(
                "solution must contain one exact value for every declared variable"
            )
        _require_bounded_rationals(self.values)
        if (
            self.producer.provider != "python-flint"
            or self.producer.availability
            is not CapabilityProviderAvailability.AVAILABLE
            or self.producer.version != "0.9.0"
        ):
            raise ValueError(
                "solution producer must be the available pinned Python-FLINT 0.9.0 runtime"
            )
        return self


class LinearRationalSolutionFindRequest(ContractModel):
    """Ask the pinned provider for one candidate vector."""

    system: LinearRationalSystem
    resource_budget: LinearRationalResourceBudget = Field(
        default_factory=LinearRationalResourceBudget
    )


class LinearRationalSolutionFindOutput(ContractModel):
    """Unverified outcome of one bounded rational-solution attempt."""

    status: Literal["SOLUTION_PRODUCED", "NO_SOLUTION_PRODUCED"]
    conclusion: Literal["UNKNOWN"] = "UNKNOWN"
    system_uri: ArtifactUri
    solution_uri: ArtifactUri | None = None
    solution: tuple[CanonicalRational, ...] | None = None
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"
    certificate_available: bool
    method: Literal["RREF_FREE_VARIABLES_ZERO"] = "RREF_FREE_VARIABLES_ZERO"
    backend: Literal["python-flint"] = "python-flint"
    backend_version: Literal["0.9.0"] = "0.9.0"
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_candidate_projection(self) -> Self:
        produced = self.status == "SOLUTION_PRODUCED"
        if produced != (
            self.solution_uri is not None
            and self.solution is not None
            and self.certificate_available
        ):
            raise ValueError(
                "produced output requires exactly one durable, directly checkable vector"
            )
        if not produced and (
            self.solution_uri is not None
            or self.solution is not None
            or self.certificate_available
        ):
            raise ValueError("not-found output cannot carry a solution candidate")
        return self


class LinearRationalSolutionVerificationRequest(ContractModel):
    """Verify one stored vector against its exact bound system."""

    solution_uri: ArtifactUri


class LinearRationalSolutionVerificationOutput(ContractModel):
    """Model-facing projection of independent ``A x = b`` replay."""

    status: Literal[
        "VERIFIED_SOLUTION",
        "REJECTED",
        "TIMEOUT",
        "CANCELLED",
        "ERROR",
    ]
    conclusion: Literal["TRUE", "UNKNOWN"]
    system_uri: ArtifactUri
    solution_uri: ArtifactUri
    witness_uri: ArtifactUri
    checker_id: CheckerUri
    verification_record_uri: ArtifactUri | None = None
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_verified_projection(self) -> Self:
        if self.status == "VERIFIED_SOLUTION":
            if self.conclusion != "TRUE" or self.verification_record_uri is None:
                raise ValueError(
                    "verified solution output requires TRUE and a verification record"
                )
        elif self.conclusion != "UNKNOWN" or self.verification_record_uri is not None:
            raise ValueError(
                "non-verified solution output cannot carry a conclusion or record"
            )
        return self


class LinearRationalInconsistencyArtifact(ContractModel):
    """One normalized left-nullspace witness that proves ``A x = b`` inconsistent."""

    certificate_schema_version: Literal["1"] = "1"
    system: LinearSystemBinding
    declared_scope: Literal["FULL_SYSTEM"] = "FULL_SYSTEM"
    left_witness: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )
    rhs_pairing: CanonicalRational
    producer: CapabilityProviderRuntime
    resource_budget: LinearRationalResourceBudget
    method: Literal["DUAL_RREF_PAIRING_ONE"] = "DUAL_RREF_PAIRING_ONE"

    @model_validator(mode="after")
    def require_normalized_bound_certificate(self) -> Self:
        if len(self.left_witness) != self.system.row_count:
            raise ValueError(
                "inconsistency witness must contain one value per system row"
            )
        if self.rhs_pairing.as_fraction() != 1:
            raise ValueError("inconsistency witness must be normalized to y^T b = 1")
        _require_bounded_rationals((*self.left_witness, self.rhs_pairing))
        if (
            self.producer.provider != "python-flint"
            or self.producer.availability
            is not CapabilityProviderAvailability.AVAILABLE
            or self.producer.version != "0.9.0"
        ):
            raise ValueError(
                "inconsistency producer must be the available pinned "
                "Python-FLINT 0.9.0 runtime"
            )
        return self


class LinearRationalInconsistencyFindRequest(ContractModel):
    """Ask the pinned provider for one normalized inconsistency witness."""

    system: LinearRationalSystem
    resource_budget: LinearRationalResourceBudget = Field(
        default_factory=LinearRationalResourceBudget
    )


class LinearRationalInconsistencyFindOutput(ContractModel):
    """Unverified outcome of one bounded inconsistency-certificate attempt."""

    status: Literal["CERTIFICATE_PRODUCED", "NO_CERTIFICATE_PRODUCED"]
    conclusion: Literal["UNKNOWN"] = "UNKNOWN"
    system_uri: ArtifactUri
    certificate_uri: ArtifactUri | None = None
    left_witness: tuple[CanonicalRational, ...] | None = None
    rhs_pairing: CanonicalRational | None = None
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"
    verification_candidate_available: bool
    method: Literal["DUAL_RREF_PAIRING_ONE"] = "DUAL_RREF_PAIRING_ONE"
    backend: Literal["python-flint"] = "python-flint"
    backend_version: Literal["0.9.0"] = "0.9.0"
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_candidate_projection(self) -> Self:
        produced = self.status == "CERTIFICATE_PRODUCED"
        if produced != (
            self.certificate_uri is not None
            and self.left_witness is not None
            and self.rhs_pairing is not None
            and self.verification_candidate_available
        ):
            raise ValueError(
                "produced output requires one durable normalized inconsistency witness"
            )
        if not produced and (
            self.certificate_uri is not None
            or self.left_witness is not None
            or self.rhs_pairing is not None
            or self.verification_candidate_available
        ):
            raise ValueError("not-found output cannot carry inconsistency evidence")
        return self


class LinearRationalInconsistencyVerificationRequest(ContractModel):
    """Verify one stored normalized left-nullspace witness."""

    certificate_uri: ArtifactUri


class LinearRationalInconsistencyVerificationOutput(ContractModel):
    """Model-facing projection of independent inconsistency replay."""

    status: Literal[
        "VERIFIED_INCONSISTENT",
        "REJECTED",
        "TIMEOUT",
        "CANCELLED",
        "ERROR",
    ]
    conclusion: Literal["TRUE", "UNKNOWN"]
    system_uri: ArtifactUri
    certificate_uri: ArtifactUri
    witness_uri: ArtifactUri
    checker_id: CheckerUri
    verification_record_uri: ArtifactUri | None = None
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_verified_projection(self) -> Self:
        if self.status == "VERIFIED_INCONSISTENT":
            if self.conclusion != "TRUE" or self.verification_record_uri is None:
                raise ValueError(
                    "verified inconsistency requires TRUE and a verification record"
                )
        elif self.conclusion != "UNKNOWN" or self.verification_record_uri is not None:
            raise ValueError(
                "non-verified inconsistency cannot carry a conclusion or record"
            )
        return self
