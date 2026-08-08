"""Exact rational linear-system and inline-result contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator

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


class LinearRationalResourceBudget(ContractModel):
    """Wall-clock bound enforced around one isolated Python-FLINT attempt."""

    budget_version: Literal["1"] = "1"
    wall_seconds: StrictInt = Field(default=10, ge=1, le=60)


class LinearRationalSolutionFindRequest(ContractModel):
    """Ask the pinned provider for one candidate vector."""

    system: LinearRationalSystem
    resource_budget: LinearRationalResourceBudget = Field(
        default_factory=LinearRationalResourceBudget
    )


class LinearRationalSolutionResult(ContractModel):
    """Inline total rational solution candidate for ordinary composition."""

    result_schema_version: Literal["1"] = "1"
    status: Literal["SOLUTION_PRODUCED", "NO_SOLUTION_PRODUCED"] = "SOLUTION_PRODUCED"
    values: tuple[CanonicalRational, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )
    method: Literal["RREF_FREE_VARIABLES_ZERO"] = "RREF_FREE_VARIABLES_ZERO"

    @model_validator(mode="after")
    def bind_values_to_status(self) -> Self:
        produced = self.status == "SOLUTION_PRODUCED"
        if produced != (self.values is not None):
            raise ValueError("solution values must agree with the result status")
        return self


class LinearRationalInconsistencyResult(ContractModel):
    """Inline normalized left witness for an inconsistent rational system."""

    result_schema_version: Literal["1"] = "1"
    status: Literal["CERTIFICATE_PRODUCED", "NO_CERTIFICATE_PRODUCED"] = (
        "CERTIFICATE_PRODUCED"
    )
    left_witness: tuple[CanonicalRational, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )
    rhs_pairing: CanonicalRational | None = None
    method: Literal["DUAL_RREF_PAIRING_ONE"] = "DUAL_RREF_PAIRING_ONE"

    @model_validator(mode="after")
    def bind_witness_to_status(self) -> Self:
        produced = self.status == "CERTIFICATE_PRODUCED"
        if produced != (self.left_witness is not None and self.rhs_pairing is not None):
            raise ValueError("inconsistency witness must agree with the result status")
        return self


class LinearRationalInconsistencyFindRequest(ContractModel):
    """Ask the pinned provider for one normalized inconsistency witness."""

    system: LinearRationalSystem
    resource_budget: LinearRationalResourceBudget = Field(
        default_factory=LinearRationalResourceBudget
    )
