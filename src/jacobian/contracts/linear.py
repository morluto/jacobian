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
    values: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )
    method: Literal["RREF_FREE_VARIABLES_ZERO"] = "RREF_FREE_VARIABLES_ZERO"


class LinearRationalInconsistencyResult(ContractModel):
    """Inline normalized left witness for an inconsistent rational system."""

    result_schema_version: Literal["1"] = "1"
    left_witness: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )
    rhs_pairing: CanonicalRational
    method: Literal["DUAL_RREF_PAIRING_ONE"] = "DUAL_RREF_PAIRING_ONE"


class LinearRationalInconsistencyFindRequest(ContractModel):
    """Ask the pinned provider for one normalized inconsistency witness."""

    system: LinearRationalSystem
    resource_budget: LinearRationalResourceBudget = Field(
        default_factory=LinearRationalResourceBudget
    )
