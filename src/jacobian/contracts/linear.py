"""Exact rational linear-system contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational, require_bounded_rational
from jacobian.contracts.matrices import RationalMatrix

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


class LinearRationalSolutionFindRequest(ContractModel):
    """Ask for one exact solution of a rational linear system."""

    system: LinearRationalSystem


class LinearRationalSolutionResult(ContractModel):
    """Whether a rational linear system has an exact solution."""

    status: Literal["SOLUTION", "INCONSISTENT"] = "SOLUTION"
    values: tuple[CanonicalRational, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )

    @model_validator(mode="after")
    def bind_values_to_status(self) -> Self:
        produced = self.status == "SOLUTION"
        if produced != (self.values is not None):
            raise ValueError("solution values must agree with the result status")
        return self


class LinearRationalInconsistencyResult(ContractModel):
    """Whether a rational linear system is inconsistent."""

    status: Literal["INCONSISTENT", "CONSISTENT"] = "INCONSISTENT"
    left_witness: tuple[CanonicalRational, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )
    rhs_pairing: CanonicalRational | None = None

    @model_validator(mode="after")
    def bind_witness_to_status(self) -> Self:
        produced = self.status == "INCONSISTENT"
        if produced != (self.left_witness is not None and self.rhs_pairing is not None):
            raise ValueError("inconsistency witness must agree with the result status")
        return self


class LinearRationalInconsistencyFindRequest(ContractModel):
    """Ask whether a rational linear system is inconsistent."""

    system: LinearRationalSystem
