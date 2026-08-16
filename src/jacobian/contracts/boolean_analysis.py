"""Typed contracts for Boolean analysis operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.base import ContractModel


class BooleanTruthTable(ContractModel):
    """A truth table for f: {-1,+1}^n -> {-1,+1}."""

    variable_names: tuple[str, ...] = Field(min_length=1, max_length=16)
    values: tuple[StrictInt, ...] = Field(min_length=2, max_length=65536)

    @model_validator(mode="after")
    def validate_truth_table(self) -> Self:
        n = len(self.variable_names)
        expected = 2 ** n
        if len(self.values) != expected:
            raise ValueError(
                f"truth table must have 2^{n}={expected} values, got {len(self.values)}"
            )
        if any(v not in (-1, 1) for v in self.values):
            raise ValueError("truth table values must be -1 or +1")
        if len(set(self.variable_names)) != len(self.variable_names):
            raise ValueError("variable names must be distinct")
        return self


class BooleanFourierRequest(ContractModel):
    """Request to compute Walsh-Fourier coefficients."""

    truth_table: BooleanTruthTable


class FourierCoefficient(ContractModel):
    """One Walsh-Fourier coefficient."""

    subset_mask: StrictInt = Field(ge=0)
    coefficient: StrictInt


class BooleanFourierResult(ContractModel):
    """Result of computing Walsh-Fourier coefficients."""

    coefficients: tuple[FourierCoefficient, ...] = Field(min_length=1)
    variable_count: StrictInt = Field(ge=1)


class BooleanMultilinearExtensionRequest(ContractModel):
    """Request to evaluate the multilinear extension at a point."""

    truth_table: BooleanTruthTable
    point: tuple[StrictInt, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_point(self) -> Self:
        n = len(self.truth_table.variable_names)
        if len(self.point) != n:
            raise ValueError("point dimension must match truth table variable count")
        return self


class BooleanMultilinearExtensionResult(ContractModel):
    """Result of multilinear extension evaluation."""

    value: StrictInt
    detail: str = Field(min_length=1, max_length=1024)


class BooleanInfluenceRequest(ContractModel):
    """Request to compute the influence of each variable."""

    truth_table: BooleanTruthTable


class BooleanInfluenceResult(ContractModel):
    """Result of computing variable influences."""

    influences: tuple[StrictInt, ...]
    total_influence: StrictInt


class BooleanErasureNoiseRequest(ContractModel):
    """Request to compute the exact erasure noise L1 expectation."""

    truth_table: BooleanTruthTable
    erasure_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_erasure(self) -> Self:
        if self.erasure_count > len(self.truth_table.variable_names):
            raise ValueError("erasure count cannot exceed variable count")
        return self


class BooleanErasureNoiseResult(ContractModel):
    """Result of exact erasure noise computation."""

    expected_absolute_value_numerator: StrictInt
    expected_absolute_value_denominator: StrictInt
    detail: str = Field(min_length=1, max_length=1024)
