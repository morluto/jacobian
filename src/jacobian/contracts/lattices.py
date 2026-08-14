"""Bounded contracts for exact lattice operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.matrices import (
    MAX_MATRIX_DIMENSION,
    IntegerMatrix,
    require_matrix_scalar_digits,
)

_MAX_LATTICE_INPUT_SCALAR_DIGITS = 256


class HermiteNormalFormRequest(ContractModel):
    """One bounded integer matrix for row Hermite normal form."""

    matrix: IntegerMatrix

    @model_validator(mode="after")
    def require_hnf_input_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.matrix.entries,
            maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
            label="Hermite normal form input",
        )
        return self


class HermiteNormalFormResult(ContractModel):
    """Exact row HNF and its left unimodular transformation."""

    normal_form: IntegerMatrix
    transformation: IntegerMatrix
    relation: Literal["NORMAL_FORM_EQUALS_TRANSFORMATION_TIMES_MATRIX"] = (
        "NORMAL_FORM_EQUALS_TRANSFORMATION_TIMES_MATRIX"
    )

    @model_validator(mode="after")
    def require_compatible_shapes(self) -> Self:
        rows = len(self.normal_form.entries)
        if len(self.transformation.entries) != rows:
            raise ValueError("HNF transformation must have one row per source row")
        if any(len(row) != rows for row in self.transformation.entries):
            raise ValueError("HNF transformation must be square")
        return self


class LatticeReductionRequest(ContractModel):
    """One bounded integer row basis for exact LLL reduction."""

    basis: IntegerMatrix

    @model_validator(mode="after")
    def require_lattice_input_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.basis.entries,
            maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
            label="basis input",
        )
        return self


class LatticeReductionResult(ContractModel):
    """An exact reduced basis and its left transformation."""

    reduced_basis: IntegerMatrix
    transformation: IntegerMatrix
    rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    relation: Literal["REDUCED_BASIS_EQUALS_TRANSFORMATION_TIMES_BASIS"] = (
        "REDUCED_BASIS_EQUALS_TRANSFORMATION_TIMES_BASIS"
    )
    representation: Literal["INTEGER_ROW_BASIS"] = "INTEGER_ROW_BASIS"
    gram_mode: Literal["EXACT"] = "EXACT"
    delta: Literal["0.99"] = "0.99"
    eta: Literal["0.51"] = "0.51"

    @model_validator(mode="after")
    def require_transformation_shape(self) -> Self:
        rows = len(self.reduced_basis.entries)
        if len(self.transformation.entries) != rows:
            raise ValueError("LLL transformation must have one row per basis row")
        if len(self.transformation.entries[0]) != rows:
            raise ValueError("LLL transformation must be square by basis row count")
        return self


__all__ = [
    "HermiteNormalFormRequest",
    "HermiteNormalFormResult",
    "LatticeReductionRequest",
    "LatticeReductionResult",
]
