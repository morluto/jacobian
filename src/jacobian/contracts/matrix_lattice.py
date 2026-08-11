"""Durable evidence contracts owned by the matrix-lattice domain."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.matrices import IntegerMatrix, require_matrix_scalar_digits
from jacobian.contracts.results import ContractModel


class HermiteNormalFormResult(ContractModel):
    """Complete H and U evidence; artifact lineage binds it to the input."""

    result_schema_version: Literal["1"] = "1"
    normal_form: IntegerMatrix
    transformation: IntegerMatrix
    method: Literal["ROW_HNF_LEFT_UNIMODULAR_TRANSFORM"] = (
        "ROW_HNF_LEFT_UNIMODULAR_TRANSFORM"
    )
    backend: Literal["python-flint"] = "python-flint"
    backend_version: Literal["0.9.0"] = "0.9.0"
    flint_library_version: Literal["3.6.0"] = "3.6.0"

    @model_validator(mode="after")
    def require_compatible_shapes(self) -> Self:
        rows = len(self.normal_form.entries)
        if len(self.transformation.entries) != rows:
            raise ValueError("HNF transformation must have one row per source row")
        if any(len(row) != rows for row in self.transformation.entries):
            raise ValueError("HNF transformation must be square")
        if any(
            len(row) != len(self.normal_form.entries[0])
            for row in self.normal_form.entries
        ):
            raise ValueError("HNF rows must have a common column count")
        return self


class HermiteNormalFormResourceBudget(ContractModel):
    """Bound the isolated HNF worker attempt."""

    budget_version: Literal["1"] = "1"
    wall_seconds: StrictInt = Field(default=10, ge=1, le=60)


class HermiteNormalFormRequest(ContractModel):
    """Bounded integer matrix input for the durable HNF operation."""

    matrix: IntegerMatrix
    resource_budget: HermiteNormalFormResourceBudget = Field(
        default_factory=HermiteNormalFormResourceBudget
    )

    @model_validator(mode="after")
    def require_hnf_input_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.matrix.entries,
            maximum=256,
            label="Hermite normal form input",
        )
        return self


__all__ = [
    "HermiteNormalFormRequest",
    "HermiteNormalFormResourceBudget",
    "HermiteNormalFormResult",
]
