"""Wire contracts for exact named level-one q-expansion construction."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.modular_forms.kernel import (
    NamedLevelOneModularForm,
    require_level_one_admission,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"modular_forms.{reason}", message)


class LevelOneNamedQExpansionRequest(StrictModel):
    """Construct one reviewed normalized level-one modular-form q-prefix."""

    form: NamedLevelOneModularForm = Field(
        description="Closed normalized family: E4, E6, or Ramanujan DELTA."
    )
    truncation_order: StrictInt = Field(
        ge=1,
        description=(
            "Return coefficients q^0 through q^(P-1); the exact work and "
            "serialized-result budgets bound P before any scan or "
            "finite-series arithmetic."
        ),
    )

    @model_validator(mode="after")
    def require_exact_bounded_prefix(self) -> Self:
        try:
            require_level_one_admission(self.form, self.truncation_order)
        except ValueError as exc:
            message = str(exc)
            reason = (
                "exact_work_bound"
                if "exact work bound" in message
                else "serialized_result_bound"
            )
            raise _validation_error(reason, message) from exc
        return self


__all__ = ["LevelOneNamedQExpansionRequest"]
