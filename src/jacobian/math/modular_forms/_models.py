"""Wire contracts for exact named level-one q-expansion construction."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.modular_forms.kernel import (
    MAX_LEVEL_ONE_TRUNCATION_ORDER,
    NamedLevelOneModularForm,
    require_level_one_admission,
)


class LevelOneNamedQExpansionRequest(StrictModel):
    """Construct one reviewed normalized level-one modular-form q-prefix."""

    form: NamedLevelOneModularForm = Field(
        description="Closed normalized family: E4, E6, or Ramanujan DELTA."
    )
    truncation_order: StrictInt = Field(
        ge=1,
        le=MAX_LEVEL_ONE_TRUNCATION_ORDER,
        description=(
            "Return coefficients q^0 through q^(P-1); P is bounded before "
            "divisor scans and finite-series arithmetic."
        ),
    )

    @model_validator(mode="after")
    def require_exact_bounded_prefix(self) -> Self:
        require_level_one_admission(self.form, self.truncation_order)
        return self


__all__ = ["LevelOneNamedQExpansionRequest"]
