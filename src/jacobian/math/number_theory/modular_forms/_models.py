"""Wire contracts for exact named level-one q-expansion construction."""

from __future__ import annotations

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.number_theory.modular_forms.kernel import NamedLevelOneModularForm


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


__all__ = ["LevelOneNamedQExpansionRequest"]
