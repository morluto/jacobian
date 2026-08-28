"""Wire adapter for exact named level-one q-expansion construction."""

from jacobian.math.number_theory.modular_forms._models import (
    LevelOneNamedQExpansionRequest,
)
from jacobian.math.number_theory.modular_forms.operations import (
    level_one_named_q_expansion,
)
from jacobian.math.number_theory.modular_forms.values import LevelOneModularQExpansion


def compute_level_one_named_q_expansion(
    request: LevelOneNamedQExpansionRequest,
) -> LevelOneModularQExpansion:
    return level_one_named_q_expansion(request.form, request.truncation_order)


__all__ = ["compute_level_one_named_q_expansion"]
