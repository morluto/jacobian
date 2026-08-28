"""Public declarations for reviewed level-one modular q-expansions."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.number_theory.modular_forms._models import (
    LevelOneNamedQExpansionRequest,
)
from jacobian.math.number_theory.modular_forms._operations import (
    compute_level_one_named_q_expansion,
)
from jacobian.math.number_theory.modular_forms.values import LevelOneModularQExpansion

TOOLS: MathTools = (
    MathTool(
        operation_id="modular_form.level_one.named_q_expansion.compute",
        title="Compute an exact named level-one modular-form q-expansion",
        description=(
            "Construct the normalized exact q-prefix of E4, E6, or Ramanujan "
            "Delta in QQ[[q]]. The closed form family and requested finite "
            "precision are admitted before complete divisor scans and Delta's "
            "finite-series identity are evaluated."
        ),
        request_type=LevelOneNamedQExpansionRequest,
        result_type=LevelOneModularQExpansion,
        run=compute_level_one_named_q_expansion,
        tags=(
            "modular-forms",
            "q-expansion",
            "level-one",
            "eisenstein-series",
            "ramanujan-delta",
            "exact",
        ),
        examples=(
            example(
                "delta_through_q5",
                "Compute Delta through q^5; the form must be one of the closed normalized level-one family.",
                {"form": "DELTA", "truncation_order": 6},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
