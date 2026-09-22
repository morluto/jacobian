"""Public declarations for reviewed level-one modular q-expansions."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.modular_forms import operations as native
from jacobian.math.number_theory.modular_forms._models import (
    LevelOneNamedQExpansionRequest,
    SpaceDimensionRequest,
    SpaceDimensionResult,
)
from jacobian.math.number_theory.modular_forms.transform_tools import (
    TOOLS as TRANSFORM_TOOLS,
)
from jacobian.math.number_theory.modular_forms.values import LevelOneModularQExpansion


def compute_level_one_named_q_expansion(
    request: LevelOneNamedQExpansionRequest,
) -> LevelOneModularQExpansion:
    return native.level_one_named_q_expansion(request.form, request.truncation_order)


def compute_space_dimension(request: SpaceDimensionRequest) -> SpaceDimensionResult:
    return native.space_dimension(request.space)


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
            OperationExample(
                name="delta_through_q5",
                description="Compute Delta through q^5; the form must be one of the closed normalized level-one family.",
                input={"form": "DELTA", "truncation_order": 6},
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.space.dimension.compute",
        title="Compute the exact dimension of a modular-form space",
        description=(
            "Return the exact complex dimension of a supported holomorphic "
            "M_k or cuspidal S_k space on Gamma0(N) with trivial character "
            "over QQ, with the level-one floor term, congruence class, and "
            "Eisenstein/cusp projections. Only level one is admitted; higher "
            "levels are rejected before computation. No q-expansion "
            "coefficients are returned."
        ),
        request_type=SpaceDimensionRequest,
        result_type=SpaceDimensionResult,
        run=compute_space_dimension,
        tags=(
            "modular-forms",
            "dimension",
            "level-one",
            "eisenstein-series",
            "cusp-forms",
            "exact",
        ),
        discovery_terms=(
            "dimension of M_k",
            "dimension of S_k",
            "level-one modular form dimension",
        ),
        examples=(
            OperationExample(
                name="dimension_of_m12",
                description="Compute dim M_12(SL_2(Z)) = 2; the space must be a trivial-character Gamma0 level-one space.",
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 1,
                        "weight": 12,
                        "kind": "M",
                        "character": "TRIVIAL",
                        "coefficient_domain": "QQ",
                    }
                },
            ),
        ),
    ),
)

TOOLS = TOOLS + TRANSFORM_TOOLS

__all__ = ["TOOLS"]
