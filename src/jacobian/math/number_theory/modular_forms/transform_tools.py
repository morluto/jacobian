"""Catalog declarations for modular-form integer and coefficient transforms."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample

from .transform_models import (
    HeckeRequest,
    NamedQExpansionRequest,
    SturmBoundRequest,
    SturmBoundResult,
    URequest,
    VRequest,
)
from .transforms import hecke, named_q_expansion, sturm_bound, u_operator, v_operator
from .values import ModularQExpansion

_Q = {
    "expansion": {
        "space": {
            "group": "GAMMA0",
            "level": 1,
            "weight": 4,
            "kind": "M",
            "character": "TRIVIAL",
            "coefficient_domain": "QQ",
        },
        "weight": 4,
        "q_expansion": {
            "variable": "q",
            "truncation_order": 2,
            "coefficients": [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
        },
        "basis_id": "canonical",
    }
}


def _n(r: NamedQExpansionRequest) -> ModularQExpansion:
    return named_q_expansion(r.space, r.form, r.precision)


def _s(r: SturmBoundRequest) -> SturmBoundResult:
    return sturm_bound(r.space)


def _h(r: HeckeRequest) -> ModularQExpansion:
    return hecke(r.expansion, r.index, r.output_precision)


def _u(r: URequest) -> ModularQExpansion:
    return u_operator(r.expansion, r.prime, r.output_precision)


def _v(r: VRequest) -> ModularQExpansion:
    return v_operator(r.expansion, r.prime, r.output_precision)


TOOLS: MathTools = (
    MathTool(
        operation_id="modular_form.named.q_expansion.compute",
        title="Compute a named modular q-prefix",
        description="Construct E4, E6, or Delta with an explicit level-one space parent.",
        request_type=NamedQExpansionRequest,
        result_type=ModularQExpansion,
        run=_n,
        tags=("modular-forms", "q-expansion", "exact"),
        examples=(
            OperationExample(
                name="named_e4",
                description="Construct E4 through q^1.",
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 1,
                        "weight": 4,
                        "kind": "M",
                        "character": "TRIVIAL",
                        "coefficient_domain": "QQ",
                    },
                    "form": "E4",
                    "precision": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.space.sturm_bound.compute",
        title="Compute a Gamma0 Sturm bound",
        description="Return the exact bounded Sturm integer for a trivial-character Gamma0(QQ) space; this operation does not compare forms.",
        request_type=SturmBoundRequest,
        result_type=SturmBoundResult,
        run=_s,
        tags=("modular-forms", "sturm", "exact"),
        examples=(
            OperationExample(
                name="level_one_sturm",
                description="Compute the bound for M_12(SL2Z).",
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
    MathTool(
        operation_id="modular_form.hecke.apply",
        title="Apply a finite Hecke transform",
        description="Apply the exact level-one/trivial-character coefficient formula to a q-prefix with sufficient source precision.",
        request_type=HeckeRequest,
        result_type=ModularQExpansion,
        run=_h,
        tags=("modular-forms", "hecke", "exact"),
        examples=(
            OperationExample(
                name="hecke_constant",
                description="Apply T_1.",
                input={**_Q, "index": 1, "output_precision": 1},
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.u_operator.apply",
        title="Apply a U operator",
        description="Return a_n mapped to a_(p n) with explicit source-precision admission.",
        request_type=URequest,
        result_type=ModularQExpansion,
        run=_u,
        tags=("modular-forms", "u-operator", "exact"),
        examples=(
            OperationExample(
                name="u_constant",
                description="Apply U_2.",
                input={**_Q, "prime": 2, "output_precision": 1},
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.v_operator.apply",
        title="Apply a V operator",
        description="Return the exact coefficient dilation by a prime index.",
        request_type=VRequest,
        result_type=ModularQExpansion,
        run=_v,
        tags=("modular-forms", "v-operator", "exact"),
        examples=(
            OperationExample(
                name="v_constant",
                description="Apply V_2.",
                input={**_Q, "prime": 2, "output_precision": 1},
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
