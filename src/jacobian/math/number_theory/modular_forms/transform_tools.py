"""Catalog declarations for modular-form integer and coefficient transforms."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.polynomials.series._models import TruncatedSeries

from .transform_models import (
    FormalQSeriesURequest,
    FormalQSeriesVRequest,
    NamedQExpansionRequest,
    SturmBoundRequest,
    SturmBoundResult,
)
from .transforms import (
    formal_q_series_u_operator,
    formal_q_series_v_operator,
    named_q_expansion,
    sturm_bound,
)
from .values import ModularQExpansion


def _n(r: NamedQExpansionRequest) -> ModularQExpansion:
    return named_q_expansion(r.space, r.form, r.precision)


def _s(r: SturmBoundRequest) -> SturmBoundResult:
    return sturm_bound(r.space)


def _u(r: FormalQSeriesURequest) -> TruncatedSeries:
    return formal_q_series_u_operator(r.series, r.prime, r.output_precision)


def _v(r: FormalQSeriesVRequest) -> TruncatedSeries:
    return formal_q_series_v_operator(r.series, r.prime, r.output_precision)


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
        description=(
            "Return the exact bounded Sturm integer for represented Gamma0 "
            "spaces of level at most 10,000 over QQ or their declared rational cyclotomic "
            "coefficient field. The bound depends on level and weight; "
            "the exact character and coefficient parent remain attached. "
            "This operation does not compare forms."
        ),
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
            OperationExample(
                name="order_four_character_sturm",
                description=(
                    "Compute a Sturm bound for a character space over "
                    "Q(zeta_12), retaining its exact coefficient parent."
                ),
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 13,
                        "weight": 2,
                        "kind": "M",
                        "character": {
                            "group": {
                                "modulus": 13,
                                "unit_residues": list(range(1, 13)),
                                "character_count": 12,
                                "invariant_factors": [12],
                                "generators": [2],
                                "generator_orders": [12],
                                "unit_coordinates": [
                                    [0],
                                    [1],
                                    [4],
                                    [2],
                                    [9],
                                    [5],
                                    [11],
                                    [3],
                                    [8],
                                    [10],
                                    [7],
                                    [6],
                                ],
                                "exponent": 12,
                            },
                            "coordinates": [3],
                        },
                        "coefficient_domain": {
                            "domain": "QQ_CYCLOTOMIC",
                            "order": 12,
                            "generator": "CLASS_OF_X",
                        },
                    }
                },
            ),
            OperationExample(
                name="gamma0_four_chi_minus4_weight_three_sturm",
                description="Compute bound 1 while retaining the exact chi_{-4} parent.",
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 4,
                        "weight": 3,
                        "kind": "M",
                        "character": {
                            "group": {
                                "modulus": 4,
                                "unit_residues": [1, 3],
                                "character_count": 2,
                                "invariant_factors": [2],
                                "generators": [3],
                                "generator_orders": [2],
                                "unit_coordinates": [[0], [1]],
                                "exponent": 2,
                            },
                            "coordinates": [1],
                        },
                        "coefficient_domain": "QQ",
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.formal_q_series.u_operator.compute",
        title="Apply the formal U_p map to a finite q-series prefix",
        description=(
            "Return the exact finite coefficient selection U_p(sum a_n q^n) = "
            "sum a_(p n) q^n. This is a formal q-series map only: it does not "
            "establish that the source or output is a modular form, and carries "
            "no modular-space parent. Prime, required source precision, source "
            "coefficient digits, work, and aggregate JSON output bytes are "
            "admitted before returning the prefix."
        ),
        request_type=FormalQSeriesURequest,
        result_type=TruncatedSeries,
        run=_u,
        tags=("formal-series", "q-series", "u-operator", "exact"),
        discovery_terms=(
            "formal U_p q-series coefficient map",
            "select coefficients a_(p n) from a truncated q-series",
        ),
        examples=(
            OperationExample(
                name="formal_u2_on_finite_q_prefix",
                description=(
                    "Apply only the formal coefficient map to a prefix; no "
                    "modularity claim is returned."
                ),
                input={
                    "series": {
                        "variable": "q",
                        "truncation_order": 5,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "2", "den": "1"},
                            {"num": "3", "den": "1"},
                            {"num": "4", "den": "1"},
                            {"num": "5", "den": "1"},
                        ],
                    },
                    "prime": 2,
                    "output_precision": 3,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.formal_q_series.v_operator.compute",
        title="Apply the formal V_p map to a finite q-series prefix",
        description=(
            "Return the exact finite q-index dilation V_p(sum a_n q^n) = "
            "sum a_n q^(p n). This is a formal q-series map only: it does not "
            "establish that the source or output is a modular form, and carries "
            "no modular-space parent. Prime, required source precision, source "
            "coefficient digits, work, and aggregate JSON output bytes are "
            "admitted before returning the prefix."
        ),
        request_type=FormalQSeriesVRequest,
        result_type=TruncatedSeries,
        run=_v,
        tags=("formal-series", "q-series", "v-operator", "exact"),
        discovery_terms=(
            "formal V_p q-series index dilation",
            "place q-series coefficient a_n at q^(p n)",
        ),
        examples=(
            OperationExample(
                name="formal_v2_on_finite_q_prefix",
                description=(
                    "Dilate the q-indices of a finite prefix and pad with exact "
                    "zero coefficients; no modularity claim is returned."
                ),
                input={
                    "series": {
                        "variable": "q",
                        "truncation_order": 3,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "2", "den": "1"},
                            {"num": "3", "den": "1"},
                        ],
                    },
                    "prime": 2,
                    "output_precision": 5,
                },
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
