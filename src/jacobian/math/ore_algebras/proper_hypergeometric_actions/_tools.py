"""Catalog declaration for proper-hypergeometric shift-operator action."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.ore_algebras.proper_hypergeometric_actions._models import (
    ProperHypergeometricOperatorActionRequest,
    ProperHypergeometricOperatorActionResult,
)
from jacobian.math.ore_algebras.proper_hypergeometric_actions.operations import (
    proper_hypergeometric_operator_action,
)

_ONE_N = {
    "domain": "QQ",
    "variables": ["n"],
    "numerator": {
        "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}]
    },
    "denominator": {
        "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}]
    },
}
_BINOMIAL = {
    "polynomial": {
        "variables": ["n", "k"],
        "polynomial": {
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0, 0]}]
        },
    },
    "factorial_factors": [
        {"n_coefficient": 0, "k_coefficient": 1, "offset": "0", "power": -1},
        {"n_coefficient": 1, "k_coefficient": -1, "offset": "0", "power": -1},
        {"n_coefficient": 1, "k_coefficient": 0, "offset": "0", "power": 1},
    ],
}

TOOLS = (
    MathTool(
        operation_id="ore.proper_hypergeometric.apply_shift_operator.compute",
        title="Apply a shift operator to a proper hypergeometric term",
        description=(
            "Compute the exact relative multiplier R(n,k) in QQ(n,k) for the "
            "formal action P*T = R*T of a left-coefficient QQ(n) shift Ore "
            "operator on a proper hypergeometric term. The identity is on the "
            "common generic nonzero locus; it does not assign quotients at "
            "zeros or reciprocal-factorial support boundaries."
        ),
        request_type=ProperHypergeometricOperatorActionRequest,
        result_type=ProperHypergeometricOperatorActionResult,
        run=lambda request: proper_hypergeometric_operator_action(
            request.operator, request.term
        ),
        tags=("ore-algebra", "holonomic", "hypergeometric", "exact"),
        discovery_terms=(
            "apply shift operator to hypergeometric term",
            "Ore operator action on summand",
            "creative telescoping operator action",
            "apply recurrence operator to proper hypergeometric term",
        ),
        examples=(
            OperationExample(
                name="shift_binomial_summand",
                description=(
                    "Apply S_n to binomial(n,k); the returned multiplier is "
                    "(n+1)/(n+1-k) on the generic nonzero locus."
                ),
                input={
                    "operator": {
                        "variable": "n",
                        "terms": [{"exponent": 1, "coefficient": _ONE_N}],
                    },
                    "term": _BINOMIAL,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
