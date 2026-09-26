"""Public operation declaration for generic-j twist-class decisions."""

from jacobian._models import StrictModel
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.elliptic_curves.finite_field import (
    FiniteFieldShortWeierstrassCurve,
)
from jacobian.math.number_theory.elliptic_curves.twist_class.operations import (
    FiniteFieldTwistClassResult,
    finite_field_twist_class,
)


class FiniteFieldTwistClassRequest(StrictModel):
    """Two curves over the same exact field presentation."""

    source: FiniteFieldShortWeierstrassCurve
    target: FiniteFieldShortWeierstrassCurve


_F5 = {
    "characteristic": "5",
    "modulus_coefficients": ["0", "1"],
    "generator": "a",
}


def _element(coordinate: int) -> dict[str, object]:
    return {"presentation": _F5, "coordinates": [str(coordinate)]}


def _curve(a: int, b: int) -> dict[str, object]:
    return {
        "field": _F5,
        "coefficient_a": _element(a),
        "coefficient_b": _element(b),
    }


TOOLS = (
    MathTool(
        operation_id="elliptic_curve.finite_field.twist_class.decide",
        title="Decide a generic-j finite-field elliptic-curve twist class",
        description=(
            "For nonsingular short-Weierstrass curves over the same exact finite "
            "field and with j outside {0, 1728}, classify unequal j, isomorphism, "
            "or quadratic-twist relation using complete bounded scaling searches."
        ),
        request_type=FiniteFieldTwistClassRequest,
        result_type=FiniteFieldTwistClassResult,
        run=lambda request: finite_field_twist_class(request.source, request.target),
        tags=("elliptic-curve", "finite-field", "quadratic-twist", "exact"),
        discovery_terms=(
            "classify finite-field elliptic curves up to quadratic twist",
            "compare equal-j finite-field elliptic curves",
            "decide generic finite-field elliptic curve twist relation",
        ),
        examples=(
            OperationExample(
                name="classify_canonical_twist_over_five",
                description=(
                    "Classify the canonical quadratic twist of y^2=x^3+x+1 over F5."
                ),
                input={"source": _curve(1, 1), "target": _curve(4, 3)},
            ),
        ),
    ),
)

__all__ = ["TOOLS", "FiniteFieldTwistClassRequest"]
