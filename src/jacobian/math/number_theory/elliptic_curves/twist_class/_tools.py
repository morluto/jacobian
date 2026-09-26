"""Public operation declaration for generic-j twist-class decisions."""

import rfc8785

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import (
    MathTool,
    OperationExample,
    OperationResourceAdmissionError,
)
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


def _run_twist_class(
    request: FiniteFieldTwistClassRequest,
) -> FiniteFieldTwistClassResult:
    field = request.source.field
    maximum_coordinate = str(field.characteristic - 1)
    element = {
        "presentation": field.model_dump(mode="json"),
        "coordinates": [maximum_coordinate] * field.degree,
    }
    maximum_curve = {
        "field": field.model_dump(mode="json"),
        "coefficient_a": element,
        "coefficient_b": element,
    }
    maximum_result = {
        "source": request.source.model_dump(mode="json"),
        "target": request.target.model_dump(mode="json"),
        "relation": "QUADRATIC_TWIST",
        "source_j_invariant": element,
        "target_j_invariant": element,
        "isomorphism": None,
        "twist": {
            "source_curve": request.source.model_dump(mode="json"),
            "twisted_curve": maximum_curve,
            "parameter": element,
        },
        "twist_to_target": {
            "source": maximum_curve,
            "target": request.target.model_dump(mode="json"),
            "isomorphic": True,
            "scaling": element,
        },
    }
    if len(rfc8785.dumps(maximum_result)) > CanonicalLimits().max_output_bytes:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="elliptic_curve.finite_field.twist_class_output_bound",
            message="twist-class result exceeds the delivery output envelope",
        )
    return finite_field_twist_class(request.source, request.target)


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
        run=_run_twist_class,
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
