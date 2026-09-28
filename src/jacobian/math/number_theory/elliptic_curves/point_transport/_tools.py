"""Built-in operation declarations for finite-field point transport."""

from typing import Any

from jacobian.catalog.models import (
    MathTool,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.elliptic_curves.point_transport._models import (
    FiniteFieldPointTransportRequest,
    FiniteFieldPointTransportResult,
)
from jacobian.math.number_theory.elliptic_curves.point_transport.operations import (
    MAX_POINT_TRANSPORT_OUTPUT_CELLS,
    transport_point,
)


def _run_transport(
    request: FiniteFieldPointTransportRequest,
) -> FiniteFieldPointTransportResult:
    field = request.isomorphism.target.field
    # Structural output bound: the transported point and its retained
    # isomorphism carry a fixed number of field elements, each with `degree`
    # coordinates below the characteristic. This replaces a serialized-byte
    # estimate over a worst-case JSON payload.
    output_digits = 6 * field.degree * len(str(field.characteristic))
    if output_digits > MAX_POINT_TRANSPORT_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="elliptic_curve.finite_field.point_transport_output_bound",
            message=(
                "transported point result exceeds the "
                f"{MAX_POINT_TRANSPORT_OUTPUT_CELLS}-digit output envelope"
            ),
        )
    return transport_point(request.isomorphism, request.point)


_F5 = {
    "characteristic": "5",
    "modulus_coefficients": ["0", "1"],
    "generator": "a",
}


def _element(value: int) -> dict[str, Any]:
    return {"presentation": _F5, "coordinates": [str(value)]}


def _curve(a: int, b: int) -> dict[str, Any]:
    return {
        "field": _F5,
        "coefficient_a": _element(a),
        "coefficient_b": _element(b),
    }


_SOURCE = _curve(1, 1)
_TARGET = _curve(1, 4)
_ISOMORPHISM = {
    "source": _SOURCE,
    "target": _TARGET,
    "isomorphic": True,
    "scaling": _element(2),
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="elliptic_curve.finite_field.isomorphism.transport_point.compute",
        title="Transport a finite-field elliptic point through a model isomorphism",
        description=(
            "Apply the witnessed short-Weierstrass coordinate map "
            "(x,y) -> (u^2*x,u^3*y), after checking the exact field, endpoint "
            "curves, and scaling identities."
        ),
        request_type=FiniteFieldPointTransportRequest,
        result_type=FiniteFieldPointTransportResult,
        run=_run_transport,
        tags=("elliptic-curve", "finite-field", "isomorphism", "exact"),
        discovery_terms=(
            "transport a point between isomorphic short Weierstrass curves over a finite field",
            "apply finite-field elliptic curve isomorphism to a point",
        ),
        examples=(
            OperationExample(
                name="transport_point_over_five",
                description="Map (0,1) on y^2=x^3+x+1 to the isomorphic model with scaling u=2.",
                input={
                    "isomorphism": _ISOMORPHISM,
                    "point": {
                        "curve": _SOURCE,
                        "at_infinity": False,
                        "x": _element(0),
                        "y": _element(1),
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
