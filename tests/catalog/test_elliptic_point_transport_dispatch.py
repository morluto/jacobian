"""Dispatch-level publication check for the elliptic point transport operation.

This exercises the published operation boundary, so it lives in the catalog
lane rather than under ``tests/math``.
"""

import json

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.finite_fields.values import FiniteFieldElement
from jacobian.math.number_theory.elliptic_curves.finite_field import (
    FiniteFieldEllipticPoint,
    FiniteFieldShortWeierstrassCurve,
    finite_field_isomorphism,
)
from jacobian.math.number_theory.elliptic_curves.point_transport._tools import TOOLS


def _field():
    from jacobian.math.finite_fields.values import FiniteFieldPresentation

    return FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )


def _curve(a: int, b: int) -> FiniteFieldShortWeierstrassCurve:
    field = _field()
    return FiniteFieldShortWeierstrassCurve(
        field=field,
        coefficient_a=FiniteFieldElement(presentation=field, coordinates=(a,)),
        coefficient_b=FiniteFieldElement(presentation=field, coordinates=(b,)),
    )


def _isomorphism():
    return finite_field_isomorphism(_curve(1, 1), _curve(1, 4))


def test_transport_public_dispatch_roundtrips_and_maps_infinity() -> None:
    source_curve = _curve(1, 1)
    operation = TOOLS[0]
    input_value = {
        "isomorphism": _isomorphism().model_dump(mode="json"),
        "point": FiniteFieldEllipticPoint.infinity(source_curve).model_dump(
            mode="json"
        ),
    }
    result = invoke_operation(
        operation.operation_id, input_value, Catalog((operation,))
    )
    assert result.output["target_point"]["at_infinity"] is True
    decoded = operation.result_type.model_validate_json(json.dumps(result.output))
    assert decoded.target_point.at_infinity
    assert decoded.target_point.curve == _curve(1, 4)
