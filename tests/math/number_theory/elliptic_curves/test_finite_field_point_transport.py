from __future__ import annotations

import json
from functools import lru_cache

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)
from jacobian.math.number_theory.elliptic_curves.finite_field import (
    FiniteFieldEllipticPoint,
    FiniteFieldIsomorphismResult,
    FiniteFieldShortWeierstrassCurve,
    finite_field_isomorphism,
)
from jacobian.math.number_theory.elliptic_curves.point_transport._models import (
    FiniteFieldPointTransportRequest,
)
from jacobian.math.number_theory.elliptic_curves.point_transport._tools import TOOLS
from jacobian.math.number_theory.elliptic_curves.point_transport.operations import (
    transport_point,
)


def _field() -> FiniteFieldPresentation:
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


def _point(
    curve: FiniteFieldShortWeierstrassCurve, x: int, y: int
) -> FiniteFieldEllipticPoint:
    field = curve.field
    return FiniteFieldEllipticPoint.affine(
        curve,
        FiniteFieldElement(presentation=field, coordinates=(x,)),
        FiniteFieldElement(presentation=field, coordinates=(y,)),
    )


def _f5_points(a: int, b: int) -> tuple[tuple[int, int] | None, ...]:
    """Independent prime-field oracle; None denotes projective infinity."""

    affine = tuple(
        (x, y)
        for x in range(5)
        for y in range(5)
        if (y * y - (x * x * x + a * x + b)) % 5 == 0
    )
    return (None, *affine)


@lru_cache(maxsize=1)
def _isomorphism() -> FiniteFieldIsomorphismResult:
    return finite_field_isomorphism(_curve(1, 1), _curve(1, 4))


def test_transport_is_bijective_on_complete_f5_point_sets() -> None:
    source_curve, target_curve = _curve(1, 1), _curve(1, 4)
    isomorphism = _isomorphism()
    assert isomorphism.isomorphic
    assert isomorphism.scaling is not None
    assert isomorphism.scaling.coordinates == (2,)

    source_oracle = _f5_points(1, 1)
    target_oracle = _f5_points(1, 4)
    assert len(source_oracle) == len(target_oracle) == 9

    image_oracle = tuple(
        None if point is None else ((2 * 2 * point[0]) % 5, (2**3 * point[1]) % 5)
        for point in source_oracle
    )
    assert set(image_oracle) == set(target_oracle)

    for coordinates, expected in zip(source_oracle, image_oracle, strict=True):
        source_point = (
            FiniteFieldEllipticPoint.infinity(source_curve)
            if coordinates is None
            else _point(source_curve, *coordinates)
        )
        result = transport_point(
            FiniteFieldPointTransportRequest(
                isomorphism=isomorphism,
                point=source_point,
            )
        )
        assert result.source_point.curve == source_curve
        assert result.target_point.curve == target_curve
        actual = (
            None
            if result.target_point.at_infinity
            else (
                result.target_point.x.coordinates[0],
                result.target_point.y.coordinates[0],
            )
        )
        assert actual == expected


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


def test_transport_rechecks_caller_supplied_scaling_relation() -> None:
    isomorphism = _isomorphism()
    assert isomorphism.scaling is not None
    forged = isomorphism.model_copy(
        update={
            "scaling": FiniteFieldElement(
                presentation=isomorphism.source.field, coordinates=(1,)
            )
        }
    )
    with pytest.raises(OperationDomainValidationError) as error:
        transport_point(
            FiniteFieldPointTransportRequest(
                isomorphism=forged,
                point=_point(_curve(1, 1), 0, 1),
            )
        )
    assert error.value.errors()[0]["type"] == (
        "elliptic_curve.finite_field.point_transport_invalid_witness"
    )


def test_transport_rejects_point_from_another_curve() -> None:
    with pytest.raises(ValueError, match="source curve"):
        FiniteFieldPointTransportRequest(
            isomorphism=_isomorphism(),
            point=_point(_curve(1, 4), 0, 2),
        )
