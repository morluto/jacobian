import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)
from jacobian.math.number_theory.elliptic_curves.finite_field import (
    FiniteFieldEllipticPoint,
    FiniteFieldShortWeierstrassCurve,
    finite_field_cardinality,
    finite_field_discriminant,
    finite_field_point_add,
    finite_field_point_negate,
    finite_field_points,
)


def test_finite_field_group_identities_and_cardinality() -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )
    points = finite_field_points(curve).points
    assert len(points) == 9
    for point in points:
        assert finite_field_point_add(
            curve, point, finite_field_point_negate(curve, point).point
        ).point.at_infinity
    result = finite_field_cardinality(curve)
    assert result.cardinality == len(points)
    assert result.trace == 5 + 1 - len(points)


def test_native_curve_consumers_reject_missing_authored_fields() -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    missing_coefficient = FiniteFieldShortWeierstrassCurve.model_construct(
        field=field, coefficient_a=one
    )
    with pytest.raises(OperationDomainValidationError):
        finite_field_points(missing_coefficient)

    malformed_coefficient = FiniteFieldElement.model_construct(coordinates=(1,))
    with pytest.raises(OperationDomainValidationError):
        finite_field_discriminant(field, malformed_coefficient, one)


def test_native_point_consumer_rejects_forged_coordinate_axis() -> None:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )
    forged_coordinate = FiniteFieldElement.model_construct(
        presentation=field, coordinates=(1, 2)
    )
    forged = FiniteFieldEllipticPoint.model_construct(
        curve=curve, at_infinity=False, x=one, y=forged_coordinate
    )
    with pytest.raises(OperationDomainValidationError) as error:
        finite_field_point_negate(curve, forged)
    assert error.value.errors()[0]["type"] == (
        "elliptic_curve.finite_field.point_coordinates"
    )
