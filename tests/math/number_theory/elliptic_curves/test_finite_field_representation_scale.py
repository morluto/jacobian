from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)
from jacobian.math.number_theory.elliptic_curves.finite_field import (
    FiniteFieldEllipticPoint,
    FiniteFieldShortWeierstrassCurve,
    finite_field_point_add,
)


def test_point_addition_over_next_admitted_prime_field() -> None:
    field = FiniteFieldPresentation(
        characteristic=65_537, modulus_coefficients=(0, 1), generator="a"
    )

    def element(value: int) -> FiniteFieldElement:
        return FiniteFieldElement(presentation=field, coordinates=(value,))

    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=element(1), coefficient_b=element(1)
    )
    point = FiniteFieldEllipticPoint.affine(curve, element(0), element(1))

    result = finite_field_point_add(curve, point, point)

    assert result.point.curve == curve
    assert result.point.x is not None and result.point.y is not None
    assert result.point.x.coordinates == (49_153,)
    assert result.point.y.coordinates == (8_191,)
