from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.quaternions import (
    RationalUnitQuaternion,
    conjugate_rational_unit_quaternion,
    inverse_rational_unit_quaternion,
    multiply_rational_unit_quaternions,
    rational_unit_quaternion_scalar_part,
)


def q(*coordinates: Fraction) -> RationalUnitQuaternion:
    return RationalUnitQuaternion(
        coordinates=tuple(
            CanonicalRational.from_fraction(value) for value in coordinates
        )
    )


def test_hamilton_product_is_noncommutative_and_inverse_is_two_sided() -> None:
    i = q(Fraction(0), Fraction(1), Fraction(0), Fraction(0))
    j = q(Fraction(0), Fraction(0), Fraction(1), Fraction(0))
    ij = multiply_rational_unit_quaternions(i, j)
    ji = multiply_rational_unit_quaternions(j, i)
    assert tuple(value.as_fraction() for value in ij.coordinates) == (0, 0, 0, 1)
    assert tuple(value.as_fraction() for value in ji.coordinates) == (0, 0, 0, -1)

    value = q(Fraction(3, 5), Fraction(4, 5), Fraction(0), Fraction(0))
    inverse = inverse_rational_unit_quaternion(value)
    assert inverse == conjugate_rational_unit_quaternion(value)
    identity = q(Fraction(1), Fraction(0), Fraction(0), Fraction(0))
    assert multiply_rational_unit_quaternions(value, inverse) == identity
    assert multiply_rational_unit_quaternions(inverse, value) == identity
    assert rational_unit_quaternion_scalar_part(value).as_fraction() == Fraction(3, 5)


def test_unit_norm_is_required_and_json_roundtrip_is_stable() -> None:
    with pytest.raises(ValidationError):
        RationalUnitQuaternion(
            coordinates=(CanonicalRational.from_fraction(Fraction(1)),) * 4
        )
    value = q(Fraction(3, 5), Fraction(4, 5), Fraction(0), Fraction(0))
    assert RationalUnitQuaternion.model_validate_json(value.model_dump_json()) == value


def test_component_and_product_growth_bounds_precede_expansion() -> None:
    # M=4*10^255 gives 512-digit numerators and denominators in the standard
    # rational parametrization of the unit circle.
    m = 4 * 10**255
    within = q(
        Fraction(m * m - 1, m * m + 1),
        Fraction(2 * m, m * m + 1),
        Fraction(0),
        Fraction(0),
    )
    assert len(str(within.coordinates[0].den)) == 512

    oversized_m = 10**256
    with pytest.raises(ValidationError):
        RationalUnitQuaternion(
            coordinates=(
                CanonicalRational.from_fraction(
                    Fraction(oversized_m**2 - 1, oversized_m**2 + 1)
                ),
                CanonicalRational.from_fraction(
                    Fraction(2 * oversized_m, oversized_m**2 + 1)
                ),
                CanonicalRational.from_fraction(Fraction(0)),
                CanonicalRational.from_fraction(Fraction(0)),
            )
        )

    # Each operand fits, but their denominator products exceed the output cap.
    m = 10**128
    first = q(
        Fraction(m * m - 1, m * m + 1),
        Fraction(2 * m, m * m + 1),
        Fraction(0),
        Fraction(0),
    )
    second = q(
        Fraction(m * m - 1, m * m + 1),
        Fraction(0),
        Fraction(2 * m, m * m + 1),
        Fraction(0),
    )
    with pytest.raises(OperationResourceAdmissionError):
        multiply_rational_unit_quaternions(first, second)
