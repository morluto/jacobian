"""Polar pairing for characteristic-two polynomial quadratic forms."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)
from jacobian.math.number_theory.quadratic_forms.general.characteristic_two import (
    FiniteFieldQuadraticCrossTerm,
    FiniteFieldQuadraticForm,
    FiniteFieldQuadraticPairingRequest,
    FiniteFieldQuadraticVector,
)
from jacobian.math.number_theory.quadratic_forms.general.characteristic_two_operations import (
    polar_pairing_finite_field_quadratic_form,
)

F2 = FiniteFieldPresentation(
    characteristic=2, modulus_coefficients=(0, 1), generator="a"
)
F4 = FiniteFieldPresentation(
    characteristic=2, modulus_coefficients=(1, 1, 1), generator="a"
)


def elt(field: FiniteFieldPresentation, *coordinates: int) -> FiniteFieldElement:
    return FiniteFieldElement(presentation=field, coordinates=coordinates)


def vector(
    field: FiniteFieldPresentation, x: tuple[int, ...], y: tuple[int, ...]
) -> FiniteFieldQuadraticVector:
    return FiniteFieldQuadraticVector(
        field=field,
        axis=("x", "y"),
        coordinates=(elt(field, *x), elt(field, *y)),
    )


def form(
    field: FiniteFieldPresentation,
    diagonal: tuple[FiniteFieldElement, FiniteFieldElement],
    cross: FiniteFieldElement,
) -> FiniteFieldQuadraticForm:
    return FiniteFieldQuadraticForm(
        field=field,
        axis=("x", "y"),
        diagonal_coefficients=diagonal,
        cross_terms=(
            FiniteFieldQuadraticCrossTerm(left=0, right=1, coefficient=cross),
        ),
    )


def test_f2_polar_pairing_cancels_squares_but_retains_mixed_terms() -> None:
    q = form(F2, (elt(F2, 1), elt(F2, 0)), elt(F2, 1))
    left, right = vector(F2, (1,), (0,)), vector(F2, (0,), (1,))

    result = polar_pairing_finite_field_quadratic_form(q, left, right)

    # Q(1,1)-Q(1,0)-Q(0,1) = 0-1-0 = 1 in F2.
    assert result.value.coordinates == (1,)
    assert result.value.presentation == F2
    assert result.left == left
    assert result.right == right


def test_f4_polar_pairing_uses_extension_field_coefficients() -> None:
    alpha = elt(F4, 0, 1)
    one = elt(F4, 1, 0)
    zero = elt(F4, 0, 0)
    q = form(F4, (alpha, zero), one)
    left = vector(F4, (0, 1), (1, 0))
    right = vector(F4, (1, 0), (0, 1))

    result = polar_pairing_finite_field_quadratic_form(q, left, right)

    # Mixed contribution is a*a + 1*1 = a^2+1 = a.
    assert result.value.coordinates == (0, 1)


def test_pairing_rejects_odd_characteristic() -> None:
    f3 = FiniteFieldPresentation(
        characteristic=3, modulus_coefficients=(0, 1), generator="a"
    )
    q = form(f3, (elt(f3, 1), elt(f3, 0)), elt(f3, 1))

    with pytest.raises(OperationDomainValidationError) as error:
        polar_pairing_finite_field_quadratic_form(
            q, vector(f3, (1,), (1,)), vector(f3, (1,), (1,))
        )

    assert (
        error.value.errors()[0]["type"]
        == "quadratic_form.characteristic_two.requires_characteristic_two"
    )


def test_pairing_request_requires_matching_field_and_axis() -> None:
    from pydantic import ValidationError

    q = form(F2, (elt(F2, 1), elt(F2, 0)), elt(F2, 1))
    wrong_field = FiniteFieldQuadraticVector(
        field=F4,
        axis=("x", "y"),
        coordinates=(elt(F4, 1, 0), elt(F4, 0, 0)),
    )

    with pytest.raises(ValidationError):
        FiniteFieldQuadraticPairingRequest(
            form=q,
            left=wrong_field,
            right=vector(F2, (1,), (0,)),
        )


def test_pairing_result_survives_json_round_trip() -> None:
    q = form(F2, (elt(F2, 1), elt(F2, 0)), elt(F2, 1))
    left, right = vector(F2, (1,), (0,)), vector(F2, (0,), (1,))

    produced = polar_pairing_finite_field_quadratic_form(q, left, right)
    decoded = type(produced).model_validate_json(produced.model_dump_json())

    assert decoded == produced
