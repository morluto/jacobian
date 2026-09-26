"""Independent small-field fixtures for the characteristic-two polynomial carrier."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)
from jacobian.math.number_theory.quadratic_forms.general.characteristic_two import (
    FiniteFieldQuadraticCrossTerm,
    FiniteFieldQuadraticEvaluationRequest,
    FiniteFieldQuadraticForm,
    FiniteFieldQuadraticPairingRequest,
    FiniteFieldQuadraticVector,
)
from jacobian.math.number_theory.quadratic_forms.general.characteristic_two_operations import (
    evaluate_finite_field_quadratic_form,
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


def test_f2_evaluation_retains_diagonal_square_terms_and_round_trips() -> None:
    q = form(F2, (elt(F2, 1), elt(F2, 0)), elt(F2, 1))
    point = vector(F2, (1,), (1,))
    request = FiniteFieldQuadraticEvaluationRequest(form=q, vector=point)

    result = evaluate_finite_field_quadratic_form(request)

    # Independently: Q(1,1) = 1*1^2 + 1*1*1 = 1+1 = 0 in F2.
    assert result.value.coordinates == (0,)
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert decoded.value == result.value


def test_f2_polar_pairing_cancels_squares_but_retains_mixed_terms() -> None:
    q = form(F2, (elt(F2, 1), elt(F2, 0)), elt(F2, 1))
    e_x, e_y = vector(F2, (1,), (0,)), vector(F2, (0,), (1,))

    result = polar_pairing_finite_field_quadratic_form(
        FiniteFieldQuadraticPairingRequest(form=q, left=e_x, right=e_y)
    )

    # Q(1,1)-Q(1,0)-Q(0,1) = 0-1-0 = 1 in F2.
    assert result.value.coordinates == (1,)


def test_f4_evaluation_and_polar_pairing_use_extension_field_coefficients() -> None:
    alpha = elt(F4, 0, 1)
    one = elt(F4, 1, 0)
    zero = elt(F4, 0, 0)
    q = form(F4, (alpha, zero), one)
    point = vector(F4, (0, 1), (1, 0))

    evaluated = evaluate_finite_field_quadratic_form(
        FiniteFieldQuadraticEvaluationRequest(form=q, vector=point)
    )
    # In F4, a^2=a+1 and a^3=1; Q(a,1)=a*a^2+a=1+a.
    assert evaluated.value.coordinates == (1, 1)

    left = vector(F4, (0, 1), (1, 0))
    right = vector(F4, (1, 0), (0, 1))
    paired = polar_pairing_finite_field_quadratic_form(
        FiniteFieldQuadraticPairingRequest(form=q, left=left, right=right)
    )
    # Mixed contribution is a*a + 1*1 = a^2+1 = a.
    assert paired.value.coordinates == (0, 1)


def test_operations_reject_odd_characteristic() -> None:
    f3 = FiniteFieldPresentation(
        characteristic=3, modulus_coefficients=(0, 1), generator="a"
    )
    q = form(f3, (elt(f3, 1), elt(f3, 0)), elt(f3, 1))
    request = FiniteFieldQuadraticEvaluationRequest(
        form=q, vector=vector(f3, (1,), (1,))
    )

    with pytest.raises(OperationDomainValidationError) as error:
        evaluate_finite_field_quadratic_form(request)

    assert (
        error.value.errors()[0]["type"]
        == "quadratic_form.characteristic_two.requires_characteristic_two"
    )


def test_form_rejects_coefficients_from_a_different_parent() -> None:
    with pytest.raises(ValueError):
        FiniteFieldQuadraticForm(
            field=F2,
            axis=("x",),
            diagonal_coefficients=(elt(F4, 1, 0),),
        )
