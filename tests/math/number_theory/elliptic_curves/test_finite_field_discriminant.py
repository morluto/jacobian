"""Exact and contract tests for finite-field short-Weierstrass discriminants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)
from jacobian.math.number_theory.elliptic_curves._tools import (
    compute_finite_field_discriminant,
)
from jacobian.math.number_theory.elliptic_curves.finite_field import (
    FiniteFieldDiscriminantRequest,
    FiniteFieldDiscriminantResult,
    FiniteFieldShortWeierstrassCurve,
    finite_field_discriminant,
)


def _field(characteristic: int, modulus: tuple[int, ...]) -> FiniteFieldPresentation:
    return FiniteFieldPresentation(
        characteristic=characteristic,
        modulus_coefficients=modulus,
        generator="a",
    )


def _element(
    field: FiniteFieldPresentation, coordinates: tuple[int, ...]
) -> FiniteFieldElement:
    return FiniteFieldElement(presentation=field, coordinates=coordinates)


def test_nonsingular_known_answer_over_f5() -> None:
    field = _field(5, (0, 1))
    result = finite_field_discriminant(
        field, _element(field, (1,)), _element(field, (1,))
    )

    # 4·1³ = 4, 27·1² = 2, Δ = -16·(4 + 2) = -96 ≡ 4, j = 1728·4/6 ≡ 2 (mod 5).
    assert result.four_a_cubed.coordinates == (4,)
    assert result.twentyseven_b_squared.coordinates == (2,)
    assert result.discriminant.coordinates == (4,)
    assert result.is_nonsingular is True
    assert result.j_invariant is not None
    assert result.j_invariant.coordinates == (2,)


def test_defining_invariant_replayed_with_plain_modular_arithmetic() -> None:
    for prime, a, b in ((5, 1, 1), (5, 2, 3), (7, 3, 5), (11, 0, 4), (13, 6, 0)):
        field = _field(prime, (0, 1))
        result = finite_field_discriminant(
            field, _element(field, (a,)), _element(field, (b,))
        )
        four_a_cubed = (4 * pow(a, 3, prime)) % prime
        twentyseven_b_squared = (27 * pow(b, 2, prime)) % prime
        total = (four_a_cubed + twentyseven_b_squared) % prime
        expected_delta = (-16 * total) % prime

        assert result.four_a_cubed.coordinates == (four_a_cubed,)
        assert result.twentyseven_b_squared.coordinates == (twentyseven_b_squared,)
        assert result.discriminant.coordinates == (expected_delta,)
        assert result.is_nonsingular == (expected_delta != 0)
        if expected_delta:
            expected_j = (1728 * four_a_cubed * pow(total, -1, prime)) % prime
            assert result.j_invariant is not None
            assert result.j_invariant.coordinates == (expected_j,)
        else:
            assert result.j_invariant is None


def test_singular_pair_reports_zero_without_j() -> None:
    field = _field(5, (0, 1))
    result = finite_field_discriminant(
        field, _element(field, (0,)), _element(field, (0,))
    )

    assert result.discriminant.coordinates == (0,)
    assert result.is_nonsingular is False
    assert result.j_invariant is None


def test_extension_field_pair_is_nonsingular_with_bound_j() -> None:
    # x² + 2 is irreducible over F_5 (3 is not a square mod 5).
    field = _field(5, (2, 0, 1))
    result = finite_field_discriminant(
        field, _element(field, (1, 0)), _element(field, (1, 1))
    )

    assert result.is_nonsingular is True
    assert result.j_invariant is not None
    assert result.j_invariant.presentation == field
    assert result.discriminant.presentation == field


def test_first_curve_value_binds_one_field_and_model() -> None:
    field = _field(5, (0, 1))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field,
        coefficient_a=_element(field, (1,)),
        coefficient_b=_element(field, (1,)),
        model="SHORT_WEIERSTRASS_ODD_CHAR_GT_3",
    )

    assert curve.coefficient_a.presentation == curve.field
    with pytest.raises(ValidationError):
        FiniteFieldShortWeierstrassCurve(
            field=field,
            coefficient_a=_element(field, (1,)),
            coefficient_b=_element(_field(7, (0, 1)), (1,)),
        )


def test_native_and_catalog_results_agree() -> None:
    field = _field(5, (0, 1))
    native = finite_field_discriminant(
        field, _element(field, (1,)), _element(field, (1,))
    )
    catalog = compute_finite_field_discriminant(
        FiniteFieldDiscriminantRequest(
            field=field,
            coefficient_a=_element(field, (1,)),
            coefficient_b=_element(field, (1,)),
        )
    )

    assert catalog == native
    assert (
        FiniteFieldDiscriminantResult.model_validate_json(catalog.model_dump_json())
        == catalog
    )


@pytest.mark.parametrize("characteristic", [2, 3])
def test_characteristic_two_and_three_rejected_structurally(
    characteristic: int,
) -> None:
    modulus = (1, 1, 1) if characteristic == 2 else (0, 1)
    field = _field(characteristic, modulus)
    degree = field.degree

    with pytest.raises(OperationDomainValidationError):
        finite_field_discriminant(
            field,
            _element(field, (1,) + (0,) * (degree - 1)),
            _element(field, (0,) * degree),
        )


def test_mismatched_coefficient_presentation_rejected() -> None:
    field = _field(5, (0, 1))
    other = _field(7, (0, 1))

    with pytest.raises(OperationDomainValidationError):
        finite_field_discriminant(field, _element(field, (1,)), _element(other, (1,)))


def test_reducible_modulus_rejected_before_arithmetic() -> None:
    # x² is reducible over F_5, so the presentation is not a field.
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 0, 1), generator="a"
    )

    with pytest.raises(OperationDomainValidationError):
        finite_field_discriminant(
            field, _element(field, (1, 0)), _element(field, (0, 0))
        )


def test_forged_j_on_singular_pair_rejected() -> None:
    field = _field(5, (0, 1))
    zero = _element(field, (0,))
    result = finite_field_discriminant(field, zero, zero)

    with pytest.raises(ValidationError):
        FiniteFieldDiscriminantResult(
            field=field,
            coefficient_a=zero,
            coefficient_b=zero,
            four_a_cubed=zero,
            twentyseven_b_squared=zero,
            discriminant=zero,
            is_nonsingular=False,
            j_invariant=_element(field, (1,)),
        )
    assert result.j_invariant is None
