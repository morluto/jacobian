"""Exact multiplication of forms with reconstructed target-space parents."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.modular_forms._tools import TOOLS
from jacobian.math.number_theory.modular_forms.basis import (
    BASIS_ID,
    GAMMA0_TWO_BASIS_ID,
    modular_form_coordinates_product,
    modular_form_coordinates_q_expansion,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)


def _form(
    level: int, weight: int, kind: str, basis_id: str, *coordinates: int
) -> ModularFormCoordinates:
    return ModularFormCoordinates(
        space=ModularFormSpace(level=level, weight=weight, kind=kind),
        basis_id=basis_id,
        coordinates=tuple(CanonicalRational(num=value, den=1) for value in coordinates),
    )


def _fractions(expansion) -> tuple[Fraction, ...]:
    return tuple(value.as_fraction() for value in expansion.q_expansion.coefficients)


def test_level_one_product_returns_coordinates_in_added_weight_space() -> None:
    e4 = _form(1, 4, "M", BASIS_ID, 1)

    result = modular_form_coordinates_product(e4, e4)

    assert result.space == ModularFormSpace(level=1, weight=8, kind="M")
    assert result.basis_id == BASIS_ID
    assert result.coordinates == (CanonicalRational(num=1, den=1),)
    expansion = modular_form_coordinates_q_expansion(result, 3)
    assert _fractions(expansion) == (Fraction(1), Fraction(480), Fraction(61_920))
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_cross_level_product_uses_lcm_parent_and_exact_sturm_coordinates() -> None:
    e4 = _form(1, 4, "M", BASIS_ID, 1)
    a2 = _form(2, 2, "M", GAMMA0_TWO_BASIS_ID, 1)

    result = modular_form_coordinates_product(e4, a2)

    assert result.space == ModularFormSpace(level=2, weight=6, kind="M")
    assert result.basis_id == GAMMA0_TWO_BASIS_ID
    # In the ordered basis (A2^3, A2*E4), the product is the second vector.
    assert result.coordinates == (
        CanonicalRational(num=0, den=1),
        CanonicalRational(num=1, den=1),
    )


def test_product_preserves_cuspidality_in_the_target_space() -> None:
    delta = _form(1, 12, "S", BASIS_ID, 1)
    e4 = _form(1, 4, "M", BASIS_ID, 1)

    result = modular_form_coordinates_product(delta, e4)

    assert result.space == ModularFormSpace(level=1, weight=16, kind="S")
    assert result.coordinates == (CanonicalRational(num=1, den=1),)
    assert _fractions(modular_form_coordinates_q_expansion(result, 3)) == (
        Fraction(0),
        Fraction(1),
        Fraction(216),
    )


def test_product_rejects_target_levels_without_a_supported_exact_basis() -> None:
    level_three = _form(3, 2, "M", "gamma0-three-weight-2-4-6-hypersurface-v1", 1)
    level_four = _form(4, 2, "M", "gamma0-four-weight-2-generators-v1", 1)

    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_coordinates_product(level_three, level_four)

    assert error.value.errors()[0]["type"] == (
        "modular_form.product_target_level_unsupported"
    )


def test_product_operation_is_published_with_an_exact_example() -> None:
    operation = next(
        item
        for item in TOOLS
        if item.operation_id == "modular_form.coordinates.product.compute"
    )
    assert operation.request_type is not None
    assert operation.result_type is ModularFormCoordinates
    assert operation.examples[0].name == "e4_squared"
