from fractions import Fraction
from math import isqrt

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
)
from jacobian.math.number_theory.modular_forms.basis import (
    GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID,
    modular_form_basis_q_expansions,
    modular_form_coordinates_q_expansion,
)
from jacobian.math.number_theory.modular_forms.operations import space_dimension
from jacobian.math.number_theory.modular_forms.transforms import sturm_bound
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)


def _space(weight: int = 3, character_coordinates: tuple[int, ...] = (1,)):
    return ModularFormSpace(
        level=4,
        weight=weight,
        kind="M",
        character=dirichlet_character(character_group(4), character_coordinates),
        coefficient_domain="QQ",
    )


def _integers(expansion):
    return [
        Fraction(coefficient.num, coefficient.den)
        for coefficient in expansion.q_expansion.coefficients
    ]


def _six_square_counts(precision: int) -> list[int]:
    """Count ordered signed six-tuples by independent finite enumeration."""

    one_coordinate = [0] * precision
    for value in range(-isqrt(precision - 1), isqrt(precision - 1) + 1):
        if value * value < precision:
            one_coordinate[value * value] += 1
    counts = [1] + [0] * (precision - 1)
    for _ in range(6):
        next_counts = [0] * precision
        for total, count in enumerate(counts):
            for square, multiplicity in enumerate(one_coordinate):
                if total + square < precision:
                    next_counts[total + square] += count * multiplicity
        counts = next_counts
    return counts


def test_weight_three_space_dimension_basis_and_theta_six_identity():
    space = _space()
    result = space_dimension(space)
    assert (result.dimension, result.eisenstein_dimension, result.cusp_dimension) == (
        2,
        2,
        0,
    )
    basis = modular_form_basis_q_expansions(space, 12)
    assert basis.space == space
    assert basis.basis_id == GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID
    assert [element.label for element in basis.elements] == [
        "A3_chi_minus4",
        "B3_chi_minus4",
    ]
    first, second = (_integers(element.expansion) for element in basis.elements)
    expected_first = [1, -4, -4, 32, -4, -104, 32, 192, -4, -292, -104, 480]
    expected_second = [0, 1, 4, 8, 16, 26, 32, 48, 64, 73, 104, 120]
    assert first == expected_first
    assert second == expected_second
    theta_six = [first[index] + 16 * second[index] for index in range(12)]
    assert theta_six == _six_square_counts(12)
    assert theta_six == [1, 12, 60, 160, 252, 312, 544, 960, 1020, 876, 1560, 2400]


def test_weight_three_coordinates_round_trip_and_expand():
    space = _space()
    form = ModularFormCoordinates(
        space=space,
        basis_id=GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID,
        coordinates=(
            CanonicalRational(num=1, den=1),
            CanonicalRational(num=16, den=1),
        ),
    )
    recovered = ModularFormCoordinates.model_validate_json(form.model_dump_json())
    assert recovered == form
    prefix = modular_form_coordinates_q_expansion(recovered, 12)
    assert prefix.space == space
    assert prefix.basis_id == GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID
    assert _integers(prefix) == _six_square_counts(12)
    assert form.space == recovered.space
    assert form.basis_id == recovered.basis_id
    assert form.coordinates == recovered.coordinates
    other = ModularFormCoordinates(
        space=space,
        basis_id=GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID,
        coordinates=(
            CanonicalRational(num=1, den=1),
            CanonicalRational(num=15, den=1),
        ),
    )
    assert form.coordinates != other.coordinates


def test_weight_three_sturm_bound_retains_character_parent():
    space = _space()
    bound = sturm_bound(space)
    assert bound.space == space
    assert bound.index == 6
    assert bound.bound == 1


def test_weight_three_character_target_binding_and_early_precision_bound():
    wrong_character = _space(character_coordinates=(0,))
    with pytest.raises(OperationDomainValidationError):
        modular_form_basis_q_expansions(wrong_character, 8)
    with pytest.raises(OperationResourceAdmissionError):
        modular_form_basis_q_expansions(_space(), 129)
