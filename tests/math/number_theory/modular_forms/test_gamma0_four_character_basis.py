from fractions import Fraction
from math import isqrt

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
)
from jacobian.math.number_theory.modular_forms.basis import (
    GAMMA0_FOUR_CHI4_WEIGHT_ONE_BASIS_ID,
    GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID,
    modular_form_basis_q_expansions,
    modular_form_coordinates_hecke,
    modular_form_coordinates_q_expansion,
    modular_form_coordinates_u2,
)
from jacobian.math.number_theory.modular_forms.operations import space_dimension
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)


def _chi4_space(kind: str = "M", weight: int = 1) -> ModularFormSpace:
    chi4 = dirichlet_character(character_group(4), (1,))
    return ModularFormSpace(
        level=4,
        weight=weight,
        kind=kind,
        character=chi4,
        coefficient_domain="QQ",
    )


def test_exact_gamma0_four_character_dimension_and_basis():
    space = _chi4_space()
    result = space_dimension(space)
    assert (result.dimension, result.eisenstein_dimension, result.cusp_dimension) == (
        1,
        1,
        0,
    )
    basis = modular_form_basis_q_expansions(space, 12)
    assert basis.basis_id == GAMMA0_FOUR_CHI4_WEIGHT_ONE_BASIS_ID
    assert [element.label for element in basis.elements] == ["G1_chi_minus4"]
    actual = [
        Fraction(coefficient.num, coefficient.den)
        for coefficient in basis.elements[0].expansion.q_expansion.coefficients
    ]
    assert actual == [Fraction(1, 4), 1, 1, 0, 1, 2, 0, 0, 1, 1, 2, 0]
    theta_square = [
        sum(
            1
            for first in range(-isqrt(index), isqrt(index) + 1)
            for second in range(-isqrt(index), isqrt(index) + 1)
            if first * first + second * second == index
        )
        for index in range(12)
    ]
    assert [4 * coefficient for coefficient in actual] == theta_square


def test_character_basis_coordinates_round_trip():
    space = _chi4_space()
    form = ModularFormCoordinates(
        space=space,
        basis_id=GAMMA0_FOUR_CHI4_WEIGHT_ONE_BASIS_ID,
        coordinates=(CanonicalRational(num=4, den=1),),
    )
    wire = form.model_dump_json()
    recovered = ModularFormCoordinates.model_validate_json(wire)
    assert recovered == form
    expansion = modular_form_coordinates_q_expansion(recovered, 12)
    assert [
        coefficient.num // coefficient.den
        for coefficient in expansion.q_expansion.coefficients
    ] == [1, 4, 4, 0, 4, 8, 0, 0, 4, 4, 8, 0]
    assert form.space == recovered.space
    assert form.basis_id == recovered.basis_id
    assert form.coordinates == recovered.coordinates


def test_character_target_rejects_wrong_character_or_weight():
    chi4 = dirichlet_character(character_group(4), (1,))
    for level, weight, character in (
        (4, 1, dirichlet_character(character_group(4), (0,))),
        (4, 5, chi4),
    ):
        space = ModularFormSpace(
            level=level,
            weight=weight,
            kind="M",
            character=character,
        )
        try:
            modular_form_basis_q_expansions(space, 8)
        except Exception as error:
            assert "chi_{-4}" in str(error) or "unsupported" in str(error)
        else:
            raise AssertionError("unsupported character space was admitted")


def test_weight_three_character_hecke_actions_match_exact_divisor_formula():
    space = _chi4_space(weight=3)
    a = ModularFormCoordinates(
        space=space,
        basis_id=GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID,
        coordinates=(
            CanonicalRational(num=1, den=1),
            CanonicalRational(num=0, den=1),
        ),
    )
    b = ModularFormCoordinates(
        space=space,
        basis_id=GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID,
        coordinates=(
            CanonicalRational(num=0, den=1),
            CanonicalRational(num=1, den=1),
        ),
    )

    assert tuple(
        value.as_fraction()
        for value in modular_form_coordinates_hecke(a, 3).coordinates
    ) == (
        Fraction(-8),
        Fraction(0),
    )
    assert tuple(
        value.as_fraction()
        for value in modular_form_coordinates_hecke(b, 3).coordinates
    ) == (
        Fraction(0),
        Fraction(8),
    )
    assert tuple(
        value.as_fraction()
        for value in modular_form_coordinates_hecke(a, 5).coordinates
    ) == (
        Fraction(26),
        Fraction(0),
    )
    assert tuple(
        value.as_fraction()
        for value in modular_form_coordinates_hecke(b, 5).coordinates
    ) == (
        Fraction(0),
        Fraction(26),
    )


def test_character_hecke_requires_an_index_coprime_to_level_four():
    form = ModularFormCoordinates(
        space=_chi4_space(weight=3),
        basis_id=GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID,
        coordinates=(
            CanonicalRational(num=1, den=1),
            CanonicalRational(num=0, den=1),
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_coordinates_hecke(form, 2)
    assert (
        error.value.errors()[0]["type"] == "modular_form.coordinates_hecke_not_coprime"
    )


def test_weight_one_character_hecke_actions_match_divisor_sum_eigenvalues():
    form = ModularFormCoordinates(
        space=_chi4_space(),
        basis_id=GAMMA0_FOUR_CHI4_WEIGHT_ONE_BASIS_ID,
        coordinates=(CanonicalRational(num=1, den=1),),
    )
    assert modular_form_coordinates_hecke(form, 3).coordinates == (
        CanonicalRational(num=0, den=1),
    )
    assert modular_form_coordinates_hecke(form, 5).coordinates == (
        CanonicalRational(num=2, den=1),
    )


def test_u2_on_character_spaces_matches_direct_coefficient_extraction():
    def chi_minus4(n: int) -> int:
        return 0 if n % 2 == 0 else (1 if n % 4 == 1 else -1)

    def divisors(n: int) -> tuple[int, ...]:
        return tuple(value for value in range(1, n + 1) if n % value == 0)

    def a3(n: int) -> int:
        if n == 0:
            return 1
        return -4 * sum(chi_minus4(d) * d**2 for d in divisors(n))

    def b3(n: int) -> int:
        if n == 0:
            return 0
        return sum(chi_minus4(n // d) * d**2 for d in divisors(n))

    def g1(n: int) -> Fraction:
        if n == 0:
            return Fraction(1, 4)
        return Fraction(sum(chi_minus4(d) for d in divisors(n)))

    for weight, basis_id, vectors, coefficient in (
        (
            1,
            GAMMA0_FOUR_CHI4_WEIGHT_ONE_BASIS_ID,
            ((Fraction(1),),),
            lambda n, _: g1(n),
        ),
        (
            3,
            GAMMA0_FOUR_CHI4_WEIGHT_THREE_BASIS_ID,
            ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
            lambda n, index: (a3(n), b3(n))[index],
        ),
    ):
        for vector_index, vector in enumerate(vectors):
            form = ModularFormCoordinates(
                space=_chi4_space(weight=weight),
                basis_id=basis_id,
                coordinates=tuple(
                    CanonicalRational(num=value.numerator, den=value.denominator)
                    for value in vector
                ),
            )
            image = modular_form_coordinates_u2(
                ModularFormCoordinates.model_validate_json(form.model_dump_json())
            )
            prefix = modular_form_coordinates_q_expansion(image, 12)
            expected = [
                sum(
                    (scalar * coefficient(2 * n, vector_index) for scalar in vector),
                    Fraction(0),
                )
                for n in range(12)
            ]
            assert [
                entry.as_fraction() for entry in prefix.q_expansion.coefficients
            ] == expected
            if weight == 1:
                assert tuple(value.as_fraction() for value in image.coordinates) == (
                    Fraction(1),
                )
            else:
                assert tuple(value.as_fraction() for value in image.coordinates) == (
                    (Fraction(1), Fraction(0))
                    if vector_index == 0
                    else (Fraction(0), Fraction(4))
                )
