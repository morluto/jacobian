"""Exact prime-to-level Hecke action on the Gamma0(3) basis."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.modular_forms import (
    ModularFormCoordinates,
    ModularFormSpace,
    modular_form_basis_q_expansions,
    modular_form_coordinates_hecke,
    modular_form_coordinates_q_expansion,
)

BASIS_ID = "gamma0-three-weight-2-4-6-hypersurface-v1"


def _form(weight: int, coordinates: tuple[int, ...]) -> ModularFormCoordinates:
    return ModularFormCoordinates(
        space=ModularFormSpace(level=3, weight=weight, kind="M"),
        basis_id=BASIS_ID,
        coordinates=tuple(CanonicalRational(num=value, den=1) for value in coordinates),
    )


def _hecke_t2_coefficients(
    coefficients: tuple[Fraction, ...], weight: int, precision: int
) -> tuple[Fraction, ...]:
    """Independent direct evaluation of T_2's coefficient formula."""

    output = []
    for m in range(precision):
        value = Fraction(0)
        for divisor in (1, 2):
            if m % divisor == 0:
                exponent = weight - 1
                factor = Fraction(1, divisor) if exponent == -1 else divisor**exponent
                value += factor * coefficients[(2 * m) // (divisor * divisor)]
        output.append(value)
    return tuple(output)


def test_t2_acts_exactly_on_gamma0_three_weights_two_and_four() -> None:
    weight_two = modular_form_coordinates_hecke(_form(2, (1,)), 2)
    assert tuple(c.as_fraction() for c in weight_two.coordinates) == (Fraction(3),)

    for coordinates in ((1, 0), (0, 1), (3, -2)):
        source = _form(4, coordinates)
        image = modular_form_coordinates_hecke(source, 2)
        assert image.space == source.space
        assert image.basis_id == BASIS_ID
        assert tuple(c.as_fraction() for c in image.coordinates) == tuple(
            Fraction(9 * value) for value in coordinates
        )


def test_t2_coordinate_reconstruction_matches_direct_coefficients_at_weight_40() -> (
    None
):
    weight = 40
    sturm_bound = weight // 3
    source_order = 2 * sturm_bound + 1
    basis = modular_form_basis_q_expansions(
        ModularFormSpace(level=3, weight=weight, kind="M"), source_order
    )
    source = _form(weight, (1, *([0] * (len(basis.elements) - 1))))
    image = modular_form_coordinates_hecke(source, 2)
    prefix = modular_form_coordinates_q_expansion(image, sturm_bound + 1)
    source_coefficients = tuple(
        sum(
            (
                coefficient
                * item.expansion.q_expansion.coefficients[index].as_fraction()
                for coefficient, item in zip(
                    (Fraction(1), *([Fraction(0)] * (len(basis.elements) - 1))),
                    basis.elements,
                    strict=True,
                )
            ),
            Fraction(0),
        )
        for index in range(source_order)
    )
    expected = _hecke_t2_coefficients(source_coefficients, weight, sturm_bound + 1)
    assert tuple(c.as_fraction() for c in prefix.q_expansion.coefficients) == expected


def test_gamma0_three_hecke_rejects_level_dividing_index_and_growth() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_coordinates_hecke(_form(4, (1, 0)), 3)
    assert (
        error.value.errors()[0]["type"] == "modular_form.coordinates_hecke_not_coprime"
    )

    dimension = len(
        modular_form_basis_q_expansions(
            ModularFormSpace(level=3, weight=94, kind="M"), 1
        ).elements
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        modular_form_coordinates_hecke(_form(94, (1, *([0] * (dimension - 1)))), 2)
    assert (
        error.value.errors()[0]["type"]
        == "modular_form.coordinates_operator_growth_bound"
    )
