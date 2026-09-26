"""Independent exact checks for the Gamma0(2) polynomial basis slice."""

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
    modular_form_coordinates_u2,
)


def _rank(matrix: list[list[Fraction]]) -> int:
    rows = [row.copy() for row in matrix]
    pivot_row = 0
    for column in range(len(rows[0]) if rows else 0):
        pivot = next(
            (row for row in range(pivot_row, len(rows)) if rows[row][column]), None
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        scale = rows[pivot_row][column]
        rows[pivot_row] = [value / scale for value in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row or not rows[row][column]:
                continue
            scale = rows[row][column]
            rows[row] = [
                value - scale * pivot
                for value, pivot in zip(rows[row], rows[pivot_row], strict=True)
            ]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row


@pytest.mark.parametrize("weight", range(0, 122, 2))
def test_gamma0_two_monomials_are_complete_and_independent(weight: int) -> None:
    dimension = weight // 4 + 1
    space = ModularFormSpace(level=2, weight=weight, kind="M")
    basis = modular_form_basis_q_expansions(space, dimension)

    assert basis.basis_id == "gamma0-two-weight-2-4-monomials-v1"
    assert basis.precision == dimension
    assert len(basis.elements) == dimension
    matrix = [
        [
            vector.expansion.q_expansion.coefficients[row].as_fraction()
            for vector in basis.elements
        ]
        for row in range(dimension)
    ]
    assert _rank(matrix) == dimension


def test_level_two_weight_two_generator_and_weight_four_forms() -> None:
    weight_two = modular_form_basis_q_expansions(
        ModularFormSpace(level=2, weight=2, kind="M"), 8
    )
    assert [
        value.as_fraction()
        for value in weight_two.elements[0].expansion.q_expansion.coefficients
    ] == [1, 24, 24, 96, 24, 144, 96, 192]
    assert weight_two.elements[0].label == "A2"

    weight_four = modular_form_basis_q_expansions(
        ModularFormSpace(level=2, weight=4, kind="M"), 3
    )
    assert [element.label for element in weight_four.elements] == ["A2^2", "E4"]
    assert [
        coefficient.as_fraction()
        for coefficient in weight_four.elements[1].expansion.q_expansion.coefficients
    ] == [Fraction(1), Fraction(240), Fraction(2160)]


def test_gamma0_two_coordinates_round_trip_and_expand() -> None:
    space = ModularFormSpace(level=2, weight=4, kind="M")
    form = ModularFormCoordinates(
        space=space,
        basis_id="gamma0-two-weight-2-4-monomials-v1",
        coordinates=(CanonicalRational(num=1, den=1), CanonicalRational(num=0, den=1)),
    )
    same = ModularFormCoordinates.model_validate(form.model_dump())

    expansion = modular_form_coordinates_q_expansion(form, 4)
    assert [value.as_fraction() for value in expansion.q_expansion.coefficients] == [
        1,
        48,
        624,
        1344,
    ]
    assert same == form


def test_level_two_cusp_basis_and_mismatched_basis_identity_are_rejected() -> None:
    with pytest.raises(OperationDomainValidationError):
        modular_form_basis_q_expansions(
            ModularFormSpace(level=2, weight=8, kind="S"), 4
        )

    with pytest.raises(OperationDomainValidationError):
        modular_form_coordinates_q_expansion(
            ModularFormCoordinates(
                space=ModularFormSpace(level=2, weight=4, kind="M"),
                basis_id="level-one-e4-e6-monomials-v1",
                coordinates=(
                    CanonicalRational(num=1, den=1),
                    CanonicalRational(num=0, den=1),
                ),
            ),
            4,
        )


def test_gamma0_two_basis_admits_bounded_large_prefix_and_rejects_growth() -> None:
    space = ModularFormSpace(level=2, weight=120, kind="M")
    basis = modular_form_basis_q_expansions(space, 64)
    assert len(basis.elements) == 31

    with pytest.raises(OperationResourceAdmissionError):
        modular_form_basis_q_expansions(space, 128)


def test_t3_and_u2_reconstruct_exact_gamma0_two_images() -> None:
    space = ModularFormSpace(level=2, weight=4, kind="M")
    basis = modular_form_basis_q_expansions(space, 4)
    basis_id = "gamma0-two-weight-2-4-monomials-v1"
    a2_squared = ModularFormCoordinates(
        space=space,
        basis_id=basis_id,
        coordinates=(CanonicalRational(num=1, den=1), CanonicalRational(num=0, den=1)),
    )
    e4 = ModularFormCoordinates(
        space=space,
        basis_id=basis_id,
        coordinates=(CanonicalRational(num=0, den=1), CanonicalRational(num=1, den=1)),
    )

    t3_e4 = modular_form_coordinates_hecke(e4, 3)
    u2_e4 = modular_form_coordinates_u2(e4)

    assert tuple(value.as_fraction() for value in t3_e4.coordinates) == (0, 28)
    assert tuple(value.as_fraction() for value in u2_e4.coordinates) == (-10, 11)
    assert [
        value.as_fraction()
        for value in modular_form_coordinates_q_expansion(
            t3_e4, 2
        ).q_expansion.coefficients
    ] == [28, 6720]
    assert [
        value.as_fraction()
        for value in modular_form_coordinates_q_expansion(
            u2_e4, 2
        ).q_expansion.coefficients
    ] == [1, 2160]

    # Independent transforms of basis prefixes using their defining coefficient rules.
    for source, source_element in zip((a2_squared, e4), basis.elements, strict=True):
        coefficients = tuple(
            value.as_fraction()
            for value in source_element.expansion.q_expansion.coefficients
        )
        direct_t3 = tuple(
            sum(
                (
                    Fraction(divisor**3) * coefficients[3 * m // divisor**2]
                    for divisor in range(1, 4)
                    if 3 % divisor == 0 and m % divisor == 0
                ),
                Fraction(0),
            )
            for m in range(2)
        )
        direct_u2 = (coefficients[0], coefficients[2])
        assert (
            tuple(
                value.as_fraction()
                for value in modular_form_coordinates_q_expansion(
                    modular_form_coordinates_hecke(source, 3), 2
                ).q_expansion.coefficients
            )
            == direct_t3
        )
        assert (
            tuple(
                value.as_fraction()
                for value in modular_form_coordinates_q_expansion(
                    modular_form_coordinates_u2(source), 2
                ).q_expansion.coefficients
            )
            == direct_u2
        )


def test_level_two_operator_domains_and_precision_bounds() -> None:
    space = ModularFormSpace(level=2, weight=4, kind="M")
    basis_id = "gamma0-two-weight-2-4-monomials-v1"
    e4 = ModularFormCoordinates(
        space=space,
        basis_id=basis_id,
        coordinates=(CanonicalRational(num=0, den=1), CanonicalRational(num=1, den=1)),
    )
    with pytest.raises(OperationDomainValidationError):
        modular_form_coordinates_hecke(e4, 2)
    with pytest.raises(OperationResourceAdmissionError):
        modular_form_coordinates_hecke(e4, 129)
    assert tuple(
        value.as_fraction()
        for value in modular_form_coordinates_hecke(e4, 127).coordinates
    ) == (0, 2_048_384)

    level_one = ModularFormCoordinates(
        space=ModularFormSpace(level=1, weight=4, kind="M"),
        basis_id="level-one-e4-e6-monomials-v1",
        coordinates=(CanonicalRational(num=1, den=1),),
    )
    with pytest.raises(OperationDomainValidationError):
        modular_form_coordinates_u2(level_one)
    with pytest.raises(OperationDomainValidationError):
        modular_form_coordinates_u2(
            ModularFormCoordinates(
                space=ModularFormSpace(level=2, weight=8, kind="S"),
                basis_id=basis_id,
                coordinates=(CanonicalRational(num=1, den=1),),
            )
        )


def test_weight_zero_hecke_keeps_exact_rational_normalization() -> None:
    constant = ModularFormCoordinates(
        space=ModularFormSpace(level=2, weight=0, kind="M"),
        basis_id="gamma0-two-weight-2-4-monomials-v1",
        coordinates=(CanonicalRational(num=1, den=1),),
    )

    image = modular_form_coordinates_hecke(constant, 3)

    assert tuple(value.as_fraction() for value in image.coordinates) == (
        Fraction(4, 3),
    )
