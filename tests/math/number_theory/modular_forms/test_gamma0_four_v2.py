"""Exact checks for the bounded Gamma0(4) basis and level-raising V2 map."""

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
    modular_form_coordinates_q_expansion,
    modular_form_coordinates_v2,
)


def _rank(matrix: list[list[Fraction]]) -> int:
    rows = [row.copy() for row in matrix]
    pivots = 0
    for column in range(len(rows[0]) if rows else 0):
        pivot = next((r for r in range(pivots, len(rows)) if rows[r][column]), None)
        if pivot is None:
            continue
        rows[pivots], rows[pivot] = rows[pivot], rows[pivots]
        value = rows[pivots][column]
        rows[pivots] = [entry / value for entry in rows[pivots]]
        for row in range(len(rows)):
            if row == pivots or not rows[row][column]:
                continue
            value = rows[row][column]
            rows[row] = [
                a - value * b for a, b in zip(rows[row], rows[pivots], strict=True)
            ]
        pivots += 1
    return pivots


@pytest.mark.parametrize("weight", range(0, 62, 2))
def test_gamma0_four_monomials_have_full_sturm_rank(weight: int) -> None:
    dimension = weight // 2 + 1
    basis = modular_form_basis_q_expansions(
        ModularFormSpace(level=4, weight=weight, kind="M"), dimension
    )
    assert basis.basis_id == "gamma0-four-weight-2-generators-v1"
    assert len(basis.elements) == dimension
    matrix = [
        [
            element.expansion.q_expansion.coefficients[row].as_fraction()
            for element in basis.elements
        ]
        for row in range(dimension)
    ]
    assert _rank(matrix) == dimension


def test_generators_and_weight_zero_v2_edge_case() -> None:
    basis = modular_form_basis_q_expansions(
        ModularFormSpace(level=4, weight=2, kind="M"), 8
    )
    assert [element.label for element in basis.elements] == ["B4", "D4"]
    assert [
        x.as_fraction() for x in basis.elements[0].expansion.q_expansion.coefficients
    ] == [1, 0, 24, 0, 24, 0, 96, 0]
    assert [
        x.as_fraction() for x in basis.elements[1].expansion.q_expansion.coefficients
    ] == [0, 1, 0, 4, 0, 6, 0, 8]

    one = ModularFormCoordinates(
        space=ModularFormSpace(level=2, weight=0, kind="M"),
        basis_id="gamma0-two-weight-2-4-monomials-v1",
        coordinates=(CanonicalRational(num=1, den=1),),
    )
    image = modular_form_coordinates_v2(one)
    assert image.space.level == 4
    assert image.coordinates == (CanonicalRational(num=1, den=1),)

    weight_two_generator = ModularFormCoordinates(
        space=ModularFormSpace(level=2, weight=2, kind="M"),
        basis_id="gamma0-two-weight-2-4-monomials-v1",
        coordinates=(CanonicalRational(num=1, den=1),),
    )
    raised = modular_form_coordinates_v2(weight_two_generator)
    assert tuple(value.as_fraction() for value in raised.coordinates) == (1, 0)

    odd_weight_basis = modular_form_basis_q_expansions(
        ModularFormSpace(level=4, weight=3, kind="M"), 2
    )
    assert odd_weight_basis.elements == ()


def test_v2_exact_target_coordinates_and_independent_coefficient_rule() -> None:
    for weight in range(0, 18, 2):
        source_space = ModularFormSpace(level=2, weight=weight, kind="M")
        source_basis = modular_form_basis_q_expansions(source_space, 12)
        target_bound = weight // 2
        target_precision = target_bound + 1
        target_space = ModularFormSpace(level=4, weight=weight, kind="M")
        for element in source_basis.elements:
            # Test each source basis vector using its standard unit coordinate.
            coordinate_index = source_basis.elements.index(element)
            source = ModularFormCoordinates(
                space=source_space,
                basis_id=source_basis.basis_id,
                coordinates=tuple(
                    CanonicalRational(num=int(i == coordinate_index), den=1)
                    for i in range(len(source_basis.elements))
                ),
            )
            image = modular_form_coordinates_v2(source)
            assert image.space == target_space
            assert image.basis_id == "gamma0-four-weight-2-generators-v1"
            actual = modular_form_coordinates_q_expansion(image, target_precision)
            src = [c.as_fraction() for c in element.expansion.q_expansion.coefficients]
            expected = [
                src[n // 2] if n % 2 == 0 else Fraction(0)
                for n in range(target_precision)
            ]
            assert [
                c.as_fraction() for c in actual.q_expansion.coefficients
            ] == expected
            assert (
                ModularFormCoordinates.model_validate_json(image.model_dump_json())
                == image
            )


def test_v2_rejects_unrepresented_source_or_target_spaces() -> None:
    level_one = ModularFormCoordinates(
        space=ModularFormSpace(level=1, weight=4, kind="M"),
        basis_id="level-one-e4-e6-monomials-v1",
        coordinates=(CanonicalRational(num=1, den=1),),
    )
    level_one_image = modular_form_coordinates_v2(level_one)
    assert level_one_image.space == ModularFormSpace(level=2, weight=4, kind="M")
    assert level_one_image.basis_id == "gamma0-two-weight-2-4-monomials-v1"

    # A Gamma0(4) form is not silently treated as a source level-two form.
    target = ModularFormCoordinates(
        space=ModularFormSpace(level=4, weight=2, kind="M"),
        basis_id="gamma0-four-weight-2-generators-v1",
        coordinates=(CanonicalRational(num=1, den=1), CanonicalRational(num=0, den=1)),
    )
    with pytest.raises(OperationDomainValidationError):
        modular_form_coordinates_v2(target)

    high_weight_source = ModularFormCoordinates(
        space=ModularFormSpace(level=2, weight=24, kind="M"),
        basis_id="gamma0-two-weight-2-4-monomials-v1",
        coordinates=tuple(CanonicalRational(num=int(i == 0), den=1) for i in range(7)),
    )
    with pytest.raises(OperationResourceAdmissionError):
        modular_form_coordinates_v2(high_weight_source)

    with pytest.raises(OperationDomainValidationError):
        modular_form_basis_q_expansions(
            ModularFormSpace(level=4, weight=2, kind="S"), 2
        )
