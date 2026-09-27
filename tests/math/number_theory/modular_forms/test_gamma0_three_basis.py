"""Exact canonical basis and coordinate behavior at Gamma0(3)."""

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
)

BASIS_ID = "gamma0-three-weight-2-4-6-hypersurface-v1"


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


def _dimension(weight: int) -> int:
    if weight == 0 or weight == 2:
        return 1
    if weight % 2:
        return 0
    cusp_dimension = -(weight - 1) + 2 * (weight // 2 - 1) + weight // 3
    return cusp_dimension + 2


@pytest.mark.parametrize("weight", range(0, 96, 2))
def test_gamma0_three_basis_is_complete_through_sturm_precision(weight: int) -> None:
    precision = weight // 3 + 1
    basis = modular_form_basis_q_expansions(
        ModularFormSpace(level=3, weight=weight, kind="M"), precision
    )

    assert basis.basis_id == BASIS_ID
    assert len(basis.elements) == _dimension(weight)
    matrix = [
        [
            item.expansion.q_expansion.coefficients[row].as_fraction()
            for item in basis.elements
        ]
        for row in range(precision)
    ]
    assert _rank(matrix) == len(basis.elements)


def test_gamma0_three_generators_and_reduced_weight_eight_basis() -> None:
    weight_two = modular_form_basis_q_expansions(
        ModularFormSpace(level=3, weight=2, kind="M"), 6
    )
    assert [
        c.as_fraction()
        for c in weight_two.elements[0].expansion.q_expansion.coefficients
    ] == [
        1,
        12,
        36,
        12,
        84,
        72,
    ]
    assert weight_two.elements[0].label == "A2"

    weight_four = modular_form_basis_q_expansions(
        ModularFormSpace(level=3, weight=4, kind="M"), 4
    )
    assert [item.label for item in weight_four.elements] == ["A2^2", "B4star"]
    assert [
        c.as_fraction()
        for c in weight_four.elements[1].expansion.q_expansion.coefficients
    ] == [
        1,
        -30,
        -270,
        -570,
    ]

    weight_six = modular_form_basis_q_expansions(
        ModularFormSpace(level=3, weight=6, kind="M"), 4
    )
    assert [item.label for item in weight_six.elements] == ["A2^3", "A2*B4star", "S6"]
    assert [
        c.as_fraction()
        for c in weight_six.elements[2].expansion.q_expansion.coefficients
    ] == [
        0,
        1,
        -6,
        9,
    ]

    weight_eight = modular_form_basis_q_expansions(
        ModularFormSpace(level=3, weight=8, kind="M"), 3
    )
    assert [item.label for item in weight_eight.elements] == [
        "A2^4",
        "A2^2*B4star",
        "A2*S6",
    ]
    assert type(weight_eight).model_validate(weight_eight.model_dump()) == weight_eight


def test_gamma0_three_weight_eight_relation_through_sturm_bound() -> None:
    basis = modular_form_basis_q_expansions(
        ModularFormSpace(level=3, weight=8, kind="M"), 3
    )
    by_label = {
        item.label: item.expansion.q_expansion.coefficients for item in basis.elements
    }
    a2_squared = [value.as_fraction() for value in by_label["A2^4"]]
    a2_b4 = [value.as_fraction() for value in by_label["A2^2*B4star"]]
    a2_s6 = [value.as_fraction() for value in by_label["A2*S6"]]
    # The weight-8 relation is B4star^2 = A2^4 - 108*A2*S6.
    b4 = [1, -30, -270]
    b4_squared = [sum(b4[i] * b4[n - i] for i in range(n + 1)) for n in range(3)]
    assert b4_squared == [a2_squared[n] - 108 * a2_s6[n] for n in range(3)]
    assert a2_b4[0] == 1


def test_gamma0_three_coordinates_reconstruct_exact_prefix() -> None:
    space = ModularFormSpace(level=3, weight=8, kind="M")
    form = ModularFormCoordinates(
        space=space,
        basis_id=BASIS_ID,
        coordinates=(
            CanonicalRational(num=1, den=1),
            CanonicalRational(num=-2, den=1),
            CanonicalRational(num=3, den=2),
        ),
    )
    expansion = modular_form_coordinates_q_expansion(form, 8)
    assert [c.as_fraction() for c in expansion.q_expansion.coefficients] == [
        -1,
        Fraction(123, 2),
        2565,
        Fraction(74775, 2),
        290802,
        1411857,
        4967937,
        14700480,
    ]
    assert expansion.space == space
    assert expansion.basis_id == BASIS_ID


def test_gamma0_three_upper_admitted_weight_and_precision_compute() -> None:
    basis = modular_form_basis_q_expansions(
        ModularFormSpace(level=3, weight=94, kind="M"), 63
    )
    assert len(basis.elements) == 32
    assert basis.precision == 63


def test_gamma0_three_basis_rejects_unsupported_parent_and_growth() -> None:
    with pytest.raises(OperationDomainValidationError):
        modular_form_basis_q_expansions(
            ModularFormSpace(level=3, weight=4, kind="S"), 3
        )

    with pytest.raises(OperationResourceAdmissionError):
        modular_form_basis_q_expansions(
            ModularFormSpace(level=3, weight=96, kind="M"), 33
        )

    with pytest.raises(OperationResourceAdmissionError):
        modular_form_basis_q_expansions(
            ModularFormSpace(level=3, weight=4, kind="M"), 64
        )
