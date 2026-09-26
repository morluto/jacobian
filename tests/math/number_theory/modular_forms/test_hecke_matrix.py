"""Exact Hecke matrices in the represented modular-form bases."""

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
    modular_form_coordinates_hecke,
    modular_form_hecke_matrix,
)


def _space(weight: int = 4) -> ModularFormSpace:
    return ModularFormSpace(level=2, weight=weight, kind="M")


def _form(coordinates: tuple[int, ...]) -> ModularFormCoordinates:
    return ModularFormCoordinates(
        space=_space(),
        basis_id="gamma0-two-weight-2-4-monomials-v1",
        coordinates=tuple(CanonicalRational(num=value, den=1) for value in coordinates),
    )


def test_hecke_matrix_rows_columns_and_matrix_vector_agree_with_coordinates() -> None:
    matrix = modular_form_hecke_matrix(_space(), 3)
    assert matrix.basis_id == "gamma0-two-weight-2-4-monomials-v1"
    assert matrix.index == 3
    assert matrix.row_labels == matrix.column_labels == ("A2^2", "E4")
    entries = tuple(
        tuple(value.as_fraction() for value in row) for row in matrix.entries
    )
    assert entries == ((Fraction(28), Fraction(0)), (Fraction(0), Fraction(28)))

    for coordinates in ((1, 0), (0, 1), (3, -2)):
        image = modular_form_coordinates_hecke(_form(coordinates), 3)
        matrix_image = tuple(
            sum(entries[row][column] * coordinates[column] for column in range(2))
            for row in range(2)
        )
        assert matrix_image == tuple(value.as_fraction() for value in image.coordinates)


def test_t1_matrix_is_identity_and_direct_q_formula_matches_column_action() -> None:
    matrix = modular_form_hecke_matrix(_space(), 1)
    entries = tuple(
        tuple(value.as_fraction() for value in row) for row in matrix.entries
    )
    assert entries == ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))

    # Independent q-expansion formula for T_3 on E4 gives b_0 = 1+3^3
    # and b_1 = a_3. Those agree with the matrix image 28 E4.
    e4 = (Fraction(1), Fraction(240), Fraction(2160), Fraction(6720))
    direct_constant = sum(Fraction(divisor**3) * e4[0] for divisor in (1, 3))
    direct_q = e4[3]
    assert (direct_constant, direct_q) == (Fraction(28), Fraction(6720))
    t3_entries = tuple(
        tuple(value.as_fraction() for value in row)
        for row in modular_form_hecke_matrix(_space(), 3).entries
    )
    assert tuple(
        sum(t3_entries[row][col] * (0, 1)[col] for col in range(2)) for row in range(2)
    ) == (0, 28)


def test_hecke_matrix_enforces_coprimality_and_source_precision() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_hecke_matrix(_space(), 2)
    assert error.value.errors()[0]["type"] == "modular_form.hecke_matrix_not_coprime"

    with pytest.raises(OperationResourceAdmissionError) as error:
        modular_form_hecke_matrix(ModularFormSpace(level=1, weight=120, kind="M"), 13)
    assert error.value.errors()[0]["type"] == "modular_form.hecke_matrix_source_bound"
