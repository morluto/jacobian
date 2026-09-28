from itertools import product

import pytest
from sympy import Poly, cyclotomic_poly, symbols

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory.characters import (
    DirichletCharacterFourierMatrix,
    character_group,
    dirichlet_character_fourier_matrix,
    operations,
)
from jacobian.math.number_theory.characters._models import (
    DirichletCharacterFourierMatrixRequest,
)
from jacobian.math.number_theory.characters._tools import TOOLS


def _reduce_root_sum(exponents: list[int], order: int) -> Poly:
    """Represent a sum of roots of unity in Q[zeta_order] exactly."""
    variable = symbols("z")
    polynomial = sum(variable**exponent for exponent in exponents)
    return Poly(polynomial, variable).rem(
        Poly(cyclotomic_poly(order, variable), variable)
    )


def _direct_unit_coordinates(group, residue: int) -> tuple[int, ...]:
    for coordinates in product(*(range(order) for order in group.generator_orders)):
        candidate = 1 % group.modulus
        for generator, coordinate in zip(group.generators, coordinates, strict=True):
            candidate = (
                candidate * pow(generator, coordinate, group.modulus) % group.modulus
            )
        if candidate == residue:
            return coordinates
    raise AssertionError("the declared generators did not produce a unit")


def test_complete_fourier_axes_and_entries_match_independent_small_group_enumeration():
    for modulus in (1, 3, 5, 7, 8, 12):
        group = character_group(modulus)
        matrix = dirichlet_character_fourier_matrix(group)

        expected_characters = tuple(
            product(*(range(order) for order in group.generator_orders))
        )
        assert matrix.group == group
        assert matrix.character_coordinates == expected_characters
        assert matrix.unit_residues == group.unit_residues
        assert matrix.cyclotomic_order == group.exponent
        expected = tuple(
            tuple(
                sum(
                    coordinate * unit_coordinate * (group.exponent // axis_order)
                    for coordinate, unit_coordinate, axis_order in zip(
                        character,
                        _direct_unit_coordinates(group, residue),
                        group.generator_orders,
                        strict=True,
                    )
                )
                % group.exponent
                for residue in group.unit_residues
            )
            for character in expected_characters
        )
        assert matrix.entries == expected


def test_complete_matrix_has_exact_both_sided_orthogonality():
    for modulus in (1, 3, 5, 7, 8, 12):
        matrix = dirichlet_character_fourier_matrix(character_group(modulus))
        count = len(matrix.unit_residues)
        order = matrix.cyclotomic_order
        for left, right in product(range(count), repeat=2):
            row_differences = [
                (a - b) % order
                for a, b in zip(
                    matrix.entries[left], matrix.entries[right], strict=True
                )
            ]
            target = Poly(count if left == right else 0, symbols("z"))
            assert _reduce_root_sum(row_differences, order) == target.rem(
                Poly(cyclotomic_poly(order, symbols("z")), symbols("z"))
            )
            column_differences = [
                (matrix.entries[row][left] - matrix.entries[row][right]) % order
                for row in range(count)
            ]
            target = Poly(count if left == right else 0, symbols("z"))
            assert _reduce_root_sum(column_differences, order) == target.rem(
                Poly(cyclotomic_poly(order, symbols("z")), symbols("z"))
            )


def test_fourier_matrix_roundtrips_with_parent_and_axes():
    matrix = dirichlet_character_fourier_matrix(character_group(5))
    decoded = DirichletCharacterFourierMatrix.model_validate_json(
        encode_strict_json(matrix.model_dump(mode="json"))
    )
    assert decoded == matrix


def test_fourier_matrix_rejects_excess_work_before_matrix_construction(monkeypatch):
    group = character_group(5)
    monkeypatch.setattr(operations, "MAX_CHARACTER_ORTHOGONALITY_WORK", 1)

    def unexpected_construction(cls, **fields):
        raise AssertionError("matrix construction must follow complete admission")

    monkeypatch.setattr(
        DirichletCharacterFourierMatrix,
        "_from_kernel",
        classmethod(unexpected_construction),
    )
    with pytest.raises(OperationResourceAdmissionError, match="work envelope"):
        dirichlet_character_fourier_matrix(group)


def test_fourier_matrix_catalog_contract():
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "dirichlet_character.group.fourier_matrix.compute"
    )
    assert tool.request_type is DirichletCharacterFourierMatrixRequest
    assert tool.result_type is DirichletCharacterFourierMatrix
    assert tool.run(
        DirichletCharacterFourierMatrixRequest(group=character_group(3))
    ) == (dirichlet_character_fourier_matrix(character_group(3)))
