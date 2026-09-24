from itertools import product

import pytest
from sympy import Poly, cyclotomic_poly, symbols

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.characters import (
    character_group,
    dirichlet_character_residue_indicator_expansion,
    operations,
)
from jacobian.math.number_theory.characters._tools import TOOLS


def _root_sum(exponents: list[int], order: int) -> Poly:
    z = symbols("z")
    return Poly(sum(z**exponent for exponent in exponents), z).rem(
        Poly(cyclotomic_poly(order, z), z)
    )


def _unit_coordinates(group, residue: int) -> tuple[int, ...]:
    for coords in product(*(range(n) for n in group.generator_orders)):
        value = 1 % group.modulus
        for generator, power in zip(group.generators, coords, strict=True):
            value = value * pow(generator, power, group.modulus) % group.modulus
        if value == residue:
            return coords
    raise AssertionError("group generators did not produce the supplied unit")


def test_character_coefficients_reconstruct_unit_indicator_exactly():
    for modulus in (1, 3, 5, 8, 12):
        group = character_group(modulus)
        for target in group.unit_residues:
            result = dirichlet_character_residue_indicator_expansion(group, target)
            assert result.denominator == group.character_count
            assert result.cyclotomic_order == group.exponent
            for value in group.unit_residues:
                x_coordinates = _unit_coordinates(group, value)
                terms = []
                for character, coefficient in zip(
                    result.character_coordinates,
                    result.coefficient_exponents,
                    strict=True,
                ):
                    character_exponent = sum(
                        c * x * (group.exponent // axis_order)
                        for c, x, axis_order in zip(
                            character,
                            x_coordinates,
                            group.generator_orders,
                            strict=True,
                        )
                    )
                    terms.append((coefficient + character_exponent) % group.exponent)
                expected = Poly(
                    group.character_count if value == target else 0, symbols("z")
                )
                assert _root_sum(terms, group.exponent) == expected.rem(
                    Poly(cyclotomic_poly(group.exponent, symbols("z")), symbols("z"))
                )


def test_nonunit_target_is_rejected():
    with pytest.raises(OperationDomainValidationError, match="unit residues"):
        dirichlet_character_residue_indicator_expansion(character_group(8), 2)


def test_work_is_admitted_before_dual_coordinates_are_built(monkeypatch):
    group = character_group(5)
    monkeypatch.setattr(operations, "MAX_CHARACTER_ORTHOGONALITY_WORK", 1)

    def unexpected_product(*_args, **_kwargs):
        raise AssertionError("dual coordinates must follow admission")

    monkeypatch.setattr(operations, "product", unexpected_product)
    with pytest.raises(OperationResourceAdmissionError, match="work envelope"):
        dirichlet_character_residue_indicator_expansion(group, 2)


def test_public_tool_example_computes_mod3_indicator():
    tool = next(
        item
        for item in TOOLS
        if item.operation_id
        == "dirichlet_character.residue_class_indicator_expansion.compute"
    )
    result = tool.run(
        tool.request_type.model_validate(
            {
                "group": {
                    "modulus": 3,
                    "unit_residues": [1, 2],
                    "character_count": 2,
                    "invariant_factors": [2],
                    "generators": [2],
                    "generator_orders": [2],
                    "unit_coordinates": [[0], [1]],
                    "exponent": 2,
                },
                "residue": 2,
            }
        )
    )
    assert result.character_coordinates == ((0,), (1,))
    assert result.coefficient_exponents == (0, 1)
    assert result.denominator == 2
