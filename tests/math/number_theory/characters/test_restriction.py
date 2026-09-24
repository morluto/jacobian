"""Exact factor-down of Dirichlet characters along unit reduction."""

from __future__ import annotations

from itertools import product

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.characters import (
    character_group,
    dirichlet_character_inflate,
    dirichlet_character_restrict_modulus,
)
from jacobian.math.number_theory.characters.operations import (
    dirichlet_character,
    dirichlet_character_value,
)


def _small_divisors(modulus: int) -> tuple[int, ...]:
    return tuple(divisor for divisor in range(1, modulus + 1) if modulus % divisor == 0)


def test_restriction_agrees_with_independent_small_fiber_tables():
    for source_modulus in (1, 2, 3, 4, 5, 8, 12, 15, 20):
        source_group = character_group(source_modulus)
        coordinates = tuple(
            product(*(range(order) for order in source_group.generator_orders))
        )
        for source_coordinates in coordinates:
            source = dirichlet_character(source_group, source_coordinates)
            for target_modulus in _small_divisors(source_modulus):
                target_group = character_group(target_modulus)
                fibers: dict[int, list[tuple[int, object]]] = {
                    residue: [] for residue in target_group.unit_residues
                }
                for source_residue in source_group.unit_residues:
                    value = dirichlet_character_value(source, source_residue).value
                    assert value is not None
                    fibers[source_residue % target_modulus].append(
                        (source_residue, value)
                    )
                expected_to_factor = all(
                    all(value == fiber[0][1] for _residue, value in fiber)
                    for fiber in fibers.values()
                )

                result = dirichlet_character_restrict_modulus(source, target_modulus)

                if not expected_to_factor:
                    assert result.status == "does_not_factor"
                    assert result.obstruction is not None
                    witness = result.obstruction
                    assert witness.target_unit_residue in fibers
                    assert witness.source_unit_residues[0] % target_modulus == (
                        witness.source_unit_residues[1] % target_modulus
                    )
                    assert witness.values[0] != witness.values[1]
                    for residue, value in zip(
                        witness.source_unit_residues, witness.values, strict=True
                    ):
                        assert dirichlet_character_value(source, residue).value == value
                    continue

                assert result.status == "descended"
                assert result.target is not None
                assert result.target_unit_residues == target_group.unit_residues
                assert result.source_unit_lifts is not None
                assert (
                    tuple(lift % target_modulus for lift in result.source_unit_lifts)
                    == result.target_unit_residues
                )
                for target_residue, source_lift in zip(
                    result.target_unit_residues,
                    result.source_unit_lifts,
                    strict=True,
                ):
                    target_value = dirichlet_character_value(
                        result.target, target_residue
                    ).value
                    source_value = dirichlet_character_value(source, source_lift).value
                    assert target_value is not None and source_value is not None
                    assert (
                        target_value.exponent
                        * (source_group.exponent // target_group.exponent)
                        == source_value.exponent
                    )
                assert (
                    dirichlet_character_inflate(result.target, source_modulus).target
                    == source
                )


def test_mod8_character_has_a_concrete_mod4_nonfactor_obstruction():
    source = dirichlet_character(character_group(8), (0, 1))

    result = dirichlet_character_restrict_modulus(source, 4)

    assert result.status == "does_not_factor"
    assert result.obstruction is not None
    assert result.obstruction.target_unit_residue == 1
    assert result.obstruction.source_unit_residues == (1, 5)
    assert result.obstruction.values[0] != result.obstruction.values[1]


def test_restriction_rejects_a_target_that_is_not_a_positive_divisor():
    source = dirichlet_character(character_group(8), (1, 0))

    with pytest.raises(OperationDomainValidationError) as error:
        dirichlet_character_restrict_modulus(source, 6)

    assert error.value.errors()[0]["type"] == (
        "dirichlet_character.restriction.modulus_not_divisor"
    )


def test_nontrivial_mod8_character_descends_and_inflation_recovers_it():
    source = dirichlet_character(character_group(8), (1, 0))

    result = dirichlet_character_restrict_modulus(source, 4)

    assert result.status == "descended"
    assert result.target is not None
    assert result.target.coordinates == (1,)
    pulled_back = dirichlet_character_inflate(result.target, 8)
    assert pulled_back.target == source
