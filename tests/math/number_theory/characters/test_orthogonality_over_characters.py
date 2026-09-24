"""Exact dual-group orthogonality over Dirichlet characters."""

from __future__ import annotations

import json
from itertools import product

import pytest
from sympy import Poly, cyclotomic_poly, symbols

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.characters import _tools
from jacobian.math.number_theory.characters import operations as native_operations
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character_orthogonality_over_characters,
)


def _direct_cyclotomic_sum(group, left: int, right: int) -> int:
    """Brute-force every dual coordinate and reduce its exact root sum."""
    modulus = group.modulus
    left_residue = left % modulus
    right_residue = right % modulus
    unit_rows = dict(zip(group.unit_residues, group.unit_coordinates, strict=True))
    left_row = unit_rows.get(left_residue)
    right_row = unit_rows.get(right_residue)
    if left_row is None or right_row is None:
        return 0

    root_order = group.exponent
    x = symbols("x")
    terms = []
    for coordinates in product(*(range(order) for order in group.generator_orders)):
        exponent = sum(
            coordinate
            * (root_order // axis_order)
            * (left_coordinate - right_coordinate)
            for coordinate, axis_order, left_coordinate, right_coordinate in zip(
                coordinates,
                group.generator_orders,
                left_row,
                right_row,
                strict=True,
            )
        ) % root_order
        terms.append(x**exponent)

    polynomial = Poly(sum(terms), x, domain="ZZ")
    root_polynomial = Poly(cyclotomic_poly(root_order, x), x, domain="ZZ")
    remainder = polynomial.rem(root_polynomial)
    assert remainder.degree() <= 0
    return int(remainder.nth(0))


@pytest.mark.parametrize("modulus", (1, 2, 3, 4, 5, 7, 8, 12))
def test_all_residue_pairs_match_independent_cyclotomic_sum(modulus: int) -> None:
    group = character_group(modulus)
    for left in range(modulus):
        for right in range(modulus):
            result = dirichlet_character_orthogonality_over_characters(
                group, left, right
            )
            assert result.value == _direct_cyclotomic_sum(group, left, right)
            assert result.group == group
            assert result.left_residue == left
            assert result.right_residue == right


def test_negative_and_large_inputs_normalize_to_the_same_unit_class() -> None:
    group = character_group(5)
    result = dirichlet_character_orthogonality_over_characters(group, -3, 7)

    assert (result.left_residue, result.right_residue, result.value) == (2, 2, 4)
    assert result.left_integer == -3 and result.right_integer == 7
    assert result.model_validate_json(result.model_dump_json()) == result


def test_modulus_one_residue_zero_is_the_unique_unit_class() -> None:
    group = character_group(1)
    result = dirichlet_character_orthogonality_over_characters(group, 123, -456)

    assert (result.left_residue, result.right_residue, result.value) == (0, 0, 1)


def test_rejects_forged_group_and_integer_outside_the_input_envelope() -> None:
    group = character_group(5)
    forged = group.model_copy(update={"unit_residues": (1, 2, 3)})
    with pytest.raises(OperationDomainValidationError, match="complete canonical unit"):
        dirichlet_character_orthogonality_over_characters(forged, 1, 1)

    with pytest.raises(OperationResourceAdmissionError, match="digit bound"):
        dirichlet_character_orthogonality_over_characters(group, 10**256, 1)


def test_output_bound_precedes_residue_class_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    group = character_group(5)

    class TinyOutputLimit:
        max_output_bytes = 1

    def unexpected_lookup(*_args: object, **_kwargs: object) -> None:
        pytest.fail("residue lookup ran before output admission")

    monkeypatch.setattr(native_operations, "CanonicalLimits", TinyOutputLimit)
    monkeypatch.setattr(native_operations, "bisect_left", unexpected_lookup)
    with pytest.raises(OperationResourceAdmissionError, match="output bound"):
        dirichlet_character_orthogonality_over_characters(group, 1, 1)


def test_catalog_example_runs_and_result_round_trips() -> None:
    tool = next(
        tool
        for tool in _tools.TOOLS
        if tool.operation_id
        == "dirichlet_character.orthogonality_over_characters.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)

    assert result.value == 4
    assert tool.result_type.model_validate_json(
        json.dumps(result.model_dump(mode="json"))
    ) == result
