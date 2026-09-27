"""Defining-equation and exact Smith tests for finite lattice extensions."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.crystallographic.extensions._models import (
    FiniteLatticeExtension,
)
from jacobian.math.geometry.crystallographic.extensions.operations import (
    decide_extension_torsion,
)


def _extension(
    *,
    table: list[list[int]],
    action: list[list[list[int]]],
    cocycle: list[list[list[int]]],
) -> FiniteLatticeExtension:
    return FiniteLatticeExtension.model_validate(
        {
            "multiplication_table": table,
            "action_matrices": action,
            "factor_set": cocycle,
        }
    )


def _klein_extension() -> FiniteLatticeExtension:
    return _extension(
        table=[[0, 1], [1, 0]],
        action=[[[1, 0], [0, 1]], [[1, 0], [0, -1]]],
        cocycle=[
            [[0, 0], [0, 0]],
            [[0, 0], [1, 0]],
        ],
    )


def test_klein_bottle_extension_is_torsion_free_by_odd_norm_obstruction() -> None:
    result = decide_extension_torsion(_klein_extension())

    assert result.torsion_free
    assert result.conclusion == "TORSION_FREE"
    assert result.torsion_witness is None
    (obstruction,) = result.lift_obstructions
    assert obstruction.holonomy_element == 1
    assert obstruction.holonomy_order == 2
    assert obstruction.norm_matrix == ((2, 0), (0, 0))
    assert obstruction.power_offset == (1, 0)
    assert obstruction.obstruction_vector == (1, 0)
    assert obstruction.modulus == 2
    assert obstruction.pairing == -1
    assert all(
        sum(
            obstruction.obstruction_vector[i] * obstruction.norm_matrix[i][j]
            for i in range(2)
        )
        % obstruction.modulus
        == 0
        for j in range(2)
    )
    assert obstruction.pairing % obstruction.modulus != 0


def test_split_reflection_extension_has_explicit_order_two_lift() -> None:
    source = _extension(
        table=[[0, 1], [1, 0]],
        action=[[[1, 0], [0, 1]], [[1, 0], [0, -1]]],
        cocycle=[
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ],
    )

    result = decide_extension_torsion(source)

    assert not result.torsion_free
    assert result.conclusion == "HAS_TORSION"
    witness = result.torsion_witness
    assert witness is not None
    assert witness.holonomy_element == 1
    assert witness.translation_part == (0, 0)
    assert witness.power_offset == (0, 0)
    assert witness.norm_matrix == ((2, 0), (0, 0))


def test_trivial_holonomy_extension_is_the_translation_lattice() -> None:
    source = _extension(
        table=[[0]],
        action=[[[1]]],
        cocycle=[[[0]]],
    )

    result = decide_extension_torsion(source)

    assert result.torsion_free
    assert result.lift_obstructions == ()


def test_cocycle_identity_is_checked_before_torsion_testing() -> None:
    source = _extension(
        table=[[0, 1], [1, 0]],
        action=[[[1]], [[-1]]],
        cocycle=[[[0], [0]], [[0], [1]]],
    )

    with pytest.raises(OperationDomainValidationError) as error:
        decide_extension_torsion(source)

    assert (
        error.value.errors()[0]["type"] == "crystallographic.extension.cocycle_identity"
    )


def test_nonfaithful_linear_holonomy_is_rejected() -> None:
    source = _extension(
        table=[[0, 1], [1, 0]],
        action=[[[1]], [[1]]],
        cocycle=[[[0], [0]], [[0], [1]]],
    )

    with pytest.raises(OperationDomainValidationError) as error:
        decide_extension_torsion(source)

    assert (
        error.value.errors()[0]["type"]
        == "crystallographic.extension.action_not_faithful"
    )


def test_catalog_example_is_an_executable_canonical_request() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "crystallographic.extension.torsion_freeness.decide"
    )
    import json

    source = tool.request_type.model_validate_json(
        json.dumps(tool.examples[0].input), strict=True
    )
    result = tool.run(source)
    assert result.torsion_free


def test_integer_change_of_section_preserves_torsion_freeness() -> None:
    original = _klein_extension()
    changed = _extension(
        table=[[0, 1], [1, 0]],
        action=[[[1, 0], [0, 1]], [[1, 0], [0, -1]]],
        # Add delta(u) for u_0=(0,0), u_1=(1,2): f'(g,g)=f(g,g)+u_g+rho(g)u_g=(3,0).
        cocycle=[[[0, 0], [0, 0]], [[0, 0], [3, 0]]],
    )

    first = decide_extension_torsion(original)
    second = decide_extension_torsion(changed)

    assert first.torsion_free and second.torsion_free
    assert first.lift_obstructions[0].power_offset == (1, 0)
    assert second.lift_obstructions[0].power_offset == (3, 0)


def test_hantzsche_wendt_extension_checks_every_nonidentity_lift() -> None:
    # Affine lifts with rational shifts (1/2,0,0), (0,1/2,1/2), and
    # (1/2,-1/2,-1/2) induce this integral cocycle on the diagonal C2 x C2
    # holonomy group. Each element has a different fixed lattice direction.
    source = _extension(
        table=[
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0],
        ],
        action=[
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        ],
        cocycle=[
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [-1, 1, 1], [0, 1, 0], [-1, 0, 1]],
            [[0, 0, 0], [0, -1, -1], [0, -1, 0], [0, 0, -1]],
        ],
    )

    from fractions import Fraction

    shifts = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1, 2), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1, 2), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(-1, 2), Fraction(-1, 2)),
    )
    for g in range(4):
        for h in range(4):
            product = source.multiplication_table[g][h]
            affine_offset = tuple(
                shifts[g][i]
                + sum(source.action_matrices[g][i][j] * shifts[h][j] for j in range(3))
                - shifts[product][i]
                for i in range(3)
            )
            assert all(value.denominator == 1 for value in affine_offset)
            assert (
                tuple(int(value) for value in affine_offset) == source.factor_set[g][h]
            )

    result = decide_extension_torsion(source)

    assert result.torsion_free
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result
    assert tuple(item.holonomy_element for item in result.lift_obstructions) == (
        1,
        2,
        3,
    )
    assert tuple(item.power_offset for item in result.lift_obstructions) == (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, -1),
    )
    for item in result.lift_obstructions:
        norm_pairing = tuple(
            sum(item.obstruction_vector[i] * item.norm_matrix[i][j] for i in range(3))
            for j in range(3)
        )
        if item.modulus == 0:
            assert norm_pairing == (0, 0, 0)
            assert item.pairing != 0
        else:
            assert all(value % item.modulus == 0 for value in norm_pairing)
            assert item.pairing % item.modulus != 0
