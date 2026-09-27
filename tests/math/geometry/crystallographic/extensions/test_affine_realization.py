"""Exact affine realization of finite lattice-extension sections."""

from __future__ import annotations

import json
from fractions import Fraction

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.geometry.crystallographic.extensions._models import (
    FiniteLatticeExtension,
)
from jacobian.math.geometry.crystallographic.extensions.operations import (
    affine_section_realization,
)

_S3_TABLE = (
    (0, 1, 2, 3, 4, 5),
    (1, 0, 4, 5, 2, 3),
    (2, 5, 0, 4, 3, 1),
    (3, 4, 5, 0, 1, 2),
    (4, 3, 1, 2, 5, 0),
    (5, 2, 3, 1, 0, 4),
)
_S3_ACTION = (
    ((1, 0), (0, 1)),
    ((-1, 1), (0, 1)),
    ((1, 0), (1, -1)),
    ((0, -1), (-1, 0)),
    ((0, -1), (1, -1)),
    ((-1, 1), (-1, 0)),
)
_S3_COCYCLE = (
    ((0, 0),) * 6,
    ((0, 0), (0, 0), (-1, 0), (0, 0), (-1, 0), (0, 0)),
    ((0, 0), (1, 0), (2, 1), (1, 0), (1, 0), (1, 0)),
    ((0, 0), (0, 0), (0, -1), (0, 0), (0, 0), (-1, 0)),
    ((0, 0), (0, 0), (0, 1), (-1, 0), (0, 0), (0, 0)),
    ((0, 0), (-1, 0), (-1, -1), (0, 0), (0, 0), (0, 0)),
)


def _s3_extension() -> FiniteLatticeExtension:
    return FiniteLatticeExtension(
        multiplication_table=_S3_TABLE,
        action_matrices=_S3_ACTION,
        factor_set=_S3_COCYCLE,
    )


def _translation_extension() -> FiniteLatticeExtension:
    return FiniteLatticeExtension(
        multiplication_table=((0,),),
        action_matrices=(((1, 0), (0, 1)),),
        factor_set=(((0, 0),),),
    )


def _shift(result, element: int) -> tuple[Fraction, ...]:
    return tuple(
        value.as_fraction() for value in result.section_maps[element].section_shift
    )


def _matvec(
    matrix: tuple[tuple[int, ...], ...], vector: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    return tuple(
        sum(
            (matrix[row][column] * vector[column] for column in range(len(vector))),
            Fraction(),
        )
        for row in range(len(matrix))
    )


def test_s3_average_cochain_and_affine_section_law() -> None:
    source = _s3_extension()

    result = affine_section_realization(source)

    assert result.source == source
    assert tuple(item.holonomy_element for item in result.section_maps) == tuple(
        range(6)
    )
    shifts = tuple(_shift(result, g) for g in range(6))
    for g in range(6):
        independently_averaged = tuple(
            Fraction(sum(source.factor_set[g][h][i] for h in range(6)), 6)
            for i in range(2)
        )
        assert shifts[g] == independently_averaged

    for g in range(6):
        for h in range(6):
            gh = _S3_TABLE[g][h]
            composed = tuple(
                shifts[g][i] + _matvec(_S3_ACTION[g], shifts[h])[i] for i in range(2)
            )
            expected_extension_product = tuple(
                shifts[gh][i] + _S3_COCYCLE[g][h][i] for i in range(2)
            )
            assert composed == expected_extension_product

    # This square is deliberately nontrivial: the chosen section maps do not
    # falsely become a representation of S3 when the source cocycle is nonzero.
    assert _S3_TABLE[2][2] == 0
    assert _S3_COCYCLE[2][2] == (2, 1)
    composed_b_b = tuple(
        shifts[2][i] + _matvec(_S3_ACTION[2], shifts[2])[i] for i in range(2)
    )
    assert composed_b_b != shifts[0]
    assert composed_b_b == tuple(shifts[0][i] + _S3_COCYCLE[2][2][i] for i in range(2))


def test_full_extension_affine_maps_obey_extension_multiplication() -> None:
    source = _s3_extension()
    result = affine_section_realization(source)
    g, h = 2, 2
    v = (2, -1)
    w = (-3, 4)
    gh = source.multiplication_table[g][h]

    # T_v o A_g o T_w o A_h has translation
    # v + rho(g)w + q_g + rho(g)q_h.
    composed = tuple(
        Fraction(v[i])
        + sum(_S3_ACTION[g][i][j] * w[j] for j in range(2))
        + _shift(result, g)[i]
        + _matvec(_S3_ACTION[g], _shift(result, h))[i]
        for i in range(2)
    )
    product_translation = tuple(
        v[i]
        + sum(_S3_ACTION[g][i][j] * w[j] for j in range(2))
        + _S3_COCYCLE[g][h][i]
        + _shift(result, gh)[i]
        for i in range(2)
    )
    assert composed == product_translation


def test_trivial_holonomy_realizes_the_translation_lattice() -> None:
    result = affine_section_realization(_translation_extension())

    assert len(result.section_maps) == 1
    identity = result.section_maps[0]
    assert identity.linear_part == ((1, 0), (0, 1))
    assert _shift(result, 0) == (Fraction(0), Fraction(0))
    # T_v o A_0 is the ordinary translation x -> x+v.
    lattice_vector = (3, -2)
    assert tuple(_shift(result, 0)[i] + lattice_vector[i] for i in range(2)) == (
        Fraction(3),
        Fraction(-2),
    )


def test_affine_realization_round_trips_and_manifest_examples_execute() -> None:
    result = affine_section_realization(_s3_extension())
    assert type(result).model_validate_json(result.model_dump_json()) == result

    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "crystallographic.extension.affine_realization.compute"
    )
    assert len(tool.examples) == 2
    for example in tool.examples:
        request = tool.request_type.model_validate_json(
            json.dumps(example.input), strict=True
        )
        realized = tool.run(request)
        assert realized.source == request
