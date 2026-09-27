"""Finite positive-root posets composed with the canonical poset value."""

from __future__ import annotations

import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.posets.core.operations import width
from jacobian.math.groups.root_systems._models import CartanMatrix, RootPosetResult
from jacobian.math.groups.root_systems._tools import TOOLS
from jacobian.math.groups.root_systems.operations import (
    _admit_root_poset_work,
    root_poset,
)

Matrix = tuple[tuple[int, ...], ...]


def _pairs(result: RootPosetResult) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    roots_by_label = dict(
        zip(result.poset.elements, result.positive_roots, strict=True)
    )
    return {
        (roots_by_label[pair.lower], roots_by_label[pair.upper])
        for pair in result.poset.strict_order_pairs
    }


@pytest.mark.parametrize(
    ("matrix", "roots", "strict_pairs", "covers", "incomparables"),
    (
        (
            ((2, -1), (-1, 2)),
            ((0, 1), (1, 0), (1, 1)),
            {((0, 1), (1, 1)), ((1, 0), (1, 1))},
            {((0, 1), (1, 1)), ((1, 0), (1, 1))},
            {frozenset(((0, 1), (1, 0)))},
        ),
        (
            ((2, -2), (-1, 2)),
            ((0, 1), (1, 0), (1, 1), (2, 1)),
            {
                ((0, 1), (1, 1)),
                ((0, 1), (2, 1)),
                ((1, 0), (1, 1)),
                ((1, 0), (2, 1)),
                ((1, 1), (2, 1)),
            },
            {
                ((0, 1), (1, 1)),
                ((1, 0), (1, 1)),
                ((1, 1), (2, 1)),
            },
            {frozenset(((0, 1), (1, 0)))},
        ),
    ),
)
def test_a2_and_b2_root_orders_match_explicit_coordinate_fixtures(
    matrix: Matrix,
    roots: tuple[tuple[int, ...], ...],
    strict_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]],
    covers: set[tuple[tuple[int, ...], tuple[int, ...]]],
    incomparables: set[frozenset[tuple[int, ...]]],
) -> None:
    result = root_poset(CartanMatrix.model_validate(matrix))

    assert result.positive_roots == roots
    assert result.datum.cartan_matrix.entries == matrix
    assert result.datum.cartan_matrix.simple_root_axis == tuple(range(len(matrix)))
    assert _pairs(result) == strict_pairs
    roots_by_label = dict(
        zip(result.poset.elements, result.positive_roots, strict=True)
    )
    assert {
        (roots_by_label[pair.lower], roots_by_label[pair.upper])
        for pair in result.poset.cover_relations
    } == covers
    assert {
        frozenset((roots_by_label[pair.left], roots_by_label[pair.right]))
        for pair in result.poset.incomparable_pairs
    } == incomparables


def test_root_poset_roundtrips_and_composes_with_existing_poset_consumer() -> None:
    result = root_poset(((2, -1), (-1, 2)))

    restored = RootPosetResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert width(restored.poset).width == 2


def test_root_poset_admits_the_64_root_boundary_and_rejects_65() -> None:
    _admit_root_poset_work(64, 8)
    with pytest.raises(OperationDomainValidationError) as exc:
        _admit_root_poset_work(65, 8)
    assert exc.value.errors()[0]["type"] == "root_system.root_poset_bounds"


def test_b8_is_admitted_and_e8_is_rejected_before_poset_expansion() -> None:
    b8 = root_poset(
        CartanMatrix.model_validate(
            tuple(
                tuple(
                    2
                    if row == column
                    else -2
                    if (row, column) == (6, 7)
                    else -1
                    if abs(row - column) == 1
                    else 0
                    for column in range(8)
                )
                for row in range(8)
            )
        )
    )
    assert len(b8.positive_roots) == 64
    assert len(b8.poset.elements) == 64

    e8 = (
        (2, -1, 0, 0, 0, 0, 0, 0),
        (-1, 2, -1, 0, 0, 0, 0, 0),
        (0, -1, 2, -1, 0, 0, 0, -1),
        (0, 0, -1, 2, -1, 0, 0, 0),
        (0, 0, 0, -1, 2, -1, 0, 0),
        (0, 0, 0, 0, -1, 2, -1, 0),
        (0, 0, 0, 0, 0, -1, 2, 0),
        (0, 0, -1, 0, 0, 0, 0, 2),
    )
    with pytest.raises(OperationDomainValidationError) as exc:
        root_poset(e8)
    assert exc.value.errors()[0]["type"] == "root_system.root_poset_bounds"


def test_root_poset_tool_is_declared_and_executes_its_typed_example() -> None:
    tool = next(
        item for item in TOOLS if item.operation_id == "root_system.root_poset.compute"
    )
    request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )

    result = tool.run(request)

    assert isinstance(result, RootPosetResult)
    assert _pairs(result) == {((0, 1), (1, 1)), ((1, 0), (1, 1))}
