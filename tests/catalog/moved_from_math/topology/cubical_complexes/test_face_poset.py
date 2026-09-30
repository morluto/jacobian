"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/topology/cubical_complexes/test_face_poset.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

from itertools import product

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.topology.cubical_complexes._models import (
    CubicalCell,
)
from jacobian.math.topology.cubical_complexes.operations import face_poset

_OPERATION_ID = "topology.cubical_complex.face_poset.compute"


def _oracle_face_closure(cells: tuple[CubicalCell, ...]) -> tuple[CubicalCell, ...]:
    faces: set[tuple[tuple[int, int], ...]] = set()
    for cell in cells:
        coordinate_choices = tuple(
            ((start, end),)
            if start == end
            else ((start, start), (start, end), (end, end))
            for start, end in cell.intervals
        )
        faces.update(product(*coordinate_choices))
    return tuple(CubicalCell(intervals=face) for face in sorted(faces))


def _oracle_order(
    cells: tuple[CubicalCell, ...],
) -> set[tuple[CubicalCell, CubicalCell]]:
    return {
        (lower, upper)
        for lower in cells
        for upper in cells
        if lower != upper
        and all(
            outer_start <= inner_start and inner_end <= outer_end
            for (inner_start, inner_end), (outer_start, outer_end) in zip(
                lower.intervals, upper.intervals, strict=True
            )
        )
    }


def _assert_matches_independent_oracle(request_cells: tuple[CubicalCell, ...]) -> None:
    result = face_poset(request_cells)
    expected_cells = _oracle_face_closure(request_cells)
    expected_order = _oracle_order(expected_cells)
    label_to_cell = {entry.element: entry.cell for entry in result.cell_elements}
    actual_order = {
        (label_to_cell[pair.lower], label_to_cell[pair.upper])
        for pair in result.poset.strict_order_pairs
    }
    actual_covers = {
        (label_to_cell[pair.lower], label_to_cell[pair.upper])
        for pair in result.poset.cover_relations
    }
    expected_covers = {
        pair
        for pair in expected_order
        if not any(
            (pair[0], middle) in expected_order and (middle, pair[1]) in expected_order
            for middle in expected_cells
            if middle != pair[0] and middle != pair[1]
        )
    }

    assert result.complex.cells == expected_cells
    assert tuple(entry.cell for entry in result.cell_elements) == expected_cells
    assert tuple(entry.dimension for entry in result.cell_elements) == tuple(
        cell.dimension for cell in expected_cells
    )
    assert actual_order == expected_order
    assert actual_covers == expected_covers


def test_face_poset_catalog_example_executes_and_round_trips() -> None:
    operation = next(
        tool for tool in BUILTIN_TOOLS if tool.operation_id == _OPERATION_ID
    )
    request = operation.request_type.model_validate(operation.examples[0].input)
    result = operation.run(request)
    assert len(result.cell_elements) == 9
    assert type(result).model_validate_json(result.model_dump_json()) == result
