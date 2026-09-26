"""Exact cubical filtrations specified on maximal cells."""

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.chain_complexes._filtered_operations import (
    associated_graded,
)
from jacobian.math.topology.cubical_complexes._models import (
    CubicalCell,
    CubicalTopCellFiltrationRequest,
    CubicalTopCellFiltrationValue,
    FilteredCubicalComplexFromTopCells,
)
from jacobian.math.topology.cubical_complexes._tools import TOOLS
from jacobian.math.topology.cubical_complexes.operations import from_top_cell_values


def _cell(*intervals: tuple[int, int]) -> CubicalCell:
    return CubicalCell(intervals=tuple(intervals))


def _value(cell: CubicalCell, value: Fraction) -> CubicalTopCellFiltrationValue:
    return CubicalTopCellFiltrationValue(
        cell=cell, value=CanonicalRational.from_fraction(value)
    )


def _contains(container: CubicalCell, cell: CubicalCell) -> bool:
    return all(
        outer[0] <= inner[0] and inner[1] <= outer[1]
        for outer, inner in zip(container.intervals, cell.intervals, strict=True)
    )


def _from_top_cell_values(request: CubicalTopCellFiltrationRequest):
    """Call the native top-cell filtration with unpacked domain arguments."""
    return from_top_cell_values(request.cells, request.top_cell_values, request.prime)


def test_top_cell_values_give_exact_face_closure_sublevels_and_witnesses() -> None:
    left = _cell((0, 1), (0, 1))
    right = _cell((1, 2), (0, 1))
    values = {left: Fraction(3, 2), right: Fraction(-2, 3)}
    result = _from_top_cell_values(
        CubicalTopCellFiltrationRequest(
            cells=(left, right),
            top_cell_values=tuple(
                _value(cell, value) for cell, value in values.items()
            ),
            prime=3,
        )
    )

    # Independent incidence oracle: a cell's birth is the least value among
    # maximal input cells containing it.
    expected = {
        cell: min(value for top, value in values.items() if _contains(top, cell))
        for cell in result.complex.cells
    }
    births = {entry.cell: entry.value.as_fraction() for entry in result.cell_births}
    assert births == expected
    for entry in result.cell_births:
        witness_values = {top: values[top] for top in entry.minimizing_top_cells}
        assert witness_values
        assert all(_contains(top, entry.cell) for top in witness_values)
        assert set(witness_values.values()) == {expected[entry.cell]}

    assert result.critical_values == tuple(
        CanonicalRational.from_fraction(value) for value in sorted(set(values.values()))
    )
    assert result.filtered_chain_complex.complex.prime == 3
    assert (
        associated_graded(
            result.filtered_chain_complex.complex,
            result.filtered_chain_complex.filtration,
        ).complex
        == result.filtered_chain_complex.complex
    )

    for level, filtration in zip(
        result.critical_values, result.filtered_chain_complex.filtration, strict=True
    ):
        for degree, basis in enumerate(result.cell_bases):
            included = {
                basis.cells[index]
                for vector in filtration.subspaces[degree].vectors
                for index, coefficient in enumerate(vector)
                if coefficient == 1
            }
            assert included == {
                cell for cell in basis.cells if expected[cell] <= level.as_fraction()
            }

    restored = FilteredCubicalComplexFromTopCells.model_validate_json(
        result.model_dump_json()
    )
    assert restored == result


def test_a_single_point_is_a_valid_degenerate_top_cell() -> None:
    point = _cell((8, 8), (-1, -1))
    result = _from_top_cell_values(
        CubicalTopCellFiltrationRequest(
            cells=(point,), top_cell_values=(_value(point, Fraction(0)),)
        )
    )
    assert len(result.complex.cells) == 1
    assert result.cell_births[0].minimizing_top_cells == (point,)
    assert result.cell_bases[0].cells == (point,)
    assert result.filtered_chain_complex.filtration[0].subspaces[0].vectors == ((1,),)


def test_top_cell_values_must_cover_exactly_maximal_generators() -> None:
    left = _cell((0, 1), (0, 1))
    right = _cell((1, 2), (0, 1))
    shared_edge = _cell((1, 1), (0, 1))
    with pytest.raises(OperationDomainValidationError, match="cover exactly"):
        _from_top_cell_values(
            CubicalTopCellFiltrationRequest(
                cells=(left, right),
                top_cell_values=(_value(left, Fraction(0)),),
            )
        )
    with pytest.raises(OperationDomainValidationError, match="cover exactly"):
        _from_top_cell_values(
            CubicalTopCellFiltrationRequest(
                cells=(left, right),
                top_cell_values=(
                    _value(left, Fraction(0)),
                    _value(right, Fraction(1)),
                    _value(shared_edge, Fraction(2)),
                ),
            )
        )
    with pytest.raises(ValidationError):
        CubicalTopCellFiltrationRequest(cells=(), top_cell_values=())


def test_top_cell_filtration_is_published_with_a_composable_example() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id
        == "topology.filtered_cubical_complex.from_top_cells.compute"
    )
    request = CubicalTopCellFiltrationRequest.model_validate_json(
        json.dumps(tool.examples[0].input)
    )
    result = tool.run(request)
    assert isinstance(result, FilteredCubicalComplexFromTopCells)
    shared = _cell((1, 1), (0, 1))
    assert next(
        entry.value.as_fraction()
        for entry in result.cell_births
        if entry.cell == shared
    ) == Fraction(0)
