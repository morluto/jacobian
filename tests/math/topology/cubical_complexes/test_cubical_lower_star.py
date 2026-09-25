"""Independent exact oracles for cubical vertex lower-star filtrations."""

from fractions import Fraction
from itertools import product

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.chain_complexes._filtered_operations import (
    associated_graded,
)
from jacobian.math.topology.cubical_complexes._models import (
    CubicalCell,
    CubicalLowerStarRequest,
    CubicalVertexFiltrationValue,
    FilteredCubicalComplex,
)
from jacobian.math.topology.cubical_complexes.operations import (
    lower_star_from_vertices,
)


def _cell(intervals: tuple[tuple[int, int], ...]) -> CubicalCell:
    return CubicalCell(intervals=intervals)


def _vertex_value(
    coordinates: tuple[int, ...], value: Fraction
) -> CubicalVertexFiltrationValue:
    return CubicalVertexFiltrationValue(
        vertex=_cell(tuple((coordinate, coordinate) for coordinate in coordinates)),
        value=CanonicalRational.from_fraction(value),
    )


def _request(
    cells: tuple[CubicalCell, ...],
    values: dict[tuple[int, ...], Fraction],
    *,
    prime: int = 2,
) -> CubicalLowerStarRequest:
    entries = tuple(
        _vertex_value(coordinates, value)
        for coordinates, value in sorted(values.items())
    )
    return CubicalLowerStarRequest(cells=cells, vertex_values=entries, prime=prime)


def _lower_star(request: CubicalLowerStarRequest) -> FilteredCubicalComplex:
    """Call the native lower-star operation with unpacked domain arguments."""
    return lower_star_from_vertices(request.cells, request.vertex_values, request.prime)


def _vertices(cell: CubicalCell) -> tuple[tuple[int, ...], ...]:
    choices = tuple(
        (lower,) if lower == upper else (lower, upper)
        for lower, upper in cell.intervals
    )
    return tuple(product(*choices))


def _faces(cell: CubicalCell) -> set[tuple[tuple[int, int], ...]]:
    """Independent direct face enumeration by ternary axis choices."""
    choices = tuple(
        ((lower, lower), (lower, upper), (upper, upper))
        if lower != upper
        else ((lower, lower),)
        for lower, upper in cell.intervals
    )
    return {tuple(intervals) for intervals in product(*choices)}


def _births(
    result: FilteredCubicalComplex,
) -> dict[tuple[tuple[int, int], ...], Fraction]:
    return {
        entry.cell.intervals: entry.value.as_fraction() for entry in result.cell_births
    }


def test_lower_star_matches_independent_maximum_and_face_monotonicity_oracle() -> None:
    source = (_cell(((0, 1), (0, 1))), _cell(((1, 2), (0, 1))))
    vertex_values = {
        (0, 0): Fraction(-1, 3),
        (0, 1): Fraction(1, 2),
        (1, 0): Fraction(4, 5),
        (1, 1): Fraction(1, 2),
        (2, 0): Fraction(1, 2),
        (2, 1): Fraction(-2, 7),
    }
    result = _lower_star(_request(source, vertex_values, prime=3))
    birth_by_cell = _births(result)

    expected = {}
    for cell in result.complex.cells:
        expected[cell.intervals] = max(
            vertex_values[vertex] for vertex in _vertices(cell)
        )
    assert birth_by_cell == expected
    for cell in result.complex.cells:
        for face in _faces(cell):
            assert face in birth_by_cell
            assert birth_by_cell[face] <= birth_by_cell[cell.intervals]

    assert tuple(value.as_fraction() for value in result.critical_values) == tuple(
        sorted(set(vertex_values.values()))
    )
    assert len(result.cell_bases) == len(
        result.filtered_chain_complex.complex.basis_sizes
    )
    assert result.filtered_chain_complex.complex.prime == 3
    # The existing filtered-chain consumer independently checks nesting and
    # that every cubical boundary preserves each generated filtration level.
    graded = associated_graded(
        result.filtered_chain_complex.complex,
        result.filtered_chain_complex.filtration,
    )
    assert graded.complex == result.filtered_chain_complex.complex

    previous_sublevel: set[tuple[tuple[int, int], ...]] = set()
    for level, filtration in zip(
        result.critical_values, result.filtered_chain_complex.filtration, strict=True
    ):
        level_cells = set()
        for degree, basis in enumerate(result.cell_bases):
            actual = {
                basis.cells[index].intervals
                for vector in filtration.subspaces[degree].vectors
                for index, coefficient in enumerate(vector)
                if coefficient == 1
            }
            direct = {
                cell.intervals
                for cell in basis.cells
                if expected[cell.intervals] <= level.as_fraction()
            }
            assert actual == direct
            level_cells.update(direct)
        assert all(_faces(_cell(cell)).issubset(level_cells) for cell in level_cells)
        assert previous_sublevel.issubset(level_cells)
        previous_sublevel = level_cells

    assert (
        FilteredCubicalComplex.model_validate_json(result.model_dump_json()) == result
    )


def test_lower_star_commutes_with_integer_translation() -> None:
    source = (_cell(((0, 1), (0, 1))), _cell(((1, 2), (0, 1))))
    values = {
        (0, 0): Fraction(0),
        (0, 1): Fraction(1, 2),
        (1, 0): Fraction(2),
        (1, 1): Fraction(3, 2),
        (2, 0): Fraction(-1),
        (2, 1): Fraction(1, 2),
    }
    offset = (7, -4)
    original = _lower_star(_request(source, values))
    translated_cells = tuple(
        _cell(
            tuple(
                (lower + offset[i], upper + offset[i])
                for i, (lower, upper) in enumerate(cell.intervals)
            )
        )
        for cell in source
    )
    translated_values = {
        tuple(coordinate + offset[i] for i, coordinate in enumerate(vertex)): value
        for vertex, value in values.items()
    }
    translated = _lower_star(_request(translated_cells, translated_values))

    def shifted(intervals: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
        return tuple(
            (lower + offset[i], upper + offset[i])
            for i, (lower, upper) in enumerate(intervals)
        )

    assert {shifted(cell.intervals) for cell in original.complex.cells} == {
        cell.intervals for cell in translated.complex.cells
    }
    assert {
        shifted(cell): (birth, tuple(shifted(vertex) for vertex in maximizing))
        for cell, birth, maximizing in (
            (
                entry.cell.intervals,
                entry.value.as_fraction(),
                tuple(v.intervals for v in entry.maximizing_vertices),
            )
            for entry in original.cell_births
        )
    } == {
        entry.cell.intervals: (
            entry.value.as_fraction(),
            tuple(vertex.intervals for vertex in entry.maximizing_vertices),
        )
        for entry in translated.cell_births
    }
    assert translated.critical_values == original.critical_values
    assert translated.filtered_chain_complex.complex.differential_matrices == (
        original.filtered_chain_complex.complex.differential_matrices
    )
    assert translated.filtered_chain_complex.filtration == (
        original.filtered_chain_complex.filtration
    )


def test_lower_star_rejects_missing_or_extra_vertex_values() -> None:
    square = (_cell(((0, 1), (0, 1))),)
    values = {(0, 0): Fraction(0), (0, 1): Fraction(0), (1, 0): Fraction(0)}
    with pytest.raises(
        OperationDomainValidationError,
        match="vertex values must cover exactly",
    ):
        _lower_star(_request(square, values))

    with pytest.raises(ValidationError):
        CubicalVertexFiltrationValue(
            vertex=_cell(((0, 1),)), value=CanonicalRational.from_fraction(Fraction(0))
        )


def test_lower_star_preflights_filtered_chain_levels_and_face_growth() -> None:
    interval_chain = tuple(_cell(((index, index + 1),)) for index in range(9))
    too_many_levels = {(index,): Fraction(index) for index in range(10)}
    with pytest.raises(
        OperationResourceAdmissionError, match="distinct lower-star values"
    ):
        _lower_star(_request(interval_chain, too_many_levels))

    six_cube = (_cell(((0, 1),) * 6),)
    all_vertices = {vertex: Fraction(0) for vertex in product((0, 1), repeat=6)}
    with pytest.raises(OperationResourceAdmissionError, match="256-cell output bound"):
        _lower_star(_request(six_cube, all_vertices))
