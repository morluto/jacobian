"""Exact exposed-facet boundaries for pure cubical complexes."""

import json
from collections import Counter
from itertools import product

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.cubical_complexes import operations
from jacobian.math.topology.cubical_complexes._models import (
    CubicalBoundarySubcomplexResult,
    CubicalCell,
    CubicalComplexRequest,
)
from jacobian.math.topology.cubical_complexes._tools import TOOLS


def _cell(*intervals):
    return CubicalCell(intervals=intervals)


def test_square_boundary_has_four_exposed_edges_and_complete_vertex_closure():
    square = _cell((0, 1), (0, 1))
    result = operations.boundary_subcomplex((square,))

    assert result.exposed_facets == (
        _cell((0, 0), (0, 1)),
        _cell((0, 1), (0, 0)),
        _cell((0, 1), (1, 1)),
        _cell((1, 1), (0, 1)),
    )
    assert tuple(cell.dimension for cell in result.boundary.cells).count(1) == 4
    assert tuple(cell.dimension for cell in result.boundary.cells).count(0) == 4
    assert result.boundary.ambient_dimension == result.complex.ambient_dimension == 2


def test_adjacent_squares_exclude_the_shared_interior_edge():
    left = _cell((0, 1), (0, 1))
    right = _cell((1, 2), (0, 1))
    shared = _cell((1, 1), (0, 1))
    result = operations.boundary_subcomplex((left, right))

    assert len(result.exposed_facets) == 6
    assert shared not in result.exposed_facets
    assert shared not in result.boundary.cells


def test_closed_cubical_surface_has_an_empty_boundary_value():
    cube_faces = (
        _cell((0, 0), (0, 1), (0, 1)),
        _cell((1, 1), (0, 1), (0, 1)),
        _cell((0, 1), (0, 0), (0, 1)),
        _cell((0, 1), (1, 1), (0, 1)),
        _cell((0, 1), (0, 1), (0, 0)),
        _cell((0, 1), (0, 1), (1, 1)),
    )
    result = operations.boundary_subcomplex(cube_faces)

    assert result.boundary.cells == ()
    assert result.boundary.ambient_dimension == 3
    assert CubicalBoundarySubcomplexResult.model_validate_json(
        result.model_dump_json()
    ) == result


def test_impure_zero_dimensional_and_mixed_axis_sources_are_rejected():
    square = _cell((0, 1), (0, 1))
    with pytest.raises(OperationDomainValidationError) as impure:
        operations.boundary_subcomplex((square, _cell((10, 10), (10, 10))))
    assert impure.value.errors()[0]["type"] == (
        "cubical_complex.boundary_subcomplex_not_pure"
    )

    with pytest.raises(OperationDomainValidationError) as zero_dimensional:
        operations.boundary_subcomplex((_cell((2, 2),),))
    assert zero_dimensional.value.errors()[0]["type"] == (
        "cubical_complex.boundary_subcomplex_dimension"
    )

    with pytest.raises(OperationDomainValidationError) as axes:
        operations.boundary_subcomplex((square, _cell((0, 1),)))
    assert axes.value.errors()[0]["type"] == (
        "cubical_complex.boundary_subcomplex_ambient_axis"
    )


def test_coordinate_and_face_output_admission_precede_expansion(monkeypatch):
    def fail_if_expanded(*_args, **_kwargs):
        raise AssertionError("cell expansion ran before admission")

    monkeypatch.setattr(operations, "_canonical_complex", fail_if_expanded)
    huge_square = _cell((10**70, 10**70 + 1), (0, 1))
    with pytest.raises(OperationResourceAdmissionError) as digits:
        operations.boundary_subcomplex((huge_square,))
    assert digits.value.errors()[0]["type"] == (
        "cubical_complex.boundary_subcomplex_coordinate_bound"
    )

    first = _cell(*((0, 1) for _ in range(10)))
    second = _cell((2, 3), *((0, 1) for _ in range(9)))
    with pytest.raises(OperationResourceAdmissionError) as output:
        operations.boundary_subcomplex((first, second))
    assert output.value.errors()[0]["type"] == (
        "cubical_complex.boundary_subcomplex_bounds"
    )


def test_non_pure_check_uses_exact_face_closure_and_roundtrips_result():
    square = _cell((0, 1), (0, 1))
    result = operations.boundary_subcomplex((square,))
    assert result.complex.cells == operations.face_closure((square,)).complex.cells
    assert CubicalBoundarySubcomplexResult.model_validate_json(
        result.model_dump_json()
    ) == result


def test_all_square_subfamilies_match_an_independent_incidence_oracle():
    squares = tuple(
        _cell((column, column + 1), (row, row + 1))
        for row in range(2)
        for column in range(2)
    )
    for mask in range(1, 1 << len(squares)):
        source = tuple(
            square for index, square in enumerate(squares) if mask & (1 << index)
        )
        incidence = Counter()
        for square in source:
            intervals = square.intervals
            for axis in range(2):
                for endpoint in intervals[axis]:
                    face = list(intervals)
                    face[axis] = (endpoint, endpoint)
                    incidence[tuple(face)] += 1
        facets = {face for face, count in incidence.items() if count == 1}
        expected_boundary = set()
        for facet in facets:
            choices = tuple(
                ((lower, upper), (lower, lower), (upper, upper))
                for lower, upper in facet
            )
            expected_boundary.update(product(*choices))
        result = operations.boundary_subcomplex(source)
        assert {cell.intervals for cell in result.exposed_facets} == facets
        assert {cell.intervals for cell in result.boundary.cells} == expected_boundary


def test_tool_is_published_with_a_real_square_example():
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id
        == "topology.cubical_complex.boundary_subcomplex.compute"
    )
    request = CubicalComplexRequest.model_validate_json(
        json.dumps(tool.examples[0].input)
    )
    result = tool.run(request)
    assert isinstance(result, CubicalBoundarySubcomplexResult)
    assert len(result.exposed_facets) == 4
