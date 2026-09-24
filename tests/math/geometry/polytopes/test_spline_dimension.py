from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes import _spline as spline_kernel
from jacobian.math.geometry.polytopes.complexes._models import (
    SplineDimensionRequest,
    SplineDimensionResult,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_closure,
    spline_dimension,
    spline_space,
)


def _box(x0: int, x1: int, y0: int, y1: int, prefix: str) -> RationalVPolytope:
    points = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y")),
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=f"{prefix}{index}",
                coordinates=tuple(
                    CanonicalRational(num=value, den=1) for value in point
                ),
            )
            for index, point in enumerate(points)
        ),
    )


def _interval(left: int, right: int, prefix: str) -> RationalVPolytope:
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x",)),
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=f"{prefix}{index}",
                coordinates=(CanonicalRational(num=value, den=1),),
            )
            for index, value in enumerate((left, right))
        ),
    )


def _rank(rows: list[list[Fraction]]) -> int:
    """Small independent exact row-reduction oracle for the interval fixture."""
    if not rows:
        return 0
    matrix = [row[:] for row in rows]
    pivot_row = 0
    for column in range(len(matrix[0])):
        pivot = next(
            (row for row in range(pivot_row, len(matrix)) if matrix[row][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        scale = matrix[pivot_row][column]
        matrix[pivot_row] = [value / scale for value in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row or not matrix[row][column]:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    matrix[row], matrix[pivot_row], strict=True
                )
            ]
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    return pivot_row


def _interval_continuity_matrix(degree: int, smoothness: int) -> list[list[Fraction]]:
    """Match derivatives at x=1 for two independent degree-d polynomials."""
    rows = []
    for derivative in range(smoothness + 1):
        row = [Fraction(0) for _ in range(2 * (degree + 1))]
        for exponent in range(derivative, degree + 1):
            coefficient = Fraction(
                1,
            )
            for factor in range(derivative):
                coefficient *= exponent - factor
            row[exponent] = coefficient
            row[degree + 1 + exponent] = -coefficient
        rows.append(row)
    return rows


def test_spline_dimension_matches_interval_derivative_oracle_and_full_space():
    complex_value = polytopal_complex_closure(
        (_interval(0, 1, "a"), _interval(1, 2, "b"))
    )
    degree, smoothness = 4, 2

    result = spline_dimension(
        SplineDimensionRequest(
            complex=complex_value, degree=degree, smoothness=smoothness
        )
    )
    full = spline_space(complex_value, degree, smoothness)
    oracle_rank = _rank(_interval_continuity_matrix(degree, smoothness))

    assert result.rank == oracle_rank == full.rank
    assert result.nullity == 2 * (degree + 1) - oracle_rank == full.nullity
    assert result.compatibility_matrix == full.compatibility_matrix
    assert result.coefficient_axis == full.coefficient_axis


def test_one_cell_zero_row_dimension_roundtrips_and_catalog_invokes():
    complex_value = polytopal_complex_closure((_interval(0, 1, "a"),))
    request = SplineDimensionRequest(complex=complex_value, degree=5, smoothness=1)

    result = spline_dimension(request)
    replayed = SplineDimensionResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json"))
    )

    assert result.compatibility_matrix.row_count == 0
    assert result.compatibility_matrix.column_count == 6
    assert result.rank == 0 and result.nullity == 6
    assert replayed == result
    invoked = invoke_operation(
        "polyhedral_complex.spline_dimension.compute",
        {
            "complex": complex_value.model_dump(mode="json"),
            "degree": 0,
            "smoothness": 0,
        },
        Catalog.open(),
    )
    assert invoked.output["nullity"] == 1 and invoked.output["rank"] == 0


def test_dimension_output_estimate_is_conservative_at_its_boundary(monkeypatch):
    complex_value = polytopal_complex_closure(
        (_interval(0, 1, "a"), _interval(1, 2, "b"))
    )
    request = SplineDimensionRequest(complex=complex_value, degree=3, smoothness=0)
    result = spline_dimension(request)
    rows = tuple(
        tuple(entry.as_fraction() for entry in row)
        for row in result.compatibility_matrix.entries
    )
    estimate = spline_kernel._spline_dimension_output_upper_bound(
        result.complex,
        result.degree,
        result.smoothness,
        result.coefficient_axis,
        rows,
        result.compatibility_matrix.column_count,
    )
    actual = len(encode_strict_json(result.model_dump(mode="json")))

    assert (
        CanonicalLimits().max_output_bytes
        == spline_kernel.MAX_SPLINE_DIMENSION_OUTPUT_BYTES
    )
    assert estimate >= actual
    monkeypatch.setattr(spline_kernel, "MAX_SPLINE_DIMENSION_OUTPUT_BYTES", estimate)
    assert spline_dimension(request).nullity == result.nullity
    monkeypatch.setattr(
        spline_kernel, "MAX_SPLINE_DIMENSION_OUTPUT_BYTES", estimate - 1
    )
    with pytest.raises(OperationResourceAdmissionError, match="output envelope"):
        spline_dimension(request)


def test_dimension_admits_matrix_when_full_basis_output_exceeds_its_bound():
    cells = tuple(_box(index, index + 1, 0, 1, f"c{index}-") for index in range(10))
    complex_value = polytopal_complex_closure(cells)
    request = SplineDimensionRequest(complex=complex_value, degree=12, smoothness=0)

    with pytest.raises(OperationResourceAdmissionError, match="exact result"):
        spline_space(complex_value, 12, 0)

    result = spline_dimension(request)

    # Ten cellwise degree-12 polynomials in two variables have 910 coordinates.
    # Each of the nine vertical interfaces equates 13 independent y-polynomial
    # coefficients, so the dimension is 910 - 9*13 = 793.
    assert result.compatibility_matrix.row_count == 117
    assert result.compatibility_matrix.column_count == 910
    assert result.rank == 117
    assert result.nullity == 793
