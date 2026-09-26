from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes._models import (
    SplineRefinementMapRequest,
    SplineRefinementMapResult,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_closure,
    spline_refinement_map,
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


def _dot(left, right) -> Fraction:
    return sum(
        (a.as_fraction() * b.as_fraction() for a, b in zip(left, right, strict=True)),
        Fraction(0),
    )


def _evaluate(coefficients, axis, point: int) -> Fraction:
    return sum(
        (
            coefficient.as_fraction() * Fraction(point) ** exponents[0]
            for (_, exponents), coefficient in zip(axis, coefficients, strict=True)
        ),
        Fraction(0),
    )


def test_refinement_map_preserves_spline_basis_exactly_and_roundtrips():
    coarse = polytopal_complex_closure((_interval(0, 2, "coarse"),))
    refined = polytopal_complex_closure(
        (_interval(0, 1, "left"), _interval(1, 2, "right"))
    )

    result = spline_refinement_map(
        SplineRefinementMapRequest(
            coarse=coarse, refined=refined, degree=1, smoothness=0
        )
    )

    assert result.coarse_compatibility_matrix.column_count == 2
    assert result.coarse_nullspace_basis.row_count == 2
    assert result.refined_compatibility_matrix.row_count == 1
    assert result.refined_compatibility_matrix.column_count == 4
    assert len(result.cell_lineage) == 2
    assert {row.coarse_cell_id for row in result.cell_lineage} == {"M0"}
    assert {row.refined_cell_id for row in result.cell_lineage} == {
        cell.cell_id for cell in result.refined_complex.maximal_cells
    }

    # This independent coefficient-copy oracle verifies both the map's exact
    # image in the fine constraint kernel and equality at all domain vertices.
    for coarse_basis_row in result.coarse_nullspace_basis.entries:
        refined_coefficients = tuple(coarse_basis_row) * 2
        assert all(
            _dot(constraint, refined_coefficients) == 0
            for constraint in result.refined_compatibility_matrix.entries
        )
        for point in (0, 1, 2):
            coarse_value = _evaluate(
                coarse_basis_row, result.coarse_coefficient_axis, point
            )
            for cell_index in (0, 1):
                start = cell_index * 2
                fine_value = _evaluate(
                    refined_coefficients[start : start + 2],
                    result.refined_coefficient_axis[start : start + 2],
                    point,
                )
                assert fine_value == coarse_value

    decoded = SplineRefinementMapResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json"))
    )
    assert decoded == result


def test_refinement_map_rejects_a_coarsening_with_equal_support():
    coarse = polytopal_complex_closure(
        (_interval(0, 1, "left"), _interval(1, 2, "right"))
    )
    one_cell = polytopal_complex_closure((_interval(0, 2, "whole"),))

    with pytest.raises(
        OperationDomainValidationError,
        match="target complex must be a face-to-face refinement",
    ):
        spline_refinement_map(
            SplineRefinementMapRequest(
                coarse=coarse, refined=one_cell, degree=1, smoothness=0
            )
        )
