from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes._models import (
    CommonRefinementRequest,
    ComplexPoint,
    MaximalCellRecord,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_closure,
    polytopal_complex_common_refinement,
)

SPACE = RationalCoordinateSpace(axes=("x", "y"))


def poly(points: tuple[tuple[int, int], ...], prefix: str) -> RationalVPolytope:
    return RationalVPolytope(
        space=SPACE,
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=f"{prefix}{i}",
                coordinates=tuple({"num": x, "den": 1} for x in point),
            )
            for i, point in enumerate(points)
        ),
    )


def complex_of(*cells: RationalVPolytope):
    return polytopal_complex_closure(tuple(cells))


def _shoelace(points) -> Fraction:
    """Independent exact polygon-area oracle, not the polytope volume kernel."""
    ordered = sorted(
        (
            tuple(
                Fraction(*coordinate.as_integer_ratio()) for coordinate in p.coordinates
            )
            for p in points
        ),
        key=lambda point: (point[1], point[0]),
    )

    # A monotone-chain hull is sufficient for the convex overlay cells.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return (
        abs(
            sum(
                hull[i][0] * hull[(i + 1) % len(hull)][1]
                - hull[(i + 1) % len(hull)][0] * hull[i][1]
                for i in range(len(hull))
            )
        )
        / 2
    )


def test_common_refinement_produces_face_closed_overlay_and_pair_provenance():
    left = complex_of(
        poly(((0, 0), (1, 0), (0, 1)), "a"),
        poly(((1, 0), (1, 1), (0, 1)), "b"),
    )
    right = complex_of(poly(((0, 0), (1, 0), (1, 1), (0, 1)), "r0"))

    result = polytopal_complex_common_refinement(
        CommonRefinementRequest(left=left, right=right)
    )

    areas = sorted(_shoelace(cell.vertices) for cell in result.refinement.maximal_cells)
    assert areas == [Fraction(1, 2), Fraction(1, 2)]
    assert len(result.cell_pairs) == 2
    assert {row.right_cell_id for row in result.cell_pairs} == {"M0"}
    assert {row.refined_cell_id for row in result.cell_pairs} == {
        cell.cell_id for cell in result.refinement.maximal_cells
    }
    assert result.refinement.f_vector == (1, 4, 5, 2)


def test_common_refinement_independent_oracle_for_diagonal_and_vertical_splits():
    left = complex_of(
        poly(((0, 0), (1, 0), (1, 1)), "a"),
        poly(((0, 0), (1, 1), (0, 1)), "b"),
    )
    # A 2x1 rectangle triangulated along one diagonal and split vertically.
    left = complex_of(
        poly(((0, 0), (2, 0), (2, 1)), "c"),
        poly(((0, 0), (2, 1), (0, 1)), "d"),
    )
    right = complex_of(
        poly(((0, 0), (1, 0), (1, 1), (0, 1)), "e"),
        poly(((1, 0), (2, 0), (2, 1), (1, 1)), "f"),
    )
    result = polytopal_complex_common_refinement(
        CommonRefinementRequest(left=left, right=right)
    )

    assert len(result.refinement.maximal_cells) == 4
    assert sorted(
        _shoelace(cell.vertices) for cell in result.refinement.maximal_cells
    ) == [Fraction(1, 4), Fraction(1, 4), Fraction(3, 4), Fraction(3, 4)]
    assert (
        sum(_shoelace(cell.vertices) for cell in result.refinement.maximal_cells) == 2
    )


def test_common_refinement_rejects_different_supports():
    left = complex_of(poly(((0, 0), (1, 0), (1, 1), (0, 1)), "a"))
    right = complex_of(poly(((0, 0), (2, 0), (2, 1), (0, 1)), "b"))

    with pytest.raises(OperationDomainValidationError, match="exactly equal support"):
        polytopal_complex_common_refinement(
            CommonRefinementRequest(left=left, right=right)
        )


def test_common_refinement_rejects_unmeasured_lower_dimensional_components():
    square = poly(((0, 0), (1, 0), (1, 1), (0, 1)), "square")
    left = complex_of(square)
    isolated_point = MaximalCellRecord.model_construct(
        cell_id="M1",
        source_indices=(1,),
        dimension=1,
        vertices=(
            ComplexPoint(
                coordinates=(
                    CanonicalRational(num=2, den=1),
                    CanonicalRational(num=2, den=1),
                )
            ),
        ),
        facet_face_ids=("Fpoint",),
    )
    left = left.model_copy(
        update={"maximal_cells": (*left.maximal_cells, isolated_point)}, deep=True
    )
    right = complex_of(poly(((0, 0), (1, 0), (1, 1), (0, 1)), "right"))

    with pytest.raises(
        OperationDomainValidationError,
        match="every maximal cell must have the full ambient dimension",
    ):
        polytopal_complex_common_refinement(
            CommonRefinementRequest(left=left, right=right)
        )


def test_common_refinement_rejects_vertex_coordinates_with_wrong_ambient_width():
    square = complex_of(poly(((0, 0), (1, 0), (1, 1), (0, 1)), "left"))
    right = complex_of(poly(((0, 0), (1, 0), (1, 1), (0, 1)), "right"))
    cell = square.maximal_cells[0]
    malformed_vertex = cell.vertices[0].model_copy(
        update={"coordinates": (CanonicalRational(num=0, den=1),)},
        deep=True,
    )
    malformed_cell = cell.model_copy(
        update={"vertices": (malformed_vertex, *cell.vertices[1:])},
        deep=True,
    )
    malformed_square = square.model_copy(
        update={"maximal_cells": (malformed_cell,)},
        deep=True,
    )

    with pytest.raises(
        OperationDomainValidationError,
        match="complex ambient coordinate axes",
    ):
        polytopal_complex_common_refinement(
            CommonRefinementRequest(left=malformed_square, right=right)
        )


def test_common_refinement_catalog_example_is_composable():
    catalog = Catalog.open()
    operation = catalog.operation("polytopal_complex.common_refinement.compute")
    example = operation.examples[0]
    result = invoke_operation(operation.operation_id, example.input, catalog)
    assert result.output["cell_pairs"][0]["refined_cell_id"] == "M0"
