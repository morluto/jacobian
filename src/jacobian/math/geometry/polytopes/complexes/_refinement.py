"""Exact bounded common refinement of two rational polytopal complexes."""

from __future__ import annotations

import math
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes._models import (
    MAX_COMPLEX_COORDINATE_DIGITS,
    MAX_COMPLEX_DIMENSION,
    MAX_COMPLEX_INTERSECTION_WORK,
    CommonRefinementCellPair,
    CommonRefinementRequest,
    CommonRefinementResult,
    PolytopalComplexClosureResult,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    _affine_dimension,
    _cell_facets,
    _intersection_key,
    _point,
    polytopal_complex_closure,
)
from jacobian.math.geometry.polytopes.operations import convex_hull_volume
from jacobian.math.geometry.polytopes.values import Vertex

_MAX_SOURCE_CELLS = 8
_MAX_SOURCE_VERTICES = 16
_MAX_REFINEMENT_PAIRS = 16


def _reject(code: str, message: str, *, location: tuple[str, ...] = ()) -> None:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _validate_source_coordinates(
    side: str, complex_value: object, dimension: int
) -> None:
    for index, cell in enumerate(complex_value.maximal_cells):
        if any(len(vertex.coordinates) != dimension for vertex in cell.vertices):
            _reject(
                "polytopal_complex.refinement_vertex_ambient_dimension",
                "every maximal-cell vertex must use the complex ambient coordinate axes",
                location=(side, "maximal_cells", str(index)),
            )
        if len(cell.vertices) > _MAX_SOURCE_VERTICES:
            raise OperationResourceAdmissionError(
                location=(side, "maximal_cells", str(index)),
                code="polytopal_complex.refinement_vertices_over_envelope",
                message=f"each source cell admits at most {_MAX_SOURCE_VERTICES} vertices",
            )
        for vertex in cell.vertices:
            for coordinate in vertex.coordinates:
                if (
                    max(len(str(abs(coordinate.num))), len(str(coordinate.den)))
                    > MAX_COMPLEX_COORDINATE_DIGITS
                ):
                    raise OperationResourceAdmissionError(
                        location=(side, "maximal_cells", str(index)),
                        code="polytopal_complex.refinement_coordinate_digits_over_envelope",
                        message="source coordinates exceed the common-refinement digit envelope",
                    )


def _admit(request: CommonRefinementRequest) -> int:
    left, right = request.left, request.right
    if left.space != right.space:
        _reject(
            "polytopal_complex.refinement_mixed_spaces",
            "both complexes must use the same labelled rational affine space",
        )
    dimension = len(left.space.axes)
    if dimension > 3 or dimension > MAX_COMPLEX_DIMENSION:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="polytopal_complex.refinement_dimension_over_envelope",
            message="common refinement currently admits ambient dimension at most three",
        )
    if (
        len(left.maximal_cells) > _MAX_SOURCE_CELLS
        or len(right.maximal_cells) > _MAX_SOURCE_CELLS
    ):
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="polytopal_complex.refinement_source_cells_over_envelope",
            message=f"each source complex admits at most {_MAX_SOURCE_CELLS} maximal cells",
        )
    if len(left.maximal_cells) * len(right.maximal_cells) > _MAX_REFINEMENT_PAIRS:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="polytopal_complex.refinement_pairs_over_envelope",
            message=f"the worst-case nonempty overlay exceeds {_MAX_REFINEMENT_PAIRS} cells",
        )
    # A d-polytope on v vertices has at most v edges for d=2 and at most
    # 2v-4 facets for d=3. These bounds admit every candidate-system solve
    # before the exact facet/enumeration kernels are entered.
    facet_bound = (
        2
        if dimension == 1
        else _MAX_SOURCE_VERTICES
        if dimension == 2
        else 2 * _MAX_SOURCE_VERTICES - 4
    )
    total_work = 0
    for side, complex_value in (("left", left), ("right", right)):
        _validate_source_coordinates(side, complex_value, dimension)
    total_work = (
        len(left.maximal_cells)
        * len(right.maximal_cells)
        * math.comb(2 * facet_bound, dimension)
    )
    if total_work > MAX_COMPLEX_INTERSECTION_WORK:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="polytopal_complex.refinement_work_over_envelope",
            message=f"the conservative exact intersection bound exceeds {MAX_COMPLEX_INTERSECTION_WORK} candidate systems",
        )
    for side, complex_value in (("left", left), ("right", right)):
        for index, cell in enumerate(complex_value.maximal_cells):
            cell_dimension = _affine_dimension(
                tuple(_point(vertex.coordinates) for vertex in cell.vertices)
            )
            if cell_dimension != dimension:
                _reject(
                    "polytopal_complex.refinement_non_full_dimensional_cell",
                    "every maximal cell must have the full ambient dimension",
                    location=(side, "maximal_cells", str(index)),
                )
    return dimension


def _polytope_from_cell(
    space: RationalCoordinateSpace, cell: object, prefix: str
) -> RationalVPolytope:
    vertices = tuple(
        RationalPolytopeVertex(
            vertex_id=f"{prefix}{index:02d}", coordinates=point.coordinates
        )
        for index, point in enumerate(cell.vertices)
    )
    return RationalVPolytope(space=space, vertices=vertices)


def _canonical_source(
    source: PolytopalComplexClosureResult, name: str
) -> PolytopalComplexClosureResult:
    cells = tuple(
        _polytope_from_cell(source.space, cell, f"{name}{index:02d}v")
        for index, cell in enumerate(source.maximal_cells)
    )
    canonical = polytopal_complex_closure(cells)
    # Source IDs and incidence rows are caller-supplied claims. Rebuild from
    # the geometric cells so malformed topology cannot leak into the output.
    return canonical


def common_refinement(request: CommonRefinementRequest) -> CommonRefinementResult:
    """Overlay two exact complexes with an independently checkable cell map.

    The supports must be equal. The construction intersects every pair of
    maximal cells; exact volume conservation on every source cell proves that
    the pair intersections cover both supports. The resulting maximal cells
    and their complete face closure are then canonicalized by the complex
    owner.
    """

    dimension = _admit(request)
    left = _canonical_source(request.left, "L")
    right = _canonical_source(request.right, "R")
    left_facets = []
    right_facets = []
    for cell in left.maximal_cells:
        verts = tuple(
            Vertex(coordinates=vertex.coordinates) for vertex in cell.vertices
        )
        left_facets.append(_cell_facets(verts, dimension))
    for cell in right.maximal_cells:
        verts = tuple(
            Vertex(coordinates=vertex.coordinates) for vertex in cell.vertices
        )
        right_facets.append(_cell_facets(verts, dimension))

    candidate_rows: list[tuple[int, int, tuple[tuple[Fraction, ...], ...]]] = []
    left_covered = [Fraction(0) for _ in left.maximal_cells]
    right_covered = [Fraction(0) for _ in right.maximal_cells]
    left_volumes = [
        Fraction(
            *convex_hull_volume(
                tuple(_point(v.coordinates) for v in c.vertices)
            ).as_integer_ratio()
        )
        for c in left.maximal_cells
    ]
    right_volumes = [
        Fraction(
            *convex_hull_volume(
                tuple(_point(v.coordinates) for v in c.vertices)
            ).as_integer_ratio()
        )
        for c in right.maximal_cells
    ]
    for i, lf in enumerate(left_facets):
        for j, rf in enumerate(right_facets):
            key = _intersection_key(lf, rf, dimension)
            if key is None or _affine_dimension(key) != dimension:
                continue
            volume = Fraction(*convex_hull_volume(key).as_integer_ratio())
            if volume <= 0:
                continue
            left_covered[i] += volume
            right_covered[j] += volume
            candidate_rows.append((i, j, key))
    if len(candidate_rows) > _MAX_REFINEMENT_PAIRS:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="polytopal_complex.refinement_cells_over_envelope",
            message=f"the overlay exceeds {_MAX_REFINEMENT_PAIRS} maximal cells",
        )
    if left_covered != left_volumes or right_covered != right_volumes:
        _reject(
            "polytopal_complex.refinement_support_mismatch",
            "the two complexes must have exactly equal support; exact cell-volume conservation failed",
        )
    if not candidate_rows:
        _reject(
            "polytopal_complex.refinement_empty",
            "equal nonempty supports must produce maximal overlay cells",
        )

    overlay_cells = tuple(
        RationalVPolytope(
            space=left.space,
            vertices=tuple(
                RationalPolytopeVertex(
                    vertex_id=f"P{position:02d}v{vertex_index:02d}",
                    coordinates=tuple(
                        CanonicalRational.from_fraction(x) for x in point
                    ),
                )
                for vertex_index, point in enumerate(key)
            ),
        )
        for position, (_, _, key) in enumerate(candidate_rows)
    )
    refinement = polytopal_complex_closure(overlay_cells)
    pairs = tuple(
        CommonRefinementCellPair(
            left_cell_id=left.maximal_cells[i].cell_id,
            right_cell_id=right.maximal_cells[j].cell_id,
            refined_cell_id=refinement.source_cell_map[position].cell_id,
        )
        for position, (i, j, _) in enumerate(candidate_rows)
    )
    return CommonRefinementResult(
        left=left, right=right, refinement=refinement, cell_pairs=pairs
    )


__all__ = ["common_refinement"]
