"""Domain-owned exact face closure of a finite family of rational maximal cells.

One operation is exposed.  Given maximal bounded rational polytopes in one
shared labelled ambient coordinate space, it computes the canonical
face-closed polytopal complex:

* the complete nonempty face family of every maximal cell, obtained by
  intersecting the cell with the supporting hyperplanes of its facets;
* the exact pairwise intersection of every unordered pair of maximal
  cells, which must be empty or a face of both (the face-to-face axiom);
* canonical geometric deduplication of maximal cells and faces with
  presentation provenance retained;
* the codimension-one cover relations, the f-vector, and the Euler
  characteristics.

The empty face is admitted and represented with dimension ``-1``.  Every
exact computation reuses the existing rational polytope facet kernel; this
owner adds only the bounded cell-face enumeration, canonicalization, and
incidence bookkeeping.  No new hull routine is introduced.

Admission happens before expansion.  Structural failures (mixed spaces,
non-face-to-face intersections, lower-dimensional maximal cells) are
``OperationDomainValidationError``; envelope overflows are
``OperationResourceAdmissionError``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import TypedDict

from sympy import Rational

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes._polyhedral_conversion import rational_rank
from jacobian.math.geometry.polytopes._rational_geometry import vertices_from_halfspaces
from jacobian.math.geometry.polytopes.complexes._models import (
    MAX_AFFINE_TRANSFORM_COMPONENT_DIGITS,
    MAX_AFFINE_TRANSFORM_OUTPUT_BYTES,
    MAX_AFFINE_TRANSFORM_WORK,
    MAX_COMPLEX_CELLS,
    MAX_COMPLEX_COORDINATE_DIGITS,
    MAX_COMPLEX_COVER_RELATIONS,
    MAX_COMPLEX_DIMENSION,
    MAX_COMPLEX_FACE_ENUMERATION_WORK,
    MAX_COMPLEX_INTERSECTION_WORK,
    MAX_COMPLEX_TOTAL_FACES,
    CommonRefinementRequest,
    CommonRefinementResult,
    ComplexCellTransport,
    ComplexFace,
    ComplexFaceTransport,
    ComplexPoint,
    FaceCoverRelation,
    MaximalCellRecord,
    PairwiseIntersectionRecord,
    PolytopalComplexAffineTransformRequest,
    PolytopalComplexAffineTransformResult,
    PolytopalComplexClosureResult,
    SourceCellTransport,
)
from jacobian.math.geometry.polytopes.complexes._spline import (
    piecewise_polynomial_add,
    piecewise_polynomial_evaluate,
    piecewise_polynomial_from_maximal_pieces,
    piecewise_polynomial_multiply,
    piecewise_polynomial_scalar_multiply,
    piecewise_polynomial_smoothness,
    spline_coordinates,
    spline_dimension,
    spline_evaluate,
    spline_refinement_map,
    spline_space,
)
from jacobian.math.geometry.polytopes.operations import facet_incidence
from jacobian.math.geometry.polytopes.values import Vertex

__all__ = [
    "piecewise_polynomial_add",
    "piecewise_polynomial_evaluate",
    "piecewise_polynomial_from_maximal_pieces",
    "piecewise_polynomial_multiply",
    "piecewise_polynomial_scalar_multiply",
    "piecewise_polynomial_smoothness",
    "polytopal_complex_affine_transform",
    "polytopal_complex_closure",
    "polytopal_complex_common_refinement",
    "spline_coordinates",
    "spline_dimension",
    "spline_evaluate",
    "spline_refinement_map",
    "spline_space",
]

Point = tuple[Fraction, ...]
FaceKey = tuple[Point, ...]
FacetRows = tuple[tuple[Fraction, ...], Fraction]


class _CanonicalCellRecord(TypedDict):
    sources: list[int]
    facets: list[FacetRows]
    faces: set[FaceKey]


type _CanonicalCells = dict[FaceKey, _CanonicalCellRecord]


def _point(coordinates: tuple[CanonicalRational, ...]) -> Point:
    return tuple(Fraction(*coordinate.as_integer_ratio()) for coordinate in coordinates)


def _canonical_key(points: Iterable[Point]) -> FaceKey:
    """Return the canonical geometric key of a finite point family."""

    return tuple(sorted(set(points)))


def _affine_dimension(points: Sequence[Point]) -> int:
    """Exact affine dimension of a point family; ``-1`` for the empty family."""

    if not points:
        return -1
    if len(points) == 1:
        return 0
    ambient = len(points[0])
    reference = points[0]
    differences = [
        [point[axis] - reference[axis] for axis in range(ambient)]
        for point in points[1:]
    ]
    return rational_rank(differences, ambient)


def _cell_facets(vertices: tuple[Vertex, ...], dimension: int) -> list[FacetRows]:
    """Enumerate the primitive supporting facets of one cell via the shared kernel."""

    try:
        profile = facet_incidence(vertices, dimension)
    except OperationDomainValidationError as exc:
        raise OperationDomainValidationError(
            location=("cells",),
            code="polytopal_complex.facet_profile_not_admitted",
            message=str(exc),
        ) from exc
    return [
        (
            tuple(
                Fraction(*c.as_integer_ratio()) for c in facet.halfspace.coefficients
            ),
            Fraction(*facet.halfspace.offset.as_integer_ratio()),
        )
        for facet in profile.facets
    ]


def _extreme_points(
    points: Sequence[Point], dimension: int, facets: Sequence[FacetRows]
) -> list[Point]:
    """Retain exactly the extreme points of ``conv(points)``.

    A source point is extreme iff the facet normals active at it span the
    ambient space, the standard exact rank test.  Redundant interior rows
    are dropped so that geometrically equal cells share one canonical key.
    """

    normals = [list(coefficients) for coefficients, _offset in facets]
    extreme: list[Point] = []
    for point in points:
        active = [
            normals[index]
            for index, (coefficients, offset) in enumerate(facets)
            if sum(coefficients[axis] * point[axis] for axis in range(dimension))
            == offset
        ]
        if active and rational_rank(active, dimension) == dimension:
            extreme.append(point)
    return extreme


def _enumerate_faces(
    points: Sequence[Point], facets: Sequence[FacetRows]
) -> set[FaceKey]:
    """Enumerate every nonempty face of ``conv(points)``.

    Every face is an intersection of the polytope with some collection of
    its supporting facet hyperplanes.  The closure is generated exactly by
    repeatedly intersecting a known face with each facet hyperplane: the
    intersection with a supporting hyperplane is the convex hull of the
    face vertices lying on it, so no new points are created.
    """

    dimension = len(points[0])
    start = _canonical_key(points)
    faces: set[FaceKey] = {start}
    frontier = [start]
    while frontier:
        current = frontier.pop()
        for coefficients, offset in facets:
            on_hyperplane = tuple(
                sorted(
                    point
                    for point in current
                    if sum(
                        coefficients[axis] * point[axis] for axis in range(dimension)
                    )
                    == offset
                )
            )
            if not on_hyperplane or on_hyperplane == current:
                continue
            if on_hyperplane not in faces:
                faces.add(on_hyperplane)
                frontier.append(on_hyperplane)
    return faces


def _intersection_key(
    first: Sequence[FacetRows], second: Sequence[FacetRows], dimension: int
) -> FaceKey | None:
    """Compute the exact vertex key of one maximal-cell intersection, or ``None``."""

    rows = [
        (
            [Rational(c.numerator, c.denominator) for c in coefficients],
            Rational(offset.numerator, offset.denominator),
        )
        for coefficients, offset in (*first, *second)
    ]
    vertices = vertices_from_halfspaces(rows, dimension)
    if not vertices:
        return None
    points = tuple(
        tuple(Fraction(int(entry.p), int(entry.q)) for entry in vertex)
        for vertex in vertices
    )
    return _canonical_key(points)


def _admit_polytopal_complex(
    cells: Sequence[object],
) -> tuple[int, RationalCoordinateSpace]:
    """Enforce the closure envelope shared by native and catalog calls."""

    if len(cells) == 0:
        raise OperationDomainValidationError(
            location=("cells",),
            code="polytopal_complex.empty_cell_family",
            message="a polytopal complex requires at least one maximal cell",
        )
    if len(cells) > MAX_COMPLEX_CELLS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="polytopal_complex.cell_count_over_envelope",
            message=(
                f"maximal-cell presentations exceed the {MAX_COMPLEX_CELLS}-cell envelope"
            ),
        )
    typed: list[RationalVPolytope] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, RationalVPolytope):
            raise OperationDomainValidationError(
                location=("cells", index),
                code="polytopal_complex.cell_not_a_v_polytope",
                message="every maximal cell must be a labelled RationalVPolytope value",
            )
        typed.append(cell)
    space = typed[0].space
    for index in range(1, len(typed)):
        if typed[index].space != space:
            raise OperationDomainValidationError(
                location=("cells", index),
                code="polytopal_complex.mixed_ambient_spaces",
                message=(
                    "all maximal cells must share one identical labelled ambient "
                    "coordinate space"
                ),
            )
    dimension = len(space.axes)
    if dimension > MAX_COMPLEX_DIMENSION:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="polytopal_complex.dimension_over_envelope",
            message=(
                f"ambient dimension {dimension} exceeds the "
                f"{MAX_COMPLEX_DIMENSION}-dimension envelope"
            ),
        )
    for index, cell in enumerate(typed):
        for vertex in cell.vertices:
            for coordinate in vertex.coordinates:
                try:
                    require_bounded_rational(
                        coordinate,
                        max_digits=MAX_COMPLEX_COORDINATE_DIGITS,
                        label="complex cell coordinate",
                    )
                except ValueError as exc:
                    raise OperationResourceAdmissionError(
                        location=("cells", index),
                        code="polytopal_complex.coordinate_digits_over_envelope",
                        message=str(exc),
                    ) from exc
    return dimension, space


def _require_intersection_work(facet_counts: Sequence[int], dimension: int) -> None:
    """Bound the exact candidate systems solved across all pairs before intersecting."""

    total = 0
    for first in range(len(facet_counts)):
        for second in range(first + 1, len(facet_counts)):
            rows = facet_counts[first] + facet_counts[second]
            if rows < dimension:
                continue
            total += math.comb(rows, dimension)
            if total > MAX_COMPLEX_INTERSECTION_WORK:
                raise OperationResourceAdmissionError(
                    location=("cells",),
                    code="polytopal_complex.intersection_work_over_envelope",
                    message=(
                        "pairwise-intersection candidate systems exceed the "
                        f"{MAX_COMPLEX_INTERSECTION_WORK}-system envelope"
                    ),
                )


class _Components:
    """Small deterministic disjoint-set structure over canonical cells."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, index: int) -> int:
        while self._parent[index] != index:
            self._parent[index] = self._parent[self._parent[index]]
            index = self._parent[index]
        return index

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self._parent[second_root] = first_root

    def count(self) -> int:
        return len({self.find(index) for index in range(len(self._parent))})


def _complex_point(point: Point) -> ComplexPoint:
    return ComplexPoint(
        coordinates=tuple(CanonicalRational.from_fraction(value) for value in point)
    )


def _collect_canonical_cells(
    cells: Sequence[RationalVPolytope], dimension: int
) -> tuple[_CanonicalCells, list[int]]:
    """Deduplicate maximal cells and enumerate each cell's face closure."""

    canonical_cells: _CanonicalCells = {}
    facet_counts: list[int] = []
    estimated_work = 0
    for index, cell in enumerate(cells):
        points = [_point(vertex.coordinates) for vertex in cell.vertices]
        vertices = tuple(
            Vertex(coordinates=vertex.coordinates) for vertex in cell.vertices
        )
        facets = _cell_facets(vertices, dimension)
        extreme = _extreme_points(points, dimension, facets)
        if _affine_dimension(extreme) != dimension:
            raise OperationDomainValidationError(
                location=("cells", index),
                code="polytopal_complex.maximal_cell_not_full_dimensional",
                message=(
                    "every maximal cell must be full-dimensional in the shared "
                    "ambient coordinate space"
                ),
            )
        estimated_work += min(2 ** len(facets), MAX_COMPLEX_TOTAL_FACES + 1)
        if estimated_work > MAX_COMPLEX_FACE_ENUMERATION_WORK:
            raise OperationResourceAdmissionError(
                location=("cells", index),
                code="polytopal_complex.face_enumeration_work_over_envelope",
                message=(
                    "per-cell face-closure work exceeds the "
                    f"{MAX_COMPLEX_FACE_ENUMERATION_WORK}-step envelope"
                ),
            )
        key = _canonical_key(extreme)
        record = canonical_cells.get(key)
        if record is None:
            canonical_cells[key] = {
                "sources": [index],
                "facets": facets,
                "faces": _enumerate_faces(extreme, facets),
            }
            facet_counts.append(len(facets))
        else:
            record["sources"].append(index)
    return canonical_cells, facet_counts


def _union_global_faces(
    canonical_cells: _CanonicalCells,
) -> set[FaceKey]:
    """Union every cell face into the canonical global face family."""

    global_faces: set[FaceKey] = {()}
    for record in canonical_cells.values():
        global_faces |= record["faces"]
        if len(global_faces) > MAX_COMPLEX_TOTAL_FACES:
            raise OperationResourceAdmissionError(
                location=("cells",),
                code="polytopal_complex.face_closure_over_envelope",
                message=(
                    "the canonical face closure exceeds the "
                    f"{MAX_COMPLEX_TOTAL_FACES}-face envelope"
                ),
            )
    return global_faces


def _build_pairwise_rows(
    canonical_cells: _CanonicalCells,
    ordered_cell_keys: list[FaceKey],
    faces_of_cell: dict[str, set[FaceKey]],
    face_id_of: dict[FaceKey, str],
    cell_id_of: dict[FaceKey, str],
    dimension: int,
    components: _Components,
) -> tuple[PairwiseIntersectionRecord, ...]:
    """Intersect every maximal-cell pair and enforce the face-to-face axiom."""

    pairwise_rows: list[PairwiseIntersectionRecord] = []
    for first in range(len(ordered_cell_keys)):
        for second in range(first + 1, len(ordered_cell_keys)):
            first_key = ordered_cell_keys[first]
            second_key = ordered_cell_keys[second]
            first_facets = canonical_cells[first_key]["facets"]
            second_facets = canonical_cells[second_key]["facets"]
            intersection = _intersection_key(first_facets, second_facets, dimension)
            first_sources = canonical_cells[first_key]["sources"]
            second_sources = canonical_cells[second_key]["sources"]
            first_source = min(first_sources)
            second_source = min(second_sources)
            first_id = cell_id_of[first_key]
            second_id = cell_id_of[second_key]
            if intersection is None:
                pairwise_rows.append(
                    PairwiseIntersectionRecord(
                        first_cell_id=first_id,
                        second_cell_id=second_id,
                        first_source_index=first_source,
                        second_source_index=second_source,
                        status="empty",
                        intersection_face_id=None,
                    )
                )
                continue
            if (
                intersection not in faces_of_cell[first_id]
                or intersection not in faces_of_cell[second_id]
            ):
                raise OperationDomainValidationError(
                    location=("cells", first_source, second_source),
                    code="polytopal_complex.non_face_intersection",
                    message=(
                        "maximal cells at presentation positions "
                        f"{first_source} and {second_source} intersect in a "
                        "nonempty set that is not a face of both"
                    ),
                )
            components.union(first, second)
            pairwise_rows.append(
                PairwiseIntersectionRecord(
                    first_cell_id=first_id,
                    second_cell_id=second_id,
                    first_source_index=first_source,
                    second_source_index=second_source,
                    status="face",
                    intersection_face_id=face_id_of[intersection],
                )
            )
    return tuple(pairwise_rows)


def _build_cover_rows(
    ordered_face_keys: list[FaceKey],
    faces_by_dimension: dict[int, list[FaceKey]],
    face_id_of: dict[FaceKey, str],
) -> tuple[FaceCoverRelation, ...]:
    """Build every codimension-one cover relation between canonical faces."""

    cover_rows: list[FaceCoverRelation] = []
    for upper in ordered_face_keys:
        upper_dimension = _affine_dimension(upper)
        upper_points = set(upper)
        for lower in faces_by_dimension.get(upper_dimension - 1, ()):
            if set(lower) <= upper_points:
                cover_rows.append(
                    FaceCoverRelation(
                        lower_face_id=face_id_of[lower],
                        upper_face_id=face_id_of[upper],
                    )
                )
                if len(cover_rows) > MAX_COMPLEX_COVER_RELATIONS:
                    raise OperationResourceAdmissionError(
                        location=("cells",),
                        code="polytopal_complex.cover_relations_over_envelope",
                        message=(
                            "face cover relations exceed the "
                            f"{MAX_COMPLEX_COVER_RELATIONS}-row envelope"
                        ),
                    )
    return tuple(cover_rows)


def _build_face_models(
    ordered_face_keys: list[FaceKey],
    face_id_of: dict[FaceKey, str],
    face_to_cell_ids: dict[FaceKey, set[str]],
) -> tuple[ComplexFace, ...]:
    """Materialize the canonical face values."""

    return tuple(
        ComplexFace(
            face_id=face_id_of[face_key],
            dimension=_affine_dimension(face_key),
            vertices=tuple(_complex_point(point) for point in face_key),
            maximal_cell_ids=tuple(sorted(face_to_cell_ids[face_key])),
        )
        for face_key in ordered_face_keys
    )


def _build_cell_models(
    canonical_cells: _CanonicalCells,
    ordered_cell_keys: list[FaceKey],
    cell_id_of: dict[FaceKey, str],
    face_id_of: dict[FaceKey, str],
    face_index_of: dict[FaceKey, int],
    dimension: int,
) -> tuple[MaximalCellRecord, ...]:
    """Materialize the canonical maximal-cell values with their facets."""

    cell_models: list[MaximalCellRecord] = []
    for cell_key in ordered_cell_keys:
        cell_id = cell_id_of[cell_key]
        record = canonical_cells[cell_key]
        sources = record["sources"]
        cell_faces = record["faces"]
        facet_keys = sorted(
            (face for face in cell_faces if _affine_dimension(face) == dimension - 1),
            key=lambda key: face_index_of[key],
        )
        cell_models.append(
            MaximalCellRecord(
                cell_id=cell_id,
                source_indices=tuple(sorted(sources)),
                dimension=dimension,
                vertices=tuple(_complex_point(point) for point in cell_key),
                facet_face_ids=tuple(sorted(face_id_of[key] for key in facet_keys)),
            )
        )
    return tuple(cell_models)


def _build_source_rows(
    canonical_cells: _CanonicalCells,
    ordered_cell_keys: list[FaceKey],
    cell_id_of: dict[FaceKey, str],
    source_count: int,
) -> tuple[SourceCellTransport, ...]:
    """Build one presentation-position transport row per source cell."""

    source_to_cell: dict[int, str] = {}
    for cell_key in ordered_cell_keys:
        sources = canonical_cells[cell_key]["sources"]
        for source in sources:
            source_to_cell[source] = cell_id_of[cell_key]
    return tuple(
        SourceCellTransport(source_index=index, cell_id=source_to_cell[index])
        for index in range(source_count)
    )


def polytopal_complex_closure(
    cells: tuple[RationalVPolytope, ...],
) -> PolytopalComplexClosureResult:
    """Compute the canonical face-closed complex of a family of maximal cells."""

    dimension, space = _admit_polytopal_complex(cells)
    canonical_cells, facet_counts = _collect_canonical_cells(cells, dimension)
    ordered_cell_keys = sorted(canonical_cells)
    if len(ordered_cell_keys) > MAX_COMPLEX_CELLS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="polytopal_complex.cell_count_over_envelope",
            message="canonical maximal cells exceed the admitted envelope",
        )

    global_faces = _union_global_faces(canonical_cells)
    _require_intersection_work(facet_counts, dimension)

    ordered_face_keys = sorted(
        global_faces, key=lambda key: (_affine_dimension(key), key)
    )
    face_id_of = {key: f"f{position}" for position, key in enumerate(ordered_face_keys)}
    face_index_of = {key: position for position, key in enumerate(ordered_face_keys)}
    cell_id_of = {key: f"M{position}" for position, key in enumerate(ordered_cell_keys)}

    face_to_cell_ids: dict[FaceKey, set[str]] = {
        (): {cell_id_of[key] for key in ordered_cell_keys}
    }
    faces_of_cell: dict[str, set[FaceKey]] = {}
    for cell_key in ordered_cell_keys:
        cell_faces = canonical_cells[cell_key]["faces"]
        cell_id = cell_id_of[cell_key]
        faces_of_cell[cell_id] = set(cell_faces)
        for face_key in cell_faces:
            face_to_cell_ids.setdefault(face_key, set()).add(cell_id)

    components = _Components(len(ordered_cell_keys))
    pairwise_rows = _build_pairwise_rows(
        canonical_cells,
        ordered_cell_keys,
        faces_of_cell,
        face_id_of,
        cell_id_of,
        dimension,
        components,
    )

    faces_by_dimension: dict[int, list[FaceKey]] = {}
    for face_key in ordered_face_keys:
        faces_by_dimension.setdefault(_affine_dimension(face_key), []).append(face_key)

    cover_rows = _build_cover_rows(ordered_face_keys, faces_by_dimension, face_id_of)
    face_models = _build_face_models(ordered_face_keys, face_id_of, face_to_cell_ids)
    cell_models = _build_cell_models(
        canonical_cells,
        ordered_cell_keys,
        cell_id_of,
        face_id_of,
        face_index_of,
        dimension,
    )
    source_rows = _build_source_rows(
        canonical_cells, ordered_cell_keys, cell_id_of, len(cells)
    )

    dimension_counts = {level: len(keys) for level, keys in faces_by_dimension.items()}
    f_vector = tuple(
        [1] + [dimension_counts.get(level, 0) for level in range(dimension + 1)]
    )
    euler_characteristic = sum(
        (-1) ** level * dimension_counts.get(level, 0) for level in range(dimension + 1)
    )

    return PolytopalComplexClosureResult._from_kernel(
        space=space,
        dimension=dimension,
        faces=face_models,
        maximal_cells=cell_models,
        cover_relations=cover_rows,
        pairwise_intersections=pairwise_rows,
        source_cell_map=source_rows,
        f_vector=f_vector,
        euler_characteristic=euler_characteristic,
        reduced_euler_characteristic=euler_characteristic - 1,
        component_count=components.count(),
    )


def _rational_digits(value: Fraction) -> int:
    return max(len(str(abs(value.numerator))), len(str(value.denominator)))


def _determinant(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    """Compute a small exact determinant by fraction-preserving elimination."""
    work = [list(row) for row in matrix]
    determinant = Fraction(1)
    for column in range(len(work)):
        pivot = next(
            (row for row in range(column, len(work)) if work[row][column]), None
        )
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = -determinant
        pivot_value = work[column][column]
        determinant *= pivot_value
        for row in range(column + 1, len(work)):
            if not work[row][column]:
                continue
            multiplier = work[row][column] / pivot_value
            for entry in range(column + 1, len(work)):
                work[row][entry] -= multiplier * work[column][entry]
            work[row][column] = Fraction(0)
    return determinant


def _affine_transport_preflight(
    request: PolytopalComplexAffineTransformRequest,
) -> tuple[list[list[Fraction]], list[Fraction]]:
    """Bound arithmetic and output before rebuilding or transforming geometry."""
    complex_value = request.complex
    dimension = len(complex_value.space.axes)
    matrix = [
        [Fraction(*entry.as_integer_ratio()) for entry in row] for row in request.matrix
    ]
    translation = [Fraction(*entry.as_integer_ratio()) for entry in request.translation]
    matrix_digits = max(_rational_digits(value) for row in matrix for value in row)
    translation_digits = max(_rational_digits(value) for value in translation)
    source_points = tuple(
        vertex.coordinates
        for cell in complex_value.maximal_cells
        for vertex in cell.vertices
    ) + tuple(
        vertex.coordinates for face in complex_value.faces for vertex in face.vertices
    )
    if any(len(point) != dimension for point in source_points):
        raise OperationDomainValidationError(
            location=("complex",),
            code="polytopal_complex.affine_transform_coordinate_dimension",
            message="every complex point must use exactly the labelled space axes",
        )
    coordinate_digits = max(
        _rational_digits(Fraction(*coordinate.as_integer_ratio()))
        for point in source_points
        for coordinate in point
    )
    for component, digits in (
        ("matrix", matrix_digits),
        ("translation", translation_digits),
        ("coordinates", coordinate_digits),
    ):
        if digits > MAX_COMPLEX_COORDINATE_DIGITS:
            raise OperationResourceAdmissionError(
                location=(component,),
                code="polytopal_complex.affine_transform_input_digits_over_envelope",
                message=(
                    f"affine transform {component} exceed the "
                    f"{MAX_COMPLEX_COORDINATE_DIGITS}-digit input envelope"
                ),
            )

    point_count = sum(len(cell.vertices) for cell in complex_value.maximal_cells)
    face_point_count = sum(len(face.vertices) for face in complex_value.faces)
    arithmetic_points = point_count + face_point_count
    work_bound = arithmetic_points * dimension * dimension
    if work_bound > MAX_AFFINE_TRANSFORM_WORK:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="polytopal_complex.affine_transform_work_over_envelope",
            message=(
                "matrix-coordinate products exceed the "
                f"{MAX_AFFINE_TRANSFORM_WORK}-operation affine transport envelope"
            ),
        )

    # A product uses at most the sum of operand digit lengths. Summing at most
    # d products and one translation with a common denominator gives this
    # conservative bound for either reduced numerator or denominator.
    term_digits = max(matrix_digits + coordinate_digits, translation_digits)
    output_component_bound = (dimension + 1) * term_digits + dimension
    if output_component_bound > MAX_AFFINE_TRANSFORM_COMPONENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="polytopal_complex.affine_transform_growth_over_envelope",
            message=(
                "the conservative transformed-coordinate height exceeds the "
                f"{MAX_AFFINE_TRANSFORM_COMPONENT_DIGITS}-digit envelope"
            ),
        )

    determinant_digits_bound = (
        dimension * matrix_digits + math.ceil(math.log10(math.factorial(dimension)))
        if dimension > 1
        else matrix_digits
    )
    if determinant_digits_bound > MAX_AFFINE_TRANSFORM_COMPONENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="polytopal_complex.affine_transform_determinant_growth_over_envelope",
            message="the determinant height bound exceeds the affine transport envelope",
        )

    input_bytes = len(complex_value.model_dump_json())
    coordinate_slots = 2 * arithmetic_points * dimension
    output_bound = (
        2 * input_bytes
        + coordinate_slots * (2 * output_component_bound + 48)
        + 256 * (len(complex_value.faces) + len(complex_value.maximal_cells))
        + 128 * len(complex_value.cover_relations)
    )
    if output_bound > min(
        MAX_AFFINE_TRANSFORM_OUTPUT_BYTES, CanonicalLimits().max_output_bytes
    ):
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="polytopal_complex.affine_transform_output_over_envelope",
            message=(
                "the conservative source, target, and transport output exceeds "
                f"the {MAX_AFFINE_TRANSFORM_OUTPUT_BYTES}-byte envelope"
            ),
        )

    if _determinant(matrix) == 0:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="polytopal_complex.affine_transform_singular",
            message="an affine complex transport requires an invertible matrix",
        )
    return matrix, translation


def _affine_image(
    point: Point,
    matrix: Sequence[Sequence[Fraction]],
    translation: Sequence[Fraction],
) -> Point:
    return tuple(
        sum(
            (matrix[row][column] * point[column] for column in range(len(point))),
            translation[row],
        )
        for row in range(len(point))
    )


def _polytope_from_complex_cell(
    space: RationalCoordinateSpace, points: Sequence[Point], prefix: str
) -> RationalVPolytope:
    ordered = sorted(set(points))
    return RationalVPolytope(
        space=space,
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=f"{prefix}{index:03d}",
                coordinates=tuple(
                    CanonicalRational.from_fraction(value) for value in point
                ),
            )
            for index, point in enumerate(ordered)
        ),
    )


def polytopal_complex_affine_transform(
    request: PolytopalComplexAffineTransformRequest,
) -> PolytopalComplexAffineTransformResult:
    """Transport a complete complex through one invertible rational affine map."""
    matrix, translation = _affine_transport_preflight(request)
    # The incidence ledger is caller-supplied mathematical data. Rebuild it
    # from source maximal cells before relying on it for face transport.
    from jacobian.math.geometry.polytopes.complexes._spline import _admit_complex

    source = _admit_complex(request.complex)
    transformed_point_map: dict[Point, Point] = {}
    for point in (
        _point(vertex.coordinates) for face in source.faces for vertex in face.vertices
    ):
        if point not in transformed_point_map:
            image = _affine_image(point, matrix, translation)
            if any(
                _rational_digits(coordinate) > MAX_COMPLEX_COORDINATE_DIGITS
                for coordinate in image
            ):
                raise OperationResourceAdmissionError(
                    location=("matrix",),
                    code="polytopal_complex.affine_transform_result_digits_over_envelope",
                    message=(
                        "a transformed coordinate exceeds the "
                        f"{MAX_COMPLEX_COORDINATE_DIGITS}-digit complex envelope"
                    ),
                )
            transformed_point_map[point] = image
    transformed_cells = tuple(
        _polytope_from_complex_cell(
            source.space,
            tuple(
                transformed_point_map[_point(vertex.coordinates)]
                for vertex in cell.vertices
            ),
            f"t{cell_index:02d}v",
        )
        for cell_index, cell in enumerate(source.maximal_cells)
    )
    target = polytopal_complex_closure(transformed_cells)

    target_cells_by_source_index = {
        row.source_index: row.cell_id for row in target.source_cell_map
    }
    cell_transport = tuple(
        ComplexCellTransport(
            source_cell_id=source_cell.cell_id,
            target_cell_id=target_cells_by_source_index[index],
        )
        for index, source_cell in enumerate(source.maximal_cells)
    )
    target_face_by_key = {
        _canonical_key(
            _point(vertex.coordinates) for vertex in face.vertices
        ): face.face_id
        for face in target.faces
    }
    face_transport = tuple(
        ComplexFaceTransport(
            source_face_id=face.face_id,
            target_face_id=target_face_by_key[
                _canonical_key(
                    transformed_point_map[_point(vertex.coordinates)]
                    for vertex in face.vertices
                )
            ],
        )
        for face in source.faces
    )
    return PolytopalComplexAffineTransformResult(
        source=source,
        target=target,
        matrix=request.matrix,
        translation=request.translation,
        cell_transport=cell_transport,
        face_transport=face_transport,
    )


def polytopal_complex_common_refinement(
    request: CommonRefinementRequest,
) -> CommonRefinementResult:
    """Compute an exact support-preserving common refinement."""
    from jacobian.math.geometry.polytopes.complexes._refinement import common_refinement

    return common_refinement(request)
