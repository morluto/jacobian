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
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes._polyhedral_conversion import rational_rank
from jacobian.math.geometry.polytopes._rational_geometry import vertices_from_halfspaces
from jacobian.math.geometry.polytopes.complexes._models import (
    MAX_COMPLEX_CELLS,
    MAX_COMPLEX_COORDINATE_DIGITS,
    MAX_COMPLEX_COVER_RELATIONS,
    MAX_COMPLEX_DIMENSION,
    MAX_COMPLEX_FACE_ENUMERATION_WORK,
    MAX_COMPLEX_INTERSECTION_WORK,
    MAX_COMPLEX_TOTAL_FACES,
    ComplexFace,
    ComplexPoint,
    FaceCoverRelation,
    MaximalCellRecord,
    PairwiseIntersectionRecord,
    PolytopalComplexClosureResult,
    SourceCellTransport,
)
from jacobian.math.geometry.polytopes.complexes._spline import (
    piecewise_polynomial_evaluate,
    piecewise_polynomial_from_maximal_pieces,
    spline_space,
)
from jacobian.math.geometry.polytopes.operations import facet_incidence
from jacobian.math.geometry.polytopes.values import Vertex

__all__ = [
    "piecewise_polynomial_evaluate",
    "piecewise_polynomial_from_maximal_pieces",
    "polytopal_complex_closure",
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
