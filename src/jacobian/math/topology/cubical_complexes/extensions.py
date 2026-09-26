"""Canonical cubical boundary, relative, and triangulation transforms."""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Self

from pydantic import Field, ValidationError, model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_FACES,
    MAX_TOPOLOGY_FACETS,
    MAX_TOPOLOGY_VERTICES,
    FiniteSimplicialComplex,
    Simplex,
    VertexLabel,
    face_closure,
)
from jacobian.math.topology._models import (
    canonical_complex as canonical_simplicial_complex,
)
from jacobian.math.topology.chain_complexes.values import MAX_OPERATION_MATRIX_CELLS
from jacobian.math.topology.cubical_complexes._models import (
    MAX_CELLS,
    MAX_CUBICAL_BITMAP_RESULT_SIZE,
    MAX_CUBICAL_BITMAP_SIDE,
    MAX_CUBICAL_CHAIN_GROUP,
    MAX_CUBICAL_TRIANGULATION_RESULT_BYTES,
    MAX_DIM,
    MAX_TRIANGULATION_FACE_CANDIDATES,
    MAX_TRIANGULATION_POINTS,
    MAX_TRIANGULATION_SOURCE_CELLS,
    CubicalBitmapPixelCell,
    CubicalBitmapRequest,
    CubicalBitmapResult,
    CubicalCell,
    CubicalComplex,
)
from jacobian.math.topology.cubical_complexes.operations import (
    _boundary_terms,
    _canonical_complex,
    _cells_by_dimension,
)


class CubicalBoundaryRequest(StrictModel):
    cells: tuple[CubicalCell, ...] = Field(min_length=1)


class CubicalBoundaryTerm(StrictModel):
    source: CubicalCell
    face: CubicalCell
    coefficient: int


class CubicalBoundaryResult(StrictModel):
    complex: CubicalComplex
    maximal_cells: tuple[CubicalCell, ...]
    terms: tuple[CubicalBoundaryTerm, ...]
    boundary_cells: tuple[CubicalCell, ...]


class RelativeCubicalHomologyRequest(StrictModel):
    cells: tuple[CubicalCell, ...] = Field(min_length=1)
    subcomplex_cells: tuple[CubicalCell, ...] = Field(min_length=1)
    prime: int = Field(ge=2, le=1000003, default=2)


class RelativeCubicalHomologyResult(StrictModel):
    complex: CubicalComplex
    subcomplex: CubicalComplex
    prime: int
    betti_numbers: tuple[int, ...]


class CubicalTriangulationRequest(StrictModel):
    cells: tuple[CubicalCell, ...] = Field(min_length=1)


class CubicalTriangulationCellMap(StrictModel):
    """The target maximal simplices subdividing one maximal source cube."""

    source_cell: CubicalCell
    simplices: tuple[Simplex, ...] = Field(min_length=1, max_length=MAX_TOPOLOGY_FACETS)


class CubicalVertexMap(StrictModel):
    """A simplicial vertex's exact source lattice coordinate."""

    vertex: VertexLabel
    coordinate: tuple[int, ...] = Field(min_length=1, max_length=MAX_DIM)


class CubicalTriangulationResult(StrictModel):
    source_complex: CubicalComplex
    simplicial_complex: FiniteSimplicialComplex
    vertex_map: tuple[CubicalVertexMap, ...] = Field(
        min_length=1, max_length=MAX_TOPOLOGY_VERTICES
    )
    cell_maps: tuple[CubicalTriangulationCellMap, ...] = Field(
        min_length=1, max_length=MAX_TOPOLOGY_FACETS
    )

    @model_validator(mode="after")
    def require_source_axis(self) -> Self:
        if len(self.source_complex.cells) > MAX_TOPOLOGY_FACES:
            raise ValueError(
                "source cubical complex exceeds the triangulation face bound"
            )
        coordinates = tuple(entry.coordinate for entry in self.vertex_map)
        vertices = tuple(entry.vertex for entry in self.vertex_map)
        if (
            vertices != self.simplicial_complex.vertices
            or coordinates != tuple(sorted(set(coordinates)))
            or vertices != tuple(f"v{index:04d}" for index in range(len(vertices)))
            or any(
                len(point) != self.source_complex.ambient_dimension
                for point in coordinates
            )
        ):
            raise ValueError(
                "vertex map must bind the canonical target axis to ordered, distinct source lattice points"
            )
        mapped_cells = tuple(entry.source_cell for entry in self.cell_maps)
        if (
            mapped_cells
            != tuple(sorted(set(mapped_cells), key=lambda cell: cell.intervals))
            or any(cell not in self.source_complex.cells for cell in mapped_cells)
            or sum(len(mapping.simplices) for mapping in self.cell_maps)
            > MAX_TOPOLOGY_FACETS
        ):
            raise ValueError(
                "cell maps must use bounded, distinct source cells and simplices"
            )
        target_faces = {
            face
            for dimension in self.simplicial_complex.faces_by_dimension
            for face in dimension.faces
        }
        for cell_map in self.cell_maps:
            if any(
                tuple(sorted(simplex)) != simplex
                or any(vertex not in vertices for vertex in simplex)
                or simplex not in target_faces
                for simplex in cell_map.simplices
            ):
                raise ValueError("cell simplex maps must name canonical target faces")
        return self

    def require_valid_transport(self) -> None:
        """Check the exact cubical-to-simplicial relation when a consumer relies on it.

        This claim check is deliberately explicit: construction and decoding
        validate bounded typed axes without replaying the triangulation.
        """
        _require_valid_triangulation_transport(self)


def _require_valid_triangulation_transport(
    result: CubicalTriangulationResult,
) -> None:
    """Check a source/target map after admitting its complete exact workload."""
    source_cells = result.source_complex.cells
    if len(source_cells) > MAX_TOPOLOGY_FACES:
        raise OperationResourceAdmissionError(
            location=("source_complex",),
            code="cubical_complex.triangulation_claim_source_budget",
            message="transport claim checking is limited to 2,048 source cells",
        )
    if any(
        abs(coordinate).bit_length() > 256 or len(str(abs(coordinate))) > 64
        for cell in source_cells
        for interval in cell.intervals
        for coordinate in interval
    ):
        raise OperationResourceAdmissionError(
            location=("source_complex",),
            code="cubical_complex.triangulation_claim_coordinate_budget",
            message="transport claim coordinates are limited to 64 digits",
        )
    source_keys = {cell.intervals for cell in source_cells}
    closure_work_bound = len(source_cells) * MAX_DIM * 2
    if closure_work_bound > MAX_TOPOLOGY_FACES * MAX_DIM * 2:
        raise OperationResourceAdmissionError(
            location=("source_complex",),
            code="cubical_complex.triangulation_claim_face_work",
            message="transport source face check exceeds its work bound",
        )
    if any(
        (*cell.intervals[:axis], (endpoint, endpoint), *cell.intervals[axis + 1 :])
        not in source_keys
        for cell in source_cells
        for axis, (start, end) in enumerate(cell.intervals)
        if start < end
        for endpoint in (start, end)
    ):
        raise ValueError("transport source cubical complex is not face closed")

    maximal_cells = _maximal_cubical_cells(source_cells)
    simplex_count = sum(
        1 if cell.dimension == 0 else _factorial(cell.dimension)
        for cell in maximal_cells
    )
    face_candidate_count = sum(
        (1 if cell.dimension == 0 else _factorial(cell.dimension))
        * ((1 << (cell.dimension + 1)) - 1)
        for cell in maximal_cells
    )
    point_candidate_count = sum(1 << cell.dimension for cell in maximal_cells)
    if (
        simplex_count > MAX_TOPOLOGY_FACETS
        or face_candidate_count > MAX_TRIANGULATION_FACE_CANDIDATES
        or point_candidate_count > MAX_TRIANGULATION_POINTS
    ):
        raise OperationResourceAdmissionError(
            location=("cell_maps",),
            code="cubical_complex.triangulation_claim_expansion",
            message="transport claim exceeds the admitted simplex, point, or face work",
        )
    if len(result.simplicial_complex.vertices) > MAX_TOPOLOGY_VERTICES:
        raise OperationResourceAdmissionError(
            location=("simplicial_complex",),
            code="cubical_complex.triangulation_claim_vertex_budget",
            message="transport target exceeds the 64-vertex bound",
        )
    if result.simplicial_complex.closure_size > MAX_TOPOLOGY_FACES:
        raise OperationResourceAdmissionError(
            location=("simplicial_complex",),
            code="cubical_complex.triangulation_claim_face_budget",
            message="transport target exceeds the 2,048-face bound",
        )

    coordinates = tuple(entry.coordinate for entry in result.vertex_map)
    vertex_for_point = {entry.coordinate: entry.vertex for entry in result.vertex_map}
    expected_coordinates = tuple(
        sorted({point for cell in maximal_cells for point in _cube_choices(cell)})
    )
    mapped_cells = tuple(entry.source_cell for entry in result.cell_maps)
    if mapped_cells != maximal_cells or coordinates != expected_coordinates:
        raise ValueError(
            "cell and vertex maps do not cover the exact maximal source presentation"
        )

    expected_facets: set[Simplex] = set()
    target_faces = {
        face
        for dimension in result.simplicial_complex.faces_by_dimension
        for face in dimension.faces
    }
    for cell_map in result.cell_maps:
        expected_simplices = tuple(
            sorted(
                tuple(sorted(vertex_for_point[point] for point in path))
                for path in _freudenthal_simplex_families(cell_map.source_cell)
            )
        )
        if cell_map.simplices != expected_simplices:
            raise ValueError(
                "cell simplex maps do not equal the source cube's Freudenthal subdivision"
            )
        expected_facets.update(expected_simplices)
        if any(simplex not in target_faces for simplex in expected_simplices):
            raise ValueError("cell simplex maps are absent from the target complex")
    if tuple(sorted(expected_facets)) != result.simplicial_complex.maximal_simplices:
        raise ValueError("target facets do not equal the mapped source subdivisions")
    expected_closure = face_closure(tuple(sorted(expected_facets)))
    actual_closure = tuple(
        group.faces for group in result.simplicial_complex.faces_by_dimension
    )
    if actual_closure != expected_closure:
        raise ValueError("target faces do not equal the mapped subdivisions' closure")


def _maximal_cubical_cells(cells: tuple[CubicalCell, ...]) -> tuple[CubicalCell, ...]:
    """Derive maximal cells from a face-closed cubical complex."""
    cell_keys = {cell.intervals for cell in cells}
    maximal = []
    for cell in cells:
        has_coface = False
        for axis, (start, end) in enumerate(cell.intervals):
            if start != end:
                continue
            for interval in ((start - 1, start), (start, start + 1)):
                candidate = list(cell.intervals)
                candidate[axis] = interval
                if tuple(candidate) in cell_keys:
                    has_coface = True
                    break
            if has_coface:
                break
        if not has_coface:
            maximal.append(cell)
    return tuple(maximal)


def _freudenthal_simplex_families(
    cell: CubicalCell,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return the path simplices of one elementary cube as lattice points."""
    active = [i for i, (start, end) in enumerate(cell.intervals) if start < end]
    base = tuple(start for start, _ in cell.intervals)
    paths = []
    for order in permutations(active):
        points = [base]
        current = list(base)
        for axis in order:
            current = current.copy()
            current[axis] = cell.intervals[axis][1]
            points.append(tuple(current))
        paths.append(tuple(points))
    return tuple(paths)


def _strictly_contains(container: CubicalCell, cell: CubicalCell) -> bool:
    return container != cell and all(
        outer_start <= inner_start and inner_end <= outer_end
        for (outer_start, outer_end), (inner_start, inner_end) in zip(
            container.intervals, cell.intervals, strict=True
        )
    )


def bitmap_to_complex(request: CubicalBitmapRequest) -> CubicalBitmapResult:
    """Convert true pixels to closed unit squares on the (column, row) grid.

    Rows increase downward and columns increase rightward.  Foreground pixel
    ``(r, c)`` denotes the closed 2-cell ``([c,c+1], [r,r+1])``.  The full
    cubical face closure is returned; false pixels contribute no cells.
    """
    if type(request) is not CubicalBitmapRequest:
        raise OperationDomainValidationError(
            location=("request",),
            code="cubical_complex.bitmap_request_type",
            message="bitmap_to_complex requires a canonical bitmap request",
        )
    try:
        if (
            type(request.pixels) is not tuple
            or not request.pixels
            or len(request.pixels) > MAX_CUBICAL_BITMAP_SIDE
            or any(
                type(row) is not tuple or len(row) > MAX_CUBICAL_BITMAP_SIDE
                for row in request.pixels
            )
        ):
            raise ValueError("bitmap axes exceed the canonical request bounds")
        request = CubicalBitmapRequest.model_validate(request.model_dump(mode="python"))
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="cubical_complex.bitmap_request_invalid",
            message="bitmap request must satisfy its canonical rectangular contract",
        ) from exc
    row_count = len(request.pixels)
    column_count = len(request.pixels[0])
    selected_count = sum(pixel for row in request.pixels for pixel in row)
    if selected_count == 0:
        raise OperationDomainValidationError(
            location=("pixels",),
            code="cubical_complex.bitmap_empty_foreground",
            message=(
                "an all-background bitmap has no value in the current nonempty "
                "CubicalComplex representation"
            ),
        )
    if selected_count > MAX_CELLS:
        raise OperationResourceAdmissionError(
            location=("pixels",),
            code="cubical_complex.bitmap_top_cell_budget",
            message=f"bitmap foreground exceeds the {MAX_CELLS}-pixel top-cell bound",
        )

    pixels = request.pixels
    horizontal_edges = sum(
        (row > 0 and pixels[row - 1][column])
        or (row < row_count and pixels[row][column])
        for row in range(row_count + 1)
        for column in range(column_count)
    )
    vertical_edges = sum(
        (column > 0 and pixels[row][column - 1])
        or (column < column_count and pixels[row][column])
        for row in range(row_count)
        for column in range(column_count + 1)
    )
    vertices = sum(
        any(
            pixels[adjacent_row][adjacent_column]
            for adjacent_row in (row - 1, row)
            for adjacent_column in (column - 1, column)
            if 0 <= adjacent_row < row_count and 0 <= adjacent_column < column_count
        )
        for row in range(row_count + 1)
        for column in range(column_count + 1)
    )
    closed_cell_count = selected_count + horizontal_edges + vertical_edges + vertices
    if closed_cell_count > MAX_CELLS:
        raise OperationResourceAdmissionError(
            location=("pixels",),
            code="cubical_complex.bitmap_face_budget",
            message=(
                "the exact bitmap face closure exceeds the "
                f"{MAX_CELLS}-cell composable complex bound"
            ),
        )
    # A 2D cell record with bounded 8-bit coordinates occupies fewer than 96
    # JSON bytes.  Bound complete closure output before constructing any cells.
    output_bytes_bound = 128 + closed_cell_count * 96 + selected_count * 256
    if output_bytes_bound > MAX_CUBICAL_BITMAP_RESULT_SIZE:
        raise OperationResourceAdmissionError(
            location=("pixels",),
            code="cubical_complex.bitmap_result_size",
            message=(
                "the bitmap cubical complex exceeds the "
                f"{MAX_CUBICAL_BITMAP_RESULT_SIZE}-byte result bound"
            ),
        )

    pixel_to_cell = tuple(
        CubicalBitmapPixelCell(
            row=row,
            column=column,
            cell=CubicalCell(intervals=((column, column + 1), (row, row + 1))),
        )
        for row, row_values in enumerate(request.pixels)
        for column, foreground in enumerate(row_values)
        if foreground
    )
    complex_, _ = _canonical_complex(tuple(entry.cell for entry in pixel_to_cell))
    return CubicalBitmapResult(
        complex=complex_,
        row_count=row_count,
        column_count=column_count,
        pixel_to_cell=pixel_to_cell,
    )


def boundary(cells: tuple[CubicalCell, ...]) -> CubicalBoundaryResult:
    complex_, source = _canonical_complex(cells)
    maximal = tuple(
        cell
        for cell in source
        if not any(
            candidate.dimension > cell.dimension and _strictly_contains(candidate, cell)
            for candidate in source
        )
    )
    # Use every inclusion-maximal source cell.  A non-pure complex can have
    # maximal cells in several dimensions; incidence cancellation remains exact.
    coefficients: dict[CubicalCell, int] = {}
    terms = []
    for cell in maximal:
        for face, coefficient in _boundary_terms(cell):
            coefficients[face] = coefficients.get(face, 0) + coefficient
            terms.append(
                CubicalBoundaryTerm(source=cell, face=face, coefficient=coefficient)
            )
    boundary_cells = tuple(
        sorted(
            (face for face, value in coefficients.items() if value != 0),
            key=lambda item: item.intervals,
        )
    )
    return CubicalBoundaryResult(
        complex=complex_,
        maximal_cells=maximal,
        terms=tuple(terms),
        boundary_cells=boundary_cells,
    )


def _rank(matrix: list[list[int]], p: int) -> int:
    if not matrix or not matrix[0]:
        return 0
    a = [[x % p for x in row] for row in matrix]
    r = 0
    for c in range(len(a[0])):
        pivot = next((i for i in range(r, len(a)) if a[i][c]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        inv = pow(a[r][c], -1, p)
        a[r] = [(v * inv) % p for v in a[r]]
        for i in range(len(a)):
            if i != r and a[i][c]:
                q = a[i][c]
                a[i] = [(x - q * y) % p for x, y in zip(a[i], a[r], strict=False)]
        r += 1
    return r


def relative_homology(
    request: RelativeCubicalHomologyRequest,
) -> RelativeCubicalHomologyResult:
    if request.prime < 2 or any(
        request.prime % divisor == 0
        for divisor in range(2, int(request.prime**0.5) + 1)
    ):
        raise OperationDomainValidationError(
            location=("prime",),
            code="cubical_complex.relative_prime_not_prime",
            message="relative homology coefficients require a prime field modulus",
        )
    ambient, _ = _canonical_complex(request.cells)
    sub, _ = _canonical_complex(request.subcomplex_cells)
    if ambient.ambient_dimension != sub.ambient_dimension or not set(
        sub.cells
    ).issubset(set(ambient.cells)):
        raise OperationDomainValidationError(
            location=("subcomplex_cells",),
            code="cubical_complex.relative_not_subcomplex",
            message="the subcomplex must use the ambient axis and be contained in the face-closed complex",
        )
    groups = _cells_by_dimension(ambient)
    raw_sub_groups = _cells_by_dimension(sub)
    sub_groups = raw_sub_groups + tuple(
        () for _ in range(len(groups) - len(raw_sub_groups))
    )
    if any(len(group) > MAX_CUBICAL_CHAIN_GROUP for group in groups) or any(
        len(group) > MAX_CUBICAL_CHAIN_GROUP for group in raw_sub_groups
    ):
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.relative_chain_group_budget",
            message=(
                "a relative cubical chain group exceeds the "
                f"{MAX_CUBICAL_CHAIN_GROUP}-cell per-degree bound"
            ),
        )
    quotient_sizes = tuple(
        len(group) - len(sub_groups[d]) for d, group in enumerate(groups)
    )
    aggregate_cells = sum(
        quotient_sizes[d - 1] * quotient_sizes[d] for d in range(1, len(quotient_sizes))
    )
    if aggregate_cells > MAX_OPERATION_MATRIX_CELLS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.relative_matrix_cell_budget",
            message=(
                "the relative cubical boundary matrices exceed the "
                f"{MAX_OPERATION_MATRIX_CELLS}-cell aggregate bound"
            ),
        )
    betti = []
    ranks = list(quotient_sizes)
    # quotient differential rows/columns omit subcomplex basis cells
    differentials = []
    for d in range(1, len(groups)):
        rows = [c for c in groups[d - 1] if c not in sub_groups[d - 1]]
        cols = [c for c in groups[d] if c not in sub_groups[d]]
        row_for = {c: i for i, c in enumerate(rows)}
        m = [[0] * len(cols) for _ in rows]
        for j, c in enumerate(cols):
            for f, v in _boundary_terms(c):
                if f in row_for:
                    m[row_for[f]][j] = (m[row_for[f]][j] + v) % request.prime
        differentials.append(m)
    for d, size in enumerate(ranks):
        outgoing = (
            _rank(differentials[d], request.prime) if d < len(differentials) else 0
        )
        incoming = _rank(differentials[d - 1], request.prime) if d > 0 else 0
        betti.append(size - outgoing - incoming)
    return RelativeCubicalHomologyResult(
        complex=ambient, subcomplex=sub, prime=request.prime, betti_numbers=tuple(betti)
    )


def triangulate(request: CubicalTriangulationRequest) -> CubicalTriangulationResult:
    source, maximal_cells = _admit_triangulation_input(request)
    _admit_triangulation_expansion(maximal_cells)
    return _build_triangulation_result(source, maximal_cells)


def _admit_triangulation_input(
    request: CubicalTriangulationRequest,
) -> tuple[tuple[CubicalCell, ...], tuple[CubicalCell, ...]]:
    if len(request.cells) > MAX_TRIANGULATION_SOURCE_CELLS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.triangulation_source_budget",
            message=(
                "triangulation input exceeds the "
                f"{MAX_TRIANGULATION_SOURCE_CELLS}-cell work bound"
            ),
        )
    ambient_dimension = len(request.cells[0].intervals)
    if any(len(cell.intervals) != ambient_dimension for cell in request.cells):
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.invalid_ambient_axis",
            message="all cells must use one ambient coordinate axis",
        )
    if any(
        abs(coordinate).bit_length() > 256 or len(str(abs(coordinate))) > 64
        for cell in request.cells
        for interval in cell.intervals
        for coordinate in interval
    ):
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.triangulation_coordinate_budget",
            message="triangulation coordinates exceed the 64-digit bound",
        )
    output_bytes_bound = (
        512
        + MAX_TOPOLOGY_FACES * (ambient_dimension * (2 * 64 + 12) + 32)
        + MAX_TOPOLOGY_FACES * 96
        + MAX_TOPOLOGY_FACETS * (ambient_dimension * (2 * 64 + 12) + 64)
        + MAX_TOPOLOGY_VERTICES * (ambient_dimension * (64 + 8) + 32)
    )
    if output_bytes_bound > MAX_CUBICAL_TRIANGULATION_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.triangulation_result_size",
            message=(
                "the conservative cubical and simplicial result estimate exceeds "
                f"{MAX_CUBICAL_TRIANGULATION_RESULT_BYTES} bytes"
            ),
        )
    if any(cell.dimension > 7 for cell in request.cells):
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.triangulation_dimension_budget",
            message="the simplicial target supports dimensions at most 7",
        )

    source_set = {cell.intervals for cell in request.cells}
    comparison_work = len(source_set) * len(source_set) * ambient_dimension
    if comparison_work > 2_621_440:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.triangulation_maximal_cell_work",
            message="maximal cubical-cell comparison work exceeds its admitted bound",
        )
    source = tuple(
        sorted(
            (CubicalCell(intervals=key) for key in source_set),
            key=lambda cell: cell.intervals,
        )
    )
    maximal_cells = tuple(
        cell
        for cell in source
        if not any(
            candidate.dimension > cell.dimension
            and all(
                outer_start <= inner_start and inner_end <= outer_end
                for (outer_start, outer_end), (inner_start, inner_end) in zip(
                    candidate.intervals, cell.intervals, strict=True
                )
            )
            for candidate in source
        )
    )

    return source, maximal_cells


def _admit_triangulation_expansion(maximal_cells: tuple[CubicalCell, ...]) -> None:
    simplex_count = sum(
        1 if cell.dimension == 0 else _factorial(cell.dimension)
        for cell in maximal_cells
    )
    if simplex_count > MAX_TOPOLOGY_FACETS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.triangulation_output_budget",
            message=(
                "the canonical simplicial target would exceed its "
                f"{MAX_TOPOLOGY_FACETS}-facet bound"
            ),
        )
    point_candidate_count = sum(1 << cell.dimension for cell in maximal_cells)
    if point_candidate_count > MAX_TRIANGULATION_POINTS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.triangulation_point_work",
            message="triangulation point expansion exceeds its admitted work bound",
        )
    face_candidate_count = sum(
        (1 if cell.dimension == 0 else _factorial(cell.dimension))
        * ((1 << (cell.dimension + 1)) - 1)
        for cell in maximal_cells
    )
    if face_candidate_count > MAX_TRIANGULATION_FACE_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.triangulation_face_work",
            message="triangulation face expansion exceeds its admitted work bound",
        )


def _build_triangulation_result(
    source: tuple[CubicalCell, ...],
    maximal_cells: tuple[CubicalCell, ...],
) -> CubicalTriangulationResult:
    source_complex, _ = _canonical_complex(source, face_output_limit=MAX_TOPOLOGY_FACES)
    point_axis = tuple(
        sorted({point for cell in maximal_cells for point in _cube_choices(cell)})
    )
    if len(point_axis) > MAX_TOPOLOGY_VERTICES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.triangulation_point_budget",
            message=(
                "the canonical simplicial target would exceed its "
                f"{MAX_TOPOLOGY_VERTICES}-vertex bound"
            ),
        )
    point_label = {point: f"v{index:04d}" for index, point in enumerate(point_axis)}
    facets: set[Simplex] = set()
    cell_maps = []
    for cell in maximal_cells:
        active = [i for i, (a, b) in enumerate(cell.intervals) if a < b]
        base = tuple(a for a, _ in cell.intervals)
        simplices = []
        for permutation in permutations(active):
            points = [base]
            current = list(base)
            for axis in permutation:
                current = current.copy()
                current[axis] = cell.intervals[axis][1]
                points.append(tuple(current))
            simplex = tuple(sorted(point_label[point] for point in points))
            facets.add(simplex)
            simplices.append(simplex)
        cell_maps.append(
            CubicalTriangulationCellMap(
                source_cell=cell,
                simplices=tuple(sorted(simplices)),
            )
        )

    closure: list[set[Simplex]] = []
    for facet in facets:
        while len(closure) < len(facet):
            closure.append(set())
        for size in range(1, len(facet) + 1):
            closure[size - 1].update(combinations(facet, size))
    while closure and not closure[-1]:
        closure.pop()
    closure_size = sum(map(len, closure))
    if closure_size > MAX_TOPOLOGY_FACES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.triangulation_face_budget",
            message=(
                "the canonical simplicial target would exceed its "
                f"{MAX_TOPOLOGY_FACES}-face bound"
            ),
        )
    simplicial_complex = canonical_simplicial_complex(
        tuple(point_label[point] for point in point_axis),
        tuple(sorted(facets)),
        closure=tuple(tuple(sorted(faces)) for faces in closure),
    )
    return CubicalTriangulationResult(
        source_complex=source_complex,
        simplicial_complex=simplicial_complex,
        vertex_map=tuple(
            CubicalVertexMap(vertex=point_label[point], coordinate=point)
            for point in point_axis
        ),
        cell_maps=tuple(cell_maps),
    )


def _factorial(value: int) -> int:
    result = 1
    for factor in range(2, value + 1):
        result *= factor
    return result


def _cube_choices(cell: CubicalCell) -> list[tuple[int, ...]]:
    choices = [(a,) if a == b else (a, b) for a, b in cell.intervals]
    out: list[tuple[int, ...]] = [()]
    for axis in choices:
        out = [(*prefix, v) for prefix in out for v in axis]
    return out


__all__ = [
    "CubicalBitmapPixelCell",
    "CubicalBitmapRequest",
    "CubicalBitmapResult",
    "CubicalBoundaryRequest",
    "CubicalBoundaryResult",
    "CubicalBoundaryTerm",
    "CubicalTriangulationCellMap",
    "CubicalTriangulationRequest",
    "CubicalTriangulationResult",
    "CubicalVertexMap",
    "RelativeCubicalHomologyRequest",
    "RelativeCubicalHomologyResult",
    "bitmap_to_complex",
    "boundary",
    "relative_homology",
    "triangulate",
]
