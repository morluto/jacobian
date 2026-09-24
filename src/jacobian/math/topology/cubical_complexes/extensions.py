"""Canonical cubical boundary, relative, and triangulation transforms."""

from __future__ import annotations

from itertools import permutations
from typing import Annotated, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.chain_complexes.values import MAX_OPERATION_MATRIX_CELLS
from jacobian.math.topology.cubical_complexes._models import (
    MAX_CELLS,
    MAX_CUBICAL_BITMAP_RESULT_BYTES,
    MAX_CUBICAL_CHAIN_GROUP,
    MAX_TRIANGULATION_CELL_SIMPLICES,
    MAX_TRIANGULATION_POINTS,
    MAX_TRIANGULATION_SIMPLICES,
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


TriangulationCell = Annotated[
    tuple[tuple[int, ...], ...], Field(max_length=MAX_TRIANGULATION_CELL_SIMPLICES)
]


class CubicalTriangulationResult(StrictModel):
    complex: CubicalComplex
    # The triangulation groups are indexed by this retained source-cell axis,
    # not by ``complex.cells`` (which is the face closure).
    source_cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)
    simplex_vertices: tuple[tuple[int, ...], ...] = Field(
        max_length=MAX_TRIANGULATION_POINTS
    )
    simplices_by_cell: tuple[TriangulationCell, ...] = Field(max_length=MAX_CELLS)

    @model_validator(mode="after")
    def require_source_axis(self) -> Self:
        if (
            tuple(sorted(self.source_cells, key=lambda cell: cell.intervals))
            != self.source_cells
        ):
            raise ValueError("source_cells must be canonically sorted")
        if len(set(self.source_cells)) != len(self.source_cells):
            raise ValueError("source_cells must be distinct")
        if len(self.simplices_by_cell) != len(self.source_cells):
            raise ValueError("simplices_by_cell must match the source_cells axis")
        if any(cell not in self.complex.cells for cell in self.source_cells):
            raise ValueError("source_cells must belong to the face-closed complex")
        return self


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
    row_count = len(request.pixels)
    column_count = len(request.pixels[0])
    selected_count = sum(pixel for row in request.pixels for pixel in row)
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
    if output_bytes_bound > MAX_CUBICAL_BITMAP_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("pixels",),
            code="cubical_complex.bitmap_result_size",
            message=(
                "the bitmap cubical complex exceeds the "
                f"{MAX_CUBICAL_BITMAP_RESULT_BYTES}-byte result bound"
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
    if pixel_to_cell:
        complex_, _ = _canonical_complex(tuple(entry.cell for entry in pixel_to_cell))
    else:
        complex_ = CubicalComplex(ambient_dimension=2, cells=())
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
    complex_, source = _canonical_complex(request.cells)
    # Admit the unavoidable staircase expansion before allocating its result.
    point_axis_set: set[tuple[int, ...]] = set()
    simplex_count = 0
    for cell in source:
        active_count = cell.dimension
        simplex_count += 1 if active_count == 0 else _factorial(active_count)
        if simplex_count > MAX_TRIANGULATION_SIMPLICES:
            raise OperationResourceAdmissionError(
                location=("cells",),
                code="cubical_complex.triangulation_output_budget",
                message="staircase simplex output exceeds the admitted bound",
            )
        point_axis_set.update(_cube_choices(cell))
        if len(point_axis_set) > MAX_TRIANGULATION_POINTS:
            raise OperationResourceAdmissionError(
                location=("cells",),
                code="cubical_complex.triangulation_point_budget",
                message="triangulation lattice-point axis exceeds the admitted bound",
            )
    # Vertices are encoded as coordinate indices into the finite point axis.
    point_axis = tuple(
        sorted(
            {
                tuple(
                    a if a == b else x
                    for (a, b), x in zip(cell.intervals, choices, strict=False)
                )
                for cell in source
                for choices in _cube_choices(cell)
            }
        )
    )
    point_index = {p: i for i, p in enumerate(point_axis)}
    by_cell: list[tuple[tuple[int, ...], ...]] = []
    for cell in source:
        active = [i for i, (a, b) in enumerate(cell.intervals) if a < b]
        if not active:
            by_cell.append(((point_index[tuple(a for a, b in cell.intervals)],),))
            continue
        simplices = []
        base = tuple(a for a, b in cell.intervals)
        for perm in permutations(active):
            points = [base]
            current = list(base)
            for axis in perm:
                current = current.copy()
                current[axis] = cell.intervals[axis][1]
                points.append(tuple(current))
            simplices.append(tuple(point_index[p] for p in points))
        by_cell.append(tuple(simplices))
    return CubicalTriangulationResult(
        complex=complex_,
        source_cells=source,
        simplex_vertices=point_axis,
        simplices_by_cell=tuple(by_cell),
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
    "CubicalTriangulationRequest",
    "CubicalTriangulationResult",
    "RelativeCubicalHomologyRequest",
    "RelativeCubicalHomologyResult",
    "bitmap_to_complex",
    "boundary",
    "relative_homology",
    "triangulate",
]
