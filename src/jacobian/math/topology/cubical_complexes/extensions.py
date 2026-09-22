"""Canonical cubical boundary, relative, and triangulation transforms."""

from __future__ import annotations

from itertools import permutations
from math import factorial

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.chain_complexes.values import (
    CoefficientRing,
    require_prime_field_admission,
)
from jacobian.math.topology.cubical_complexes._models import (
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


class CubicalTriangulationResult(StrictModel):
    complex: CubicalComplex
    simplex_vertices: tuple[tuple[int, ...], ...]
    simplices_by_cell: tuple[tuple[tuple[int, ...], ...], ...]


def boundary(cells: tuple[CubicalCell, ...]) -> CubicalBoundaryResult:
    complex_, source = _canonical_complex(cells)
    maximal = tuple(
        cell
        for cell in source
        if cell.dimension == max(item.dimension for item in source)
    )
    # use the canonical top-dimensional source cells; incidence cancellation is exact
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
    ambient, _ = _canonical_complex(request.cells)
    sub, _ = _canonical_complex(request.subcomplex_cells)
    if ambient.ambient_dimension != sub.ambient_dimension or not set(
        sub.cells
    ).issubset(set(ambient.cells)):
        raise OperationDomainValidationError(
            location=("subcomplex_cells",),
            code="cubical_complex.relative_not_subcomplex",
            message=(
                "the subcomplex must use the ambient axis and be contained in "
                "the face-closed complex"
            ),
        )
    try:
        require_prime_field_admission(CoefficientRing.PRIME_FIELD, request.prime)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("prime",),
            code="cubical_complex.relative_prime_invalid",
            message=str(exc),
        ) from exc
    groups = _cells_by_dimension(ambient)
    raw_sub_groups = _cells_by_dimension(sub)
    sub_groups = raw_sub_groups + tuple(
        () for _ in range(len(groups) - len(raw_sub_groups))
    )
    betti = []
    ranks = []
    for d, group in enumerate(groups):
        ranks.append(len(group) - len(sub_groups[d]))
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


MAX_TRIANGULATION_POINTS = 65_536
MAX_TRIANGULATION_SIMPLICES = 100_000


def triangulate(request: CubicalTriangulationRequest) -> CubicalTriangulationResult:
    complex_, source = _canonical_complex(request.cells)
    simplex_count = sum(
        1 if cell.dimension == 0 else factorial(cell.dimension) for cell in source
    )
    point_upper_bound = sum(2 ** cell.dimension for cell in source)
    if simplex_count > MAX_TRIANGULATION_SIMPLICES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.triangulation_simplex_budget",
            message=(
                f"triangulation would materialize {simplex_count} simplices, "
                f"above the {MAX_TRIANGULATION_SIMPLICES}-simplex bound"
            ),
        )
    if point_upper_bound > MAX_TRIANGULATION_POINTS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.triangulation_point_budget",
            message="the triangulation point axis exceeds its admitted bound",
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
        complex=complex_, simplex_vertices=point_axis, simplices_by_cell=tuple(by_cell)
    )


def _cube_choices(cell: CubicalCell) -> list[tuple[int, ...]]:
    choices = [(a,) if a == b else (a, b) for a, b in cell.intervals]
    out: list[tuple[int, ...]] = [()]
    for axis in choices:
        out = [(*prefix, v) for prefix in out for v in axis]
    return out


__all__ = [
    "CubicalBoundaryRequest",
    "CubicalBoundaryResult",
    "CubicalBoundaryTerm",
    "CubicalTriangulationRequest",
    "CubicalTriangulationResult",
    "RelativeCubicalHomologyRequest",
    "RelativeCubicalHomologyResult",
    "boundary",
    "relative_homology",
    "triangulate",
]
