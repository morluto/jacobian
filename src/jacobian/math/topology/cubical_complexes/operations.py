"""Exact cubical complex operations."""

from __future__ import annotations

from jacobian.math.topology.cubical_complexes._models import (
    CubicalCell,
    CubicalComplex,
    FaceClosureResult,
    FVector,
    FVectorResult,
)


def _face_cells(cells: tuple[CubicalCell, ...]) -> tuple[CubicalCell, ...]:
    """Materialize the canonical face closure once during operation admission."""
    all_cells: set[tuple[tuple[int, int], ...]] = set()

    def add_faces(intervals: tuple[tuple[int, int], ...]) -> None:
        if intervals in all_cells:
            return
        all_cells.add(intervals)
        for i, (a, b) in enumerate(intervals):
            if b > a:
                for endpoint in (a, b):
                    face = list(intervals)
                    face[i] = (endpoint, endpoint)
                    add_faces(tuple(face))

    for cell in cells:
        add_faces(cell.intervals)
    return tuple(CubicalCell(intervals=intervals) for intervals in sorted(all_cells))


def _canonical_complex(cells: tuple[CubicalCell, ...]) -> tuple[
    CubicalComplex, tuple[CubicalCell, ...]
]:
    if not cells:
        raise ValueError("at least one cell is required")
    ambient_dimension = len(cells[0].intervals)
    if any(len(cell.intervals) != ambient_dimension for cell in cells):
        raise ValueError("all cells must use one ambient coordinate axis")
    source_cells = tuple(sorted(set(cells), key=lambda cell: cell.intervals))
    closed_cells = _face_cells(source_cells)
    return (
        CubicalComplex(
            ambient_dimension=ambient_dimension,
            cells=closed_cells,
        ),
        source_cells,
    )


def _counts(complex_: CubicalComplex) -> FVector:
    by_dimension = [0] * (complex_.ambient_dimension + 1)
    for cell in complex_.cells:
        by_dimension[cell.dimension] += 1
    return FVector(
        dimension_axis=tuple(range(complex_.ambient_dimension + 1)),
        counts=tuple(by_dimension),
    )


def f_vector(cells: tuple[CubicalCell, ...]) -> FVectorResult:
    """Compute the f-vector and Euler characteristic of a cubical complex.

    The f-vector counts all faces (including the supplied maximal cells) by
    dimension.  A single square [0,1]x[0,1] has 4 vertices, 4 edges, 1 square,
    so its f-vector is (4, 4, 1).
    """
    complex_, source_cells = _canonical_complex(cells)
    vector = _counts(complex_)
    euler = sum((-1) ** d * count for d, count in enumerate(vector.counts))
    return FVectorResult(
        complex=complex_,
        source_cells=source_cells,
        f_vector=vector,
        euler_characteristic=euler,
    )


def face_closure(cells: tuple[CubicalCell, ...]) -> FaceClosureResult:
    """Compute the full face closure of a set of cells."""
    complex_, source_cells = _canonical_complex(cells)
    cells_by_dimension = _counts(complex_)

    return FaceClosureResult(
        complex=complex_,
        source_cells=source_cells,
        original_cells=len(source_cells),
        total_cells=len(complex_.cells),
        cells_by_dimension=cells_by_dimension,
    )


def verify_f_vector(claim: FVectorResult) -> bool:
    """Verify f-vector and Euler claims against retained source cells."""
    try:
        return f_vector(claim.source_cells) == claim
    except (TypeError, ValueError, RuntimeError):
        return False


def verify_face_closure(claim: FaceClosureResult) -> bool:
    """Verify the canonical face closure and count summary."""
    try:
        return face_closure(claim.source_cells) == claim
    except (TypeError, ValueError, RuntimeError):
        return False


__all__ = ["f_vector", "face_closure", "verify_f_vector", "verify_face_closure"]
