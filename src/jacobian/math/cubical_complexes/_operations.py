"""Exact cubical complex operations."""

from __future__ import annotations

from jacobian.math.cubical_complexes._models import (
    CubicalComplexRequest,
    FaceClosureRequest,
    FaceClosureResult,
    FVectorResult,
)


def compute_f_vector(request: CubicalComplexRequest) -> FVectorResult:
    """Compute the f-vector and Euler characteristic of a cubical complex.

    The f-vector counts all faces (including the supplied maximal cells) by
    dimension.  A single square [0,1]x[0,1] has 4 vertices, 4 edges, 1 square,
    so its f-vector is (4, 4, 1).
    """
    cells = request.cells

    # Generate all faces including lower and upper degenerations
    all_cells: set[tuple[tuple[int, int], ...]] = set()

    def add_faces(intervals: tuple[tuple[int, int], ...]) -> None:
        if intervals in all_cells:
            return
        all_cells.add(intervals)
        for i, (a, b) in enumerate(intervals):
            if b > a:
                face_lower = list(intervals)
                face_lower[i] = (a, a)
                add_faces(tuple(face_lower))
                face_upper = list(intervals)
                face_upper[i] = (b, b)
                add_faces(tuple(face_upper))

    for cell in cells:
        add_faces(cell.intervals)

    # Count by dimension (number of non-degenerate intervals)
    by_dim: dict[int, int] = {}
    for c in all_cells:
        dim = sum(1 for a, b in c if b > a)
        by_dim[dim] = by_dim.get(dim, 0) + 1

    max_dim = max(by_dim.keys()) if by_dim else 0
    f_vector = [by_dim.get(d, 0) for d in range(max_dim + 1)]
    euler = sum((-1) ** d * f for d, f in enumerate(f_vector))

    return FVectorResult(
        dimension=max_dim,
        f_vector=tuple(f_vector),
        euler_characteristic=euler,
    )


def compute_face_closure(request: FaceClosureRequest) -> FaceClosureResult:
    """Compute the full face closure of a set of cells."""
    cells = request.cells

    all_cells: set[tuple[tuple[int, int], ...]] = set()

    def add_faces(intervals: tuple[tuple[int, int], ...]) -> None:
        if intervals in all_cells:
            return
        all_cells.add(intervals)
        for i, (a, b) in enumerate(intervals):
            if b > a:
                for new_b in (a, b):
                    if new_b == a:
                        face = list(intervals)
                        face[i] = (a, a)
                        add_faces(tuple(face))
                    else:
                        face = list(intervals)
                        face[i] = (b, b)
                        add_faces(tuple(face))

    for cell in cells:
        add_faces(cell.intervals)

    by_dim: dict[int, int] = {}
    for c in all_cells:
        dim = sum(1 for a, b in c if b > a)
        by_dim[dim] = by_dim.get(dim, 0) + 1

    max_dim = max(by_dim.keys()) if by_dim else 0
    cells_by_dimension = tuple(by_dim.get(d, 0) for d in range(max_dim + 1))

    return FaceClosureResult(
        original_cells=len(cells),
        total_cells=len(all_cells),
        cells_by_dimension=cells_by_dimension,
    )
