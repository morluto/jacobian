"""Exact cubical complex operations."""

from __future__ import annotations

from jacobian.math.cubical_complexes._models import (
    CubicalComplexRequest,
    FaceClosureRequest,
    FaceClosureResult,
    FVectorResult,
)


def compute_f_vector(request: CubicalComplexRequest) -> FVectorResult:
    """Compute the f-vector and Euler characteristic of a cubical complex."""
    cells = request.cells

    # Find all cells by dimension (including faces)
    all_cells_by_dim: dict[int, set[tuple[tuple[int, int], ...]]] = {}

    for cell in cells:
        dim = cell.dimension
        key = cell.intervals
        if dim not in all_cells_by_dim:
            all_cells_by_dim[dim] = set()
        all_cells_by_dim[dim].add(key)

    max_dim = max((cell.dimension for cell in cells), default=0)

    f_vector = []
    for d in range(max_dim + 1):
        count = len(all_cells_by_dim.get(d, set()))
        f_vector.append(count)

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

    for cell in cells:
        all_cells.add(cell.intervals)
        # Add all proper faces
        dim = cell.dimension
        for mask in range(1, 1 << dim):
            face = []
            for i in range(dim):
                if mask & (1 << i):
                    _a, _b = cell.intervals[i]
                    face.append((_a, _a))
                else:
                    face.append(cell.intervals[i])
            face_tuple = tuple(face)
            all_cells.add(face_tuple)

    # Count by dimension
    by_dim: dict[int, int] = {}
    for c in all_cells:
        dim = sum(1 for a, b in c if a < b)
        by_dim[dim] = by_dim.get(dim, 0) + 1

    max_dim = max(by_dim.keys()) if by_dim else 0
    cells_by_dimension = tuple(by_dim.get(d, 0) for d in range(max_dim + 1))

    return FaceClosureResult(
        original_cells=len(cells),
        total_cells=len(all_cells),
        cells_by_dimension=cells_by_dimension,
    )
