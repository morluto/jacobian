"""Domain-owned chain complex operations."""

from __future__ import annotations

from jacobian.math.chain_complexes._models import (
    ChainComplex,
    HomologyGroup,
    HomologyRequest,
    HomologyResult,
    MappingConeRequest,
    MappingConeResult,
    MatrixEntry,
)


def _build_matrix(
    entries: tuple[MatrixEntry, ...],
    rows: int,
    cols: int,
    prime: int,
) -> list[list[int]]:
    """Build a dense matrix from sparse entries over GF(prime)."""
    mat = [[0] * cols for _ in range(rows)]
    for entry in entries:
        mat[entry.row][entry.col] = int(entry.value) % prime
    return mat


def _pivot_row(
    aug: list[list[int]], col: int, start: int, rows: int, prime: int
) -> int | None:
    """First row at or below ``start`` with a nonzero entry in ``col``."""
    for r in range(start, rows):
        if aug[r][col] % prime != 0:
            return r
    return None


def _scale_and_clear(
    aug: list[list[int]], rank: int, col: int, cols: int, rows: int, prime: int
) -> None:
    """Scale the pivot row to a unit leading entry and clear its column."""
    inv_pivot = pow(aug[rank][col] % prime, prime - 2, prime)
    for c in range(cols):
        aug[rank][c] = (aug[rank][c] * inv_pivot) % prime
    for r, row in enumerate(aug):
        factor = row[col] % prime
        if r != rank and factor != 0:
            for c in range(cols):
                row[c] = (row[c] - factor * aug[rank][c]) % prime


def _gaussian_rank(matrix: list[list[int]], prime: int) -> int:
    """Compute rank of a matrix over GF(prime) via Gaussian elimination."""
    rows = len(matrix)
    if rows == 0:
        return 0
    cols = len(matrix[0]) if rows > 0 else 0
    if cols == 0:
        return 0
    aug = [row[:] for row in matrix]
    rank = 0
    for col in range(cols):
        pivot = _pivot_row(aug, col, rank, rows, prime)
        if pivot is None:
            continue
        aug[rank], aug[pivot] = aug[pivot], aug[rank]
        _scale_and_clear(aug, rank, col, cols, rows, prime)
        rank += 1
        if rank >= rows:
            break
    return rank


def _homology_groups(cx: ChainComplex) -> tuple[HomologyGroup, ...]:
    """Exact homology groups of one chain complex over its prime field.

    H_n = ker(d_n) / im(d_{n+1})
    Betti_n = dim(C_n) - rank(d_n) - rank(d_{n+1})
    where d_n: C_n -> C_{n-1} has matrix dims[n-1] x dims[n] (target x source)
    and differentials[i] is d_{min+i+1} with source dims[i+1] and target dims[i].

    Shared by the operation and the result validator so an authored result
    replays against the identical kernel.
    """
    prime = cx.prime
    n_min = cx.min_degree
    dims = cx.dimensions
    diffs = cx.differentials

    ranks = []
    for i, diff_entries in enumerate(diffs):
        # differential i is d_{min+i+1}: C_{min+i+1} -> C_{min+i}
        source_dim = dims[i + 1]
        target_dim = dims[i]
        mat = _build_matrix(diff_entries, target_dim, source_dim, prime)
        ranks.append(_gaussian_rank(mat, prime))

    groups = []
    for k in range(len(dims)):
        # outgoing d_k: C_k -> C_{k-1} contributes rank ranks[k-1] when k > 0
        out_rank = ranks[k - 1] if k > 0 and k - 1 < len(ranks) else 0
        in_rank = ranks[k] if k < len(ranks) else 0
        cycle_rank = dims[k] - out_rank
        boundary_rank = in_rank
        betti = cycle_rank - boundary_rank
        # The validator already ensures d^2=0, so betti should be >=0.
        # Keep max(0, ...) as safety but a valid complex never triggers it.
        groups.append(
            HomologyGroup(
                degree=n_min + k,
                betti=max(0, betti),
                dimension=dims[k],
                boundary_rank=boundary_rank,
                cycle_rank=cycle_rank,
            )
        )

    return tuple(groups)


def compute_homology(request: HomologyRequest) -> HomologyResult:
    """Compute the homology of a chain complex over GF(prime)."""
    cx = request.complex
    groups = _homology_groups(cx)
    return HomologyResult(
        complex=cx,
        groups=groups,
        prime=cx.prime,
        min_degree=cx.min_degree,
        max_degree=cx.max_degree,
    )


def _degree_dimension(
    dimensions: tuple[int, ...], min_degree: int, max_degree: int, deg: int
) -> int:
    """Dimension of the group at one degree, zero outside the complex."""
    if min_degree <= deg <= max_degree:
        return dimensions[deg - min_degree]
    return 0


def _boundary_matrix(
    complex_: ChainComplex,
    deg: int,
    prime: int,
) -> list[list[int]]:
    """d^complex_deg: C_deg -> C_{deg-1} as a dense GF(prime) matrix."""
    if deg <= complex_.min_degree or deg > complex_.max_degree:
        return []
    idx = deg - complex_.min_degree - 1
    if idx < 0 or idx >= len(complex_.differentials):
        return []
    rows = _degree_dimension(
        complex_.dimensions, complex_.min_degree, complex_.max_degree, deg - 1
    )
    cols = _degree_dimension(
        complex_.dimensions, complex_.min_degree, complex_.max_degree, deg
    )
    return _build_matrix(complex_.differentials[idx], rows, cols, prime)


def _chain_map_matrix(
    chain_map: tuple[tuple[MatrixEntry, ...], ...],
    source: ChainComplex,
    target: ChainComplex,
    deg: int,
    prime: int,
) -> list[list[int]]:
    """f_deg: C_deg -> D_deg as a dense GF(prime) matrix."""
    if (
        deg < source.min_degree
        or deg > source.max_degree
        or deg < target.min_degree
        or deg > target.max_degree
    ):
        return []
    # chain_map is aligned to the source degree range: f_maps[i] = f_{s_min + i}
    idx = deg - source.min_degree
    if idx < 0 or idx >= len(chain_map):
        return []
    rows = _degree_dimension(
        target.dimensions, target.min_degree, target.max_degree, deg
    )
    cols = _degree_dimension(
        source.dimensions, source.min_degree, source.max_degree, deg
    )
    # An empty chain-map entry represents the zero map.
    if not chain_map[idx]:
        return [[0] * cols for _ in range(rows)] if rows and cols else []
    return _build_matrix(chain_map[idx], rows, cols, prime)


def _dense_to_entries(mat: list[list[int]], prime: int) -> tuple[MatrixEntry, ...]:
    entries = []
    for r, row in enumerate(mat):
        for c_idx, value in enumerate(row):
            reduced = value % prime
            if reduced != 0:
                entries.append(MatrixEntry(row=r, col=c_idx, value=str(reduced)))
    return tuple(entries)


def _paste_block(
    mat: list[list[int]],
    block: list[list[int]],
    row_offset: int,
    col_offset: int,
    sign: int,
    prime: int,
) -> None:
    """Write one signed block into a dense assembly matrix."""
    for r in range(min(len(block), len(mat) - row_offset)):
        width = len(block[r]) if block else 0
        for c_idx in range(min(width, len(mat[0]) - col_offset)):
            mat[row_offset + r][col_offset + c_idx] = (sign * block[r][c_idx]) % prime


def _cone_differential(
    gap: int,
    cone_min: int,
    source: ChainComplex,
    target: ChainComplex,
    chain_map: tuple[tuple[MatrixEntry, ...], ...],
    prime: int,
) -> tuple[MatrixEntry, ...]:
    """Build the cone differential leaving degree cone_min + gap + 1.

    d_cone_n = [ -d^C_{n-1}   0        ]
                [  f_{n-1}     d^D_n    ]   on Cone(f)_n = C_{n-1} ⊕ D_n.
    """
    n = cone_min + gap + 1
    source_above = _degree_dimension(
        source.dimensions, source.min_degree, source.max_degree, n - 1
    )
    target_here = _degree_dimension(
        target.dimensions, target.min_degree, target.max_degree, n
    )
    source_two_below = _degree_dimension(
        source.dimensions, source.min_degree, source.max_degree, n - 2
    )
    target_below = _degree_dimension(
        target.dimensions, target.min_degree, target.max_degree, n - 1
    )
    rows = source_two_below + target_below
    cols = source_above + target_here
    if rows == 0 or cols == 0:
        return ()
    mat = [[0] * cols for _ in range(rows)]
    negated_source = _boundary_matrix(source, n - 1, prime)
    _paste_block(mat, negated_source, 0, 0, -1, prime)
    map_block = _chain_map_matrix(chain_map, source, target, n - 1, prime)
    _paste_block(mat, map_block, source_two_below, 0, 1, prime)
    negated_target = _boundary_matrix(target, n, prime)
    _paste_block(mat, negated_target, source_two_below, source_above, 1, prime)
    # Top-right remains zero.
    return _dense_to_entries(mat, prime)


def compute_mapping_cone(request: MappingConeRequest) -> MappingConeResult:
    """Compute the mapping cone of a chain map f: C -> D.

    The mapping cone has groups Cone(f)_n = C_{n-1} ⊕ D_n and the
    differential is

        d_cone_n = [ -d^C_{n-1}   0
                      f_{n-1}   d^D_n ]

    where d^C, d^D are the source/target differentials and f is the chain map.
    """
    source = request.source
    target = request.target
    if source.prime != target.prime:
        raise ValueError("source and target must have the same prime")
    prime = source.prime
    cone_min = min(source.min_degree + 1, target.min_degree)
    cone_max = max(source.max_degree + 1, target.max_degree)

    def cone_dimension(deg: int) -> int:
        below = _degree_dimension(
            source.dimensions, source.min_degree, source.max_degree, deg - 1
        )
        here = _degree_dimension(
            target.dimensions, target.min_degree, target.max_degree, deg
        )
        return below + here

    cone_dims = [cone_dimension(cone_min + k) for k in range(cone_max - cone_min + 1)]
    # One differential per gap between consecutive cone degrees.
    cone_diffs = [
        _cone_differential(k, cone_min, source, target, request.chain_map, prime)
        for k in range(len(cone_dims) - 1)
    ]

    cone_complex = ChainComplex(
        prime=prime,
        min_degree=cone_min,
        max_degree=cone_max,
        dimensions=tuple(cone_dims),
        differentials=tuple(cone_diffs),
    )

    return MappingConeResult(cone=cone_complex)


__all__ = [
    "compute_homology",
    "compute_mapping_cone",
]
