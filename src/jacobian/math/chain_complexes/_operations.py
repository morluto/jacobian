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


def _pivot_row(aug: list[list[int]], col: int, start: int, rows: int, prime: int) -> int | None:
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
    n_max = cx.max_degree
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
        # outgoing d_k: C_k -> C_{k-1} is diffs[k-1] if k>0
        outgoing_rank = ranks[k - 1] if 0 < k < len(ranks) + 1 and k - 1 < len(ranks) else 0
        # incoming d_{k+1}: C_{k+1} -> C_k is diffs[k] if k < len(ranks)
        incoming_rank = ranks[k] if 0 <= k < len(ranks) else 0
        # Actually for k index, outgoing = ranks[k-1] (d_{min+k}), incoming = ranks[k]
        # But need to map: for k=0, outgoing=0, incoming=ranks[0] if exists
        # For general, outgoing = ranks[k-1] when k>0
        # Let's recompute cleanly:
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
    prime = source.prime
    if source.prime != target.prime:
        raise ValueError("source and target must have the same prime")
    s_dims = source.dimensions
    t_dims = target.dimensions
    s_min = source.min_degree
    t_min = target.min_degree
    s_max = source.max_degree
    t_max = target.max_degree
    s_diffs = source.differentials
    t_diffs = target.differentials
    f_maps = request.chain_map

    def dim_C(deg: int) -> int:
        if s_min <= deg <= s_max:
            return s_dims[deg - s_min]
        return 0

    def dim_D(deg: int) -> int:
        if t_min <= deg <= t_max:
            return t_dims[deg - t_min]
        return 0

    def mat_C(deg: int) -> list[list[int]]:
        # d^C_deg: C_deg -> C_{deg-1}, matrix dim_C(deg-1) x dim_C(deg)
        if deg <= s_min or deg > s_max:
            return []
        idx = deg - s_min - 1
        if idx < 0 or idx >= len(s_diffs):
            return []
        entries = s_diffs[idx]
        rows = dim_C(deg - 1)
        cols = dim_C(deg)
        return _build_matrix(entries, rows, cols, prime)

    def mat_D(deg: int) -> list[list[int]]:
        if deg <= t_min or deg > t_max:
            return []
        idx = deg - t_min - 1
        if idx < 0 or idx >= len(t_diffs):
            return []
        entries = t_diffs[idx]
        rows = dim_D(deg - 1)
        cols = dim_D(deg)
        return _build_matrix(entries, rows, cols, prime)

    def mat_F(deg: int) -> list[list[int]]:
        # f_deg: C_deg -> D_deg
        if deg < s_min or deg > s_max or deg < t_min or deg > t_max:
            return []
        # chain_map is aligned to source degree range: f_maps[i] = f_{s_min + i}
        idx = deg - s_min
        if idx < 0 or idx >= len(f_maps):
            return []
        entries = f_maps[idx]
        rows = dim_D(deg)
        cols = dim_C(deg)
        # If chain_map entry is empty tuple, it represents zero map
        if not entries:
            return [[0] * cols for _ in range(rows)] if rows and cols else []
        return _build_matrix(entries, rows, cols, prime)

    cone_min = min(s_min + 1, t_min)
    cone_max = max(s_max + 1, t_max)
    n_cone = cone_max - cone_min + 1
    cone_dims = []
    for k in range(n_cone):
        deg = cone_min + k
        cone_dims.append(dim_C(deg - 1) + dim_D(deg))

    # Build cone differentials: one per gap between cone degrees
    cone_diffs: list[tuple[MatrixEntry, ...]] = []
    for k in range(n_cone - 1):
        # differential from Cone_{cone_min + k +1} -> Cone_{cone_min + k}
        n = cone_min + k + 1
        c_n_minus_1 = dim_C(n - 1)
        d_n = dim_D(n)
        c_n_minus_2 = dim_C(n - 2)
        d_n_minus_1 = dim_D(n - 1)
        rows = c_n_minus_2 + d_n_minus_1
        cols = c_n_minus_1 + d_n
        if rows == 0 or cols == 0:
            cone_diffs.append(())
            continue
        # Build block matrix
        mat = [[0] * cols for _ in range(rows)]
        # Top-left: -d^C_{n-1}
        dC = mat_C(n - 1)
        if dC:
            for r in range(min(len(dC), c_n_minus_2)):
                for c_idx in range(min(len(dC[0]) if dC else 0, c_n_minus_1)):
                    mat[r][c_idx] = (-dC[r][c_idx]) % prime
        # Bottom-left: f_{n-1}
        f = mat_F(n - 1)
        if f:
            for r in range(min(len(f), d_n_minus_1)):
                for c_idx in range(min(len(f[0]) if f else 0, c_n_minus_1)):
                    mat[c_n_minus_2 + r][c_idx] = f[r][c_idx] % prime
        # Bottom-right: d^D_n
        dD = mat_D(n)
        if dD:
            for r in range(min(len(dD), d_n_minus_1)):
                for c_idx in range(min(len(dD[0]) if dD else 0, d_n)):
                    mat[c_n_minus_2 + r][c_n_minus_1 + c_idx] = dD[r][c_idx] % prime
        # Top-right remains zero
        # Convert dense to sparse entries
        entries = []
        for r in range(rows):
            for c_idx in range(cols):
                val = mat[r][c_idx] % prime
                if val != 0:
                    entries.append(MatrixEntry(row=r, col=c_idx, value=str(val)))
        cone_diffs.append(tuple(entries))

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
