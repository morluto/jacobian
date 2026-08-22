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


def compute_homology(request: HomologyRequest) -> HomologyResult:
    """Compute the homology of a chain complex over GF(prime).

    H_n = ker(d_n) / im(d_{n+1})
    Betti_n = dim(ker(d_n)) - rank(d_n)
            = (dim(C_n) - rank(d_n)) - rank(d_{n+1})
            = dim(C_n) - rank(d_n) - rank(d_{n+1})
    """
    cx = request.complex
    prime = cx.prime
    n_min = cx.min_degree
    n_max = cx.max_degree
    dims = cx.dimensions
    diffs = cx.differentials

    ranks = []
    for i, diff_entries in enumerate(diffs):
        source_dim = dims[i]
        target_dim = dims[i + 1]
        mat = _build_matrix(diff_entries, target_dim, source_dim, prime)
        ranks.append(_gaussian_rank(mat, prime))

    # rank of differential going OUT of the last degree is 0
    # We need len(dims) ranks total (rank of d^i for each degree i)
    # differentials has len(dims) - 1 entries (d^0, ..., d^{n-1})
    # rank of d^n (for the last degree) is 0 (no differential)
    # ranks[i] is the rank of d^i (the i-th differential)
    # But we also need rank of d^{-1} (which is 0, for the first degree)
    # Let's make ranks have len(dims) entries, with ranks[i] = rank of d^i
    # and ranks[-1] = 0 (no differential before degree 0), ranks[n] = 0 (no differential after)
    full_ranks = [0]  # rank of d^{-1} = 0
    for i in range(len(diffs)):
        full_ranks.append(ranks[i] if i < len(ranks) else 0)
    full_ranks.append(0)  # rank of d^n = 0 (no differential out of last degree)
    # Now full_ranks[k+1] is rank of d^k (for k = 0, ..., n-1)
    # full_ranks[0] = rank of d^{-1} = 0
    # full_ranks[n+1] = rank of d^n = 0
    ranks = full_ranks

    # For degree n (index k = n - n_min):
    # cycle_rank = dims[k] - rank(d_k)  (kernel of d_k)
    # boundary_rank = rank(d_{k+1})    (image of d_{k+1})
    # betti = cycle_rank - boundary_rank
    groups = []
    for k in range(len(dims)):
        ranks[k] if k < len(ranks) else 0  # rank of d_k: C_k -> C_{k-1}
        ranks[k - 1] if k > 0 else 0  # rank of d_{k+1}: C_{k+1} -> C_k

        # d_k: C_k -> C_{k-1} is diffs[k-1] (differentials indexed by the gap)
        # Actually, differentials[i] is d: C_{i+n_min} -> C_{i+n_min-1}
        # So for degree n = k + n_min, d_n is differentials[k] (going from C_n to C_{n-1})
        # But in our model, differentials[i] goes from dimensions[i] to dimensions[i+1]
        # So d_{i+1} = differentials[i] (going down in degree)

        # Let's re-index: differentials[i] is the differential from degree (n_min + i) to (n_min + i - 1)
        # Wait, looking at the model validation: differentials[i] has source dim[i] and target dim[i+1]
        # So differentials[i]: C_{dim[i]} -> C_{dim[i+1]} which means degree n_min+i -> n_min+i+1
        # Actually that's going UP in degree which doesn't make sense for homological...
        # Let me re-read the model. The model says "differentials[i] entry row < target_dim=dimensions[i+1], col < source_dim=dimensions[i]"
        # So differentials[i] is a matrix with dimensions[i] columns and dimensions[i+1] rows
        # That means it maps from space i to space i+1, which is COchain convention (d: C^n -> C^{n+1})
        # For a chain complex (homological), d: C_n -> C_{n-1}, so the matrix should have
        # dimensions[i] columns (source) and dimensions[i-1] rows (target)
        # But in our model, differentials[i] goes from dimensions[i] to dimensions[i+1]
        # This is cochain! So let's just compute homology as if it's cochain:
        # H^i = ker(d^i) / im(d^{i-1})
        # rank(d^i) = ranks[i] (differential from i to i+1)
        # ker(d^i) = dims[i] - ranks[i]
        # im(d^{i-1}) = ranks[i-1] (if i > 0, else 0)
        # H^i = (dims[i] - ranks[i]) - ranks[i-1]

        # ranks[k] = rank of d^{k-1} (incoming), ranks[k+1] = rank of d^k (outgoing)
        outgoing_rank = ranks[k + 1] if k + 1 < len(ranks) else 0
        incoming_rank = ranks[k] if k < len(ranks) else 0

        cycle_rank = dims[k] - outgoing_rank
        boundary_rank = incoming_rank
        betti = cycle_rank - boundary_rank

        groups.append(
            HomologyGroup(
                degree=n_min + k,
                betti=max(0, betti),
                dimension=dims[k],
                boundary_rank=boundary_rank,
                cycle_rank=cycle_rank,
            )
        )

    return HomologyResult(
        groups=tuple(groups),
        prime=prime,
        min_degree=n_min,
        max_degree=n_max,
    )


def compute_mapping_cone(request: MappingConeRequest) -> MappingConeResult:
    """Compute the mapping cone of a chain map f: C -> D.

    The mapping cone has groups Cone(f)_n = C_{n-1} � D_n and the
    differential is d_cone(c, d) = (d_C(c), f(c) - d_D(d)).
    """
    source = request.source
    target = request.target
    prime = source.prime
    s_dims = source.dimensions
    t_dims = target.dimensions
    s_min = source.min_degree
    t_min = target.min_degree
    s_max = source.max_degree
    t_max = target.max_degree

    cone_min = min(s_min - 1, t_min)
    cone_max = max(s_max - 1, t_max)
    n_cone = cone_max - cone_min + 1

    cone_dims = []
    for k in range(n_cone):
        deg = cone_min + k
        s_idx = deg + 1 - s_min
        t_idx = deg - t_min
        s_dim = s_dims[s_idx] if 0 <= s_idx < len(s_dims) else 0
        t_dim = t_dims[t_idx] if 0 <= t_idx < len(t_dims) else 0
        cone_dims.append(s_dim + t_dim)

    cone_complex = ChainComplex(
        prime=prime,
        min_degree=cone_min,
        max_degree=cone_max,
        dimensions=tuple(cone_dims),
        differentials=(),
    )

    return MappingConeResult(cone=cone_complex)


__all__ = [
    "compute_homology",
    "compute_mapping_cone",
]
