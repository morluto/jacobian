"""Domain-owned chain complex operations."""

from __future__ import annotations

from jacobian.math.chain_complexes._models import (
    ChainComplex,
    HomologyGroup,
    HomologyRequest,
    HomologyResult,
    MappingConeRequest,
    MappingConeResult,
    _boundary_matrix,
    _chain_map_degree,
    _chain_map_matrix,
)
from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix, rank


def _negate(matrix: PrimeFieldMatrix) -> PrimeFieldMatrix:
    """Negate every residue of a prime-field matrix."""

    prime = matrix.prime
    return PrimeFieldMatrix(
        prime=prime,
        entries=tuple(
            tuple((prime - value) % prime for value in row) for row in matrix.entries
        ),
        columns=matrix.columns,
    )


def _zero_block(rows: int, columns: int, prime: int) -> PrimeFieldMatrix:
    return PrimeFieldMatrix(
        prime=prime,
        entries=tuple((0,) * columns for _ in range(rows)),
        columns=columns,
    )


def _homology_groups(cx: ChainComplex) -> tuple[HomologyGroup, ...]:
    """Exact homology groups of one chain complex over its prime field.

    H_n = ker(d_n) / im(d_{n+1})
    Betti_n = dim(C_n) - rank(d_n) - rank(d_{n+1})
    where d_n: C_n -> C_{n-1} has matrix dims[n-1] x dims[n] (target x source)
    and differentials[i] is d_{min+i+1} with source dims[i+1] and target dims[i].

    Every rank comes from the shared prime-field kernel. Shared by the
    operation and the result validator so an authored result replays against
    the identical kernel; im(d_{n+1}) is contained in ker(d_n) because
    admission verified d^2 = 0, so each Betti number is nonnegative without
    clamping.
    """
    n_min = cx.min_degree
    dims = cx.dimensions

    ranks = [rank(diff) for diff in cx.differentials]

    groups = []
    for k in range(len(dims)):
        # outgoing d_k: C_k -> C_{k-1} contributes rank ranks[k-1] when k > 0
        out_rank = ranks[k - 1] if k > 0 and k - 1 < len(ranks) else 0
        in_rank = ranks[k] if k < len(ranks) else 0
        cycle_rank = dims[k] - out_rank
        boundary_rank = in_rank
        groups.append(
            HomologyGroup(
                degree=n_min + k,
                betti=cycle_rank - boundary_rank,
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


def _cone_differential(
    gap: int,
    cone_min: int,
    source: ChainComplex,
    target: ChainComplex,
    chain_map: tuple[PrimeFieldMatrix, ...],
    prime: int,
) -> PrimeFieldMatrix:
    """Build the cone differential leaving degree cone_min + gap + 1.

    d_cone_n = [ -d^C_{n-1}   0        ]
                [  f_{n-1}     d^D_n    ]   on Cone(f)_n = C_{n-1} (+) D_n.
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
    top_left = _negate(_boundary_matrix(source, n - 1))
    top_right = _zero_block(source_two_below, target_here, prime)
    bottom_left = _chain_map_matrix(chain_map, source, target, n - 1)
    bottom_right = _boundary_matrix(target, n)

    entries = []
    for left_row, right_row in zip(top_left.entries, top_right.entries, strict=True):
        entries.append(tuple(left_row) + tuple(right_row))
    for left_row, right_row in zip(
        bottom_left.entries, bottom_right.entries, strict=True
    ):
        entries.append(tuple(left_row) + tuple(right_row))
    return PrimeFieldMatrix(
        prime=prime,
        entries=tuple(entries),
        columns=source_above + target_here,
    )


def _build_cone_complex(request: MappingConeRequest) -> ChainComplex:
    """Pure mapping-cone construction used by execution and result replay."""
    source = request.source
    target = request.target
    if source.prime != target.prime:
        raise ValueError("source and target must have the same prime")
    prime = source.prime
    cone_min = min(source.min_degree + 1, target.min_degree)
    cone_max = max(source.max_degree + 1, target.max_degree)

    def cone_dimension(deg: int) -> int:
        return _chain_map_degree(source, deg - 1) + _chain_map_degree(target, deg)

    cone_dims = [cone_dimension(cone_min + k) for k in range(cone_max - cone_min + 1)]
    # One differential per gap between consecutive cone degrees.
    cone_diffs = [
        _cone_differential(k, cone_min, source, target, request.chain_map, prime)
        for k in range(len(cone_dims) - 1)
    ]

    return ChainComplex(
        prime=prime,
        min_degree=cone_min,
        max_degree=cone_max,
        dimensions=tuple(cone_dims),
        differentials=tuple(cone_diffs),
    )


def compute_mapping_cone(request: MappingConeRequest) -> MappingConeResult:
    """Compute the mapping cone of a chain map f: C -> D.

    The mapping cone has groups Cone(f)_n = C_{n-1} (+) D_n and the
    differential is

        d_cone_n = [ -d^C_{n-1}   0
                     f_{n-1}    d^D_n ]

    where d^C, d^D are the source/target differentials and f is the chain map.
    """
    return MappingConeResult(request=request, cone=_build_cone_complex(request))


__all__ = [
    "compute_homology",
    "compute_mapping_cone",
]
