"""Domain-owned group cohomology operations."""

from __future__ import annotations

from itertools import product as iproduct

from jacobian.math.group._models import PermutationGroupRequest
from jacobian.math.group_cohomology._models import (
    CohomologyGroup,
    GroupCohomologyRequest,
    GroupCohomologyResult,
)


def _enumerate_group_elements(group: PermutationGroupRequest) -> list[tuple[int, ...]]:
    """Enumerate all elements of a permutation group."""
    from sympy.combinatorics import Permutation, PermutationGroup

    perms = [Permutation(list(g)) for g in group.generators]
    pg = PermutationGroup(perms)
    return [tuple(p.array_form) for p in pg.elements]


def _cayley_table(elements: list[tuple[int, ...]]) -> list[list[int]]:
    """Index the group multiplication over the enumerated elements."""
    from sympy.combinatorics import Permutation, PermutationGroup

    index = {element: i for i, element in enumerate(elements)}
    perms = [Permutation(list(e)) for e in elements]
    PermutationGroup(perms)
    table = [[0] * len(elements) for _ in elements]
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            product_form = tuple((a * b).array_form)
            if product_form not in index:
                raise ValueError("enumerated set is not closed under multiplication")
            table[i][j] = index[product_form]
    return table


def _gaussian_rank(matrix: list[list[int]], prime: int) -> int:
    rows = len(matrix)
    if rows == 0:
        return 0
    cols = len(matrix[0])
    aug = [row[:] for row in matrix]
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if aug[r][col] % prime != 0:
                pivot = r
                break
        if pivot is None:
            continue
        aug[rank], aug[pivot] = aug[pivot], aug[rank]
        inv_pivot = pow(aug[rank][col] % prime, prime - 2, prime)
        aug[rank] = [(v * inv_pivot) % prime for v in aug[rank]]
        for r in range(rows):
            factor = aug[r][col] % prime
            if r != rank and factor != 0:
                aug[r] = [
                    (aug[r][cc] - factor * aug[rank][cc]) % prime for cc in range(cols)
                ]
        rank += 1
        if rank >= rows:
            break
    return rank


def _encode(indexes: tuple[int, ...], base: int) -> int:
    value = 0
    for digit in indexes:
        value = value * base + digit
    return value


def _bar_delta_rank(group_size: int, n: int, cayley: list[list[int]], p: int) -> int:
    """GF(p)-rank of the inhomogeneous bar coboundary delta^n: C^n -> C^{n+1}.

    (delta f)(g_1,...,g_{n+1}) =
        f(g_2,...,g_{n+1})
      + sum_{i=1}^{n} (-1)^i f(g_1,..., g_i g_{i+1},...,g_{n+1})
      + (-1)^{n+1} f(g_1,...,g_n)
    """
    rows = group_size ** (n + 1)
    cols = group_size**n if n > 0 else 1
    matrix = [[0] * cols for _ in range(rows)]

    def add(row: int, arg: tuple[int, ...], coeff: int) -> None:
        col = _encode(arg, group_size)
        matrix[row][col] = (matrix[row][col] + coeff) % p

    for h in iproduct(range(group_size), repeat=n + 1):
        row = _encode(h, group_size)
        add(row, h[1:], 1)
        for i in range(n):
            merged = cayley[h[i]][h[i + 1]]
            add(row, (*h[:i], merged, *h[i + 2 :]), (-1) ** (i + 1))
        add(row, h[:n], (-1) ** (n + 1))
    return _gaussian_rank(matrix, p)


def _cohomology_profile(
    request: GroupCohomologyRequest,
) -> tuple[tuple[CohomologyGroup, ...], int]:
    """Exact cochain dimensions and Betti numbers for degrees 0..max_degree."""
    p = request.prime
    max_deg = request.max_degree

    elements = _enumerate_group_elements(request.group)
    group_order = len(elements)
    cayley = _cayley_table(elements)

    dims = [group_order**k if k > 0 else 1 for k in range(max_deg + 1)]
    ranks = [_bar_delta_rank(group_order, n, cayley, p) for n in range(max_deg + 1)]

    groups = []
    for k in range(max_deg + 1):
        rank_out = ranks[k]
        rank_in = ranks[k - 1] if k > 0 else 0
        betti = dims[k] - rank_out - rank_in
        if betti < 0:
            raise ValueError(
                "internal inconsistency: coboundaries do not square to zero"
            )
        groups.append(CohomologyGroup(degree=k, betti=betti, cochain_dimension=dims[k]))

    return tuple(groups), group_order


def compute_group_cohomology(request: GroupCohomologyRequest) -> GroupCohomologyResult:
    """Compute H^n(G, GF(p)) with trivial action via the exact unnormalized
    inhomogeneous bar complex.

    The cochain groups are C^n = {functions G^n -> GF(p)} of dimension
    |G|^n (reported as ``cochain_dimension``) and the inhomogeneous bar
    coboundaries are materialized exactly; each ``betti`` number is
    dim ker(delta^n) - rank(im(delta^{n-1})) = dim H^n(G, GF(p)).
    """
    groups, group_order = _cohomology_profile(request)
    return GroupCohomologyResult._from_kernel(request, groups, group_order)


def verify_group_cohomology_result(result: GroupCohomologyResult) -> bool:
    """Replay a separately supplied claim inside its admitted owner envelope."""

    groups, group_order = _cohomology_profile(result.request)
    return result.groups == groups and result.group_order == group_order


__all__ = ["compute_group_cohomology", "verify_group_cohomology_result"]
