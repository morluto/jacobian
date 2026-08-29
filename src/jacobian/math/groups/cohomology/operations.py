"""Domain-owned group cohomology operations."""

from __future__ import annotations

from itertools import product as iproduct

from sympy.combinatorics import Permutation
from sympy.combinatorics import PermutationGroup as SympyPermutationGroup

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups._models import PermutationGroup
from jacobian.math.groups.cohomology._models import (
    MAX_BAR_MATRIX_CELLS,
    MAX_COCHAIN_DEGREE,
    MAX_COCHAIN_TENSOR_ELEMENTS,
    MAX_GROUP_ORDER,
    MAX_PRIME,
    CohomologyGroup,
    GroupCohomologyResult,
)


def _admitted_max_degree(order: int) -> int:
    if order == 1:
        return MAX_COCHAIN_DEGREE
    degree = 0
    while (
        order ** (degree + 1) <= MAX_COCHAIN_TENSOR_ELEMENTS
        and order ** (2 * degree + 1) <= MAX_BAR_MATRIX_CELLS
    ):
        degree += 1
    return degree - 1


def _admit_request(
    group: PermutationGroup, prime: int, max_degree: int
) -> list[tuple[int, ...]]:
    from sympy import isprime

    if type(prime) is not int or prime < 2 or prime > MAX_PRIME:
        raise OperationDomainValidationError(
            location=("prime",),
            code="group_cohomology.prime_out_of_range",
            message=f"prime must be an integer between 2 and {MAX_PRIME}",
        )
    if not isprime(prime):
        raise OperationDomainValidationError(
            location=("prime",),
            code="group_cohomology.prime_not_prime",
            message="prime must be a prime integer",
        )
    if type(max_degree) is not int or max_degree < 0 or max_degree > MAX_COCHAIN_DEGREE:
        raise OperationDomainValidationError(
            location=("max_degree",),
            code="group_cohomology.max_degree_out_of_range",
            message=(
                f"max_degree must be an integer between 0 and {MAX_COCHAIN_DEGREE}"
            ),
        )
    elements = _enumerate_group_elements(group)
    order = len(elements)
    if order > MAX_GROUP_ORDER:
        raise OperationDomainValidationError(
            location=("group",),
            code="group_cohomology.group_order_exceeds_bound",
            message=(
                f"enumerated group order {order} exceeds the bounded maximum "
                f"{MAX_GROUP_ORDER}"
            ),
        )
    admitted_degree = _admitted_max_degree(order)
    if max_degree > admitted_degree:
        raise OperationDomainValidationError(
            location=("max_degree",),
            code="group_cohomology.max_degree_exceeds_work_budget",
            message=(
                f"max_degree {max_degree} exceeds the work-derived degree "
                f"budget {admitted_degree} for enumerated group order {order}"
            ),
        )
    return elements


def _enumerate_group_elements(group: PermutationGroup) -> list[tuple[int, ...]]:
    """Enumerate all elements of a permutation group."""
    perms = [Permutation(list(g)) for g in group.generators]
    pg = SympyPermutationGroup(perms)
    return [tuple(p.array_form) for p in pg.elements]


def _cayley_table(elements: list[tuple[int, ...]]) -> list[list[int]]:
    """Index the group multiplication over the enumerated elements."""
    index = {element: i for i, element in enumerate(elements)}
    perms = [Permutation(list(e)) for e in elements]
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
    prime: int,
    max_degree: int,
    elements: list[tuple[int, ...]],
) -> tuple[tuple[CohomologyGroup, ...], int]:
    """Exact cochain dimensions and Betti numbers for degrees 0..max_degree."""
    p = prime
    max_deg = max_degree

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


def group_cohomology(
    group: PermutationGroup,
    prime: int,
    max_degree: int,
) -> GroupCohomologyResult:
    """Compute H^n(G, GF(p)) with trivial action via the exact unnormalized
    inhomogeneous bar complex.

    The cochain groups are C^n = {functions G^n -> GF(p)} of dimension
    |G|^n (reported as ``cochain_dimension``) and the inhomogeneous bar
    coboundaries are materialized exactly; each ``betti`` number is
    dim ker(delta^n) - rank(im(delta^{n-1})) = dim H^n(G, GF(p)).
    """
    elements = _admit_request(group, prime, max_degree)
    groups, group_order = _cohomology_profile(prime, max_degree, elements)
    return GroupCohomologyResult._from_kernel(
        group, prime, max_degree, groups, group_order
    )


__all__ = ["group_cohomology"]
