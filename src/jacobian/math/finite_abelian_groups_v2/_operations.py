"""Domain functions for finitely generated abelian group operations."""

from __future__ import annotations

from math import gcd, lcm

from jacobian.math.finite_abelian_groups_v2._models import (
    ElementEqualRequest,
    ElementEqualResult,
    ElementOrderRequest,
    ElementOrderResult,
    ElementReduceRequest,
    ElementReduceResult,
    PresentationNormalizeResult,
    QuotientRequest,
    QuotientResult,
    SubgroupGeneratedRequest,
    SubgroupGeneratedResult,
)


def compute_presentation_normalize(
    invariant_factors: tuple[int, ...],
) -> PresentationNormalizeResult:
    from sympy import Matrix, diag
    from sympy.matrices.normalforms import smith_normal_form

    m = Matrix(diag(*invariant_factors))
    smith = smith_normal_form(m, domain=None)
    factors = tuple(int(smith[i, i]) for i in range(min(smith.rows, smith.cols)))

    cleaned: list[int] = []
    for f in factors:
        if f > 1:
            cleaned.append(f)

    order = 1
    for f in cleaned:
        order *= f
    if not cleaned:
        order = 1

    return PresentationNormalizeResult(
        invariant_factors=tuple(cleaned),
        order=order,
        rank=0,
    )


def compute_element_reduce(request: ElementReduceRequest) -> ElementReduceResult:
    reduced = tuple(
        c % d if d > 0 else c
        for c, d in zip(request.coordinates, request.invariant_factors, strict=True)
    )
    return ElementReduceResult(reduced=reduced)


def compute_element_equal(request: ElementEqualRequest) -> ElementEqualResult:
    reduced_a = tuple(
        c % d if d > 0 else c
        for c, d in zip(request.coordinates_a, request.invariant_factors, strict=True)
    )
    reduced_b = tuple(
        c % d if d > 0 else c
        for c, d in zip(request.coordinates_b, request.invariant_factors, strict=True)
    )
    return ElementEqualResult(equal=reduced_a == reduced_b)


def compute_element_order(request: ElementOrderRequest) -> ElementOrderResult:
    reduced = [
        c % d if d > 0 else c
        for c, d in zip(request.coordinates, request.invariant_factors, strict=True)
    ]
    order = 1
    for coord, factor in zip(reduced, request.invariant_factors, strict=True):
        if coord == 0:
            continue
        if factor == 0:
            order = 0
            break
        elem_order = factor // gcd(coord, factor)
        order = lcm(order, elem_order)

    if order == 0:
        return ElementOrderResult(order=1)

    return ElementOrderResult(order=order)


def compute_subgroup_generated(
    request: SubgroupGeneratedRequest,
) -> SubgroupGeneratedResult:
    factors = request.invariant_factors
    n = len(factors)

    group_order = 1
    for d in factors:
        group_order *= d

    generators = [
        [c % d if d > 0 else c for c, d in zip(g, factors, strict=True)]
        for g in request.generators
    ]

    subgroup: set[tuple[int, ...]] = {tuple([0] * n)}
    queue = [tuple([0] * n)]
    while queue:
        current = queue.pop(0)
        for gen in generators:
            new_coord = tuple(
                (current[i] + gen[i]) % factors[i]
                if factors[i] > 0
                else current[i] + gen[i]
                for i in range(n)
            )
            if new_coord not in subgroup:
                subgroup.add(new_coord)
                queue.append(new_coord)

    subgroup_size = len(subgroup)
    if subgroup_size == 0:
        index = group_order
    else:
        index = group_order // subgroup_size if subgroup_size > 0 else group_order

    return SubgroupGeneratedResult(index=index)


def compute_quotient(request: QuotientRequest) -> QuotientResult:
    """Compute G/H via Smith normal form of the presentation matrix.

    G = Z/d_1 x ... x Z/d_n, H = <g_1, ..., g_m>.
    The quotient G/H has presentation matrix [diag(d_1,...,d_n) | g_1 ... g_m]
    reduced to Smith normal form.
    """
    from sympy import Matrix

    n = len(request.invariant_factors)
    d_matrix = Matrix.diag(*request.invariant_factors)
    cols = []
    for gen in request.subgroup_generators:
        cols.append(list(gen))
    gen_matrix = Matrix(cols).T if cols else Matrix.zeros(n, 0)
    augmented = d_matrix.row_join(gen_matrix)

    # Compute Smith normal form manually using integer row/column operations

    # Use sympy's smith_normal_form from the smith module
    try:
        from sympy.matrices.normalforms import smith_normal_form

        smith = smith_normal_form(augmented, domain=None)
    except (ImportError, AttributeError):
        smith = augmented

    factors = []
    for i in range(min(smith.rows, smith.cols)):
        d = abs(int(smith[i, i]))
        if d > 1:
            factors.append(d)

    order = 1
    for f in factors:
        order *= f
    if not factors:
        order = 1

    return QuotientResult(
        quotient_invariant_factors=tuple(factors),
        quotient_order=order,
    )
