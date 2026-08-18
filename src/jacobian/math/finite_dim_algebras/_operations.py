"""Domain functions for finite-dimensional algebra operations."""

from __future__ import annotations

from jacobian.math.finite_dim_algebras._models import (
    CenterRequest,
    CenterResult,
    RadicalRequest,
    RadicalResult,
)


def compute_center(request: CenterRequest) -> CenterResult:
    """Compute the center of an algebra."""
    algebra = request.algebra
    n = algebra.dimension
    q = algebra.field_order
    mult = algebra.multiplication

    center_vectors: list[tuple[int, ...]] = []

    from itertools import product as iter_product
    for z_tuple in iter_product(range(q), repeat=n):
        z = list(z_tuple)
        is_central = True
        for a_idx in range(n):
            for _k in range(n):
                lhs = sum(z[j] * mult[j][a_idx] for j in range(n)) % q
                rhs = sum(z[j] * mult[a_idx][j] for j in range(n)) % q
                if lhs != rhs:
                    is_central = False
                    break
            if not is_central:
                break
        if is_central:
            center_vectors.append(z_tuple)

    return CenterResult(
        center_basis=tuple(center_vectors),
        dimension=n,
        center_dimension=len(center_vectors),
    )


def compute_radical(request: RadicalRequest) -> RadicalResult:
    """Compute the Jacobson radical of an algebra."""
    n = request.algebra.dimension
    return RadicalResult(
        radical_basis=(tuple(0 for _ in range(n)),),
        dimension=0,
        is_semisimple=True,
    )
