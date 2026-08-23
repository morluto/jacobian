"""Domain-owned coalgebra operations."""

from __future__ import annotations

from itertools import product

from jacobian.math.coalgebras._models import (
    Coalgebra,
    ComultiplicationRequest,
    ComultiplicationResult,
    CounitRequest,
    CounitResult,
    GroupLikeElement,
    GroupLikeElementsRequest,
    GroupLikeElementsResult,
)
from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix


def compute_comultiplication(
    request: ComultiplicationRequest,
) -> ComultiplicationResult:
    """Compute Delta(c_i) for a basis element of a coalgebra.

    Returns the comultiplication as a dimension x dimension matrix of coefficients
    over GF(p), where entry (j, k) is the coefficient of c_j ⊗ c_k.
    """
    ca = request.coalgebra
    i = request.element_index
    n = ca.dimension
    p = ca.prime

    coeffs = tuple(
        tuple(ca.comultiplication[i][j][k] % p for k in range(n)) for j in range(n)
    )

    return ComultiplicationResult(
        coalgebra=ca,
        element_index=i,
        matrix=PrimeFieldMatrix(prime=p, entries=coeffs, columns=n),
        dimension=n,
    )


def compute_counit(request: CounitRequest) -> CounitResult:
    """Compute epsilon(c_i) for a basis element of a coalgebra."""
    ca = request.coalgebra
    i = request.element_index
    p = ca.prime

    return CounitResult(
        coalgebra=ca,
        element_index=i,
        value=ca.counit[i] % p,
    )


def _group_like_coefficients(
    ca: Coalgebra,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate every group-like coefficient vector of a coalgebra.

    An element g = sum a_i c_i is group-like when epsilon(g) = 1 and
    Delta(g) = g (x) g modulo p. The request model bounds the derived scan
    work -- candidates times per-candidate reconstruction -- within the
    documented budget, so this scan is exhaustive and deterministic; the
    result validator replays the identical enumeration. Delta(g) = g (x) g
    is decided row by row so a mismatching row skips the remaining rows.
    """
    n = ca.dimension
    p = ca.prime

    comult = tuple(
        tuple(
            tuple(ca.comultiplication[i][j][k] % p for k in range(n)) for j in range(n)
        )
        for i in range(n)
    )
    counit = tuple(ca.counit[i] % p for i in range(n))

    found: list[tuple[int, ...]] = []
    for coeffs in product(range(p), repeat=n):
        if sum(coeffs[i] * counit[i] for i in range(n)) % p != 1:
            continue

        group_like = True
        for j in range(n):
            delta_row = tuple(
                sum(coeffs[i] * comult[i][j][k] for i in range(n)) % p for k in range(n)
            )
            coefficient_j = coeffs[j]
            tensor_row = tuple(coefficient_j * coeffs[k] % p for k in range(n))
            if delta_row != tensor_row:
                group_like = False
                break
        if group_like:
            found.append(coeffs)
    return tuple(found)


def find_group_like_elements(
    request: GroupLikeElementsRequest,
) -> GroupLikeElementsResult:
    """Find all group-like elements g in a coalgebra over GF(p).

    An element g is group-like if Delta(g) = g ⊗ g and epsilon(g) = 1.
    Writing g = sum a_i c_i, the defining conditions are:

    1. epsilon(g) = sum a_i * epsilon(c_i) = 1 (mod p)
    2. Delta(g) = sum_{i,j,k} a_i d_i^{jk} c_j ⊗ c_k equals
       g ⊗ g = sum_{j,k} a_j a_k c_j ⊗ c_k

    The request model admits only coalgebras whose derived scan work --
    every candidate's counit filter plus each surviving candidate's
    reconstruction -- fits the documented budget, so this scan is
    exhaustive and the result lists every group-like element.
    """
    ca = request.coalgebra
    found = _group_like_coefficients(ca)

    return GroupLikeElementsResult(
        coalgebra=ca,
        elements=tuple(GroupLikeElement(coefficients=coeffs) for coeffs in found),
        count=len(found),
    )


__all__ = [
    "compute_comultiplication",
    "compute_counit",
    "find_group_like_elements",
]
