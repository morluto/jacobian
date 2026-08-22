"""Domain-owned cohomology operations."""

from __future__ import annotations

from jacobian.math.cohomology_operations._models import (
    BocksteinRequest,
    BocksteinResult,
    SteenrodSquareRequest,
    SteenrodSquareResult,
)


def _cup_product(
    left_simplices: tuple[tuple[int, ...], ...],
    left_coeffs: tuple[int, ...],
    left_degree: int,
    right_simplices: tuple[tuple[int, ...], ...],
    right_coeffs: tuple[int, ...],
    right_degree: int,
) -> tuple[list[tuple[int, ...]], list[int]]:
    """Compute the cup product of two cochains over GF(2) via Alexander-Whitney.

    For simplicial cochains, the cup product of alpha (degree p) and
    beta (degree q) on a (p+q)-simplex [v_0, ..., v_{p+q}] is:
    (alpha cup beta)([v_0, ..., v_{p+q}]) = alpha([v_0, ..., v_p]) * beta([v_p, ..., v_{p+q}])

    Over GF(2), all signs are 1. Only pairs where the front face of the
    combined simplex equals the left simplex and the back face equals the
    right simplex contribute.
    """
    result_map: dict[tuple[int, ...], int] = {}
    for ls, lc in zip(left_simplices, left_coeffs, strict=False):
        lc_mod = lc % 2
        if lc_mod == 0:
            continue
        ls_sorted = tuple(sorted(ls))
        if len(ls_sorted) != left_degree + 1:
            continue
        for rs, rc in zip(right_simplices, right_coeffs, strict=False):
            rc_mod = rc % 2
            if rc_mod == 0:
                continue
            rs_sorted = tuple(sorted(rs))
            if len(rs_sorted) != right_degree + 1:
                continue
            # Alexander-Whitney requires the faces meet at exactly one vertex
            # and that vertex is the last of the front face and first of the back face.
            if ls_sorted[-1] != rs_sorted[0]:
                continue
            # Combined simplex is the sorted union; it must have size p+q+1
            # and its front p+1 vertices must be ls and its last q+1 must be rs.
            combined = tuple(sorted(set(ls_sorted) | set(rs_sorted)))
            if len(combined) != left_degree + right_degree + 1:
                continue
            # Check that ls is the front face and rs is the back face of combined
            if combined[: left_degree + 1] != ls_sorted:
                continue
            if combined[left_degree :] != rs_sorted:
                continue
            product = (lc_mod * rc_mod) % 2
            if product != 0:
                result_map[combined] = (result_map.get(combined, 0) + product) % 2

    # Filter zero results
    result_map = {k: v for k, v in result_map.items() if v % 2 != 0}
    simplices = sorted(result_map.keys())
    coeffs = [result_map[s] for s in simplices]
    return simplices, coeffs


def compute_steenrod_square(request: SteenrodSquareRequest) -> SteenrodSquareResult:
    """Compute the Steenrod square Sq^k(x) for a cocycle x over GF(2).

    Sq^k(x) is nonzero only when 0 <= k <= deg(x).
    Sq^0(x) = x (identity)
    Sq^n(x) = x cup x when n = deg(x) (where x is a degree-n cocycle)
    Sq^k(x) = 0 when k > deg(x) (instability / cessation)

    For a cocycle x of degree n:
    - Sq^0(x) = x
    - Sq^n(x) = x cup x
    - Sq^k(x) = 0 for k > n (instability)
    - Sq^k(x) for 0 < k < n requires the cup-i product structure
    """
    p = request.cochain_degree
    k = request.square_degree

    if k > p:
        return SteenrodSquareResult(
            result_degree=p + k,
            result_simplex_values=(),
            result_simplex_coefficients=(),
            is_zero=True,
            square_degree=k,
        )

    if k == 0:
        # Reduce coefficients mod 2 and drop zero support
        reduced = tuple(c % 2 for c in request.simplex_coefficients)
        filtered_values = tuple(v for v, c in zip(request.simplex_values, reduced, strict=False) if c != 0)
        filtered_coeffs = tuple(c for c in reduced if c != 0)
        is_zero = len(filtered_coeffs) == 0
        return SteenrodSquareResult(
            result_degree=p,
            result_simplex_values=filtered_values if not is_zero else (),
            result_simplex_coefficients=filtered_coeffs if not is_zero else (),
            is_zero=is_zero,
            square_degree=k,
        )

    if k == p:
        simplices, coeffs = _cup_product(
            request.simplex_values,
            tuple(c % 2 for c in request.simplex_coefficients),
            p,
            request.simplex_values,
            tuple(c % 2 for c in request.simplex_coefficients),
            p,
        )
        is_zero = len(coeffs) == 0 or all(c == 0 for c in coeffs)
        return SteenrodSquareResult(
            result_degree=2 * p,
            result_simplex_values=tuple(simplices) if not is_zero else (),
            result_simplex_coefficients=tuple(coeffs) if not is_zero else (),
            is_zero=is_zero,
            square_degree=k,
        )

    # Intermediate squares 0<k<p are not supported; request validation should have rejected.
    raise ValueError(
        "intermediate Steenrod squares 0<k<deg require cup-i products and are not supported"
    )


def compute_bockstein(request: BocksteinRequest) -> BocksteinResult:
    """Compute the Bockstein homomorphism beta: H^n(Z/p) -> H^{n+1}(Z/p).

    For the short exact sequence 0 -> Z/p -> Z/p^2 -> Z/p -> 0,
    the Bockstein of a cocycle x is beta(x) = (1/p) * dx where dx is
    the coboundary of x modulo p. Computing it requires the ambient
    simplicial complex to evaluate the coboundary. This bounded operation
    only supports the trivial cocycle (all coefficients 0 mod p), for which
    the Bockstein is provably zero. Non-zero inputs are rejected at the
    request boundary as unsupported.
    """
    p = request.prime
    n = request.cochain_degree

    # Only zero cocycles are supported; non-zero requires the complex.
    if not request.simplex_coefficients or all(c % p == 0 for c in request.simplex_coefficients):
        return BocksteinResult(
            result_degree=n + 1,
            result_simplex_values=(),
            result_simplex_coefficients=(),
            is_zero=True,
            prime=p,
        )

    raise ValueError(
        "non-zero Bockstein requires the ambient simplicial complex and is not supported"
    )


__all__ = [
    "compute_bockstein",
    "compute_steenrod_square",
]
