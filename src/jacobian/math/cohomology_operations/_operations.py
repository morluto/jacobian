"""Domain-owned cohomology operations."""

from __future__ import annotations

from jacobian.math.cohomology_operations._models import (
    BocksteinRequest,
    BocksteinResult,
    SteenrodSquareRequest,
    SteenrodSquareResult,
)


def _reduce_support(
    simplices: tuple[tuple[int, ...], ...],
    coeffs: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Combine repeated simplex keys modulo 2 and drop vanished support.

    The sparse cochain notation denotes one cochain, so a repeated key
    contributes the sum of its coefficients in GF(2).
    """
    combined: dict[tuple[int, ...], int] = {}
    for simplex, coeff in zip(simplices, coeffs, strict=True):
        key = tuple(sorted(simplex))
        combined[key] = (combined.get(key, 0) + coeff) % 2
    surviving = {key: value for key, value in combined.items() if value != 0}
    keys = sorted(surviving)
    return tuple(keys), tuple(surviving[key] for key in keys)


def _combined_face(
    front: tuple[int, ...],
    back: tuple[int, ...],
    left_degree: int,
    right_degree: int,
) -> tuple[int, ...] | None:
    """Return the simplex carrying ``front cup back`` via Alexander-Whitney.

    The faces must meet at exactly one shared vertex that closes the front
    face and opens the back face, and the sorted union must have the front
    ``left_degree + 1`` vertices followed by the back ``right_degree + 1``.
    """
    if front[-1] != back[0]:
        return None
    combined = tuple(sorted(set(front) | set(back)))
    if len(combined) != left_degree + right_degree + 1:
        return None
    if combined[: left_degree + 1] != front or combined[left_degree:] != back:
        return None
    return combined


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
            combined = _combined_face(ls_sorted, rs_sorted, left_degree, right_degree)
            if combined is None:
                continue
            result_map[combined] = (result_map.get(combined, 0) + lc_mod * rc_mod) % 2

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

    support_values, support_coeffs = _reduce_support(
        request.simplex_values, request.simplex_coefficients
    )

    if k == 0:
        is_zero = not support_coeffs
        return SteenrodSquareResult(
            result_degree=p,
            result_simplex_values=support_values,
            result_simplex_coefficients=support_coeffs,
            is_zero=is_zero,
            square_degree=k,
        )

    if k == p:
        simplices, coeffs = _cup_product(
            support_values,
            support_coeffs,
            p,
            support_values,
            support_coeffs,
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
