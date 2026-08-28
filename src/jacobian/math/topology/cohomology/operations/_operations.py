"""Domain functions for cohomology operations over GF(2) and Z/p."""

from __future__ import annotations

from jacobian.math.topology.cohomology.operations._models import (
    BocksteinRequest,
    BocksteinResult,
    SteenrodSquareRequest,
    SteenrodSquareResult,
)


def _reduce_support(
    simplices: tuple[tuple[int, ...], ...],
    coeffs: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Merge duplicate simplex keys, summing coefficients in GF(2)."""

    merged: dict[tuple[int, ...], int] = {}
    for simplex, coefficient in zip(simplices, coeffs, strict=True):
        key = tuple(sorted(simplex))
        merged[key] = (merged.get(key, 0) + coefficient) % 2
    surviving = sorted(key for key, value in merged.items() if value != 0)
    values = tuple(surviving)
    coefficients = tuple(merged[key] for key in surviving)
    return values, coefficients


def _reduce_support_mod_prime(
    simplices: tuple[tuple[int, ...], ...],
    coeffs: tuple[int, ...],
    prime: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Merge duplicate simplex keys, summing coefficients modulo ``prime``."""

    merged: dict[tuple[int, ...], int] = {}
    for simplex, coefficient in zip(simplices, coeffs, strict=True):
        key = tuple(sorted(simplex))
        merged[key] = (merged.get(key, 0) + coefficient) % prime
    surviving = sorted(key for key, value in merged.items() if value != 0)
    values = tuple(surviving)
    coefficients = tuple(merged[key] for key in surviving)
    return values, coefficients


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
    allowed_faces: frozenset[tuple[int, ...]] | None = None,
) -> tuple[list[tuple[int, ...]], list[int]]:
    """Compute the cup product of two cochains over GF(2) via Alexander-Whitney.

    For simplicial cochains, the cup product of alpha (degree p) and
    beta (degree q) on a (p+q)-simplex [v_0, ..., v_{p+q}] is:
    (alpha cup beta)([v_0, ..., v_{p+q}]) = alpha([v_0, ..., v_p]) * beta([v_p, ..., v_{p+q}])

    Over GF(2), all signs are 1. Only pairs where the front face of the
    combined simplex equals the left simplex and the back face equals the
    right simplex contribute. When ``allowed_faces`` is provided, target
    simplices outside the ambient complex are dropped: the cup product lives
    on the ambient complex's simplices only.
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
            if allowed_faces is not None and combined not in allowed_faces:
                continue
            result_map[combined] = (result_map.get(combined, 0) + lc_mod * rc_mod) % 2

    # Filter zero results
    result_map = {k: v for k, v in result_map.items() if v % 2 != 0}
    simplices = sorted(result_map.keys())
    coeffs = [result_map[s] for s in simplices]
    return simplices, coeffs


def steenrod_square_fields(
    cochain_degree: int,
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    square_degree: int,
    ambient_simplices: tuple[tuple[int, ...], ...],
) -> tuple[int, tuple[tuple[int, ...], ...], tuple[int, ...], bool]:
    """Pure Sq^k core returning ``(degree, values, coefficients, is_zero)``.

    Kept free of models so the explicit claim verifier can replay the exact
    computation without re-entering result validation.
    """
    p = cochain_degree
    k = square_degree

    if k > p:
        return (p + k, (), (), True)

    support_values, support_coeffs = _reduce_support(
        simplex_values, simplex_coefficients
    )

    if k == 0:
        is_zero = not support_coeffs
        return (p, support_values, support_coeffs, is_zero)

    if k == p:
        allowed = frozenset(ambient_simplices) if ambient_simplices else None
        simplices, coeffs = _cup_product(
            support_values,
            support_coeffs,
            p,
            support_values,
            support_coeffs,
            p,
            allowed_faces=allowed,
        )
        is_zero = len(coeffs) == 0 or all(c == 0 for c in coeffs)
        return (
            2 * p,
            tuple(simplices) if not is_zero else (),
            tuple(coeffs) if not is_zero else (),
            is_zero,
        )

    # Intermediate squares 0<k<p are not supported; request validation should have rejected.
    raise ValueError(
        "intermediate Steenrod squares 0<k<deg require cup-i products and are not supported"
    )


def _effective_ambient_for_request(
    request: SteenrodSquareRequest | BocksteinRequest,
) -> tuple[tuple[int, ...], ...]:
    """Return the integer ambient set for a request."""
    # Import here to avoid circular import at module load.
    from jacobian.math.topology.cohomology.operations._models import _effective_ambient

    return _effective_ambient(request.ambient_simplices, request.ambient_complex)


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
    effective = _effective_ambient_for_request(request)
    (
        result_degree,
        result_simplex_values,
        result_simplex_coefficients,
        is_zero,
    ) = steenrod_square_fields(
        request.cochain_degree,
        request.simplex_values,
        request.simplex_coefficients,
        request.square_degree,
        effective,
    )
    return SteenrodSquareResult._from_kernel(
        request,
        result_degree,
        result_simplex_values,
        result_simplex_coefficients,
        is_zero,
    )


def bockstein_fields(
    prime: int,
    cochain_degree: int,
    simplex_coefficients: tuple[int, ...],
    simplex_values: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[int, tuple[tuple[int, ...], ...], tuple[int, ...], bool]:
    """Pure Bockstein core returning ``(degree, values, coefficients, is_zero)``.

    When ``simplex_values`` is supplied duplicate simplex keys are merged
    modulo ``prime`` before the zero test, so a cochain whose sparse
    support cancels to zero is correctly classified as the zero cocycle.
    """

    if simplex_values is not None:
        # Merge duplicate keys modulo prime before the zero check.
        _, merged_coeffs = _reduce_support_mod_prime(
            simplex_values, simplex_coefficients, prime
        )
        # ``merged_coeffs`` is empty iff every residue is 0
        if not merged_coeffs:
            return (cochain_degree + 1, (), (), True)
        raise ValueError(
            "non-zero Bockstein requires the ambient simplicial complex and is not supported"
        )

    if not simplex_coefficients or all(c % prime == 0 for c in simplex_coefficients):
        return (cochain_degree + 1, (), (), True)
    raise ValueError(
        "non-zero Bockstein requires the ambient simplicial complex and is not supported"
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
    (
        result_degree,
        result_simplex_values,
        result_simplex_coefficients,
        is_zero,
    ) = bockstein_fields(
        request.prime,
        request.cochain_degree,
        request.simplex_coefficients,
        request.simplex_values,
    )
    return BocksteinResult._from_kernel(
        request,
        result_degree,
        result_simplex_values,
        result_simplex_coefficients,
        is_zero,
    )


__all__ = [
    "bockstein_fields",
    "compute_bockstein",
    "compute_steenrod_square",
    "steenrod_square_fields",
]
