"""Domain functions for cohomology operations over GF(2) and Z/p."""

from __future__ import annotations

from collections.abc import Callable

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology._models import FiniteSimplicialComplex
from jacobian.math.topology.cohomology.operations._models import (
    MAX_AMBIENT_SIMPLEX_VERTICES,
    MAX_RESULT_COCHAIN_DEGREE,
    BocksteinResult,
    SteenrodSquareResult,
    _effective_ambient,
    _validate_simplex_entries,
    _validation_error,
)


def _run_admission(admission: Callable[[], None]) -> None:
    """Expose owner admission as a typed native-domain failure."""

    try:
        admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("request",), code=exc.type, message=exc.message()
        ) from exc


def _require_downward_closed(simplices: tuple[tuple[int, ...], ...]) -> None:
    """Require every codimension-one face of each ambient simplex."""

    known = set(simplices)
    for simplex in simplices:
        for index in range(len(simplex)):
            face = simplex[:index] + simplex[index + 1 :]
            if face and face not in known:
                raise _validation_error(
                    "ambient_not_downward_closed",
                    "ambient_simplices must be downward closed: the face "
                    f"{face} of {simplex} is absent",
                )


def _require_cocycle(
    cochain_degree: int,
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    ambient_simplices: tuple[tuple[int, ...], ...],
) -> None:
    """Require the GF(2) coboundary to vanish on ambient simplices."""

    values_by_face: dict[tuple[int, ...], int] = {}
    for simplex, coefficient in zip(simplex_values, simplex_coefficients, strict=True):
        key = tuple(simplex)
        values_by_face[key] = (values_by_face.get(key, 0) + coefficient) % 2
    for sigma in ambient_simplices:
        if len(sigma) != cochain_degree + 2:
            continue
        coboundary = 0
        for index in range(len(sigma)):
            face = sigma[:index] + sigma[index + 1 :]
            coboundary = (coboundary + values_by_face.get(face, 0)) % 2
        if coboundary:
            raise _validation_error(
                "not_cocycle",
                "the supplied cochain is not a cocycle: its coboundary does not "
                "vanish on the ambient complex",
            )


def _validate_ambient_complex(
    cochain_degree: int,
    support: tuple[tuple[int, ...], ...],
    coefficients: tuple[int, ...],
    ambient: tuple[tuple[int, ...], ...],
) -> None:
    """Verify ambient shape, support containment, closure, and cocyclicity."""

    _validate_simplex_entries(ambient, "ambient simplex")
    if any(len(simplex) > MAX_AMBIENT_SIMPLEX_VERTICES for simplex in ambient):
        raise _validation_error(
            "ambient_simplex_bound",
            "each ambient simplex may carry at most "
            f"{MAX_AMBIENT_SIMPLEX_VERTICES} vertices",
        )
    known = set(ambient)
    for simplex in support:
        if simplex not in known:
            raise _validation_error(
                "support_outside_ambient",
                "cochain support must lie inside the ambient complex",
            )
    _require_downward_closed(ambient)
    _require_cocycle(cochain_degree, support, coefficients, ambient)


def _is_zero_mod2_cochain(
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
) -> bool:
    """Return whether the GF(2) cochain represented by sparse support is zero."""

    merged: dict[tuple[int, ...], int] = {}
    for simplex, coefficient in zip(simplex_values, simplex_coefficients, strict=True):
        key = tuple(simplex)
        merged[key] = (merged.get(key, 0) + coefficient) % 2
    return not any(value != 0 for value in merged.values())


def _is_zero_mod_prime_cochain(
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    prime: int,
) -> bool:
    """Return whether the Z/p cochain represented by sparse support is zero."""

    merged: dict[tuple[int, ...], int] = {}
    for simplex, coefficient in zip(simplex_values, simplex_coefficients, strict=True):
        key = tuple(simplex)
        merged[key] = (merged.get(key, 0) + coefficient) % prime
    return not any(value != 0 for value in merged.values())


def _admit_steenrod_square(
    cochain_degree: int,
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    square_degree: int,
    ambient_simplices: tuple[tuple[int, ...], ...],
    ambient_complex: FiniteSimplicialComplex | None,
) -> None:
    """Admit one exact Steenrod-square invocation and verify its cocycle."""

    def admission() -> None:
        result_degree = cochain_degree + square_degree
        if result_degree > MAX_RESULT_COCHAIN_DEGREE:
            raise _validation_error(
                "result_degree_bound",
                f"Sq^{square_degree} of a degree-{cochain_degree} "
                f"cochain returns degree {result_degree}, above the "
                f"{MAX_RESULT_COCHAIN_DEGREE}-degree exact-result budget",
            )
        if 0 < square_degree < cochain_degree:
            raise _validation_error(
                "intermediate_square_unsupported",
                "intermediate Steenrod squares 0<k<deg require cup-i products "
                "and are not supported",
            )
        effective_ambient = _effective_ambient(ambient_simplices, ambient_complex)
        if (
            not _is_zero_mod2_cochain(simplex_values, simplex_coefficients)
            and not effective_ambient
        ):
            raise _validation_error(
                "ambient_required_for_nonzero",
                "Steenrod squares are cohomology operations: the supplied cochain "
                "must be verified as a cocycle against an ambient simplicial "
                "complex; supply ambient_simplices or ambient_complex",
            )
        if (
            square_degree == cochain_degree
            and cochain_degree >= 1
            and not effective_ambient
        ):
            raise _validation_error(
                "ambient_required_for_top_square",
                "the top Steenrod square requires the ambient simplicial complex; "
                "supply ambient_simplices or ambient_complex",
            )
        if effective_ambient:
            _validate_ambient_complex(
                cochain_degree,
                simplex_values,
                simplex_coefficients,
                effective_ambient,
            )

    _run_admission(admission)


def _admit_bockstein(
    prime: int,
    cochain_degree: int,
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    ambient_simplices: tuple[tuple[int, ...], ...],
    ambient_complex: FiniteSimplicialComplex | None,
) -> None:
    """Admit the supported zero-cocycle Bockstein branch."""

    def admission() -> None:
        from sympy import isprime

        if not isprime(prime):
            raise _validation_error("prime_not_prime", "prime must be a prime integer")
        if not _is_zero_mod_prime_cochain(
            simplex_values,
            simplex_coefficients,
            prime,
        ):
            raise _validation_error(
                "nonzero_bockstein_unsupported",
                "non-zero Bockstein requires the ambient simplicial complex; "
                "unsupported in this bounded operation",
            )
        effective_ambient = _effective_ambient(ambient_simplices, ambient_complex)
        if effective_ambient:
            _validate_ambient_complex(
                cochain_degree,
                simplex_values,
                simplex_coefficients,
                effective_ambient,
            )

    _run_admission(admission)


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


def steenrod_square(
    cochain_degree: int,
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    square_degree: int,
    ambient_simplices: tuple[tuple[int, ...], ...] = (),
    ambient_complex: FiniteSimplicialComplex | None = None,
) -> SteenrodSquareResult:
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
    _admit_steenrod_square(
        cochain_degree,
        simplex_values,
        simplex_coefficients,
        square_degree,
        ambient_simplices,
        ambient_complex,
    )
    effective = _effective_ambient(ambient_simplices, ambient_complex)
    (
        result_degree,
        result_simplex_values,
        result_simplex_coefficients,
        is_zero,
    ) = steenrod_square_fields(
        cochain_degree,
        simplex_values,
        simplex_coefficients,
        square_degree,
        effective,
    )
    return SteenrodSquareResult._from_kernel(
        cochain_degree,
        simplex_values,
        simplex_coefficients,
        square_degree,
        ambient_simplices,
        ambient_complex,
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


def bockstein(
    prime: int,
    cochain_degree: int,
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    ambient_simplices: tuple[tuple[int, ...], ...] = (),
    ambient_complex: FiniteSimplicialComplex | None = None,
) -> BocksteinResult:
    """Compute the Bockstein homomorphism beta: H^n(Z/p) -> H^{n+1}(Z/p).

    For the short exact sequence 0 -> Z/p -> Z/p^2 -> Z/p -> 0,
    the Bockstein of a cocycle x is beta(x) = (1/p) * dx where dx is
    the coboundary of x modulo p. Computing it requires the ambient
    simplicial complex to evaluate the coboundary. This bounded operation
    only supports the trivial cocycle (all coefficients 0 mod p), for which
    the Bockstein is provably zero. Non-zero inputs are rejected at the
    request boundary as unsupported.
    """
    _admit_bockstein(
        prime,
        cochain_degree,
        simplex_values,
        simplex_coefficients,
        ambient_simplices,
        ambient_complex,
    )
    (
        result_degree,
        result_simplex_values,
        result_simplex_coefficients,
        is_zero,
    ) = bockstein_fields(
        prime,
        cochain_degree,
        simplex_coefficients,
        simplex_values,
    )
    return BocksteinResult._from_kernel(
        prime,
        cochain_degree,
        simplex_values,
        simplex_coefficients,
        ambient_simplices,
        ambient_complex,
        result_degree,
        result_simplex_values,
        result_simplex_coefficients,
        is_zero,
    )


__all__ = [
    "bockstein",
    "bockstein_fields",
    "steenrod_square",
    "steenrod_square_fields",
]
