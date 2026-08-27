"""Domain-owned p-adic number theory operations."""

from __future__ import annotations

from collections.abc import Sequence

from jacobian.canonical import format_canonical_integer
from jacobian.math.padic_arithmetic._models import (
    HenselFactorLiftRequest,
    HenselFactorLiftResult,
    HenselRootRequest,
    HenselRootResult,
    IntegerPolynomial,
    PAdicRootEntry,
    PAdicRootsRequest,
    PAdicRootsResult,
    _kernel_coefficients,
)


def _trim_asc(coefficients: Sequence[int]) -> list[int]:
    """Drop zero highest-degree coefficients (ascending representation)."""
    trimmed = list(coefficients)
    while len(trimmed) > 1 and trimmed[-1] == 0:
        trimmed.pop()
    return trimmed


def _wire_polynomial(coefficients: Sequence[int]) -> IntegerPolynomial:
    """Convert ascending int coefficients to the canonical descending value."""
    trimmed = _trim_asc(coefficients)
    return IntegerPolynomial(
        coefficients=tuple(
            format_canonical_integer(coefficient) for coefficient in reversed(trimmed)
        )
    )


def _eval_poly(coeffs: tuple[int, ...], x: int, modulus: int) -> int:
    """Evaluate a polynomial at x modulo modulus (Horner's method)."""
    result = 0
    for coeff in reversed(coeffs):
        result = (result * x + coeff) % modulus
    return result


def _eval_deriv(coeffs: tuple[int, ...], x: int, modulus: int) -> int:
    """Evaluate f'(x) mod modulus."""
    n = len(coeffs)
    if n <= 1:
        return 0
    deriv_coeffs = tuple(i * coeffs[i] for i in range(1, n))
    return _eval_poly(deriv_coeffs, x, modulus)


def _hensel_lift_root(
    coeffs: tuple[int, ...],
    p: int,
    r: int,
    k: int,
) -> int:
    """Lift the simple root r of f mod p to its unique root mod p^k.

    The caller must have established f(r) = 0 and f'(r) != 0 (mod p); Hensel's
    lemma then guarantees a unique lift, constructed step by step.
    """
    f_r = _eval_poly(coeffs, r, p)
    if f_r != 0:
        raise ValueError(f"{r} is not a root mod {p}")
    fp_r = _eval_deriv(coeffs, r, p)
    if fp_r % p == 0:
        raise ValueError("root is not simple; Hensel lifting does not apply")

    pow(fp_r % p, -1, p)

    current = r
    current_mod = p
    for _ in range(1, k):
        next_mod = current_mod * p
        f_val = _eval_poly(coeffs, current, next_mod)
        # Newton step: r_next = r - f(r) * (f'(r))^{-1} mod next_mod
        fp_val = _eval_deriv(coeffs, current, next_mod)
        if fp_val % next_mod == 0:
            raise ValueError("derivative vanished during lifting")
        inv_fp_val = pow(int(fp_val), -1, next_mod)
        correction = (f_val * inv_fp_val) % next_mod
        current = (current - correction) % next_mod
        current_mod = next_mod

    return current


def hensel_lift_root(request: HenselRootRequest) -> HenselRootResult:
    """Lift the simple root of f(x) mod p validated by the request model."""
    coeffs = _kernel_coefficients(request.polynomial)
    lifted = _hensel_lift_root(
        coeffs, request.prime, request.root_mod_p, request.precision
    )
    return HenselRootResult._from_kernel(request, lifted)


def _poly_mul_exact_mod(
    a: Sequence[int],
    b: Sequence[int],
    modulus: int,
) -> list[int]:
    """Multiply two ascending polynomials, reducing each coefficient."""
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            result[i + j] = (result[i + j] + ai * bj) % modulus
    return result


def _poly_add_mod(a: Sequence[int], b: Sequence[int], modulus: int) -> list[int]:
    """Return ``a + b`` reduced modulo ``modulus`` (ascending)."""
    length = max(len(a), len(b))
    return [
        ((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)) % modulus
        for i in range(length)
    ]


def _poly_sub_mod(a: Sequence[int], b: Sequence[int], modulus: int) -> list[int]:
    """Return ``a - b`` reduced modulo ``modulus`` (ascending)."""
    length = max(len(a), len(b))
    return [
        ((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)) % modulus
        for i in range(length)
    ]


def _is_zero_polynomial(coefficients: Sequence[int]) -> bool:
    return all(coefficient == 0 for coefficient in coefficients)


def _poly_divmod_mod(
    dividend: Sequence[int], divisor: Sequence[int], p: int
) -> tuple[list[int], list[int]]:
    """Divide ``dividend`` by a nonzero ``divisor`` over GF(p).

    Returns ``(quotient, remainder)`` in ascending order, both trimmed.
    """
    if len(divisor) == 1 and divisor[0] == 0:
        raise ValueError("polynomial division by the zero polynomial")
    work = _trim_asc(dividend)
    divisor_degree = len(divisor) - 1
    quotient = [0] * max(len(work) - divisor_degree, 1)
    lead_inverse = pow(divisor[divisor_degree], -1, p)
    while len(work) - 1 >= divisor_degree and not (len(work) == 1 and work[0] == 0):
        shift = len(work) - 1 - divisor_degree
        factor = (work[-1] * lead_inverse) % p
        quotient[shift] = factor
        for i, coeff in enumerate(divisor):
            work[shift + i] = (work[shift + i] - factor * coeff) % p
        while len(work) > 1 and work[-1] == 0:
            work.pop()
    return _trim_asc(quotient), _trim_asc(work)


def _bezout_unit_mod_p(
    g_bar: Sequence[int], h_bar: Sequence[int], p: int
) -> tuple[list[int], list[int]]:
    """Compute ``(s, t)`` with ``s*g + t*h ≡ 1 (mod p)`` over GF(p)[x].

    Raises when the factors share a nonconstant common factor modulo
    ``p``; a nonzero constant greatest common divisor is normalized away
    so the returned relation has right-hand side exactly one.
    """
    r_prev = _trim_asc([coefficient % p for coefficient in g_bar])
    r_curr = _trim_asc([coefficient % p for coefficient in h_bar])
    s_prev: list[int] = [1]
    t_prev: list[int] = [0]
    s_curr: list[int] = [0]
    t_curr: list[int] = [1]
    while not (len(r_curr) == 1 and r_curr[0] == 0):
        quotient, remainder = _poly_divmod_mod(r_prev, r_curr, p)
        r_prev, r_curr = r_curr, remainder
        s_prev, s_curr = (
            s_curr,
            _poly_sub_mod(s_prev, _poly_mul_exact_mod(quotient, s_curr, p), p),
        )
        t_prev, t_curr = (
            t_curr,
            _poly_sub_mod(t_prev, _poly_mul_exact_mod(quotient, t_curr, p), p),
        )
    gcd = _trim_asc(r_prev)
    if len(gcd) != 1 or gcd[0] == 0:
        raise ValueError("factor_g and factor_h are not coprime mod p")
    unit_inverse = pow(gcd[0], -1, p)
    s = [coefficient * unit_inverse % p for coefficient in s_prev]
    t = [coefficient * unit_inverse % p for coefficient in t_prev]
    return s, t


def hensel_lift_factors(request: HenselFactorLiftRequest) -> HenselFactorLiftResult:
    """Lift a coprime factorization f ≡ g*h (mod p) to f ≡ g*h (mod p^k).

    Standard quadratic Hensel lifting: every step derives both factor
    corrections from the fixed Bézout relation ``s*g + t*h ≡ 1 (mod p)``,
    preserving the product congruence exactly. The reconstructed product
    is validated against ``f`` modulo ``p^k`` before returning.
    """
    f_asc = _kernel_coefficients(request.polynomial)
    g_asc = _kernel_coefficients(request.factor_g)
    h_asc = _kernel_coefficients(request.factor_h)
    p = request.prime
    k = request.precision

    g_bar = _trim_asc([coefficient % p for coefficient in g_asc])
    h_bar = _trim_asc([coefficient % p for coefficient in h_asc])
    if len(g_bar) == 1 and g_bar[0] == 0:
        raise ValueError("factor_g must be nonzero mod p")
    if len(h_bar) == 1 and h_bar[0] == 0:
        raise ValueError("factor_h must be nonzero mod p")
    f_bar = _trim_asc([coefficient % p for coefficient in f_asc])
    residue = _poly_sub_mod(f_bar, _poly_mul_exact_mod(g_bar, h_bar, p), p)
    if not _is_zero_polynomial(residue):
        raise ValueError("factor_g * factor_h is not congruent to polynomial mod p")

    bezout_s, bezout_t = _bezout_unit_mod_p(g_bar, h_bar, p)

    g = [coefficient % p for coefficient in g_bar]
    h = [coefficient % p for coefficient in h_bar]
    modulus = p
    for _ in range(1, k):
        next_modulus = modulus * p
        product = _poly_mul_exact_mod(g, h, next_modulus)
        error = _poly_sub_mod(f_asc, product, next_modulus)
        if any(coefficient % modulus != 0 for coefficient in error):
            raise ValueError("intermediate lift lost the product congruence")
        scaled = [(coefficient // modulus) % p for coefficient in error]
        # sigma corrects g and pairs with h in the correction identity
        # sigma*h_bar + tau*g_bar == scaled_error; reducing sigma modulo
        # g_bar (folding the quotient into tau) keeps deg(sigma) < deg(g)
        # so the lifted factors stay degree-bounded.
        sigma = [
            coefficient % p for coefficient in _poly_mul_exact_mod(bezout_t, scaled, p)
        ]
        tau = [
            coefficient % p for coefficient in _poly_mul_exact_mod(bezout_s, scaled, p)
        ]
        sigma_quotient, sigma_reduced = _poly_divmod_mod(sigma, g_bar, p)
        tau = _poly_add_mod(tau, _poly_mul_exact_mod(sigma_quotient, h_bar, p), p)
        g = [
            (g[i] if i < len(g) else 0)
            + modulus * (sigma_reduced[i] if i < len(sigma_reduced) else 0)
            for i in range(max(len(g), len(sigma_reduced)))
        ]
        h = [
            (h[i] if i < len(h) else 0) + modulus * (tau[i] if i < len(tau) else 0)
            for i in range(max(len(h), len(tau)))
        ]
        modulus = next_modulus

    lifted_g = _trim_asc([coefficient % modulus for coefficient in g])
    lifted_h = _trim_asc([coefficient % modulus for coefficient in h])
    reconstruction = _poly_mul_exact_mod(lifted_g, lifted_h, modulus)
    residue = _poly_sub_mod(f_asc, reconstruction, modulus)
    if not _is_zero_polynomial(residue):
        raise ValueError(
            "Hensel lifting failed to reproduce the polynomial "
            f"mod {modulus}; supply factors admitting a coprime lift"
        )

    return HenselFactorLiftResult(
        lifted_g=_wire_polynomial(lifted_g),
        lifted_h=_wire_polynomial(lifted_h),
        prime=p,
        precision=k,
    )


def find_padic_roots(request: PAdicRootsRequest) -> PAdicRootsResult:
    """Find every simple root of f(x) mod p^k via Hensel lifting.

    Roots are found mod p by brute force. Each residue with nonzero
    derivative lifts uniquely to an exact root modulo p^k. Residues whose
    derivative also vanishes are reported unlifted in ``multiple_residues``:
    their mod-p^k solution sets can grow unboundedly (x^2 has five roots mod
    25), so enumerating them would not be bounded.
    """
    coeffs = _kernel_coefficients(request.polynomial)
    p = request.prime
    k = request.precision

    lifted_roots: list[PAdicRootEntry] = []
    multiple_residues: list[int] = []
    for r in range(p):
        if _eval_poly(coeffs, r, p) != 0:
            continue
        if _eval_deriv(coeffs, r, p) % p == 0:
            multiple_residues.append(r)
            continue
        lifted = _hensel_lift_root(coeffs, p, r, k)
        lifted_roots.append(PAdicRootEntry(root=lifted, root_type="SIMPLE"))

    return PAdicRootsResult._from_kernel(
        request, tuple(lifted_roots), tuple(multiple_residues)
    )


__all__ = [
    "find_padic_roots",
    "hensel_lift_factors",
    "hensel_lift_root",
]
