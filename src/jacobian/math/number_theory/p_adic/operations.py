"""Domain-owned p-adic number theory operations."""

from __future__ import annotations

from collections.abc import Sequence
from math import isqrt

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.p_adic._models import (
    MAX_PRECISION,
    MAX_PRIME,
    HenselFactorLiftResult,
    HenselRootResult,
    IntegerPolynomial,
    PAdicRootEntry,
    PAdicRootsResult,
    _kernel_coefficients,
)

_MAX_PADIC_COEFFICIENTS = 64


def _domain_error(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"padic_arithmetic.{code}",
        message=message,
    )


def _require_prime(value: int) -> None:
    if (
        value < 2
        or value > MAX_PRIME
        or any(value % divisor == 0 for divisor in range(2, isqrt(value) + 1))
    ):
        _domain_error(("prime",), "prime_not_prime", "prime must be a prime modulus")


def _require_precision(precision: int) -> None:
    if not 1 <= precision <= MAX_PRECISION:
        _domain_error(
            ("precision",),
            "precision_out_of_range",
            f"precision must lie in 1..{MAX_PRECISION}",
        )


def _require_polynomial_budget(polynomial: IntegerPolynomial, field: str) -> None:
    if len(polynomial.coefficients) > _MAX_PADIC_COEFFICIENTS:
        _domain_error(
            (field, "coefficients"),
            "polynomial_budget",
            "p-adic polynomial exceeds the "
            f"{_MAX_PADIC_COEFFICIENTS}-coefficient operation budget",
        )


def _poly_eval_mod_p(coeffs: tuple[int, ...], x: int, p: int) -> int:
    result = 0
    for coeff in reversed(coeffs):
        result = (result * x + coeff) % p
    return result


def _poly_deriv_mod_p(coeffs: tuple[int, ...], x: int, p: int) -> int:
    if len(coeffs) <= 1:
        return 0
    return _poly_eval_mod_p(tuple(i * coeffs[i] for i in range(1, len(coeffs))), x, p)


def _admit_root(
    polynomial: IntegerPolynomial, prime: int, root_mod_p: int, precision: int
) -> tuple[int, ...]:
    _require_polynomial_budget(polynomial, "polynomial")
    _require_prime(prime)
    _require_precision(precision)
    if root_mod_p < 0 or root_mod_p >= prime:
        _domain_error(
            ("root_mod_p",), "root_out_of_range", "root_mod_p must be in 0..p-1"
        )
    coeffs = _kernel_coefficients(polynomial)
    if _poly_eval_mod_p(coeffs, root_mod_p, prime) != 0:
        _domain_error(
            ("root_mod_p",),
            "root_not_root",
            "root_mod_p must satisfy f(root_mod_p) = 0 mod p",
        )
    if _poly_deriv_mod_p(coeffs, root_mod_p, prime) % prime == 0:
        _domain_error(
            ("root_mod_p",),
            "root_not_simple",
            "Hensel lifting requires a simple root: f'(root_mod_p) must be nonzero mod p",
        )
    return coeffs


def _admit_factors(
    polynomial: IntegerPolynomial,
    factor_g: IntegerPolynomial,
    factor_h: IntegerPolynomial,
    prime: int,
    precision: int,
) -> None:
    for field, candidate in (
        ("polynomial", polynomial),
        ("factor_g", factor_g),
        ("factor_h", factor_h),
    ):
        _require_polynomial_budget(candidate, field)
    _require_prime(prime)
    _require_precision(precision)


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


def hensel_lift_root(
    polynomial: IntegerPolynomial,
    prime: int,
    root_mod_p: int,
    precision: int,
) -> HenselRootResult:
    """Normalize, admit, and lift one simple root of ``f`` modulo ``p``."""
    coeffs = _admit_root(polynomial, prime, root_mod_p, precision)
    lifted = _hensel_lift_root(coeffs, prime, root_mod_p, precision)
    return HenselRootResult._from_kernel(
        polynomial, prime, root_mod_p, precision, lifted
    )


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


def hensel_lift_factors(
    polynomial: IntegerPolynomial,
    factor_g: IntegerPolynomial,
    factor_h: IntegerPolynomial,
    prime: int,
    precision: int,
) -> HenselFactorLiftResult:
    """Lift a coprime factorization f ≡ g*h (mod p) to f ≡ g*h (mod p^k).

    Standard quadratic Hensel lifting: every step derives both factor
    corrections from the fixed Bézout relation ``s*g + t*h ≡ 1 (mod p)``,
    preserving the product congruence exactly. The reconstructed product
    is validated against ``f`` modulo ``p^k`` before returning.
    """
    _admit_factors(polynomial, factor_g, factor_h, prime, precision)
    f_asc = _kernel_coefficients(polynomial)
    g_asc = _kernel_coefficients(factor_g)
    h_asc = _kernel_coefficients(factor_h)
    p = prime
    k = precision

    g_bar = _trim_asc([coefficient % p for coefficient in g_asc])
    h_bar = _trim_asc([coefficient % p for coefficient in h_asc])
    if len(g_bar) == 1 and g_bar[0] == 0:
        _domain_error(
            ("factor_g",), "factor_zero_mod_prime", "factor_g must be nonzero mod p"
        )
    if len(h_bar) == 1 and h_bar[0] == 0:
        _domain_error(
            ("factor_h",), "factor_zero_mod_prime", "factor_h must be nonzero mod p"
        )
    f_bar = _trim_asc([coefficient % p for coefficient in f_asc])
    residue = _poly_sub_mod(f_bar, _poly_mul_exact_mod(g_bar, h_bar, p), p)
    if not _is_zero_polynomial(residue):
        _domain_error(
            ("factor_g", "factor_h"),
            "factorization_not_congruent",
            "factor_g * factor_h is not congruent to polynomial mod p",
        )

    try:
        bezout_s, bezout_t = _bezout_unit_mod_p(g_bar, h_bar, p)
    except ValueError as exc:
        _domain_error(
            ("factor_g", "factor_h"),
            "factors_not_coprime",
            str(exc),
        )

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


def find_padic_roots(
    polynomial: IntegerPolynomial,
    prime: int,
    precision: int,
) -> PAdicRootsResult:
    """Find every simple root of f(x) mod p^k via Hensel lifting.

    Roots are found mod p by brute force. Each residue with nonzero
    derivative lifts uniquely to an exact root modulo p^k. Residues whose
    derivative also vanishes are reported unlifted in ``multiple_residues``:
    their mod-p^k solution sets can grow unboundedly (x^2 has five roots mod
    25), so enumerating them would not be bounded.
    """
    _require_polynomial_budget(polynomial, "polynomial")
    _require_prime(prime)
    _require_precision(precision)
    coeffs = _kernel_coefficients(polynomial)
    p = prime
    k = precision

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
        polynomial, prime, precision, tuple(lifted_roots), tuple(multiple_residues)
    )


__all__ = [
    "find_padic_roots",
    "hensel_lift_factors",
    "hensel_lift_root",
]
