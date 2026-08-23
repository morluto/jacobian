"""Domain-owned p-adic number theory operations."""

from __future__ import annotations

from jacobian.math.padic_arithmetic._models import (
    HenselFactorLiftRequest,
    HenselFactorLiftResult,
    HenselRootRequest,
    HenselRootResult,
    IntegerPolynomial,
    PAdicRootEntry,
    PAdicRootsRequest,
    PAdicRootsResult,
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
    coeffs = request.polynomial.coefficients
    lifted = _hensel_lift_root(
        coeffs, request.prime, request.root_mod_p, request.precision
    )
    return HenselRootResult(
        lifted_root=lifted,
        prime=request.prime,
        precision=request.precision,
        is_simple_root=True,
    )


def _poly_mul_mod(
    a: tuple[int, ...],
    b: tuple[int, ...],
    modulus: int,
) -> tuple[int, ...]:
    """Multiply two polynomials modulo modulus."""
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] = (result[i + j] + ai * bj) % modulus
    while len(result) > 1 and result[-1] % modulus == 0:
        result.pop()
    return tuple(result)


def hensel_lift_factors(request: HenselFactorLiftRequest) -> HenselFactorLiftResult:
    """Lift a coprime factorization f ≡ g*h (mod p) to f ≡ g*h (mod p^k)."""
    f = request.polynomial.coefficients
    g = list(request.factor_g.coefficients)
    h = list(request.factor_h.coefficients)
    p = request.prime
    k = request.precision

    for _ in range(1, k):
        modulus = p ** (_ + 1)
        f_mod = tuple(c % modulus for c in f)
        g_mod = [c % modulus for c in g]
        h_mod = [c % modulus for c in h]
        product = _poly_mul_mod(tuple(g_mod), tuple(h_mod), modulus)
        diff = [(f_mod[i] - product[i]) % modulus if i < len(product) else f_mod[i] % modulus
                for i in range(len(f_mod))]
        for i, d in enumerate(diff):
            if i < len(g):
                g[i] = g[i] + d
        for i in range(len(g), len(f)):
            if i < len(g):
                g[i] = g[i]

    return HenselFactorLiftResult(
        lifted_g=IntegerPolynomial(coefficients=tuple(c % (p**k) for c in g)),
        lifted_h=IntegerPolynomial(coefficients=tuple(c % (p**k) for c in h)),
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
    coeffs = request.polynomial.coefficients
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

    return PAdicRootsResult(
        polynomial=request.polynomial,
        roots=tuple(lifted_roots),
        prime=p,
        precision=k,
        root_count=len(lifted_roots),
        multiple_residues=tuple(multiple_residues),
    )


__all__ = [
    "find_padic_roots",
    "hensel_lift_factors",
    "hensel_lift_root",
]
