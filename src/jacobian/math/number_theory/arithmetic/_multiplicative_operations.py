"""Exact integer multiplicative normal-form operations."""

from __future__ import annotations

from math import gcd as math_gcd

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory.arithmetic._multiplicative_forms import (
    IntegerKRequest,
    IntegerRequest,
    KFreeDecompositionResult,
    NonnegativeIntegerRequest,
    NormalizedQuadraticRadicalResult,
    PerfectPowerProfileResult,
    PrimeExponentRow,
    SquarefreeDecompositionResult,
)


def _factorize_abs(value: int) -> list[tuple[int, int]]:
    """Return the complete prime factorization of |value| as sorted [(prime, exponent)]."""
    from sympy import factorint

    return sorted(factorint(abs(value)).items())


def compute_perfect_power_profile(
    request: IntegerRequest,
) -> PerfectPowerProfileResult:
    """Compute the maximal perfect-power profile of one integer."""

    n = parse_canonical_integer(request.value)

    if n == 0:
        return PerfectPowerProfileResult(kind="ZERO")
    if n == 1:
        return PerfectPowerProfileResult(kind="POSITIVE_UNIT")
    if n == -1:
        return PerfectPowerProfileResult(kind="NEGATIVE_UNIT")

    prime_exps = _factorize_abs(n)
    all_exponents = [exp for _, exp in prime_exps]

    g = all_exponents[0]
    for exp in all_exponents[1:]:
        g = math_gcd(g, exp)

    if n > 0:
        max_exp = g
        from sympy import integer_nthroot

        root, _ = integer_nthroot(abs(n), max_exp)
        base_val = root
        reconstruction = base_val**max_exp
        assert reconstruction == abs(n)
    else:
        # Negative: maximal exponent = largest odd divisor of g
        max_exp = g
        while max_exp % 2 == 0 and max_exp > 1:
            max_exp //= 2
        from sympy import integer_nthroot

        root, _ = integer_nthroot(abs(n), max_exp)
        base_val = -root
        reconstruction = base_val**max_exp
        assert reconstruction == n

    return PerfectPowerProfileResult(
        kind="NONUNIT",
        base=format_canonical_integer(base_val),
        exponent=max_exp,
        is_nontrivial_perfect_power=max_exp > 1,
        factors=tuple(
            PrimeExponentRow(
                prime=format_canonical_integer(p),
                power=e,
            )
            for p, e in prime_exps
        ),
        reconstruction=format_canonical_integer(n),
    )


def compute_k_free_decomposition(
    request: IntegerKRequest,
) -> KFreeDecompositionResult:
    """Compute the unique decomposition n = a^k * c with c k-th-power-free."""

    n = parse_canonical_integer(request.value)
    k = request.k

    if n == 0:
        return KFreeDecompositionResult(kind="ZERO")
    if n == 1 or n == -1:
        return KFreeDecompositionResult(kind="UNIT")

    prime_exps = _factorize_abs(n)

    a_val = 1
    c_sign = 1 if n > 0 else -1
    c_abs = 1

    rows: list[PrimeExponentRow] = []

    for prime, exp in prime_exps:
        q, r = divmod(exp, k)
        if q > 0:
            a_val *= prime**q
        if r > 0:
            c_abs *= prime**r
        rows.append(PrimeExponentRow(prime=format_canonical_integer(prime), power=exp))

    c_val = c_sign * c_abs
    reconstruction = (a_val**k) * c_val
    assert reconstruction == n

    return KFreeDecompositionResult(
        kind="NONUNIT",
        base=format_canonical_integer(a_val),
        cofactor=format_canonical_integer(c_val),
        factors=tuple(rows),
        reconstruction=format_canonical_integer(n),
    )


def compute_squarefree_decomposition(
    request: IntegerRequest,
) -> SquarefreeDecompositionResult:
    """Compute the unique decomposition n = s^2 * d with |d| squarefree."""

    n = parse_canonical_integer(request.value)

    if n == 0:
        return SquarefreeDecompositionResult(kind="ZERO")
    if n == 1 or n == -1:
        return SquarefreeDecompositionResult(kind="UNIT")

    prime_exps = _factorize_abs(n)

    s_val = 1
    c_sign = 1 if n > 0 else -1
    d_abs = 1

    rows: list[PrimeExponentRow] = []

    for prime, exp in prime_exps:
        q, r = divmod(exp, 2)
        if q > 0:
            s_val *= prime**q
        if r > 0:
            d_abs *= prime
        rows.append(PrimeExponentRow(prime=format_canonical_integer(prime), power=exp))

    d_val = c_sign * d_abs
    reconstruction = (s_val**2) * d_val
    assert reconstruction == n

    return SquarefreeDecompositionResult(
        kind="NONUNIT",
        square_factor=format_canonical_integer(s_val),
        squarefree_part=format_canonical_integer(d_val),
        factors=tuple(rows),
        reconstruction=format_canonical_integer(n),
    )


def compute_normalized_quadratic_radical(
    request: NonnegativeIntegerRequest,
) -> NormalizedQuadraticRadicalResult:
    """Compute the canonical positive sqrt(n) = s * sqrt(d) with d squarefree."""

    n = parse_canonical_integer(request.value)

    if n == 0:
        return NormalizedQuadraticRadicalResult(
            kind="ZERO",
            coefficient="0",
            radicand="1",
            reconstruction="0",
        )

    if n == 1:
        return NormalizedQuadraticRadicalResult(
            kind="RATIONAL_INTEGER",
            coefficient="1",
            radicand="1",
            reconstruction="1",
        )

    prime_exps = _factorize_abs(n)

    s_val = 1
    d_val = 1

    for prime, exp in prime_exps:
        q, r = divmod(exp, 2)
        if q > 0:
            s_val *= prime**q
        if r > 0:
            d_val *= prime

    assert s_val**2 * d_val == n

    return NormalizedQuadraticRadicalResult(
        kind="RATIONAL_INTEGER" if d_val == 1 else "IRRATIONAL_QUADRATIC",
        coefficient=format_canonical_integer(s_val),
        radicand=format_canonical_integer(d_val),
        reconstruction=format_canonical_integer(n),
    )
