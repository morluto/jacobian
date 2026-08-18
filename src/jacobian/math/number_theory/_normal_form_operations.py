"""Worker-safe kernels for integer multiplicative normal-form operations.

Each operation consumes one or two bounded integers, obtains a complete exact
prime factorization of ``|n|`` via SymPy's ``factorint``, and then derives
the canonical normal form deterministically.  The factorization is an
implementation detail; all public results are Jacobian-owned integers and
factorization rows.
"""

from __future__ import annotations

import math
from typing import Literal

from jacobian.math.number_theory._normal_forms import (
    KFreeDecompositionRequest,
    KFreeDecompositionResult,
    MaximalPerfectPowerResult,
    NormalizedQuadraticRadicalResult,
    PerfectPowerProfileRequest,
    PrimeExponentParityRow,
    PrimeExponentRow,
    PrimeQuotientRemainderRow,
    QuadraticRadicalNormalizeRequest,
    SquarefreeDecompositionRequest,
    SquarefreeDecompositionResult,
)


def _factorint(n: int) -> dict[int, int]:
    """Return the complete prime factorization of |n| as {prime: exponent}."""
    from sympy import factorint

    return {int(p): int(e) for p, e in factorint(abs(n)).items()}


def _gcd_all(exponents: list[int]) -> int:
    """Gcd of a list of positive integers."""
    result = exponents[0]
    for e in exponents[1:]:
        result = math.gcd(result, e)
    return result


# ---------------------------------------------------------------------------
# Perfect-power profile
# ---------------------------------------------------------------------------


def perfect_power_profile(
    request: PerfectPowerProfileRequest,
) -> MaximalPerfectPowerResult:
    """Compute the maximal perfect-power profile of one integer.

    For |n| > 1, the maximal exponent is gcd of all prime exponents.
    For negative n, only odd exponents are admissible, so the maximal exponent
    is the largest odd divisor of the gcd.
    """
    n = int(request.value)

    if n == 0:
        return MaximalPerfectPowerResult(source=request.value, classification="ZERO")
    if n == 1:
        return MaximalPerfectPowerResult(
            source=request.value, classification="POSITIVE_UNIT"
        )
    if n == -1:
        return MaximalPerfectPowerResult(
            source=request.value, classification="NEGATIVE_UNIT"
        )

    factors = _factorint(n)
    sorted_factors = sorted(factors.items())
    primes_exp = [(p, e) for p, e in sorted_factors]
    exponents = [e for _, e in primes_exp]
    g = _gcd_all(exponents)

    if n < 0:
        # For negative n, only odd exponents are admissible
        while g % 2 == 0 and g > 1:
            g //= 2
        if g == 0:
            g = 1

    exponent = g
    # Compute base: sign(n) * product(p^(e/exponent))
    base_val = 1
    for p, e in primes_exp:
        base_val *= p ** (e // exponent)
    if n < 0:
        base_val = -base_val

    factor_rows = tuple(
        PrimeExponentRow(prime=str(p), exponent=e) for p, e in primes_exp
    )

    return MaximalPerfectPowerResult(
        source=request.value,
        classification="NONUNIT",
        base=str(base_val),
        exponent=exponent,
        is_nontrivial_perfect_power=exponent > 1,
        factors=factor_rows,
    )


# ---------------------------------------------------------------------------
# K-free decomposition
# ---------------------------------------------------------------------------


def k_free_decomposition(
    request: KFreeDecompositionRequest,
) -> KFreeDecompositionResult:
    """Compute the canonical k-free decomposition: n = a^k * c.

    For nonzero n: a >= 1, c has the same sign as n, and no prime to the k-th
    power divides |c|.
    """
    n = int(request.value)
    k = request.k

    if n == 0:
        return KFreeDecompositionResult(
            source=request.value, k=k, classification="ZERO"
        )

    sign = -1 if n < 0 else 1
    abs_n = abs(n)
    factors = _factorint(abs_n)
    sorted_factors = sorted(factors.items())

    a = 1
    c_abs = 1
    rows: list[PrimeQuotientRemainderRow] = []

    for p, e in sorted_factors:
        q, r = divmod(e, k)
        a *= p**q
        if r > 0:
            c_abs *= p**r
        rows.append(PrimeQuotientRemainderRow(prime=str(p), quotient=q, remainder=r))

    c = sign * c_abs

    return KFreeDecompositionResult(
        source=request.value,
        k=k,
        classification="NONZERO",
        extracted_base=str(a),
        k_free_cofactor=str(c),
        factor_rows=tuple(rows),
    )


# ---------------------------------------------------------------------------
# Squarefree decomposition
# ---------------------------------------------------------------------------


def squarefree_decomposition(
    request: SquarefreeDecompositionRequest,
) -> SquarefreeDecompositionResult:
    """Compute the squarefree decomposition: n = s^2 * d.

    For nonzero n: s >= 1, d has the same sign as n, |d| is squarefree.
    """
    n = int(request.value)

    if n == 0:
        return SquarefreeDecompositionResult(
            source=request.value, classification="ZERO"
        )

    sign = -1 if n < 0 else 1
    abs_n = abs(n)
    factors = _factorint(abs_n)
    sorted_factors = sorted(factors.items())

    s = 1
    d_abs = 1
    parity_rows: list[PrimeExponentParityRow] = []

    for p, e in sorted_factors:
        half, rem = divmod(e, 2)
        s *= p**half
        if rem > 0:
            d_abs *= p
        parity_rows.append(PrimeExponentParityRow(prime=str(p), exponent=e, parity=rem))

    d = sign * d_abs

    return SquarefreeDecompositionResult(
        source=request.value,
        classification="NONZERO",
        square_factor=str(s),
        signed_squarefree_part=str(d),
        parity_rows=tuple(parity_rows),
    )


# ---------------------------------------------------------------------------
# Normalized positive quadratic radical
# ---------------------------------------------------------------------------


def normalize_positive_quadratic_radical(
    request: QuadraticRadicalNormalizeRequest,
) -> NormalizedQuadraticRadicalResult:
    """Normalize sqrt(n) = s * sqrt(d) for nonnegative integer n.

    For n = 0: s = 0, d = 1.
    For n > 0: n = s^2 * d from squarefree decomposition, d >= 1 squarefree.
    """
    n = int(request.value)

    if n == 0:
        return NormalizedQuadraticRadicalResult(
            source=request.value,
            coefficient="0",
            radicand="1",
            classification="ZERO",
        )

    factors = _factorint(n)
    sorted_factors = sorted(factors.items())

    s = 1
    d = 1

    for p, e in sorted_factors:
        half, rem = divmod(e, 2)
        s *= p**half
        if rem > 0:
            d *= p

    if d == 1:
        classification: Literal["ZERO", "RATIONAL_INTEGER", "IRRATIONAL_QUADRATIC"] = (
            "RATIONAL_INTEGER"
        )
    else:
        classification = "IRRATIONAL_QUADRATIC"

    return NormalizedQuadraticRadicalResult(
        source=request.value,
        coefficient=str(s),
        radicand=str(d),
        classification=classification,
    )
