"""Exact p-adic valuation profiles of binomial coefficients via Kummer's theorem."""

from __future__ import annotations

from jacobian.math.number_theory._binomial_valuation_models import (
    BinomialValuationProfileRequest,
    BinomialValuationProfileResult,
    BinomialValuationProfileRow,
)


def _count_carries(n: int, k: int, p: int) -> int:
    """Count carries when adding k and (n-k) in base p (Kummer's theorem)."""
    a = k
    b = n - k
    carries = 0
    carry = 0
    while a > 0 or b > 0 or carry > 0:
        da = a % p
        db = b % p
        s = da + db + carry
        if s >= p:
            carries += 1
            carry = 1
        else:
            carry = 0
        a //= p
        b //= p
    return carries


def compute_binomial_valuation_profile(
    request: BinomialValuationProfileRequest,
) -> BinomialValuationProfileResult:
    """Compute v_p(C(n,k)) for all k from 0 to n using Kummer's theorem.

    v_p(C(n,k)) = number of carries when adding k and (n-k) in base p.
    """
    n = request.n
    p = request.prime
    rows = []
    for k in range(n + 1):
        valuation = _count_carries(n, k, p)
        rows.append(BinomialValuationProfileRow(k=k, valuation=valuation))
    return BinomialValuationProfileResult(n=n, prime=p, rows=rows)


__all__ = ["compute_binomial_valuation_profile"]
