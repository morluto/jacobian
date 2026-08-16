"""Domain adapter for certified factoring."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.contracts.certified_factoring import (
    CertifiedFactorRequest,
    CertifiedFactorResult,
)


def compute_certified_factor(request: CertifiedFactorRequest) -> CertifiedFactorResult:
    from sympy import factorint

    n = parse_canonical_integer(request.n)
    factors = factorint(n)
    return CertifiedFactorResult(
        factors=tuple(
            (format_canonical_integer(base), exp) for base, exp in factors.items()
        )
    )
