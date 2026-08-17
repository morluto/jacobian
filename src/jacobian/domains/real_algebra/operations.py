"""Domain adapter for real algebra operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.real_algebra import (
    PolynomialTerm,
    RootCountRequest,
    RootCountResult,
    SturmChainRequest,
    SturmChainResult,
    UnivariatePolynomial,
)
from jacobian.math.real_algebra import root_count, sturm_chain


def _poly_to_terms(poly: UnivariatePolynomial) -> list[tuple[Fraction, int]]:
    return [(t.coefficient.as_fraction(), t.exponent) for t in poly.terms]


def _terms_to_poly(terms: list[tuple[Fraction, int]]) -> UnivariatePolynomial:
    return UnivariatePolynomial(
        terms=tuple(
            PolynomialTerm(
                coefficient=CanonicalRational.from_fraction(coeff),
                exponent=exp,
            )
            for coeff, exp in terms
            if coeff != 0
        )
    )


def compute_sturm_chain(request: SturmChainRequest) -> SturmChainResult:
    poly = request.polynomial
    terms = _poly_to_terms(poly)
    chain = sturm_chain(terms)
    degree = max(t.exponent for t in poly.terms)
    return SturmChainResult(
        chain=tuple(_terms_to_poly(c) for c in chain),
        degree=degree,
    )


def compute_root_count(request: RootCountRequest) -> RootCountResult:
    poly = request.polynomial
    terms = _poly_to_terms(poly)
    lower = request.lower.as_fraction()
    upper = request.upper.as_fraction()
    count = root_count(terms, lower, upper)
    return RootCountResult(
        root_count=count,
        lower=request.lower,
        upper=request.upper,
    )
