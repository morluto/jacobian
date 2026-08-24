"""Domain-owned real algebra operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.real_algebra import root_count, sturm_chain
from jacobian.math.polynomials.real_algebra._models import (
    PolynomialTerm,
    RootCountRequest,
    RootCountResult,
    SturmChainRequest,
    SturmChainResult,
    UnivariatePolynomial,
)
from jacobian.math.polynomials.real_algebra._strict_sublevel import (
    compute_strict_sublevel_payload,
)
from jacobian.math.polynomials.real_algebra._strict_sublevel_models import (
    StrictSublevelMeasureRequest,
    StrictSublevelMeasureResult,
)


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
    terms = _poly_to_terms(request.polynomial)
    lower = request.lower.as_fraction()
    upper = request.upper.as_fraction()
    count = root_count(terms, lower, upper)
    return RootCountResult(
        root_count=count,
        lower=request.lower,
        upper=request.upper,
    )


def compute_strict_sublevel_measure(
    request: StrictSublevelMeasureRequest,
) -> StrictSublevelMeasureResult:
    payload = compute_strict_sublevel_payload(request)
    # The payload is built entirely from the already validated request and the
    # exact kernel. External result validation still replays the source once;
    # the trusted producer must not repeat root isolation merely to accept its
    # own derived fields.
    return StrictSublevelMeasureResult.model_construct(
        source_polynomial=request.polynomial,
        threshold=request.threshold,
        lower=request.lower,
        upper=request.upper,
        components=payload.components,
        measure=payload.measure,
    )
