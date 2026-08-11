"""Exact rational univariate polynomial interval enclosure contracts.

The contracts describe one atomic mathematical outcome: an exact rational
enclosure of the values of one univariate rational polynomial on one closed
rational interval, derived from the Bernstein-coefficient bound.

The enclosure is a valid superset of the polynomial's range on the interval.
It is not, in general, the exact range: the Bernstein-coefficient bound equals
the exact range only in special cases (for example, degree at most one, or when
the extremal values are attained at the interval endpoints with monotone
behaviour). Contracts carry ``enclosure_kind`` and ``range_exactness`` so that
callers cannot mistake a valid bound for the exact image.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.common import ArtifactUri, CheckerUri
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.polynomials import (
    PolynomialVariable,
    SparseRationalPolynomial,
)
from jacobian.contracts.results import ContractModel

_MAX_DEGREE = 64
_MAX_TERMS = 1024


class RationalInterval(ContractModel):
    interval_schema_version: Literal["1"] = "1"
    lo: CanonicalRational
    hi: CanonicalRational

    @model_validator(mode="after")
    def require_nonempty_closed_interval(self) -> Self:
        if self.lo.as_fraction() >= self.hi.as_fraction():
            raise ValueError(
                "rational interval must satisfy lo < hi; a degenerate or "
                "reversed interval is not a valid enclosure scope"
            )
        return self


class UnivariateRationalPolynomial(ContractModel):
    polynomial_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    variable: PolynomialVariable
    polynomial: SparseRationalPolynomial

    @model_validator(mode="after")
    def require_univariate_and_bounded(self) -> Self:
        if any(len(term.exponents) != 1 for term in self.polynomial.terms):
            raise ValueError(
                "every term of a univariate polynomial must use a one-dimensional "
                "exponent tuple"
            )
        if any(
            term.exponents[0] < 0 or term.exponents[0] > _MAX_DEGREE
            for term in self.polynomial.terms
        ):
            raise ValueError(
                "univariate polynomial exponents must lie between zero and the "
                "bounded degree limit"
            )
        if len(self.polynomial.terms) > _MAX_TERMS:
            raise ValueError("univariate polynomial term limit exceeded")
        return self

    @property
    def degree(self) -> int:
        if not self.polynomial.terms:
            return 0
        return max(term.exponents[0] for term in self.polynomial.terms)


class PolynomialIntervalEnclosureRequest(ContractModel):
    polynomial: UnivariateRationalPolynomial
    interval: RationalInterval


class PolynomialIntervalEnclosure(ContractModel):
    enclosure_schema_version: Literal["1"] = "1"
    polynomial_uri: ArtifactUri
    interval: RationalInterval
    degree: int = Field(ge=0, le=_MAX_DEGREE)
    bernstein_coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=_MAX_DEGREE + 1,
    )
    lo: CanonicalRational
    hi: CanonicalRational
    backend: Literal["sympy"] = "sympy"
    backend_version: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_bernstein_consistency(self) -> Self:
        if len(self.bernstein_coefficients) != self.degree + 1:
            raise ValueError(
                "the number of Bernstein coefficients must equal degree + 1"
            )
        values = [c.as_fraction() for c in self.bernstein_coefficients]
        if self.lo.as_fraction() != min(values):
            raise ValueError(
                "enclosure lower bound must equal the minimum Bernstein coefficient"
            )
        if self.hi.as_fraction() != max(values):
            raise ValueError(
                "enclosure upper bound must equal the maximum Bernstein coefficient"
            )
        return self


class PolynomialIntervalEnclosureClaim(ContractModel):
    claim_schema_version: Literal["1"] = "1"
    predicate: Literal["POLYNOMIAL_INTERVAL_BERNSTEIN_ENCLOSURE"] = (
        "POLYNOMIAL_INTERVAL_BERNSTEIN_ENCLOSURE"
    )
    domain: Literal["QQ"] = "QQ"
    polynomial_uri: ArtifactUri
    interval: RationalInterval


class PolynomialIntervalEnclosureReplay(ContractModel):
    method: Literal["BERNSTEIN_COEFFICIENT_REPLAY"] = "BERNSTEIN_COEFFICIENT_REPLAY"
    polynomial_uri: ArtifactUri
    interval: RationalInterval
    degree: int = Field(ge=0, le=_MAX_DEGREE)
    bernstein_coefficients: tuple[CanonicalRational, ...]
    lo: CanonicalRational
    hi: CanonicalRational


class PolynomialIntervalEnclosureVerifyRequest(ContractModel):
    polynomial: UnivariateRationalPolynomial
    interval: RationalInterval
    claimed_bernstein_coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=_MAX_DEGREE + 1,
    )
    claimed_lo: CanonicalRational
    claimed_hi: CanonicalRational

    @model_validator(mode="after")
    def require_claimed_enclosure_consistency(self) -> Self:
        degree = self.polynomial.degree
        if len(self.claimed_bernstein_coefficients) != degree + 1:
            raise ValueError(
                "claimed Bernstein coefficient count must equal degree + 1"
            )
        values = [c.as_fraction() for c in self.claimed_bernstein_coefficients]
        if self.claimed_lo.as_fraction() != min(values):
            raise ValueError(
                "claimed lower bound must equal the minimum claimed coefficient"
            )
        if self.claimed_hi.as_fraction() != max(values):
            raise ValueError(
                "claimed upper bound must equal the maximum claimed coefficient"
            )
        return self


class PolynomialIntervalEnclosureOutput(ContractModel):
    polynomial_uri: ArtifactUri
    enclosure_uri: ArtifactUri
    interval: RationalInterval
    degree: int = Field(ge=0, le=_MAX_DEGREE)
    bernstein_coefficients: tuple[CanonicalRational, ...]
    lo: CanonicalRational
    hi: CanonicalRational
    enclosure_kind: Literal["BERNSTEIN_COEFFICIENT_BOUND"] = (
        "BERNSTEIN_COEFFICIENT_BOUND"
    )
    range_exactness: Literal["ENCLOSURE_VALID_NOT_EXACT"] = "ENCLOSURE_VALID_NOT_EXACT"
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"
    certificate_available: Literal[False] = False
    checker_id: None = None
    backend: Literal["sympy"] = "sympy"
    backend_version: str = Field(min_length=1, max_length=64)


class PolynomialIntervalEnclosureVerifyOutput(ContractModel):
    polynomial_uri: ArtifactUri
    enclosure_uri: ArtifactUri
    claim_uri: ArtifactUri
    certificate_uri: ArtifactUri
    verification_record_uri: ArtifactUri | None = None
    checker_id: CheckerUri | None = None
    interval: RationalInterval
    degree: int = Field(ge=0, le=_MAX_DEGREE)
    bernstein_coefficients: tuple[CanonicalRational, ...]
    lo: CanonicalRational
    hi: CanonicalRational
    enclosure_kind: Literal["BERNSTEIN_COEFFICIENT_BOUND"] = (
        "BERNSTEIN_COEFFICIENT_BOUND"
    )
    range_exactness: Literal["ENCLOSURE_VALID_NOT_EXACT"] = "ENCLOSURE_VALID_NOT_EXACT"
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    enclosure_assurance: Literal["COMPUTED", "VERIFIED"]
    conclusion: Literal["TRUE", "FALSE", "UNKNOWN"]

    @model_validator(mode="after")
    def preserve_truth_and_assurance(self) -> Self:
        if self.conclusion == "UNKNOWN" and self.enclosure_assurance == "VERIFIED":
            raise ValueError("an unknown conclusion cannot carry verified assurance")
        if self.enclosure_assurance == "VERIFIED" and (
            self.verification_record_uri is None
            or self.checker_id is None
            or self.conclusion == "UNKNOWN"
        ):
            raise ValueError(
                "verified enclosure assurance requires a decisive checker-backed "
                "record and checker identity"
            )
        if (
            self.enclosure_assurance != "VERIFIED"
            and self.verification_record_uri is not None
        ):
            raise ValueError(
                "a verification record requires checker-verified enclosure assurance"
            )
        return self
