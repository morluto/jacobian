"""Exact rational univariate polynomial strict positivity contracts.

The contracts describe one atomic mathematical outcome: an exact decision
whether one univariate rational polynomial is strictly positive on one closed
rational interval. The decision uses Sturm's theorem, which counts the
distinct real roots of the polynomial in the interval using only exact
rational arithmetic.

The decision is EXACT: it is a theorem, not a heuristic or enclosure. The
contract distinguishes ``positive`` (p(x) > 0 for all x in [a,b]) from
non-strict positivity (p >= 0). A polynomial that touches zero at any point
in the interval is NOT strictly positive.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.common import ArtifactUri, CheckerUri
from jacobian.contracts.polynomial_intervals import (
    RationalInterval,
    UnivariateRationalPolynomial,
)
from jacobian.contracts.polynomials import SparseRationalPolynomial
from jacobian.contracts.results import ContractModel

_MAX_DEGREE = 64


class PolynomialIntervalPositivityRequest(ContractModel):
    polynomial: UnivariateRationalPolynomial
    interval: RationalInterval


class PolynomialIntervalPositivityDecision(ContractModel):
    decision_schema_version: Literal["1"] = "1"
    polynomial_uri: ArtifactUri
    interval: RationalInterval
    degree: int = Field(ge=0, le=_MAX_DEGREE)
    sturm_sequence: tuple[SparseRationalPolynomial, ...] = Field(min_length=1)
    sign_changes_at_lo: int = Field(ge=0)
    sign_changes_at_hi: int = Field(ge=0)
    roots_in_open_interval: int = Field(ge=0)
    endpoint_root: bool
    positive: bool
    backend: Literal["sympy"] = "sympy"
    backend_version: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_root_count_consistency(self) -> Self:
        if (
            self.roots_in_open_interval
            != self.sign_changes_at_lo - self.sign_changes_at_hi
        ):
            raise ValueError(
                "roots_in_open_interval must equal sign_changes_at_lo minus "
                "sign_changes_at_hi"
            )
        return self

    @model_validator(mode="after")
    def require_positivity_consistency(self) -> Self:
        if self.positive and self.endpoint_root:
            raise ValueError(
                "a strictly positive polynomial cannot vanish at the left endpoint"
            )
        if self.positive and self.roots_in_open_interval > 0:
            raise ValueError(
                "a strictly positive polynomial cannot have roots in the interval"
            )
        return self


class PolynomialIntervalPositivityClaim(ContractModel):
    claim_schema_version: Literal["1"] = "1"
    predicate: Literal["POLYNOMIAL_INTERVAL_STRICT_POSITIVITY"] = (
        "POLYNOMIAL_INTERVAL_STRICT_POSITIVITY"
    )
    domain: Literal["QQ"] = "QQ"
    polynomial_uri: ArtifactUri
    interval: RationalInterval
    positive: bool


class PolynomialIntervalPositivityReplay(ContractModel):
    method: Literal["STURM_SEQUENCE_REPLAY"] = "STURM_SEQUENCE_REPLAY"
    polynomial_uri: ArtifactUri
    interval: RationalInterval
    degree: int = Field(ge=0, le=_MAX_DEGREE)
    sturm_sequence_length: int = Field(ge=1)
    sign_changes_at_lo: int = Field(ge=0)
    sign_changes_at_hi: int = Field(ge=0)
    roots_in_open_interval: int = Field(ge=0)
    endpoint_root: bool
    positive: bool


class PolynomialIntervalPositivityOutput(ContractModel):
    polynomial_uri: ArtifactUri
    decision_uri: ArtifactUri
    interval: RationalInterval
    degree: int = Field(ge=0, le=_MAX_DEGREE)
    sign_changes_at_lo: int = Field(ge=0)
    sign_changes_at_hi: int = Field(ge=0)
    roots_in_open_interval: int = Field(ge=0)
    endpoint_root: bool
    positive: bool


class PolynomialIntervalPositivityVerifyRequest(ContractModel):
    polynomial: UnivariateRationalPolynomial
    interval: RationalInterval
    claimed_positive: bool
    claimed_sign_changes_at_lo: int = Field(ge=0)
    claimed_sign_changes_at_hi: int = Field(ge=0)
    claimed_roots_in_open_interval: int = Field(ge=0)
    claimed_endpoint_root: bool


class PolynomialIntervalPositivityVerifyOutput(ContractModel):
    polynomial_uri: ArtifactUri
    decision_uri: ArtifactUri
    claim_uri: ArtifactUri
    certificate_uri: ArtifactUri
    verification_record_uri: ArtifactUri | None = None
    checker_id: CheckerUri | None = None
    interval: RationalInterval
    degree: int = Field(ge=0, le=_MAX_DEGREE)
    positive: bool
    sign_changes_at_lo: int = Field(ge=0)
    sign_changes_at_hi: int = Field(ge=0)
    roots_in_open_interval: int = Field(ge=0)
    endpoint_root: bool
    conclusion: Literal["TRUE", "FALSE", "UNKNOWN"]

    @model_validator(mode="after")
    def bind_verification_record(self) -> Self:
        if self.verification_record_uri is not None and (
            self.checker_id is None or self.conclusion == "UNKNOWN"
        ):
            raise ValueError(
                "a positivity verification record requires a decisive conclusion "
                "and checker identity"
            )
        return self
