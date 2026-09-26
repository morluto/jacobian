"""Wire contracts for bounded Laurent arithmetic."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.local_series.puiseux_values import (
    TruncatedPuiseuxWindow,
)
from jacobian.math.polynomials.local_series.values import (
    MAX_LOCAL_SERIES_EXPONENT,
    MAX_LOCAL_SERIES_POWER_EXPONENT,
    TruncatedLaurentWindow,
)
from jacobian.math.polynomials.series._models import TruncatedSeries
from jacobian.math.polynomials.values import RationalFunction


class RationalFunctionExpansionRequest(StrictModel):
    """Expand one univariate QQ rational function at a finite rational point."""

    function: RationalFunction
    center: CanonicalRational
    precision: StrictInt = Field(
        ge=1,
        le=MAX_LOCAL_SERIES_EXPONENT,
        description="Exclusive exponent cutoff in t = x - center.",
    )


class RationalFunctionInfinityExpansionRequest(StrictModel):
    """Expand one univariate QQ rational function at infinity using t = 1/x."""

    function: RationalFunction
    precision: StrictInt = Field(
        ge=1,
        le=MAX_LOCAL_SERIES_EXPONENT,
        description="Exclusive exponent cutoff in the reciprocal parameter t = 1/x.",
    )


class RationalFunctionExpansionResult(StrictModel):
    """Exact rational-function expansion with local orders and residual cutoff."""

    function: RationalFunction
    series: TruncatedLaurentWindow
    numerator_order: StrictInt | None
    denominator_order: StrictInt
    valuation: StrictInt | None
    pole_order: StrictInt = Field(ge=0)
    zero_order: StrictInt = Field(ge=0)
    normalized_unit_quotient: TruncatedLaurentWindow | None
    product_residual_precision: StrictInt = Field(
        description="Q times the returned prefix minus P vanishes below this exponent."
    )


class LaurentToPowerSeriesResult(StrictModel):
    """A power-series conversion or its exact negative-exponent obstruction."""

    status: Literal["CONVERTED", "HAS_NEGATIVE_EXPONENTS"]
    source: TruncatedLaurentWindow
    result: TruncatedSeries | None = None
    first_negative_exponent: StrictInt | None = None

    @model_validator(mode="after")
    def require_status_payload(self) -> Self:
        if self.status == "CONVERTED":
            if self.result is None or self.first_negative_exponent is not None:
                raise ValueError(
                    "a converted result requires a power series and no negative exponent"
                )
            return self
        if self.result is not None or self.first_negative_exponent is None:
            raise ValueError(
                "a negative-exponent result requires its first exponent and no power series"
            )
        if self.first_negative_exponent >= 0:
            raise ValueError("the reported obstruction exponent must be negative")
        return self


class LaurentBinaryRequest(StrictModel):
    left: TruncatedLaurentWindow
    right: TruncatedLaurentWindow


class PuiseuxBinaryRequest(StrictModel):
    left: TruncatedPuiseuxWindow
    right: TruncatedPuiseuxWindow


class PuiseuxUnaryRequest(StrictModel):
    series: TruncatedPuiseuxWindow


class PuiseuxResidueResult(StrictModel):
    """Exact residue of a Puiseux prefix whose window contains exponent -1."""

    series: TruncatedPuiseuxWindow
    residue: CanonicalRational


class LaurentMultiplyRequest(LaurentBinaryRequest):
    output_precision: StrictInt = Field(description="Exclusive output exponent.")


class LaurentUnaryRequest(StrictModel):
    series: TruncatedLaurentWindow


class LaurentPowerRequest(LaurentUnaryRequest):
    exponent: StrictInt = Field(
        ge=-MAX_LOCAL_SERIES_POWER_EXPONENT,
        le=MAX_LOCAL_SERIES_POWER_EXPONENT,
    )
    output_precision: StrictInt | None = Field(
        default=None,
        ge=-MAX_LOCAL_SERIES_EXPONENT,
        le=MAX_LOCAL_SERIES_EXPONENT,
    )


class LaurentTruncateRequest(LaurentUnaryRequest):
    valuation_lower: StrictInt
    precision: StrictInt


class LaurentShiftRequest(LaurentUnaryRequest):
    shift: StrictInt


class LaurentScaleRequest(LaurentUnaryRequest):
    scale: CanonicalRational


class LaurentRamifyRequest(LaurentUnaryRequest):
    ramification: StrictInt = Field(ge=1, le=256)


class LaurentIntegralResult(StrictModel):
    source: TruncatedLaurentWindow
    laurent_part: TruncatedLaurentWindow
    log_coefficient: CanonicalRational


class LaurentDeramifyResult(StrictModel):
    status: Literal["IN_IMAGE_OF_RAMIFICATION", "NOT_IN_IMAGE_OF_RAMIFICATION"]
    result: TruncatedLaurentWindow | None = None
    offending_exponents: tuple[StrictInt, ...] = ()

    @model_validator(mode="after")
    def require_status_payload(self) -> Self:
        if self.status == "IN_IMAGE_OF_RAMIFICATION":
            if self.result is None or self.offending_exponents:
                raise ValueError(
                    "an in-image ramification result requires a result and no offending exponents"
                )
        elif self.result is not None or not self.offending_exponents:
            raise ValueError(
                "a non-image ramification result requires offending exponents and no result"
            )
        return self


class LaurentResidueResult(StrictModel):
    series: TruncatedLaurentWindow
    residue: CanonicalRational


class LaurentPrincipalPartResult(StrictModel):
    series: TruncatedLaurentWindow
    principal_part: TruncatedLaurentWindow
    regular_part: TruncatedLaurentWindow

    @model_validator(mode="after")
    def require_disjoint_split(self) -> Self:
        boundary = max(self.series.valuation_lower, min(self.series.precision, 0))
        if (
            self.principal_part.valuation_lower != self.series.valuation_lower
            or self.principal_part.precision != boundary
            or self.regular_part.valuation_lower != boundary
            or self.regular_part.precision != self.series.precision
        ):
            raise ValueError(
                "principal and regular parts must be a disjoint source split"
            )
        return self


class LaurentPowerResult(StrictModel):
    source: TruncatedLaurentWindow
    exponent: StrictInt
    result: TruncatedLaurentWindow


__all__ = [
    "LaurentBinaryRequest",
    "LaurentDeramifyResult",
    "LaurentIntegralResult",
    "LaurentMultiplyRequest",
    "LaurentPowerRequest",
    "LaurentPowerResult",
    "LaurentPrincipalPartResult",
    "LaurentRamifyRequest",
    "LaurentResidueResult",
    "LaurentScaleRequest",
    "LaurentShiftRequest",
    "LaurentToPowerSeriesResult",
    "LaurentTruncateRequest",
    "LaurentUnaryRequest",
    "PuiseuxBinaryRequest",
    "PuiseuxResidueResult",
    "PuiseuxUnaryRequest",
    "RationalFunctionExpansionRequest",
    "RationalFunctionExpansionResult",
    "RationalFunctionInfinityExpansionRequest",
]
