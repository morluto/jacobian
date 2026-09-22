"""Wire contracts for bounded Laurent arithmetic."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.local_series.values import TruncatedLaurentWindow


class LaurentBinaryRequest(StrictModel):
    left: TruncatedLaurentWindow
    right: TruncatedLaurentWindow


class LaurentMultiplyRequest(LaurentBinaryRequest):
    output_precision: StrictInt = Field(description="Exclusive output exponent.")


class LaurentUnaryRequest(StrictModel):
    series: TruncatedLaurentWindow


class LaurentPowerRequest(LaurentUnaryRequest):
    exponent: StrictInt = Field(ge=-64, le=64)
    output_precision: StrictInt | None = None


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
                    "an in-image deramification must carry only its result"
                )
        elif self.result is not None or not self.offending_exponents:
            raise ValueError(
                "a non-image deramification must carry offending exponents only"
            )
        return self


class LaurentResidueResult(StrictModel):
    series: TruncatedLaurentWindow
    residue: CanonicalRational


class LaurentPrincipalPartResult(StrictModel):
    series: TruncatedLaurentWindow
    principal_part: TruncatedLaurentWindow
    regular_part: TruncatedLaurentWindow


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
    "LaurentTruncateRequest",
    "LaurentUnaryRequest",
]
