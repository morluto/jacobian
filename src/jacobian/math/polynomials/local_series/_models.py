"""Typed wire contracts for Laurent valuation profiles."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.local_series.values import TruncatedLaurentWindow


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"local_series.{reason}", message)


class ValuationProfileRequest(StrictModel):
    """Project one bounded Laurent window to its valuation profile.

    Admission runs in ``laurent_valuation_profile`` before the leading-term
    scan: the window length must fit ``MAX_LOCAL_SERIES_TERMS``. Per-entry
    coefficient and exponent bounds are stated on the nested window value.
    """

    series: TruncatedLaurentWindow = Field(
        description=(
            "Dense Laurent window at one center; admission additionally caps "
            "the retained window length at 4096 terms."
        ),
    )


class ZeroValuation(StrictModel):
    """Every retained coefficient vanishes; no finite valuation is reported."""

    status: Literal["ZERO_AT_PRECISION"] = "ZERO_AT_PRECISION"


class NonzeroValuation(StrictModel):
    """Exact valuation, leading coefficient, and pole/zero orders of a window.

    ``valuation`` is the least exponent with a nonzero coefficient;
    ``pole_order`` is ``max(-valuation, 0)`` and ``zero_order`` is
    ``max(valuation, 0)``. The leading monomial is
    ``leading_coefficient * variable^valuation`` in the window's parameter.
    """

    status: Literal["NONZERO"] = "NONZERO"
    valuation: int = Field(
        description="Least exponent with a nonzero retained coefficient."
    )
    leading_coefficient: CanonicalRational = Field(
        description="Exact nonzero coefficient of the valuation monomial."
    )
    pole_order: int = Field(ge=0, description="max(-valuation, 0).")
    zero_order: int = Field(ge=0, description="max(valuation, 0).")


class ValuationProfileResult(StrictModel):
    """A source-bound valuation profile of one Laurent window."""

    series: TruncatedLaurentWindow
    conclusion: Annotated[
        ZeroValuation | NonzeroValuation, Field(discriminator="status")
    ]

    @property
    def status(self) -> Literal["ZERO_AT_PRECISION", "NONZERO"]:
        return self.conclusion.status

    @property
    def valuation(self) -> int | None:
        if isinstance(self.conclusion, NonzeroValuation):
            return self.conclusion.valuation
        return None

    @property
    def leading_coefficient(self) -> CanonicalRational | None:
        if isinstance(self.conclusion, NonzeroValuation):
            return self.conclusion.leading_coefficient
        return None

    @property
    def pole_order(self) -> int:
        if isinstance(self.conclusion, NonzeroValuation):
            return self.conclusion.pole_order
        return 0

    @property
    def zero_order(self) -> int:
        if isinstance(self.conclusion, NonzeroValuation):
            return self.conclusion.zero_order
        return 0

    @classmethod
    def _from_kernel(
        cls,
        series: TruncatedLaurentWindow,
        *,
        conclusion: ZeroValuation | NonzeroValuation,
    ) -> Self:
        return cls.model_construct(series=series, conclusion=conclusion)


__all__ = [
    "NonzeroValuation",
    "ValuationProfileRequest",
    "ValuationProfileResult",
    "ZeroValuation",
]
