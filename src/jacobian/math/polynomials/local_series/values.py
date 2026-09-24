"""Exact bounded truncated Laurent windows over QQ at one center."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import PolynomialVariable

MAX_LOCAL_SERIES_TERMS = 4_096
MAX_LOCAL_SERIES_COEFFICIENT_DIGITS = 4_096
MAX_LOCAL_SERIES_EXPONENT = 1_000_000
MAX_LOCAL_SERIES_POWER_EXPONENT = 64
MAX_LOCAL_SERIES_POWER_WORK = 4_000_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"local_series.{reason}", message)


class TruncatedLaurentWindow(StrictModel):
    """One dense truncated Laurent window at a single rational center.

    The window holds ``f(t) = sum_{k=v}^{N-1} a_k t^k`` with explicit dense
    coefficients ``a_v, ..., a_{N-1}`` in ascending-exponent order, where
    ``t`` is the local parameter at ``center`` (``t = x - center``). The
    omitted tail beyond ``N`` is unknown, not zero. A zero window (every
    coefficient zero) is the canonical ``ZERO_AT_PRECISION`` value; arithmetic
    splits may additionally use an explicit empty window with equal lower and
    upper exponents. Nonzero windows normalize their valuation by dropping
    leading zeros.
    """

    variable: PolynomialVariable = Field(
        default="t", description="The single local parameter."
    )
    place: Literal["FINITE", "INFINITY"] = Field(
        default="FINITE",
        description=(
            "Expansion place. FINITE uses t = x - center; INFINITY uses "
            "t = 1/x and requires center = 0."
        ),
    )
    center: CanonicalRational = Field(
        default=CanonicalRational(num=0, den=1),
        description="Rational expansion center; the local parameter is x - center.",
    )
    valuation_lower: StrictInt = Field(
        description="Inclusive lower exponent v of the retained window."
    )
    precision: StrictInt = Field(
        description="Exclusive upper exponent N of the retained window."
    )
    coefficients: tuple[CanonicalRational, ...] = Field(
        description="Dense coefficients a_v through a_{N-1} in ascending order."
    )

    @model_validator(mode="after")
    def require_dense_window(self) -> Self:
        if self.place == "INFINITY" and self.center.as_fraction() != 0:
            raise _validation_error(
                "infinity_center",
                "an expansion at infinity uses the reciprocal parameter and center zero",
            )
        if self.precision < self.valuation_lower:
            raise _validation_error(
                "empty_window",
                "a Laurent window needs valuation_lower <= precision",
            )
        if len(self.coefficients) != self.precision - self.valuation_lower:
            raise _validation_error(
                "coefficient_count_mismatch",
                "coefficient tuple must hold exactly precision - valuation_lower entries",
            )
        if (
            abs(self.valuation_lower) > MAX_LOCAL_SERIES_EXPONENT
            or abs(self.precision) > MAX_LOCAL_SERIES_EXPONENT
        ):
            raise _validation_error(
                "exponent_bound",
                "Laurent window exponents exceed the shared representation limit",
            )
        for value in self.coefficients:
            try:
                require_bounded_rational(
                    value,
                    max_digits=MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
                    label="Laurent window coefficient",
                )
            except ValueError as error:
                raise _validation_error("coefficient_bound", str(error)) from error
        return self


# The Laurent window is also the canonical carrier for the first arithmetic
# slice.  Arithmetic never treats the omitted tail as zero.

__all__ = [
    "MAX_LOCAL_SERIES_COEFFICIENT_DIGITS",
    "MAX_LOCAL_SERIES_EXPONENT",
    "MAX_LOCAL_SERIES_POWER_EXPONENT",
    "MAX_LOCAL_SERIES_POWER_WORK",
    "MAX_LOCAL_SERIES_TERMS",
    "TruncatedLaurentWindow",
]
