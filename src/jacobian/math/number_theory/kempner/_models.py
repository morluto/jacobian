"""Typed wire contracts for Kempner reciprocal-series enclosures."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.number_theory._kempner_models import (
    KempnerDigitSet,
    KempnerSmallInteger,
)

MAX_KEMPNER_SERIES_NUMERALS = 50_000
MAX_KEMPNER_SERIES_DIGITS = 32_768


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"number_theory.kempner_series.{reason}", message)


class KempnerSeriesEnclosureRequest(StrictModel):
    """Enclose the reciprocal sum of one Kempner digit family through D digits.

    Semantic admission runs in ``enclose_kempner_series`` before enumeration:
    the finite numeral count ``r * sum_{m<D} s^m`` (with ``s = |S|`` and
    ``r`` the nonzero digit count), the exact-rational height estimate, and
    the serialized result size are preflighted. No fixed cutoff ceiling is
    imposed beyond those derived quantities.
    """

    digit_set: KempnerDigitSet = Field(
        description="Nonempty proper digit subset defining the numeral family."
    )
    cutoff: KempnerSmallInteger = Field(
        ge=0,
        description=(
            "Nonnegative digit cutoff D; the partial sum covers accepted "
            "positive integers with at most D digits."
        ),
    )


class KempnerSeriesEnclosure(StrictModel):
    """A source-bound exact rational interval enclosing one Kempner series.

    ``partial_sum`` is the exact sum of reciprocals over all accepted
    positive integers with at most ``cutoff`` digits; ``tail_upper_bound``
    is the exact rational ``r*(s/b)^D/(1-s/b)`` dominating every omitted
    longer numeral. ``lower`` equals the partial sum and ``upper`` equals
    the partial sum plus the tail bound, so the total infinite series lies
    in ``[lower, upper]``. An empty positive family (no nonzero digit)
    yields ``[0, 0]``.
    """

    digit_set: KempnerDigitSet
    cutoff: KempnerSmallInteger = Field(ge=0)
    partial_sum: CanonicalRational
    tail_upper_bound: CanonicalRational
    lower: CanonicalRational
    upper: CanonicalRational

    @model_validator(mode="after")
    def require_nonnegative_tail(self) -> Self:
        if self.tail_upper_bound.num < 0:
            raise _validation_error(
                "negative_tail", "the tail upper bound must be nonnegative"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        digit_set: KempnerDigitSet,
        cutoff: int,
        *,
        partial_sum: CanonicalRational,
        tail_upper_bound: CanonicalRational,
        lower: CanonicalRational,
        upper: CanonicalRational,
    ) -> Self:
        """Build one enclosure after the admitted kernel established it."""

        return cls.model_construct(
            digit_set=digit_set,
            cutoff=cutoff,
            partial_sum=partial_sum,
            tail_upper_bound=tail_upper_bound,
            lower=lower,
            upper=upper,
        )


__all__ = [
    "MAX_KEMPNER_SERIES_DIGITS",
    "MAX_KEMPNER_SERIES_NUMERALS",
    "KempnerSeriesEnclosure",
    "KempnerSeriesEnclosureRequest",
]
