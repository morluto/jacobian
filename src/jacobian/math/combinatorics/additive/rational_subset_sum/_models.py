"""Typed contracts for the rational subset-sum profile operation."""

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json

MAX_RATIONAL_SUBSET_SUM_VALUES = 16
MAX_RATIONAL_SUBSET_SUM_CANDIDATES = 1 << MAX_RATIONAL_SUBSET_SUM_VALUES


def require_rational_subset_sum_envelope(
    values: tuple[CanonicalRational, ...],
) -> None:
    if len(values) > MAX_RATIONAL_SUBSET_SUM_VALUES:
        raise ValueError("rational subset-sum exceeds the 65536-subset work bound")
    candidates = 1 << len(values)
    derived_digits = sum(
        max(len(value.num.lstrip("-")), len(value.den)) for value in values
    ) + len(str(max(len(values), 1)))
    source_bytes = len(
        encode_strict_json(
            {"values": [value.model_dump(mode="json") for value in values]},
            limits=CanonicalLimits(
                max_output_bytes=2 * CanonicalLimits().max_output_bytes
            ),
        )
    )
    if (
        source_bytes + candidates * (2 * derived_digits + 80)
        > CanonicalLimits().max_output_bytes
    ):
        raise ValueError(
            "complete rational subset-sum profile exceeds the canonical output bound"
        )


class RationalSubsetSumRequest(StrictModel):
    """Request the indexed rational subset-sum profile."""

    values: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_RATIONAL_SUBSET_SUM_VALUES
    )

    @model_validator(mode="after")
    def require_bounded_profile(self) -> Self:
        try:
            require_rational_subset_sum_envelope(self.values)
        except ValueError as exc:
            raise PydanticCustomError(
                "rational_subset_sum.envelope_exceeded", str(exc)
            ) from exc
        return self


class RationalSubsetSumEntry(StrictModel):
    """One attainable sum and its multiplicity."""

    sum: CanonicalRational
    multiplicity: int = Field(ge=1, le=MAX_RATIONAL_SUBSET_SUM_CANDIDATES)


class RationalSubsetSumResult(StrictModel):
    """The complete indexed rational subset-sum profile."""

    values: tuple[CanonicalRational, ...]
    entries: tuple[RationalSubsetSumEntry, ...] = Field(
        max_length=MAX_RATIONAL_SUBSET_SUM_CANDIDATES
    )
    support_cardinality: int = Field(ge=1, le=MAX_RATIONAL_SUBSET_SUM_CANDIDATES)


__all__ = [
    "MAX_RATIONAL_SUBSET_SUM_CANDIDATES",
    "MAX_RATIONAL_SUBSET_SUM_VALUES",
    "RationalSubsetSumEntry",
    "RationalSubsetSumRequest",
    "RationalSubsetSumResult",
    "require_rational_subset_sum_envelope",
]
