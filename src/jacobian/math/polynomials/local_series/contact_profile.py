"""Exact pairwise contact orders visible in finite Puiseux prefixes."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.local_series.puiseux_values import (
    TruncatedPuiseuxWindow,
)
from jacobian.math.polynomials.local_series.values import (
    MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
)

MAX_PUISEUX_CONTACT_PREFIXES = 32
MAX_PUISEUX_CONTACT_TERMS = 8_192
MAX_PUISEUX_CONTACT_WORK = 1_000_000
MAX_PUISEUX_CONTACT_OUTPUT_BYTES = CanonicalLimits().max_output_bytes
_PAIR_OUTPUT_BYTES = 256


class PuiseuxContactRequest(StrictModel):
    """Compare prefixes on one common known exponent window.

    This profiles the supplied finite series values. It does not establish that
    they are branches of a polynomial or that the supplied family is complete.
    """

    prefixes: tuple[TruncatedPuiseuxWindow, ...] = Field(
        min_length=2, max_length=MAX_PUISEUX_CONTACT_PREFIXES
    )

    @model_validator(mode="after")
    def require_common_window(self) -> PuiseuxContactRequest:
        first = self.prefixes[0]
        for prefix in self.prefixes:
            if (
                prefix.variable,
                prefix.center,
                prefix.valuation_lower,
                prefix.precision,
            ) != (
                first.variable,
                first.center,
                first.valuation_lower,
                first.precision,
            ):
                raise ValueError(
                    "Puiseux prefixes must share variable, center, and known exponent window"
                )
        return self


class PuiseuxContactPair(StrictModel):
    """The first differing exponent, or agreement through a finite cutoff."""

    left_index: StrictInt = Field(ge=0)
    right_index: StrictInt = Field(ge=1)
    status: Literal["DETERMINED", "UNRESOLVED"]
    common_precision: CanonicalRational
    contact_order: CanonicalRational | None = None

    @model_validator(mode="after")
    def require_status_shape(self) -> PuiseuxContactPair:
        if self.left_index >= self.right_index:
            raise ValueError("contact pair indices must be strictly increasing")
        if (self.status == "DETERMINED") != (self.contact_order is not None):
            raise ValueError("only determined contacts carry a finite contact order")
        if self.contact_order is not None and not (
            self.contact_order.as_fraction() < self.common_precision.as_fraction()
        ):
            raise ValueError(
                "a determined contact order must lie below common precision"
            )
        return self


class PuiseuxContactProfile(StrictModel):
    """Source-bound contact data for each pair of supplied finite prefixes."""

    prefixes: tuple[TruncatedPuiseuxWindow, ...] = Field(
        max_length=MAX_PUISEUX_CONTACT_PREFIXES
    )
    pairs: tuple[PuiseuxContactPair, ...]

    @model_validator(mode="after")
    def require_complete_pair_indexing(self) -> PuiseuxContactProfile:
        if len(self.prefixes) < 2:
            raise ValueError("contact profiles require at least two prefixes")
        expected = combinations(range(len(self.prefixes)), 2)
        actual = iter((row.left_index, row.right_index) for row in self.pairs)
        if (
            not all(pair == next(actual, None) for pair in expected)
            or next(actual, None) is not None
        ):
            raise ValueError("contact profile must contain every source pair in order")
        return self

    @classmethod
    def _from_kernel(
        cls,
        prefixes: tuple[TruncatedPuiseuxWindow, ...],
        pairs: tuple[PuiseuxContactPair, ...],
    ) -> PuiseuxContactProfile:
        return cls.model_construct(prefixes=prefixes, pairs=pairs)


def _contact_order(
    left: TruncatedPuiseuxWindow, right: TruncatedPuiseuxWindow
) -> Fraction | None:
    """Find the least supported exponent whose exact coefficients differ."""
    left_terms = left.terms
    right_terms = right.terms
    request_checkpoint("during Puiseux contact support comparison")
    i = j = 0
    while i < len(left_terms) or j < len(right_terms):
        if j == len(right_terms) or (
            i < len(left_terms)
            and left_terms[i].exponent.as_fraction()
            < right_terms[j].exponent.as_fraction()
        ):
            exponent = left_terms[i].exponent.as_fraction()
            if exponent < right.precision.as_fraction():
                return exponent
            i += 1
            continue
        if i == len(left_terms) or (
            j < len(right_terms)
            and right_terms[j].exponent.as_fraction()
            < left_terms[i].exponent.as_fraction()
        ):
            exponent = right_terms[j].exponent.as_fraction()
            if exponent < left.precision.as_fraction():
                return exponent
            j += 1
            continue

        exponent = left_terms[i].exponent.as_fraction()
        if exponent < left.precision.as_fraction() and (
            left_terms[i].coefficient.as_fraction()
            != right_terms[j].coefficient.as_fraction()
        ):
            return exponent
        i += 1
        j += 1
    return None


def puiseux_contact_profile(request: PuiseuxContactRequest) -> PuiseuxContactProfile:
    """Return pairwise finite-prefix contact orders under exact bounds."""
    if not isinstance(request, PuiseuxContactRequest):
        raise OperationResourceAdmissionError(
            location=("request",),
            code="local_series.puiseux_contact_request",
            message="expected a PuiseuxContactRequest",
        )
    try:
        request = PuiseuxContactRequest.model_validate(request.model_dump())
        for prefix in request.prefixes:
            for value in (prefix.center, prefix.valuation_lower, prefix.precision):
                require_bounded_rational(
                    value,
                    max_digits=MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
                    label="Puiseux request value",
                )
            for term in prefix.terms:
                require_bounded_rational(
                    term.exponent,
                    max_digits=MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
                    label="Puiseux exponent",
                )
                require_bounded_rational(
                    term.coefficient,
                    max_digits=MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
                    label="Puiseux coefficient",
                )
    except (TypeError, ValueError) as error:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="local_series.puiseux_contact_request",
            message="invalid Puiseux contact request",
        ) from error
    prefixes = request.prefixes
    request_checkpoint("before Puiseux contact admission")
    pair_count = len(prefixes) * (len(prefixes) - 1) // 2
    aggregate_terms = sum(len(prefix.terms) for prefix in prefixes)
    work = aggregate_terms * (len(prefixes) - 1)
    if aggregate_terms > MAX_PUISEUX_CONTACT_TERMS or work > MAX_PUISEUX_CONTACT_WORK:
        raise OperationResourceAdmissionError(
            location=("prefixes",),
            code="local_series.puiseux_contact_work_bound",
            message="Puiseux contact support exceeds the admitted comparison work",
        )

    # The result retains its source windows. Account for their serialized size
    # and a worst-case compact row for every pair before pair comparisons.
    source_bytes = sum(len(prefix.model_dump_json()) for prefix in prefixes)
    if (
        source_bytes + pair_count * _PAIR_OUTPUT_BYTES + 512
        > MAX_PUISEUX_CONTACT_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("prefixes",),
            code="local_series.puiseux_contact_output_bound",
            message="source prefixes and pair profile exceed the exact output bound",
        )

    precision = prefixes[0].precision
    rows: list[PuiseuxContactPair] = []
    for left_index, right_index in combinations(range(len(prefixes)), 2):
        request_checkpoint("during Puiseux contact pair comparison")
        order = _contact_order(prefixes[left_index], prefixes[right_index])
        rows.append(
            PuiseuxContactPair(
                left_index=left_index,
                right_index=right_index,
                status="UNRESOLVED" if order is None else "DETERMINED",
                common_precision=precision,
                contact_order=(
                    None if order is None else CanonicalRational.from_fraction(order)
                ),
            )
        )
    return PuiseuxContactProfile._from_kernel(prefixes, tuple(rows))


__all__ = [
    "PuiseuxContactPair",
    "PuiseuxContactProfile",
    "PuiseuxContactRequest",
    "puiseux_contact_profile",
]
