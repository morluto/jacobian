"""Contracts and kernels for exact prime-affine interval operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory.affine_forms.values import MAX_AFFINE_COMPONENT_DIGITS
from jacobian.math.number_theory.prime_affine_forms._kernel import (
    MAX_DETERMINISTIC_PRIME_INPUT,
    interval_match_summary,
    interval_matches,
)
from jacobian.math.number_theory.prime_affine_forms._models import (
    MAX_RESULT_CHARACTER_BUDGET,
    _digits,
    _run_admission,
    _source_character_upper_bound,
    _validation_error,
)
from jacobian.math.number_theory.prime_affine_forms.values import (
    MAX_AFFINE_FORMS,
    PrimeAffineTuple,
)

MAX_INTERVAL_EVALUATIONS = 100_000
MAX_INTERVAL_ENUMERATION_CELLS = 65_536

# Endpoint syntax is neutral canonical integer grammar. Its effective size is
# bounded from the source before parsing, so cancellation intervals are not
# rejected by a fixed syntax cap.
IntervalEndpointInteger = CanonicalInteger
DeterministicPrimeInteger = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=20, strict=True),
]


def _parse_interval(lower: str, upper: str) -> tuple[int, int, int]:
    lower_value = parse_canonical_integer(lower)
    upper_value = parse_canonical_integer(upper)
    if lower_value > upper_value:
        raise _validation_error(
            "interval lower endpoint must not exceed upper endpoint"
        )
    return lower_value, upper_value, upper_value - lower_value + 1


def require_bounded_affine_endpoints(
    source: PrimeAffineTuple,
    *values: str,
    label: str,
) -> None:
    """Reject endpoints that cannot yield a bounded affine value before parsing.

    If ``|a*n + b|`` has at most ``C`` digits and ``a`` is nonzero, then
    ``n`` has at most ``max(C, digits(b)) + 1`` digits.  This keeps cancelling
    shifts admissible while rejecting oversized input before bigint conversion.
    """

    maximum_digits = (
        max(
            MAX_AFFINE_COMPONENT_DIGITS,
            *(len(form.constant.lstrip("-")) for form in source.forms),
        )
        + 1
    )
    if any(len(value.lstrip("-")) > maximum_digits for value in values):
        raise _validation_error(
            f"{label} endpoint exceeds the {maximum_digits}-digit "
            "source-sensitive pre-parse bound"
        )


def _interval_value_digit_bound(
    source: PrimeAffineTuple, lower: int, upper: int
) -> int:
    maximum_digits = 1
    for form in source.forms:
        values = (form.evaluate(lower), form.evaluate(upper))
        maximum_digits = max(maximum_digits, *(_digits(value) for value in values))
        if max(values) > MAX_DETERMINISTIC_PRIME_INPUT:
            raise _validation_error(
                "a positive affine value exceeds SymPy's deterministic primality "
                f"range 0..{MAX_DETERMINISTIC_PRIME_INPUT}"
            )
    return maximum_digits


class PrimeAffineIntervalCountRequest(StrictModel):
    """Count all positive-prime affine tuples on one closed integer interval."""

    source: PrimeAffineTuple
    lower: IntervalEndpointInteger = Field(
        description="Canonical lower endpoint of a nonempty closed integer interval."
    )
    upper: IntervalEndpointInteger = Field(
        description="Canonical upper endpoint, required to be at least lower."
    )


def _admit_interval_count(
    request: PrimeAffineIntervalCountRequest,
) -> tuple[int, int, int, int]:
    require_bounded_affine_endpoints(
        request.source, request.lower, request.upper, label="interval"
    )
    lower, upper, interval_size = _parse_interval(request.lower, request.upper)
    value_digits = _interval_value_digit_bound(request.source, lower, upper)
    evaluations = interval_size * request.source.form_count
    if evaluations > MAX_INTERVAL_EVALUATIONS:
        raise _validation_error(
            f"interval needs {evaluations} affine evaluations, exceeding "
            f"{MAX_INTERVAL_EVALUATIONS}"
        )
    return lower, upper, interval_size, value_digits


def _admit_interval_enumerate(
    request: PrimeAffineIntervalEnumerateRequest,
) -> tuple[int, int, int]:
    lower, upper, interval_size, value_digits = _admit_interval_count(request)
    result_cells = interval_size * (request.source.form_count + 1)
    if result_cells > MAX_INTERVAL_ENUMERATION_CELLS:
        raise _validation_error(
            f"interval enumeration may need {result_cells} result cells, "
            f"exceeding {MAX_INTERVAL_ENUMERATION_CELLS}"
        )
    parameter_digits = max(_digits(lower), _digits(upper))
    serialized_characters = (
        _source_character_upper_bound(request.source)
        + interval_size
        * (40 + parameter_digits + request.source.form_count * (value_digits + 4))
        + 256
    )
    if serialized_characters > MAX_RESULT_CHARACTER_BUDGET:
        raise _validation_error(
            "interval enumeration exceeds the conservative serialized bound"
        )
    return lower, upper, interval_size


class PrimeAffineIntervalEnumerateRequest(PrimeAffineIntervalCountRequest):
    """Enumerate all positive-prime affine tuples on a stricter output envelope."""


class PrimePatternMatch(StrictModel):
    parameter: IntervalEndpointInteger
    prime_values: tuple[DeterministicPrimeInteger, ...] = Field(
        min_length=1, max_length=MAX_AFFINE_FORMS
    )


class PrimePatternIntervalCountResult(StrictModel):
    source: PrimeAffineTuple
    lower: IntervalEndpointInteger
    upper: IntervalEndpointInteger
    interval_size: StrictInt = Field(ge=1)
    affine_values_examined: StrictInt = Field(ge=1)
    match_count: StrictInt = Field(ge=0)
    first_match: IntervalEndpointInteger | None = None
    last_match: IntervalEndpointInteger | None = None

    @model_validator(mode="after")
    def require_interval_shape(self) -> Self:
        lower, upper, interval_size = _parse_interval(self.lower, self.upper)
        if self.interval_size != interval_size:
            raise _validation_error("interval_size must equal upper-lower+1")
        if self.affine_values_examined != interval_size * self.source.form_count:
            raise _validation_error(
                "affine_values_examined does not match the complete interval"
            )
        if (self.first_match is None) != (self.last_match is None):
            raise _validation_error(
                "first_match and last_match must be both present or absent"
            )
        if (self.match_count == 0) != (self.first_match is None):
            raise _validation_error(
                "match endpoints must be present if and only if match_count is positive"
            )
        if self.match_count > interval_size:
            raise _validation_error("match_count cannot exceed interval_size")
        if self.first_match is not None and self.last_match is not None:
            first = parse_canonical_integer(self.first_match)
            last = parse_canonical_integer(self.last_match)
            if not lower <= first <= last <= upper:
                raise _validation_error("match endpoints must lie in interval order")
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PrimeAffineIntervalCountRequest,
        *,
        count: int,
        first: int | None,
        last: int | None,
    ) -> Self:
        lower = parse_canonical_integer(request.lower)
        upper = parse_canonical_integer(request.upper)
        interval_size = upper - lower + 1
        return cls(
            source=request.source,
            lower=request.lower,
            upper=request.upper,
            interval_size=interval_size,
            affine_values_examined=interval_size * request.source.form_count,
            match_count=count,
            first_match=None if first is None else format_canonical_integer(first),
            last_match=None if last is None else format_canonical_integer(last),
        )


class PrimePatternIntervalEnumerateResult(StrictModel):
    source: PrimeAffineTuple
    lower: IntervalEndpointInteger
    upper: IntervalEndpointInteger
    interval_size: StrictInt = Field(ge=1)
    affine_values_examined: StrictInt = Field(ge=1)
    matches: tuple[PrimePatternMatch, ...] = Field(
        max_length=MAX_INTERVAL_ENUMERATION_CELLS
    )

    @model_validator(mode="after")
    def require_enumeration_shape(self) -> Self:
        result_cells = len(self.matches) + sum(
            len(match.prime_values) for match in self.matches
        )
        if result_cells > MAX_INTERVAL_ENUMERATION_CELLS:
            raise _validation_error("matches exceed the interval result-cell bound")
        lower, upper, interval_size = _parse_interval(self.lower, self.upper)
        actual = tuple(
            (
                parse_canonical_integer(match.parameter),
                tuple(parse_canonical_integer(value) for value in match.prime_values),
            )
            for match in self.matches
        )
        if actual != tuple(sorted(actual)) or len(
            {parameter for parameter, _ in actual}
        ) != len(actual):
            raise _validation_error(
                "matches must be in strictly increasing parameter order"
            )
        if any(
            not lower <= parameter <= upper
            or len(prime_values) != self.source.form_count
            for parameter, prime_values in actual
        ):
            raise _validation_error(
                "matches must lie in the interval with one value per affine form"
            )
        if self.interval_size != interval_size:
            raise _validation_error("interval_size must equal upper-lower+1")
        if self.affine_values_examined != interval_size * self.source.form_count:
            raise _validation_error(
                "affine_values_examined does not match the complete interval"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PrimeAffineIntervalEnumerateRequest,
        *,
        matches: tuple[tuple[int, tuple[int, ...]], ...],
    ) -> Self:
        lower = parse_canonical_integer(request.lower)
        upper = parse_canonical_integer(request.upper)
        interval_size = upper - lower + 1
        return cls(
            source=request.source,
            lower=request.lower,
            upper=request.upper,
            interval_size=interval_size,
            affine_values_examined=interval_size * request.source.form_count,
            matches=tuple(
                PrimePatternMatch(
                    parameter=format_canonical_integer(parameter),
                    prime_values=tuple(
                        format_canonical_integer(value) for value in values
                    ),
                )
                for parameter, values in matches
            ),
        )


def compute_interval_count(
    request: PrimeAffineIntervalCountRequest,
) -> PrimePatternIntervalCountResult:
    """Count every admitted positive-prime affine tuple in the interval."""

    lower, upper, _, _ = _run_admission(lambda: _admit_interval_count(request))
    count, first, last = interval_match_summary(request.source, lower, upper)
    return PrimePatternIntervalCountResult._from_kernel(
        request, count=count, first=first, last=last
    )


def compute_interval_enumerate(
    request: PrimeAffineIntervalEnumerateRequest,
) -> PrimePatternIntervalEnumerateResult:
    """Materialize every admitted positive-prime affine tuple in the interval."""

    lower, upper, _ = _run_admission(lambda: _admit_interval_enumerate(request))
    matches = interval_matches(request.source, lower, upper)
    return PrimePatternIntervalEnumerateResult._from_kernel(request, matches=matches)


__all__ = [
    "MAX_INTERVAL_ENUMERATION_CELLS",
    "MAX_INTERVAL_EVALUATIONS",
    "IntervalEndpointInteger",
    "PrimeAffineIntervalCountRequest",
    "PrimeAffineIntervalEnumerateRequest",
    "PrimePatternIntervalCountResult",
    "PrimePatternIntervalEnumerateResult",
    "PrimePatternMatch",
    "compute_interval_count",
    "compute_interval_enumerate",
    "require_bounded_affine_endpoints",
]
