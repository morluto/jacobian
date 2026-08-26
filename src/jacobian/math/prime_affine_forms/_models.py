"""Typed contracts for exact prime-affine local arithmetic."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError
from sympy import isprime

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.affine_forms.values import (
    MAX_AFFINE_COMPONENT_DIGITS,
    AffineFormId,
)
from jacobian.math.prime_affine_forms._kernel import (
    MAX_DETERMINISTIC_PRIME_INPUT,
    interval_match_summary,
    interval_matches,
    local_bad_residues,
    translated_tuple,
)
from jacobian.math.prime_affine_forms.values import (
    MAX_AFFINE_AGGREGATE_DIGITS,
    MAX_AFFINE_FORMS,
    PrimeAffineTuple,
)

MAX_LOCAL_PROFILE_PRIME = 8_191
MAX_LOCAL_PROFILE_WORK = 20_000
MAX_BATCH_PRIME = (1 << 53) - 1
MAX_PRIME_BATCH = 64
MAX_BATCH_ROOT_WORK = 250_000
MAX_FACTOR_COMPONENT_DIGITS = 4_096
MAX_FACTOR_PRODUCT_DIGITS = 8_192
MAX_RESULT_CHARACTER_BUDGET = 2_000_000
MAX_INTERVAL_ENDPOINT_DIGITS = 64
MAX_INTERVAL_REPLAY_EVALUATIONS = 200_000
MAX_INTERVAL_ENUMERATION_CELLS = 65_536

CompactPrime = Annotated[
    StrictInt,
    Field(
        ge=2,
        le=MAX_BATCH_PRIME,
        description=(
            "Prime in the interoperable JSON-integer range 2 through "
            f"{MAX_BATCH_PRIME}."
        ),
    ),
]
IntervalEndpointInteger = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=MAX_INTERVAL_ENDPOINT_DIGITS + 1, strict=True),
]
DeterministicPrimeInteger = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=20, strict=True),
]


def _validation_error(message: str) -> PydanticCustomError:
    """Return an actionable owner-local validation error."""

    code_by_reason = (
        ("prime must", "prime_bound"),
        ("modulus must", "prime_required"),
        ("primes must", "prime_order"),
        ("factor", "factor_bound"),
        ("residue", "residue_invariant"),
        ("local", "local_invariant"),
        ("interval", "interval_bound"),
        ("affine", "affine_bound"),
        ("wheel", "wheel_invariant"),
        ("membership", "membership_invariant"),
        ("translation", "translation_bound"),
        ("translated", "translation_invariant"),
        ("match", "interval_invariant"),
        ("first", "obstruction_invariant"),
        ("admissibility", "admissibility_invariant"),
        ("finite", "finite_product_invariant"),
    )
    suffix = next(
        (suffix for phrase, suffix in code_by_reason if phrase in message.lower()),
        "invariant",
    )
    return PydanticCustomError(f"prime_affine_form.{suffix}", message)


def _digits(value: int | str) -> int:
    return len(str(value).lstrip("-"))


def _source_character_upper_bound(source: PrimeAffineTuple) -> int:
    """Conservatively bound the source tuple's compact JSON representation."""

    return 16 + sum(
        64 + len(form.form_id) + len(form.coefficient) + len(form.constant)
        for form in source.forms
    )


def _summary_character_upper_bound(source: PrimeAffineTuple, prime: int) -> int:
    """Bound one compact local summary after its root-work preflight."""

    bad = local_bad_residues(source, prime)
    return (
        72
        + 4 * _digits(prime)
        + sum(
            32 + _digits(residue) + sum(len(form_id) + 4 for form_id in form_ids)
            for residue, form_ids in bad
        )
    )


def _require_prime(prime: int, *, maximum: int) -> None:
    """Require a prime inside SymPy's deterministic sub-2^64 domain."""

    if prime > maximum:
        raise _validation_error(f"prime must be at most {maximum}")
    if not isprime(prime):
        raise _validation_error("modulus must be prime")


def _require_prime_set(primes: tuple[int, ...], *, maximum: int) -> None:
    if primes != tuple(sorted(set(primes))):
        raise _validation_error("primes must be distinct and strictly increasing")
    for prime in primes:
        _require_prime(prime, maximum=maximum)


def _expected_summary(
    source: PrimeAffineTuple, prime: int
) -> tuple[tuple[tuple[int, tuple[str, ...]], ...], int, int]:
    bad = local_bad_residues(source, prime)
    return bad, len(bad), prime - len(bad)


def _require_summary(source: PrimeAffineTuple, summary: PrimeTupleLocalSummary) -> None:
    bad, bad_count, valid_count = _expected_summary(source, summary.prime)
    actual = tuple((row.residue, row.form_ids) for row in summary.bad_residues)
    if actual != bad:
        raise _validation_error("bad-residue rows do not match the source affine tuple")
    if summary.bad_count != bad_count or summary.valid_count != valid_count:
        raise _validation_error("local residue counts do not match the source tuple")


def _parse_interval(lower: str, upper: str) -> tuple[int, int, int]:
    if (
        _digits(lower) > MAX_INTERVAL_ENDPOINT_DIGITS
        or _digits(upper) > MAX_INTERVAL_ENDPOINT_DIGITS
    ):
        raise _validation_error(
            "interval endpoints must each have at most "
            f"{MAX_INTERVAL_ENDPOINT_DIGITS} digits"
        )
    lower_value = parse_canonical_integer(lower)
    upper_value = parse_canonical_integer(upper)
    if lower_value > upper_value:
        raise _validation_error(
            "interval lower endpoint must not exceed upper endpoint"
        )
    return lower_value, upper_value, upper_value - lower_value + 1


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


class PrimeTupleBadResidueRow(StrictModel):
    """One residue excluded by the labelled forms that vanish there."""

    residue: StrictInt = Field(ge=0, le=MAX_BATCH_PRIME)
    form_ids: tuple[AffineFormId, ...] = Field(
        min_length=1, max_length=MAX_AFFINE_FORMS
    )

    @model_validator(mode="after")
    def require_canonical_ids(self) -> Self:
        if self.form_ids != tuple(sorted(set(self.form_ids))):
            raise _validation_error("vanishing form IDs must be distinct and sorted")
        return self


class PrimeTupleResidueRow(StrictModel):
    """One residue and every source form that vanishes there."""

    residue: StrictInt = Field(ge=0, le=MAX_LOCAL_PROFILE_PRIME)
    vanishing_form_ids: tuple[AffineFormId, ...] = Field(max_length=MAX_AFFINE_FORMS)

    @model_validator(mode="after")
    def require_canonical_ids(self) -> Self:
        if self.vanishing_form_ids != tuple(sorted(set(self.vanishing_form_ids))):
            raise _validation_error("vanishing form IDs must be distinct and sorted")
        return self


class PrimeTupleLocalSummary(StrictModel):
    """Compact exact local obstruction data for one prime."""

    prime: StrictInt = Field(ge=2, le=MAX_BATCH_PRIME)
    bad_residues: tuple[PrimeTupleBadResidueRow, ...] = Field(
        max_length=MAX_AFFINE_FORMS
    )
    bad_count: StrictInt = Field(ge=0)
    valid_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_internal_partition_counts(self) -> Self:
        _require_prime(self.prime, maximum=MAX_BATCH_PRIME)
        residues = tuple(row.residue for row in self.bad_residues)
        if residues != tuple(sorted(set(residues))):
            raise _validation_error("bad residues must be distinct and sorted")
        if any(residue >= self.prime for residue in residues):
            raise _validation_error("bad residues must be canonical modulo the prime")
        if self.bad_count != len(self.bad_residues):
            raise _validation_error(
                "bad_count must equal the number of bad residue rows"
            )
        if self.bad_count + self.valid_count != self.prime:
            raise _validation_error("bad and valid counts must partition every residue")
        if sum(len(row.form_ids) for row in self.bad_residues) > MAX_AFFINE_FORMS:
            raise _validation_error(
                "bad-residue incidence count exceeds the affine-form bound"
            )
        return self


class PrimeAffineIntervalCountRequest(StrictModel):
    """Count all positive-prime affine tuples on one closed integer interval."""

    source: PrimeAffineTuple
    lower: IntervalEndpointInteger = Field(
        description="Canonical lower endpoint of a nonempty closed integer interval."
    )
    upper: IntervalEndpointInteger = Field(
        description="Canonical upper endpoint, required to be at least lower."
    )

    @model_validator(mode="after")
    def require_bounded_complete_count(self) -> Self:
        lower, upper, interval_size = _parse_interval(self.lower, self.upper)
        _interval_value_digit_bound(self.source, lower, upper)
        replay_evaluations = 2 * interval_size * self.source.form_count
        if replay_evaluations > MAX_INTERVAL_REPLAY_EVALUATIONS:
            raise _validation_error(
                f"interval result and validation need {replay_evaluations} affine "
                f"evaluations, exceeding {MAX_INTERVAL_REPLAY_EVALUATIONS}"
            )
        return self


class PrimeAffineIntervalEnumerateRequest(PrimeAffineIntervalCountRequest):
    """Enumerate all positive-prime affine tuples on a stricter output envelope."""

    @model_validator(mode="after")
    def require_bounded_complete_enumeration(self) -> Self:
        lower, upper, interval_size = _parse_interval(self.lower, self.upper)
        value_digits = _interval_value_digit_bound(self.source, lower, upper)
        result_cells = interval_size * (self.source.form_count + 1)
        if result_cells > MAX_INTERVAL_ENUMERATION_CELLS:
            raise _validation_error(
                f"interval enumeration may need {result_cells} result cells, "
                f"exceeding {MAX_INTERVAL_ENUMERATION_CELLS}"
            )
        parameter_digits = max(_digits(lower), _digits(upper))
        serialized_characters = (
            _source_character_upper_bound(self.source)
            + interval_size
            * (40 + parameter_digits + self.source.form_count * (value_digits + 4))
            + 256
        )
        if serialized_characters > MAX_RESULT_CHARACTER_BUDGET:
            raise _validation_error(
                "interval enumeration exceeds the conservative serialized bound"
            )
        return self


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
    def bind_exact_interval_count(self) -> Self:
        PrimeAffineIntervalCountRequest(
            source=self.source, lower=self.lower, upper=self.upper
        )
        lower, upper, interval_size = _parse_interval(self.lower, self.upper)
        expected_count, expected_first, expected_last = interval_match_summary(
            self.source, lower, upper
        )
        if self.interval_size != interval_size:
            raise _validation_error("interval_size must equal upper-lower+1")
        if self.affine_values_examined != interval_size * self.source.form_count:
            raise _validation_error(
                "affine_values_examined does not match the complete interval"
            )
        if self.match_count != expected_count:
            raise _validation_error(
                "match_count does not match exact interval primality"
            )
        if (
            None
            if self.first_match is None
            else parse_canonical_integer(self.first_match)
        ) != expected_first or (
            None
            if self.last_match is None
            else parse_canonical_integer(self.last_match)
        ) != expected_last:
            raise _validation_error(
                "first or last match does not match exact interval primality"
            )
        return self


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
    def bind_complete_interval_enumeration(self) -> Self:
        result_cells = len(self.matches) + sum(
            len(match.prime_values) for match in self.matches
        )
        if result_cells > MAX_INTERVAL_ENUMERATION_CELLS:
            raise _validation_error("matches exceed the interval result-cell bound")
        PrimeAffineIntervalEnumerateRequest(
            source=self.source, lower=self.lower, upper=self.upper
        )
        lower, upper, interval_size = _parse_interval(self.lower, self.upper)
        expected = interval_matches(self.source, lower, upper)
        actual = tuple(
            (
                parse_canonical_integer(match.parameter),
                tuple(parse_canonical_integer(value) for value in match.prime_values),
            )
            for match in self.matches
        )
        if actual != expected:
            raise _validation_error(
                "matches must be every and only positive-prime affine tuple"
            )
        if self.interval_size != interval_size:
            raise _validation_error("interval_size must equal upper-lower+1")
        if self.affine_values_examined != interval_size * self.source.form_count:
            raise _validation_error(
                "affine_values_examined does not match the complete interval"
            )
        return self


class PrimeAffineTranslationRequest(StrictModel):
    """Translate every source form by the variable substitution n -> n+shift."""

    source: PrimeAffineTuple
    shift: IntervalEndpointInteger

    @model_validator(mode="after")
    def require_bounded_translated_tuple(self) -> Self:
        if _digits(self.shift) > MAX_INTERVAL_ENDPOINT_DIGITS:
            raise _validation_error(
                f"translation shift must have at most {MAX_INTERVAL_ENDPOINT_DIGITS} digits"
            )
        shift = parse_canonical_integer(self.shift)
        aggregate_digits = 0
        for form in self.source.forms:
            translated_constant = form.evaluate(shift)
            if _digits(translated_constant) > MAX_AFFINE_COMPONENT_DIGITS:
                raise _validation_error(
                    "translated constant exceeds the canonical affine component bound"
                )
            aggregate_digits += _digits(form.coefficient) + _digits(translated_constant)
        if aggregate_digits > MAX_AFFINE_AGGREGATE_DIGITS:
            raise _validation_error(
                "translated affine tuple exceeds the aggregate coefficient-digit "
                f"bound {MAX_AFFINE_AGGREGATE_DIGITS}"
            )
        return self


class PrimeAffineTranslationResult(StrictModel):
    source: PrimeAffineTuple
    shift: IntervalEndpointInteger
    translated: PrimeAffineTuple

    @model_validator(mode="after")
    def bind_exact_translation(self) -> Self:
        PrimeAffineTranslationRequest(source=self.source, shift=self.shift)
        expected = translated_tuple(self.source, parse_canonical_integer(self.shift))
        if self.translated != expected:
            raise _validation_error(
                "translated tuple must equal L_i(n+shift) for every form"
            )
        return self


__all__ = [
    "MAX_BATCH_PRIME",
    "MAX_INTERVAL_ENUMERATION_CELLS",
    "MAX_INTERVAL_REPLAY_EVALUATIONS",
    "MAX_LOCAL_PROFILE_PRIME",
    "MAX_PRIME_BATCH",
    "PrimeAffineIntervalCountRequest",
    "PrimeAffineIntervalEnumerateRequest",
    "PrimeAffineTranslationRequest",
    "PrimeAffineTranslationResult",
    "PrimePatternIntervalCountResult",
    "PrimePatternIntervalEnumerateResult",
    "PrimePatternMatch",
    "PrimeTupleBadResidueRow",
    "PrimeTupleLocalSummary",
    "PrimeTupleResidueRow",
]
