"""Typed contracts for exact prime-affine local arithmetic."""

from __future__ import annotations

from fractions import Fraction
from math import prod
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictInt, StringConstraints, model_validator
from sympy import isprime, primepi

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.affine_forms.values import (
    MAX_AFFINE_COMPONENT_DIGITS,
    MAX_FORM_ID_LENGTH,
    AffineComponentInteger,
    AffineFormId,
)
from jacobian.math.prime_affine_forms._kernel import (
    MAX_DETERMINISTIC_PRIME_INPUT,
    interval_match_summary,
    interval_matches,
    local_bad_residues,
    local_counts,
    local_factor_from_bad_count,
    primes_through,
    translated_tuple,
    wheel_modulus,
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
MAX_ADMISSIBILITY_CUTOFF = MAX_AFFINE_FORMS
MAX_ADMISSIBILITY_PRIME_ROWS = 128
MAX_ADMISSIBILITY_ROOT_CELLS = 200_000
MAX_WHEEL_PRIMES = 64
MAX_WHEEL_LOCAL_RESIDUES = 32_768
MAX_WHEEL_RESIDUES = 8_192
MAX_WHEEL_RESULT_CELLS = 65_536
MAX_WHEEL_ENUMERATION_WORK = 300_000
MAX_COMPACT_WHEEL_ROOT_WORK = 250_000
MAX_COMPACT_WHEEL_SCALAR_DIGITS = 4_096
MAX_RESULT_CHARACTER_BUDGET = 2_000_000
MAX_INTERVAL_ENDPOINT_DIGITS = 64
MAX_INTERVAL_REPLAY_EVALUATIONS = 200_000
MAX_INTERVAL_ENUMERATION_CELLS = 65_536
MAX_WHEEL_INTERVAL_LENGTH = 8_192

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
CompactWheelScalar = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=MAX_COMPACT_WHEEL_SCALAR_DIGITS, strict=True),
]
IntervalEndpointInteger = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=MAX_INTERVAL_ENDPOINT_DIGITS + 1, strict=True),
]
DeterministicPrimeInteger = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=20, strict=True),
]


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
        raise ValueError(f"prime must be at most {maximum}")
    if not isprime(prime):
        raise ValueError("modulus must be prime")


def _require_prime_set(primes: tuple[int, ...], *, maximum: int) -> None:
    if primes != tuple(sorted(set(primes))):
        raise ValueError("primes must be distinct and strictly increasing")
    for prime in primes:
        _require_prime(prime, maximum=maximum)


def _factor_digit_upper_bound(source: PrimeAffineTuple, prime: int) -> int:
    bad_count, _ = local_counts(source, prime)
    if bad_count == prime:
        return 1
    numerator = _digits(prime - bad_count) + (source.form_count - 1) * _digits(prime)
    denominator = source.form_count * _digits(prime - 1)
    return max(numerator, denominator)


def _require_factor_output(source: PrimeAffineTuple, prime: int) -> None:
    bound = _factor_digit_upper_bound(source, prime)
    if bound > MAX_FACTOR_COMPONENT_DIGITS:
        raise ValueError(
            "local-factor numerator or denominator may require "
            f"{bound} digits, exceeding the bound {MAX_FACTOR_COMPONENT_DIGITS}"
        )


def _expected_summary(
    source: PrimeAffineTuple, prime: int
) -> tuple[tuple[tuple[int, tuple[str, ...]], ...], int, int]:
    bad = local_bad_residues(source, prime)
    return bad, len(bad), prime - len(bad)


def _require_summary(source: PrimeAffineTuple, summary: PrimeTupleLocalSummary) -> None:
    bad, bad_count, valid_count = _expected_summary(source, summary.prime)
    actual = tuple((row.residue, row.form_ids) for row in summary.bad_residues)
    if actual != bad:
        raise ValueError("bad-residue rows do not match the source affine tuple")
    if summary.bad_count != bad_count or summary.valid_count != valid_count:
        raise ValueError("local residue counts do not match the source tuple")


def _parse_interval(lower: str, upper: str) -> tuple[int, int, int]:
    if (
        _digits(lower) > MAX_INTERVAL_ENDPOINT_DIGITS
        or _digits(upper) > MAX_INTERVAL_ENDPOINT_DIGITS
    ):
        raise ValueError(
            "interval endpoints must each have at most "
            f"{MAX_INTERVAL_ENDPOINT_DIGITS} digits"
        )
    lower_value = parse_canonical_integer(lower)
    upper_value = parse_canonical_integer(upper)
    if lower_value > upper_value:
        raise ValueError("interval lower endpoint must not exceed upper endpoint")
    return lower_value, upper_value, upper_value - lower_value + 1


def _interval_value_digit_bound(
    source: PrimeAffineTuple, lower: int, upper: int
) -> int:
    maximum_digits = 1
    for form in source.forms:
        values = (form.evaluate(lower), form.evaluate(upper))
        maximum_digits = max(maximum_digits, *(_digits(value) for value in values))
        if max(values) > MAX_DETERMINISTIC_PRIME_INPUT:
            raise ValueError(
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
            raise ValueError("vanishing form IDs must be distinct and sorted")
        return self


class PrimeTupleResidueRow(StrictModel):
    """One residue and every source form that vanishes there."""

    residue: StrictInt = Field(ge=0, le=MAX_LOCAL_PROFILE_PRIME)
    vanishing_form_ids: tuple[AffineFormId, ...] = Field(max_length=MAX_AFFINE_FORMS)

    @model_validator(mode="after")
    def require_canonical_ids(self) -> Self:
        if self.vanishing_form_ids != tuple(sorted(set(self.vanishing_form_ids))):
            raise ValueError("vanishing form IDs must be distinct and sorted")
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
            raise ValueError("bad residues must be distinct and sorted")
        if any(residue >= self.prime for residue in residues):
            raise ValueError("bad residues must be canonical modulo the prime")
        if self.bad_count != len(self.bad_residues):
            raise ValueError("bad_count must equal the number of bad residue rows")
        if self.bad_count + self.valid_count != self.prime:
            raise ValueError("bad and valid counts must partition every residue")
        if sum(len(row.form_ids) for row in self.bad_residues) > MAX_AFFINE_FORMS:
            raise ValueError(
                "bad-residue incidence count exceeds the affine-form bound"
            )
        return self


class PrimeTupleLocalFactorRequest(StrictModel):
    """Compute one complete local residue profile and exact local factor."""

    source: PrimeAffineTuple
    prime: StrictInt = Field(ge=2, le=MAX_LOCAL_PROFILE_PRIME)

    @model_validator(mode="after")
    def require_bounded_complete_profile(self) -> Self:
        _require_prime(self.prime, maximum=MAX_LOCAL_PROFILE_PRIME)
        _require_factor_output(self.source, self.prime)
        work = 6 * self.source.form_count + 2 * self.prime
        if work > MAX_LOCAL_PROFILE_WORK:
            raise ValueError(
                f"local profile and validation need {work} bounded steps, "
                f"exceeding {MAX_LOCAL_PROFILE_WORK}"
            )
        return self


class PrimeTupleLocalFactorResult(StrictModel):
    """Source-bound complete partition modulo one prime and its local factor."""

    source: PrimeAffineTuple
    prime: StrictInt = Field(ge=2, le=MAX_LOCAL_PROFILE_PRIME)
    residue_rows: tuple[PrimeTupleResidueRow, ...] = Field(
        min_length=2, max_length=MAX_LOCAL_PROFILE_PRIME
    )
    bad_count: StrictInt = Field(ge=0)
    valid_count: StrictInt = Field(ge=0)
    locally_obstructed: StrictBool
    factor: CanonicalRational

    @model_validator(mode="after")
    def bind_complete_local_factor(self) -> Self:
        if (
            sum(len(row.vanishing_form_ids) for row in self.residue_rows)
            > self.source.form_count
        ):
            raise ValueError(
                "residue incidence count exceeds the source affine-form count"
            )
        PrimeTupleLocalFactorRequest(source=self.source, prime=self.prime)
        bad = dict(local_bad_residues(self.source, self.prime))
        expected_rows = tuple(
            (residue, bad.get(residue, ())) for residue in range(self.prime)
        )
        actual_rows = tuple(
            (row.residue, row.vanishing_form_ids) for row in self.residue_rows
        )
        if actual_rows != expected_rows:
            raise ValueError("residue rows must be the complete source-bound partition")
        expected_bad = len(bad)
        expected_valid = self.prime - expected_bad
        if self.bad_count != expected_bad or self.valid_count != expected_valid:
            raise ValueError("local counts do not match the complete residue rows")
        if self.locally_obstructed != (expected_valid == 0):
            raise ValueError("local obstruction status must equal valid_count == 0")
        if self.factor.as_fraction() != local_factor_from_bad_count(
            self.source.form_count, self.prime, expected_bad
        ):
            raise ValueError("local factor does not satisfy its defining formula")
        return self


class PrimeTupleLocalFactorsRequest(StrictModel):
    """Compute compact local factors for a canonical finite prime set."""

    source: PrimeAffineTuple
    primes: tuple[CompactPrime, ...] = Field(
        min_length=1,
        max_length=MAX_PRIME_BATCH,
        description=(
            "Distinct primes in strictly increasing order. The aggregate form/root "
            "and exact rational-output bounds are validated before computation."
        ),
    )

    @model_validator(mode="after")
    def require_bounded_factor_batch(self) -> Self:
        _require_prime_set(self.primes, maximum=MAX_BATCH_PRIME)
        root_cells = self.source.form_count * len(self.primes)
        root_work = 6 * root_cells
        if root_work > MAX_BATCH_ROOT_WORK:
            raise ValueError(
                f"local-factor computation and validation need {root_work} root "
                f"steps, exceeding {MAX_BATCH_ROOT_WORK}"
            )
        digit_bounds = tuple(
            _factor_digit_upper_bound(self.source, prime) for prime in self.primes
        )
        if any(bound > MAX_FACTOR_COMPONENT_DIGITS for bound in digit_bounds):
            raise ValueError(
                "one local factor exceeds the exact rational component-digit bound"
            )
        if sum(digit_bounds) > MAX_FACTOR_PRODUCT_DIGITS:
            raise ValueError(
                "finite factor product exceeds the conservative exact rational "
                f"digit bound {MAX_FACTOR_PRODUCT_DIGITS}"
            )
        estimated_characters = (
            _source_character_upper_bound(self.source)
            + sum(
                _summary_character_upper_bound(self.source, prime)
                for prime in self.primes
            )
            + 128 * len(self.primes)
            + 4 * sum(digit_bounds)
            + 256
        )
        if estimated_characters > MAX_RESULT_CHARACTER_BUDGET:
            raise ValueError(
                "finite factor result exceeds the conservative serialized bound"
            )
        return self


class PrimeTupleLocalFactorRow(StrictModel):
    summary: PrimeTupleLocalSummary
    factor: CanonicalRational


class FinitePrimeTupleFactorProduct(StrictModel):
    """Exact finite local-factor product, explicitly not an infinite series."""

    source: PrimeAffineTuple
    primes: tuple[CompactPrime, ...] = Field(min_length=1, max_length=MAX_PRIME_BATCH)
    rows: tuple[PrimeTupleLocalFactorRow, ...] = Field(
        min_length=1, max_length=MAX_PRIME_BATCH
    )
    product: CanonicalRational
    first_obstructing_prime: StrictInt | None = None

    @model_validator(mode="after")
    def bind_finite_factor_product(self) -> Self:
        PrimeTupleLocalFactorsRequest(source=self.source, primes=self.primes)
        if tuple(row.summary.prime for row in self.rows) != self.primes:
            raise ValueError(
                "local-factor rows must align with the canonical prime set"
            )
        expected_product = Fraction(1, 1)
        expected_first: int | None = None
        for row in self.rows:
            _require_summary(self.source, row.summary)
            expected_factor = local_factor_from_bad_count(
                self.source.form_count,
                row.summary.prime,
                row.summary.bad_count,
            )
            if row.factor.as_fraction() != expected_factor:
                raise ValueError(
                    "local factor row does not satisfy its defining formula"
                )
            expected_product *= expected_factor
            if expected_first is None and row.summary.valid_count == 0:
                expected_first = row.summary.prime
        if self.product.as_fraction() != expected_product:
            raise ValueError("finite product must equal the product of every local row")
        if self.first_obstructing_prime != expected_first:
            raise ValueError("first obstructing prime does not match the local rows")
        return self


class PrimeTupleAdmissibilityRequest(StrictModel):
    """Decide local admissibility by checking exactly the primes at most k."""

    source: PrimeAffineTuple

    @model_validator(mode="after")
    def require_bounded_cutoff_profile(self) -> Self:
        cutoff = self.source.form_count
        prime_rows = int(primepi(cutoff))
        if prime_rows > MAX_ADMISSIBILITY_PRIME_ROWS:
            raise ValueError(
                f"admissibility needs {prime_rows} prime rows, exceeding "
                f"{MAX_ADMISSIBILITY_PRIME_ROWS}"
            )
        root_cells = self.source.form_count * prime_rows
        total_root_cells = 4 * root_cells
        if total_root_cells > MAX_ADMISSIBILITY_ROOT_CELLS:
            raise ValueError(
                "admissibility computation and validation may require "
                f"{total_root_cells} root cells, "
                f"exceeding {MAX_ADMISSIBILITY_ROOT_CELLS}"
            )
        estimated_characters = (
            _source_character_upper_bound(self.source)
            + sum(
                _summary_character_upper_bound(self.source, prime)
                for prime in primes_through(cutoff)
            )
            + 16 * prime_rows
            + 256
        )
        if estimated_characters > MAX_RESULT_CHARACTER_BUDGET:
            raise ValueError(
                "admissibility profile exceeds the conservative serialized bound"
            )
        return self


class PrimeTupleAdmissibilityResult(StrictModel):
    """Closed decision: every p<=k is checked and every p>k has nu_p<=k<p."""

    source: PrimeAffineTuple
    cutoff: StrictInt = Field(ge=1, le=MAX_ADMISSIBILITY_CUTOFF)
    checked_primes: tuple[StrictInt, ...] = Field(
        max_length=MAX_ADMISSIBILITY_PRIME_ROWS
    )
    local_rows: tuple[PrimeTupleLocalSummary, ...] = Field(
        max_length=MAX_ADMISSIBILITY_PRIME_ROWS
    )
    status: Literal["LOCALLY_ADMISSIBLE", "LOCALLY_OBSTRUCTED"]
    least_obstructing_prime: StrictInt | None = None
    large_prime_lower_bound: StrictInt = Field(ge=2)
    maximum_large_prime_bad_residues: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def bind_cutoff_decision(self) -> Self:
        PrimeTupleAdmissibilityRequest(source=self.source)
        expected_cutoff = self.source.form_count
        expected_primes = primes_through(expected_cutoff)
        if self.cutoff != expected_cutoff or self.checked_primes != expected_primes:
            raise ValueError(
                "checked primes must be exactly every prime through the cutoff"
            )
        if tuple(row.prime for row in self.local_rows) != expected_primes:
            raise ValueError("local rows must align with every checked prime")
        for row in self.local_rows:
            _require_summary(self.source, row)
        obstructing = tuple(
            row.prime for row in self.local_rows if row.valid_count == 0
        )
        expected_status = "LOCALLY_OBSTRUCTED" if obstructing else "LOCALLY_ADMISSIBLE"
        if self.status != expected_status:
            raise ValueError("admissibility status does not match the local rows")
        expected_first = obstructing[0] if obstructing else None
        if self.least_obstructing_prime != expected_first:
            raise ValueError("least obstructing prime does not match the local rows")
        if (
            self.large_prime_lower_bound != expected_cutoff + 1
            or self.maximum_large_prime_bad_residues != self.source.form_count
        ):
            raise ValueError("large-prime cutoff evidence does not match the source")
        return self


class PrimeTupleResidueWheelRequest(StrictModel):
    """Construct a compact exact CRT wheel for a canonical finite prime set."""

    source: PrimeAffineTuple
    primes: tuple[CompactPrime, ...] = Field(
        max_length=MAX_WHEEL_PRIMES,
        description=(
            "Possibly empty tuple of distinct primes in strictly increasing order; "
            "the empty set returns the modulus-one identity wheel. This compact "
            "operation does not enumerate all combined residues."
        ),
    )

    @model_validator(mode="after")
    def require_bounded_compact_wheel(self) -> Self:
        _require_prime_set(self.primes, maximum=MAX_BATCH_PRIME)
        root_cells = self.source.form_count * len(self.primes)
        root_work = 6 * root_cells
        if root_work > MAX_COMPACT_WHEEL_ROOT_WORK:
            raise ValueError(
                f"compact wheel computation and validation need {root_work} root "
                f"steps, exceeding {MAX_COMPACT_WHEEL_ROOT_WORK}"
            )
        modulus_digits = sum(_digits(prime) for prime in self.primes) or 1
        if modulus_digits > MAX_COMPACT_WHEEL_SCALAR_DIGITS:
            raise ValueError(
                "compact wheel modulus exceeds the conservative exact scalar "
                f"digit bound {MAX_COMPACT_WHEEL_SCALAR_DIGITS}"
            )
        estimated_characters = (
            _source_character_upper_bound(self.source)
            + sum(
                _summary_character_upper_bound(self.source, prime)
                for prime in self.primes
            )
            + 2 * modulus_digits
            + 128
        )
        if estimated_characters > MAX_RESULT_CHARACTER_BUDGET:
            raise ValueError("compact wheel exceeds the conservative serialized bound")
        return self


class PrimeTupleWheelResidueRow(StrictModel):
    residue: CompactWheelScalar
    components: tuple[StrictInt, ...] = Field(max_length=MAX_WHEEL_PRIMES)


class PrimeTupleResidueWheel(StrictModel):
    """Compact source-bound CRT wheel as a product of valid local residue sets."""

    source: PrimeAffineTuple
    primes: tuple[CompactPrime, ...] = Field(max_length=MAX_WHEEL_PRIMES)
    local_rows: tuple[PrimeTupleLocalSummary, ...] = Field(max_length=MAX_WHEEL_PRIMES)
    modulus: CompactWheelScalar
    valid_count: CompactWheelScalar

    @model_validator(mode="after")
    def bind_compact_crt_wheel(self) -> Self:
        PrimeTupleResidueWheelRequest(source=self.source, primes=self.primes)
        if tuple(row.prime for row in self.local_rows) != self.primes:
            raise ValueError("wheel local rows must align with its canonical primes")
        for row in self.local_rows:
            _require_summary(self.source, row)
        expected_modulus = wheel_modulus(self.primes)
        if parse_canonical_integer(self.modulus) != expected_modulus:
            raise ValueError("wheel modulus must equal the product of its primes")
        expected_count = prod((row.valid_count for row in self.local_rows), start=1)
        if parse_canonical_integer(self.valid_count) != expected_count:
            raise ValueError("wheel valid_count must equal the product of local counts")
        return self


class PrimeTupleResidueWheelEnumerationRequest(StrictModel):
    """Materialize every residue of a supplied compact wheel under strict bounds."""

    wheel: PrimeTupleResidueWheel

    @model_validator(mode="after")
    def require_bounded_wheel_enumeration(self) -> Self:
        local_residue_rows = sum(self.wheel.primes)
        if local_residue_rows > MAX_WHEEL_LOCAL_RESIDUES:
            raise ValueError(
                f"wheel local residue enumeration {local_residue_rows} exceeds "
                f"{MAX_WHEEL_LOCAL_RESIDUES}"
            )
        result_count = parse_canonical_integer(self.wheel.valid_count)
        if result_count > MAX_WHEEL_RESIDUES:
            raise ValueError(
                f"wheel has {result_count} valid residues, exceeding "
                f"{MAX_WHEEL_RESIDUES}"
            )
        result_cells = result_count * (len(self.wheel.primes) + 1)
        if result_cells > MAX_WHEEL_RESULT_CELLS:
            raise ValueError(
                f"wheel result needs {result_cells} cells, exceeding "
                f"{MAX_WHEEL_RESULT_CELLS}"
            )
        root_cells = self.wheel.source.form_count * len(self.wheel.primes)
        replay_work = 2 * (
            result_count * len(self.wheel.primes) + local_residue_rows + root_cells
        )
        if replay_work > MAX_WHEEL_ENUMERATION_WORK:
            raise ValueError(
                f"wheel enumeration and validation need {replay_work} bounded "
                f"steps, exceeding {MAX_WHEEL_ENUMERATION_WORK}"
            )
        modulus_digits = _digits(self.wheel.modulus)
        component_digits = sum(_digits(prime) for prime in self.wheel.primes)
        serialized_characters = (
            len(self.wheel.model_dump_json())
            + result_count
            * (modulus_digits + component_digits + 4 * len(self.wheel.primes) + 64)
            + 128
        )
        if serialized_characters > MAX_RESULT_CHARACTER_BUDGET:
            raise ValueError(
                "wheel enumeration exceeds the conservative serialized bound"
            )
        return self


class PrimeTupleResidueWheelEnumeration(StrictModel):
    """Complete explicit residue realization of one compact CRT wheel."""

    wheel: PrimeTupleResidueWheel
    residues: tuple[PrimeTupleWheelResidueRow, ...] = Field(
        max_length=MAX_WHEEL_RESIDUES
    )

    @model_validator(mode="after")
    def bind_complete_crt_enumeration(self) -> Self:
        result_cells = len(self.residues) + sum(
            len(row.components) for row in self.residues
        )
        if result_cells > MAX_WHEEL_RESULT_CELLS:
            raise ValueError("wheel rows exceed the explicit result-cell bound")
        PrimeTupleResidueWheelEnumerationRequest(wheel=self.wheel)
        expected_count = parse_canonical_integer(self.wheel.valid_count)
        if len(self.residues) != expected_count:
            raise ValueError(
                "wheel rows do not satisfy the complete CRT reconstruction invariant"
            )
        residues = tuple(parse_canonical_integer(row.residue) for row in self.residues)
        if residues != tuple(sorted(set(residues))):
            raise ValueError(
                "wheel rows do not satisfy the complete CRT reconstruction invariant"
            )
        modulus = parse_canonical_integer(self.wheel.modulus)
        for row, residue in zip(self.residues, residues, strict=True):
            if not 0 <= residue < modulus or len(row.components) != len(
                self.wheel.primes
            ):
                raise ValueError(
                    "wheel rows do not satisfy the complete CRT reconstruction invariant"
                )
            for prime, summary, component in zip(
                self.wheel.primes,
                self.wheel.local_rows,
                row.components,
                strict=True,
            ):
                bad = {item.residue for item in summary.bad_residues}
                if (
                    not 0 <= component < prime
                    or residue % prime != component
                    or component in bad
                ):
                    raise ValueError(
                        "wheel rows do not satisfy the complete CRT reconstruction "
                        "invariant"
                    )
        return self


class PrimeTupleWheelMembershipRequest(StrictModel):
    """Check one integer against an exact residue wheel supplied unchanged."""

    wheel: PrimeTupleResidueWheel
    value: AffineComponentInteger

    @model_validator(mode="after")
    def require_bounded_value(self) -> Self:
        if _digits(self.value) > MAX_AFFINE_COMPONENT_DIGITS:
            raise ValueError(
                f"membership value must have at most {MAX_AFFINE_COMPONENT_DIGITS} digits"
            )
        result_characters = (
            len(self.wheel.model_dump_json())
            + len(self.value)
            + _digits(self.wheel.modulus)
            + sum(_digits(prime) for prime in self.wheel.primes)
            + MAX_FORM_ID_LENGTH * self.wheel.source.form_count
            + 256
        )
        if result_characters > MAX_RESULT_CHARACTER_BUDGET:
            raise ValueError(
                "wheel membership result exceeds the conservative serialized bound"
            )
        return self


class PrimeTupleWheelMembershipResult(StrictModel):
    wheel: PrimeTupleResidueWheel
    value: AffineComponentInteger
    canonical_residue: CompactWheelScalar
    components: tuple[StrictInt, ...] = Field(max_length=MAX_WHEEL_PRIMES)
    is_permitted: StrictBool
    first_excluded_prime: StrictInt | None = None
    vanishing_form_ids: tuple[AffineFormId, ...] = Field(max_length=MAX_AFFINE_FORMS)

    @model_validator(mode="after")
    def bind_wheel_membership(self) -> Self:
        PrimeTupleWheelMembershipRequest(wheel=self.wheel, value=self.value)
        value = parse_canonical_integer(self.value)
        modulus = parse_canonical_integer(self.wheel.modulus)
        expected_residue = value % modulus
        expected_components = tuple(value % prime for prime in self.wheel.primes)
        expected_prime: int | None = None
        expected_form_ids: tuple[str, ...] = ()
        for summary, component in zip(
            self.wheel.local_rows, expected_components, strict=True
        ):
            bad = {row.residue: row.form_ids for row in summary.bad_residues}
            if component in bad:
                expected_prime = summary.prime
                expected_form_ids = bad[component]
                break
        expected_permitted = expected_prime is None
        if parse_canonical_integer(self.canonical_residue) != expected_residue:
            raise ValueError("canonical residue does not match the wheel modulus")
        if self.components != expected_components:
            raise ValueError("prime components do not match the supplied integer")
        if self.is_permitted != expected_permitted:
            raise ValueError("wheel membership does not match the complete residue set")
        if (
            self.first_excluded_prime != expected_prime
            or self.vanishing_form_ids != expected_form_ids
        ):
            raise ValueError("first local exclusion does not match the source forms")
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
            raise ValueError(
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
            raise ValueError(
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
            raise ValueError(
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
            raise ValueError("interval_size must equal upper-lower+1")
        if self.affine_values_examined != interval_size * self.source.form_count:
            raise ValueError(
                "affine_values_examined does not match the complete interval"
            )
        if self.match_count != expected_count:
            raise ValueError("match_count does not match exact interval primality")
        if (
            None
            if self.first_match is None
            else parse_canonical_integer(self.first_match)
        ) != expected_first or (
            None
            if self.last_match is None
            else parse_canonical_integer(self.last_match)
        ) != expected_last:
            raise ValueError(
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
            raise ValueError("matches exceed the interval result-cell bound")
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
            raise ValueError(
                "matches must be every and only positive-prime affine tuple"
            )
        if self.interval_size != interval_size:
            raise ValueError("interval_size must equal upper-lower+1")
        if self.affine_values_examined != interval_size * self.source.form_count:
            raise ValueError(
                "affine_values_examined does not match the complete interval"
            )
        return self


class PrimeTupleIntervalResidueProfileRequest(StrictModel):
    """Enumerate local-wheel survivors on one bounded closed integer interval."""

    wheel: PrimeTupleResidueWheel
    lower: IntervalEndpointInteger
    upper: IntervalEndpointInteger

    @model_validator(mode="after")
    def require_bounded_survivor_profile(self) -> Self:
        _, _, interval_size = _parse_interval(self.lower, self.upper)
        if interval_size > MAX_WHEEL_INTERVAL_LENGTH:
            raise ValueError(
                f"wheel interval length {interval_size} exceeds "
                f"{MAX_WHEEL_INTERVAL_LENGTH}"
            )
        replay_work = 2 * interval_size * max(1, len(self.wheel.primes))
        if replay_work > MAX_INTERVAL_REPLAY_EVALUATIONS:
            raise ValueError(
                f"wheel interval profile and validation need {replay_work} "
                f"modular checks, exceeding {MAX_INTERVAL_REPLAY_EVALUATIONS}"
            )
        endpoint_digits = max(_digits(self.lower), _digits(self.upper))
        result_characters = (
            len(self.wheel.model_dump_json())
            + interval_size * (endpoint_digits + 4)
            + 192
        )
        if result_characters > MAX_RESULT_CHARACTER_BUDGET:
            raise ValueError(
                "wheel interval profile exceeds the conservative serialized bound"
            )
        return self


class PrimeTupleIntervalResidueProfileResult(StrictModel):
    """Exact local survivors; this value does not assert actual primality."""

    wheel: PrimeTupleResidueWheel
    lower: IntervalEndpointInteger
    upper: IntervalEndpointInteger
    interval_size: StrictInt = Field(ge=1, le=MAX_WHEEL_INTERVAL_LENGTH)
    survivors: tuple[IntervalEndpointInteger, ...] = Field(
        max_length=MAX_WHEEL_INTERVAL_LENGTH
    )

    @model_validator(mode="after")
    def bind_complete_survivor_family(self) -> Self:
        PrimeTupleIntervalResidueProfileRequest(
            wheel=self.wheel, lower=self.lower, upper=self.upper
        )
        lower, upper, interval_size = _parse_interval(self.lower, self.upper)
        bad_by_prime = tuple(
            {row.residue for row in summary.bad_residues}
            for summary in self.wheel.local_rows
        )
        expected = tuple(
            value
            for value in range(lower, upper + 1)
            if all(
                value % prime not in bad
                for prime, bad in zip(self.wheel.primes, bad_by_prime, strict=True)
            )
        )
        actual = tuple(parse_canonical_integer(value) for value in self.survivors)
        if actual != expected:
            raise ValueError("survivors must be the complete local wheel profile")
        if self.interval_size != interval_size:
            raise ValueError("interval_size must equal upper-lower+1")
        return self


class PrimeAffineTranslationRequest(StrictModel):
    """Translate every source form by the variable substitution n -> n+shift."""

    source: PrimeAffineTuple
    shift: IntervalEndpointInteger

    @model_validator(mode="after")
    def require_bounded_translated_tuple(self) -> Self:
        if _digits(self.shift) > MAX_INTERVAL_ENDPOINT_DIGITS:
            raise ValueError(
                f"translation shift must have at most {MAX_INTERVAL_ENDPOINT_DIGITS} digits"
            )
        shift = parse_canonical_integer(self.shift)
        aggregate_digits = 0
        for form in self.source.forms:
            translated_constant = form.evaluate(shift)
            if _digits(translated_constant) > MAX_AFFINE_COMPONENT_DIGITS:
                raise ValueError(
                    "translated constant exceeds the canonical affine component bound"
                )
            aggregate_digits += _digits(form.coefficient) + _digits(translated_constant)
        if aggregate_digits > MAX_AFFINE_AGGREGATE_DIGITS:
            raise ValueError(
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
            raise ValueError("translated tuple must equal L_i(n+shift) for every form")
        return self


__all__ = [
    "MAX_ADMISSIBILITY_CUTOFF",
    "MAX_BATCH_PRIME",
    "MAX_INTERVAL_ENUMERATION_CELLS",
    "MAX_INTERVAL_REPLAY_EVALUATIONS",
    "MAX_LOCAL_PROFILE_PRIME",
    "MAX_PRIME_BATCH",
    "MAX_WHEEL_INTERVAL_LENGTH",
    "MAX_WHEEL_PRIMES",
    "MAX_WHEEL_RESIDUES",
    "FinitePrimeTupleFactorProduct",
    "PrimeAffineIntervalCountRequest",
    "PrimeAffineIntervalEnumerateRequest",
    "PrimeAffineTranslationRequest",
    "PrimeAffineTranslationResult",
    "PrimePatternIntervalCountResult",
    "PrimePatternIntervalEnumerateResult",
    "PrimePatternMatch",
    "PrimeTupleAdmissibilityRequest",
    "PrimeTupleAdmissibilityResult",
    "PrimeTupleBadResidueRow",
    "PrimeTupleIntervalResidueProfileRequest",
    "PrimeTupleIntervalResidueProfileResult",
    "PrimeTupleLocalFactorRequest",
    "PrimeTupleLocalFactorResult",
    "PrimeTupleLocalFactorRow",
    "PrimeTupleLocalFactorsRequest",
    "PrimeTupleLocalSummary",
    "PrimeTupleResidueRow",
    "PrimeTupleResidueWheel",
    "PrimeTupleResidueWheelEnumeration",
    "PrimeTupleResidueWheelEnumerationRequest",
    "PrimeTupleResidueWheelRequest",
    "PrimeTupleWheelMembershipRequest",
    "PrimeTupleWheelMembershipResult",
    "PrimeTupleWheelResidueRow",
]
