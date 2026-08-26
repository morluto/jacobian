"""Typed contracts and bounded replay for prime-affine residue wheels."""

from __future__ import annotations

from math import prod
from typing import Annotated, Self

from pydantic import Field, StrictBool, StrictInt, StringConstraints, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.affine_forms.values import (
    MAX_AFFINE_COMPONENT_DIGITS,
    MAX_FORM_ID_LENGTH,
    AffineComponentInteger,
    AffineFormId,
)
from jacobian.math.prime_affine_forms._kernel import wheel_modulus
from jacobian.math.prime_affine_forms._models import (
    MAX_BATCH_PRIME,
    MAX_INTERVAL_REPLAY_EVALUATIONS,
    MAX_RESULT_CHARACTER_BUDGET,
    CompactPrime,
    IntervalEndpointInteger,
    PrimeTupleLocalSummary,
    _digits,
    _parse_interval,
    _require_prime_set,
    _require_summary,
    _source_character_upper_bound,
    _summary_character_upper_bound,
    _validation_error,
)
from jacobian.math.prime_affine_forms.values import (
    MAX_AFFINE_FORMS,
    PrimeAffineTuple,
)

MAX_WHEEL_PRIMES = 64
MAX_WHEEL_LOCAL_RESIDUES = 32_768
MAX_WHEEL_RESIDUES = 8_192
MAX_WHEEL_RESULT_CELLS = 65_536
MAX_WHEEL_ENUMERATION_WORK = 300_000
MAX_COMPACT_WHEEL_ROOT_WORK = 250_000
MAX_COMPACT_WHEEL_SCALAR_DIGITS = 4_096
MAX_WHEEL_INTERVAL_LENGTH = 8_192

CompactWheelScalar = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=MAX_COMPACT_WHEEL_SCALAR_DIGITS, strict=True),
]


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
            raise _validation_error(
                f"compact wheel computation and validation need {root_work} root "
                f"steps, exceeding {MAX_COMPACT_WHEEL_ROOT_WORK}"
            )
        modulus_digits = sum(_digits(prime) for prime in self.primes) or 1
        if modulus_digits > MAX_COMPACT_WHEEL_SCALAR_DIGITS:
            raise _validation_error(
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
            raise _validation_error(
                "compact wheel exceeds the conservative serialized bound"
            )
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
            raise _validation_error(
                "wheel local rows must align with its canonical primes"
            )
        for row in self.local_rows:
            _require_summary(self.source, row)
        expected_modulus = wheel_modulus(self.primes)
        if parse_canonical_integer(self.modulus) != expected_modulus:
            raise _validation_error(
                "wheel modulus must equal the product of its primes"
            )
        expected_count = prod((row.valid_count for row in self.local_rows), start=1)
        if parse_canonical_integer(self.valid_count) != expected_count:
            raise _validation_error(
                "wheel valid_count must equal the product of local counts"
            )
        return self


class PrimeTupleResidueWheelEnumerationRequest(StrictModel):
    """Materialize every residue of a supplied compact wheel under strict bounds."""

    wheel: PrimeTupleResidueWheel

    @model_validator(mode="after")
    def require_bounded_wheel_enumeration(self) -> Self:
        local_residue_rows = sum(self.wheel.primes)
        if local_residue_rows > MAX_WHEEL_LOCAL_RESIDUES:
            raise _validation_error(
                f"wheel local residue enumeration {local_residue_rows} exceeds "
                f"{MAX_WHEEL_LOCAL_RESIDUES}"
            )
        result_count = parse_canonical_integer(self.wheel.valid_count)
        if result_count > MAX_WHEEL_RESIDUES:
            raise _validation_error(
                f"wheel has {result_count} valid residues, exceeding "
                f"{MAX_WHEEL_RESIDUES}"
            )
        result_cells = result_count * (len(self.wheel.primes) + 1)
        if result_cells > MAX_WHEEL_RESULT_CELLS:
            raise _validation_error(
                f"wheel result needs {result_cells} cells, exceeding "
                f"{MAX_WHEEL_RESULT_CELLS}"
            )
        root_cells = self.wheel.source.form_count * len(self.wheel.primes)
        replay_work = 2 * (
            result_count * len(self.wheel.primes) + local_residue_rows + root_cells
        )
        if replay_work > MAX_WHEEL_ENUMERATION_WORK:
            raise _validation_error(
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
            raise _validation_error(
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
            raise _validation_error("wheel rows exceed the explicit result-cell bound")
        PrimeTupleResidueWheelEnumerationRequest(wheel=self.wheel)
        expected_count = parse_canonical_integer(self.wheel.valid_count)
        if len(self.residues) != expected_count:
            raise _validation_error(
                "wheel rows do not satisfy the complete CRT reconstruction invariant"
            )
        residues = tuple(parse_canonical_integer(row.residue) for row in self.residues)
        if residues != tuple(sorted(set(residues))):
            raise _validation_error(
                "wheel rows do not satisfy the complete CRT reconstruction invariant"
            )
        modulus = parse_canonical_integer(self.wheel.modulus)
        for row, residue in zip(self.residues, residues, strict=True):
            if not 0 <= residue < modulus or len(row.components) != len(
                self.wheel.primes
            ):
                raise _validation_error(
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
                    raise _validation_error(
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
            raise _validation_error(
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
            raise _validation_error(
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
            raise _validation_error(
                "canonical residue does not match the wheel modulus"
            )
        if self.components != expected_components:
            raise _validation_error(
                "prime components do not match the supplied integer"
            )
        if self.is_permitted != expected_permitted:
            raise _validation_error(
                "wheel membership does not match the complete residue set"
            )
        if (
            self.first_excluded_prime != expected_prime
            or self.vanishing_form_ids != expected_form_ids
        ):
            raise _validation_error(
                "first local exclusion does not match the source forms"
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
            raise _validation_error(
                f"wheel interval length {interval_size} exceeds "
                f"{MAX_WHEEL_INTERVAL_LENGTH}"
            )
        replay_work = 2 * interval_size * max(1, len(self.wheel.primes))
        if replay_work > MAX_INTERVAL_REPLAY_EVALUATIONS:
            raise _validation_error(
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
            raise _validation_error(
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
            raise _validation_error(
                "survivors must be the complete local wheel profile"
            )
        if self.interval_size != interval_size:
            raise _validation_error("interval_size must equal upper-lower+1")
        return self


__all__ = [
    "MAX_WHEEL_INTERVAL_LENGTH",
    "MAX_WHEEL_PRIMES",
    "MAX_WHEEL_RESIDUES",
    "PrimeTupleIntervalResidueProfileRequest",
    "PrimeTupleIntervalResidueProfileResult",
    "PrimeTupleResidueWheel",
    "PrimeTupleResidueWheelEnumeration",
    "PrimeTupleResidueWheelEnumerationRequest",
    "PrimeTupleResidueWheelRequest",
    "PrimeTupleWheelMembershipRequest",
    "PrimeTupleWheelMembershipResult",
    "PrimeTupleWheelResidueRow",
]
