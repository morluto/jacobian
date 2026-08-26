"""Contracts and kernels for bounded prime-affine local factors."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.prime_affine_forms._kernel import (
    local_bad_residues,
    local_counts,
    local_factor_from_bad_count,
)
from jacobian.math.prime_affine_forms._models import (
    MAX_BATCH_PRIME,
    MAX_BATCH_ROOT_WORK,
    MAX_FACTOR_COMPONENT_DIGITS,
    MAX_FACTOR_PRODUCT_DIGITS,
    MAX_LOCAL_PROFILE_PRIME,
    MAX_LOCAL_PROFILE_WORK,
    MAX_PRIME_BATCH,
    MAX_RESULT_CHARACTER_BUDGET,
    CompactPrime,
    PrimeTupleBadResidueRow,
    PrimeTupleLocalSummary,
    PrimeTupleResidueRow,
    _require_prime,
    _require_prime_set,
    _require_summary,
    _source_character_upper_bound,
    _summary_character_upper_bound,
    _validation_error,
)
from jacobian.math.prime_affine_forms.values import PrimeAffineTuple


def _digits(value: int | str) -> int:
    return len(str(value).lstrip("-"))


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
        raise _validation_error(
            "local-factor numerator or denominator may require "
            f"{bound} digits, exceeding the bound {MAX_FACTOR_COMPONENT_DIGITS}"
        )


def local_summary(source: PrimeAffineTuple, prime: int) -> PrimeTupleLocalSummary:
    """Return the canonical compact local obstruction summary."""

    bad = local_bad_residues(source, prime)
    return PrimeTupleLocalSummary(
        prime=prime,
        bad_residues=tuple(
            PrimeTupleBadResidueRow(residue=residue, form_ids=form_ids)
            for residue, form_ids in bad
        ),
        bad_count=len(bad),
        valid_count=prime - len(bad),
    )


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
            raise _validation_error(
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
            raise _validation_error(
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
            raise _validation_error(
                "residue rows must be the complete source-bound partition"
            )
        expected_bad = len(bad)
        expected_valid = self.prime - expected_bad
        if self.bad_count != expected_bad or self.valid_count != expected_valid:
            raise _validation_error(
                "local counts do not match the complete residue rows"
            )
        if self.locally_obstructed != (expected_valid == 0):
            raise _validation_error(
                "local obstruction status must equal valid_count == 0"
            )
        if self.factor.as_fraction() != local_factor_from_bad_count(
            self.source.form_count, self.prime, expected_bad
        ):
            raise _validation_error(
                "local factor does not satisfy its defining formula"
            )
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
            raise _validation_error(
                f"local-factor computation and validation need {root_work} root "
                f"steps, exceeding {MAX_BATCH_ROOT_WORK}"
            )
        digit_bounds = tuple(
            _factor_digit_upper_bound(self.source, prime) for prime in self.primes
        )
        if any(bound > MAX_FACTOR_COMPONENT_DIGITS for bound in digit_bounds):
            raise _validation_error(
                "one local factor exceeds the exact rational component-digit bound"
            )
        if sum(digit_bounds) > MAX_FACTOR_PRODUCT_DIGITS:
            raise _validation_error(
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
            raise _validation_error(
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
            raise _validation_error(
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
                raise _validation_error(
                    "local factor row does not satisfy its defining formula"
                )
            expected_product *= expected_factor
            if expected_first is None and row.summary.valid_count == 0:
                expected_first = row.summary.prime
        if self.product.as_fraction() != expected_product:
            raise _validation_error(
                "finite product must equal the product of every local row"
            )
        if self.first_obstructing_prime != expected_first:
            raise _validation_error(
                "first obstructing prime does not match the local rows"
            )
        return self


def compute_local_factor(
    request: PrimeTupleLocalFactorRequest,
) -> PrimeTupleLocalFactorResult:
    """Return one complete local residue partition and exact factor."""

    bad = dict(local_bad_residues(request.source, request.prime))
    return PrimeTupleLocalFactorResult(
        source=request.source,
        prime=request.prime,
        residue_rows=tuple(
            PrimeTupleResidueRow(
                residue=residue,
                vanishing_form_ids=bad.get(residue, ()),
            )
            for residue in range(request.prime)
        ),
        bad_count=len(bad),
        valid_count=request.prime - len(bad),
        locally_obstructed=len(bad) == request.prime,
        factor=CanonicalRational.from_fraction(
            local_factor_from_bad_count(
                request.source.form_count, request.prime, len(bad)
            )
        ),
    )


def compute_local_factors(
    request: PrimeTupleLocalFactorsRequest,
) -> FinitePrimeTupleFactorProduct:
    """Return compact exact local factors over one finite prime set."""

    product = Fraction(1, 1)
    rows: list[PrimeTupleLocalFactorRow] = []
    first_obstruction: int | None = None
    for prime in request.primes:
        summary = local_summary(request.source, prime)
        factor = local_factor_from_bad_count(
            request.source.form_count, prime, summary.bad_count
        )
        rows.append(
            PrimeTupleLocalFactorRow(
                summary=summary,
                factor=CanonicalRational.from_fraction(factor),
            )
        )
        product *= factor
        if first_obstruction is None and summary.valid_count == 0:
            first_obstruction = prime
    return FinitePrimeTupleFactorProduct(
        source=request.source,
        primes=request.primes,
        rows=tuple(rows),
        product=CanonicalRational.from_fraction(product),
        first_obstructing_prime=first_obstruction,
    )


__all__ = [
    "FinitePrimeTupleFactorProduct",
    "PrimeTupleLocalFactorRequest",
    "PrimeTupleLocalFactorResult",
    "PrimeTupleLocalFactorRow",
    "PrimeTupleLocalFactorsRequest",
    "compute_local_factor",
    "compute_local_factors",
    "local_summary",
]
