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
                f"local profile needs {work} bounded steps, "
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
    def require_local_factor_shape(self) -> Self:
        if (
            sum(len(row.vanishing_form_ids) for row in self.residue_rows)
            > self.source.form_count
        ):
            raise _validation_error(
                "residue incidence count exceeds the source affine-form count"
            )
        if tuple(row.residue for row in self.residue_rows) != tuple(range(self.prime)):
            raise _validation_error(
                "residue rows must be the canonical residue partition"
            )
        if self.bad_count + self.valid_count != self.prime:
            raise _validation_error("local counts must partition the prime modulus")
        if self.locally_obstructed != (self.valid_count == 0):
            raise _validation_error(
                "local obstruction status must equal valid_count == 0"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PrimeTupleLocalFactorRequest,
        *,
        bad: tuple[tuple[int, tuple[str, ...]], ...],
    ) -> Self:
        bad_by_residue = dict(bad)
        bad_count = len(bad)
        return cls(
            source=request.source,
            prime=request.prime,
            residue_rows=tuple(
                PrimeTupleResidueRow(
                    residue=residue,
                    vanishing_form_ids=bad_by_residue.get(residue, ()),
                )
                for residue in range(request.prime)
            ),
            bad_count=bad_count,
            valid_count=request.prime - bad_count,
            locally_obstructed=bad_count == request.prime,
            factor=CanonicalRational.from_fraction(
                local_factor_from_bad_count(
                    request.source.form_count, request.prime, bad_count
                )
            ),
        )


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
                f"local-factor computation needs {root_work} root "
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
    def require_finite_product_shape(self) -> Self:
        if tuple(row.summary.prime for row in self.rows) != self.primes:
            raise _validation_error(
                "local-factor rows must align with the canonical prime set"
            )
        for row in self.rows:
            if row.summary.bad_count + row.summary.valid_count != row.summary.prime:
                raise _validation_error("local summary counts must partition its prime")
        if (
            self.first_obstructing_prime is not None
            and self.first_obstructing_prime not in self.primes
        ):
            raise _validation_error(
                "first obstructing prime must belong to the canonical prime set"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PrimeTupleLocalFactorsRequest,
        *,
        rows: tuple[PrimeTupleLocalFactorRow, ...],
        product: Fraction,
        first_obstruction: int | None,
    ) -> Self:
        return cls(
            source=request.source,
            primes=request.primes,
            rows=rows,
            product=CanonicalRational.from_fraction(product),
            first_obstructing_prime=first_obstruction,
        )


def compute_local_factor(
    request: PrimeTupleLocalFactorRequest,
) -> PrimeTupleLocalFactorResult:
    """Return one complete local residue partition and exact factor."""

    return PrimeTupleLocalFactorResult._from_kernel(
        request, bad=local_bad_residues(request.source, request.prime)
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
    return FinitePrimeTupleFactorProduct._from_kernel(
        request,
        rows=tuple(rows),
        product=product,
        first_obstruction=first_obstruction,
    )


def verify_local_factor_result(result: PrimeTupleLocalFactorResult) -> bool:
    """Verify an independently supplied complete local-factor claim."""

    request = PrimeTupleLocalFactorRequest(source=result.source, prime=result.prime)
    return result == compute_local_factor(request)


def verify_local_factors_result(result: FinitePrimeTupleFactorProduct) -> bool:
    """Verify an independently supplied finite local-factor product claim."""

    request = PrimeTupleLocalFactorsRequest(source=result.source, primes=result.primes)
    return result == compute_local_factors(request)


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
