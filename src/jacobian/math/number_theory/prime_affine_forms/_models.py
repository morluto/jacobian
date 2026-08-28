"""Typed contracts for exact prime-affine local arithmetic."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError
from sympy import isprime

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.affine_forms.values import AffineFormId
from jacobian.math.number_theory.prime_affine_forms._kernel import (
    local_bad_residues,
)
from jacobian.math.number_theory.prime_affine_forms.values import (
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


def _run_admission[ResultT](admission: Callable[[], ResultT]) -> ResultT:
    """Run owner admission and expose its diagnostic through the domain API."""

    try:
        return admission()
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc


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


__all__ = [
    "MAX_BATCH_PRIME",
    "MAX_LOCAL_PROFILE_PRIME",
    "MAX_PRIME_BATCH",
    "PrimeTupleBadResidueRow",
    "PrimeTupleLocalSummary",
    "PrimeTupleResidueRow",
]
