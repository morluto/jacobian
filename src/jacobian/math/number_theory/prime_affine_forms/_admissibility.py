"""Contracts and kernel for bounded prime-affine local admissibility."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from sympy import primepi

from jacobian._models import StrictModel
from jacobian.math.number_theory.prime_affine_forms._kernel import primes_through
from jacobian.math.number_theory.prime_affine_forms._local_factors import local_summary
from jacobian.math.number_theory.prime_affine_forms._models import (
    MAX_RESULT_CHARACTER_BUDGET,
    PrimeTupleLocalSummary,
    _run_admission,
    _source_character_upper_bound,
    _summary_character_upper_bound,
    _validation_error,
)
from jacobian.math.number_theory.prime_affine_forms.values import (
    MAX_AFFINE_FORMS,
    PrimeAffineTuple,
)

MAX_ADMISSIBILITY_CUTOFF = MAX_AFFINE_FORMS
MAX_ADMISSIBILITY_PRIME_ROWS = 128
MAX_ADMISSIBILITY_ROOT_CELLS = 200_000


class PrimeTupleAdmissibilityRequest(StrictModel):
    """Decide local admissibility by checking exactly the primes at most k."""

    source: PrimeAffineTuple


def _admit_local_admissibility(source: PrimeAffineTuple) -> None:
    cutoff = source.form_count
    prime_rows = int(primepi(cutoff))
    if prime_rows > MAX_ADMISSIBILITY_PRIME_ROWS:
        raise _validation_error(
            f"admissibility needs {prime_rows} prime rows, exceeding "
            f"{MAX_ADMISSIBILITY_PRIME_ROWS}"
        )
    root_cells = source.form_count * prime_rows
    total_root_cells = 4 * root_cells
    if total_root_cells > MAX_ADMISSIBILITY_ROOT_CELLS:
        raise _validation_error(
            "admissibility computation may require "
            f"{total_root_cells} root cells, exceeding {MAX_ADMISSIBILITY_ROOT_CELLS}"
        )
    estimated_characters = (
        _source_character_upper_bound(source)
        + sum(
            _summary_character_upper_bound(source, prime)
            for prime in primes_through(cutoff)
        )
        + 16 * prime_rows
        + 256
    )
    if estimated_characters > MAX_RESULT_CHARACTER_BUDGET:
        raise _validation_error(
            "admissibility profile exceeds the conservative serialized bound"
        )


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
        expected_cutoff = self.source.form_count
        if self.cutoff != expected_cutoff:
            raise _validation_error("cutoff must equal the source form count")
        if self.checked_primes != tuple(sorted(set(self.checked_primes))):
            raise _validation_error("checked primes must be distinct and increasing")
        if tuple(row.prime for row in self.local_rows) != self.checked_primes:
            raise _validation_error("local rows must align with checked primes")
        obstructing = tuple(
            row.prime for row in self.local_rows if row.valid_count == 0
        )
        expected_status = "LOCALLY_OBSTRUCTED" if obstructing else "LOCALLY_ADMISSIBLE"
        if self.status != expected_status:
            raise _validation_error(
                "admissibility status does not match the local rows"
            )
        expected_first = obstructing[0] if obstructing else None
        if self.least_obstructing_prime != expected_first:
            raise _validation_error(
                "least obstructing prime does not match the local rows"
            )
        if (
            self.large_prime_lower_bound != expected_cutoff + 1
            or self.maximum_large_prime_bad_residues != self.source.form_count
        ):
            raise _validation_error(
                "large-prime cutoff evidence does not match the source"
            )
        return self


def compute_local_admissibility(
    request: PrimeTupleAdmissibilityRequest,
) -> PrimeTupleAdmissibilityResult:
    """Decide whether every prime-affine local factor has a valid residue."""

    _run_admission(lambda: _admit_local_admissibility(request.source))
    cutoff = request.source.form_count
    checked_primes = primes_through(cutoff)
    rows = tuple(local_summary(request.source, prime) for prime in checked_primes)
    obstructing = tuple(row.prime for row in rows if row.valid_count == 0)
    return PrimeTupleAdmissibilityResult(
        source=request.source,
        cutoff=cutoff,
        checked_primes=checked_primes,
        local_rows=rows,
        status="LOCALLY_OBSTRUCTED" if obstructing else "LOCALLY_ADMISSIBLE",
        least_obstructing_prime=obstructing[0] if obstructing else None,
        large_prime_lower_bound=cutoff + 1,
        maximum_large_prime_bad_residues=request.source.form_count,
    )


__all__ = [
    "PrimeTupleAdmissibilityRequest",
    "PrimeTupleAdmissibilityResult",
    "compute_local_admissibility",
]
