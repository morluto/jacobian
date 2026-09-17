"""Local admissibility decisions with a replayed finite cutoff (#1705).

A family is locally admissible when no prime's square divides any form
value on a complete residue system.  For all sufficiently large primes
this follows from an elementary bound replayed from the exact family:
with ``A`` the largest nonzero coefficient magnitude and ``M`` covering
the form count and constant magnitudes, every prime ``p > B`` with
``B = max(A, isqrt(M))`` leaves a valid residue.  Finitely many primes
``p <= B`` are decided by exact local-factor rows.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError
from sympy import isprime

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.squarefree_affine_forms._euler_product import (
    SquarefreeLocalFactorRow,
)
from jacobian.math.number_theory.squarefree_affine_forms._kernel import (
    closed_form_ledger,
    form_solution_profile,
)
from jacobian.math.number_theory.squarefree_affine_forms._models import (
    MAX_LOCAL_FACTOR_PRIME,
    admit_admissibility_cutoff,
    admit_admissibility_work,
    admit_family,
)
from jacobian.math.number_theory.squarefree_affine_forms.values import (
    SquarefreeAffineFamily,
)


def _invariant_error(message: str) -> PydanticCustomError:
    return PydanticCustomError(
        "number_theory.squarefree_affine.admissibility_invariant", message
    )


def admissibility_cutoff(source: SquarefreeAffineFamily) -> tuple[int, int, int]:
    """Return the exact ``(B, A, M)`` cutoff triple of one family.

    ``A`` is the largest nonzero coefficient magnitude (zero when every
    form is constant), ``M`` covers the form count and every nonzero
    constant magnitude, and ``B = max(A, isqrt(M))``.  Every prime
    ``p > B`` then satisfies ``p > |a_j|`` for nonzero coefficients and
    ``p^2 > M``, so each nonconstant form excludes exactly one residue,
    no constant form excludes any, and the union cannot cover ``p^2``.
    """

    coefficients = [abs(form.coefficient) for form in source.forms if form.coefficient]
    absolute = max(coefficients, default=0)
    constants = [
        abs(form.constant)
        for form in source.forms
        if form.constant and not form.coefficient
    ]
    magnitude = max([len(source.forms), *constants, 1])
    return max(absolute, isqrt(magnitude)), absolute, magnitude


def primes_up_to(bound: int) -> tuple[int, ...]:
    """Return every prime ``p <= bound`` by exact sieve."""

    if bound < 2:
        return ()
    sieve = bytearray(b"\x01") * (bound + 1)
    sieve[0:2] = b"\x00\x00"
    for factor in range(2, isqrt(bound) + 1):
        if sieve[factor]:
            sieve[factor * factor : bound + 1 : factor] = b"\x00" * (
                (bound - factor * factor) // factor + 1
            )
    return tuple(index for index, flagged in enumerate(sieve) if flagged)


class LocalAdmissibilityRequest(StrictModel):
    """Decide local square admissibility with a replayed finite cutoff."""

    source: SquarefreeAffineFamily


class LocalAdmissibilityResult(StrictModel):
    """Local admissibility with its exact cutoff rows and proof data.

    ``LOCALLY_ADMISSIBLE`` means every prime ``p <= cutoff`` has a valid
    residue (witnessed row by row) and the large-prime argument covers
    every ``p > cutoff``.  ``LOCALLY_OBSTRUCTED`` carries the first prime
    whose residues are all bad.  This decides local obstructions only --
    never existence, positive density, or infinitude.
    """

    source: SquarefreeAffineFamily
    status: Literal["LOCALLY_ADMISSIBLE", "LOCALLY_OBSTRUCTED"]
    cutoff: StrictInt = Field(ge=1, le=MAX_LOCAL_FACTOR_PRIME)
    max_abs_coefficient: StrictInt = Field(ge=0)
    large_prime_bound: StrictInt = Field(ge=1)
    rows: tuple[SquarefreeLocalFactorRow, ...]
    obstruction: SquarefreeLocalFactorRow | None = None

    @model_validator(mode="after")
    def require_cutoff_proof_data(self) -> Self:
        absolute = max(
            (abs(form.coefficient) for form in self.source.forms if form.coefficient),
            default=0,
        )
        constants = [
            abs(form.constant)
            for form in self.source.forms
            if form.constant and not form.coefficient
        ]
        magnitude = max([len(self.source.forms), *constants, 1])
        if self.max_abs_coefficient != absolute or self.large_prime_bound != magnitude:
            raise _invariant_error(
                "cutoff data must replay the family coefficient and magnitude bounds"
            )
        if self.cutoff != max(absolute, isqrt(magnitude)):
            raise _invariant_error(
                "cutoff must equal max(A, isqrt(M)) of the admitted family"
            )
        return self

    @model_validator(mode="after")
    def require_complete_prime_rows(self) -> Self:
        if tuple(row.prime for row in self.rows) != primes_up_to(self.cutoff):
            raise _invariant_error("rows must cover exactly the primes through cutoff")
        for row in self.rows:
            if not isprime(row.prime):
                raise _invariant_error("every row prime must be prime")
            modulus = row.prime * row.prime
            bad: set[int] = set()
            for form in self.source.forms:
                count, root, stride = form_solution_profile(form, row.prime)
                if count == modulus:
                    bad = set(range(modulus))
                    break
                if root is not None and stride is not None:
                    bad.update(root + offset * stride for offset in range(count))
            if len(bad) != row.bad_count or modulus - len(bad) != row.valid_count:
                raise _invariant_error("every row must replay its residue partition")
            if row.has_local_obstruction != (row.valid_count == 0):
                raise _invariant_error("row obstruction flags must match valid counts")
            if row.modulus != modulus:
                raise _invariant_error("row modulus must equal the square of its prime")
        if self.status == "LOCALLY_OBSTRUCTED":
            if self.obstruction is None:
                raise _invariant_error("an obstructed family carries its first row")
            first = next((row for row in self.rows if row.valid_count == 0), None)
            if first is None or first != self.obstruction:
                raise _invariant_error(
                    "the obstruction must be the first fully covered prime"
                )
        elif self.obstruction is not None or any(
            row.valid_count == 0 for row in self.rows
        ):
            raise _invariant_error(
                "an admissible family carries no obstruction and no empty row"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: SquarefreeAffineFamily,
        status: str,
        cutoff: int,
        absolute: int,
        magnitude: int,
        rows: tuple[SquarefreeLocalFactorRow, ...],
        obstruction: SquarefreeLocalFactorRow | None,
    ) -> Self:
        return cls.model_construct(
            source=source,
            status=status,
            cutoff=cutoff,
            max_abs_coefficient=absolute,
            large_prime_bound=magnitude,
            rows=rows,
            obstruction=obstruction,
        )


def _factor_row(source: SquarefreeAffineFamily, prime: int) -> SquarefreeLocalFactorRow:
    """Build one compact local-factor row from the closed-form ledger."""

    modulus = prime * prime
    profiles, ledger, covers_all = closed_form_ledger(source, prime)
    _ = profiles
    bad_count = modulus if covers_all else len(ledger)
    valid_count = modulus - bad_count
    return SquarefreeLocalFactorRow(
        prime=prime,
        modulus=modulus,
        bad_count=bad_count,
        valid_count=valid_count,
        local_factor=CanonicalRational.from_fraction(Fraction(valid_count, modulus)),
        has_local_obstruction=valid_count == 0,
    )


def local_admissibility(
    source: SquarefreeAffineFamily,
) -> LocalAdmissibilityResult:
    """Decide local admissibility by finite check plus large-prime proof."""

    admit_family(source)
    cutoff, absolute, magnitude = admissibility_cutoff(source)
    admit_admissibility_cutoff(cutoff)
    primes = primes_up_to(cutoff)
    admit_admissibility_work(len(source.forms), primes)
    rows = tuple(_factor_row(source, prime) for prime in primes)
    obstruction = next((row for row in rows if row.valid_count == 0), None)
    return LocalAdmissibilityResult._from_kernel(
        source=source,
        status="LOCALLY_OBSTRUCTED"
        if obstruction is not None
        else "LOCALLY_ADMISSIBLE",
        cutoff=cutoff,
        absolute=absolute,
        magnitude=magnitude,
        rows=rows,
        obstruction=obstruction,
    )


def verify_local_admissibility(claim: LocalAdmissibilityResult) -> bool:
    """Check an admissibility claim by recomputing its cutoff and rows."""
    try:
        return local_admissibility(claim.source) == claim
    except (OperationDomainValidationError, OperationResourceAdmissionError):
        return False


__all__ = [
    "LocalAdmissibilityRequest",
    "LocalAdmissibilityResult",
    "admissibility_cutoff",
    "local_admissibility",
    "primes_up_to",
    "verify_local_admissibility",
]
