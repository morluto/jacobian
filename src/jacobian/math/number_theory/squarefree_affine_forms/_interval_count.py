"""Exact bounded interval counts of simultaneous square-free values (#1705).

For each ``n`` in ``[lower, upper]`` and each form, the least prime ``p``
with ``p^2`` dividing the form value is found by a prime-power sieve:
primes ascend, and linear congruences mark solution classes, so the first
mark on ``n`` is its least square divisor.  Zero values are divisible by
every square and record prime 2.  The scalar count is always returned; the
matching list and per-rejection obstructions are returned only when the
caller requests the ledger.
"""

from __future__ import annotations

from math import gcd, isqrt
from typing import Any, Self

from pydantic import Field, StrictBool, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.affine_forms.values import AffineFormId
from jacobian.math.number_theory.squarefree_affine_forms._admissibility import (
    primes_up_to,
)
from jacobian.math.number_theory.squarefree_affine_forms._models import (
    MAX_INTERVAL_LENGTH,
    admit_interval,
)
from jacobian.math.number_theory.squarefree_affine_forms.values import (
    SquarefreeAffineFamily,
)


def _invariant_error(message: str) -> PydanticCustomError:
    return PydanticCustomError(
        "number_theory.squarefree_affine.interval_invariant", message
    )


class IntervalCountRequest(StrictModel):
    """Count interval points where every affine form is square-free."""

    source: SquarefreeAffineFamily
    lower: StrictInt = Field(description="Inclusive interval lower bound.")
    upper: StrictInt = Field(description="Inclusive interval upper bound.")
    include_ledger: StrictBool = Field(
        default=False,
        description=(
            "Also return the matching points and the first square-divisor "
            "obstruction of every rejected point."
        ),
    )

    @model_validator(mode="after")
    def require_interval_order(self) -> Self:
        if self.lower > self.upper:
            raise _invariant_error("the interval lower bound must not exceed the upper")
        return self


class IntervalObstruction(StrictModel):
    """The first square divisor excluding one rejected interval point."""

    n: StrictInt
    form_id: AffineFormId
    prime: StrictInt = Field(ge=2)

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class IntervalCountResult(StrictModel):
    """Exact simultaneous square-free count over one bounded interval.

    ``count`` is always present.  With ``include_ledger``, ``matching``
    lists every accepted point in increasing order and ``obstructions``
    carries the least square divisor of every rejected point, so accepted
    and rejected points partition the interval exactly once.
    """

    source: SquarefreeAffineFamily
    lower: StrictInt
    upper: StrictInt
    include_ledger: StrictBool = False
    count: StrictInt = Field(ge=0)
    matching: tuple[StrictInt, ...] = ()
    obstructions: tuple[IntervalObstruction, ...] = ()

    @model_validator(mode="after")
    def require_interval_partition(self) -> Self:
        if self.lower > self.upper:
            raise _invariant_error("the interval lower bound must not exceed the upper")
        length = self.upper - self.lower + 1
        if length > MAX_INTERVAL_LENGTH or not 0 <= self.count <= length:
            raise _invariant_error(
                "the count must lie within the admitted interval length"
            )
        form_ids = {form.form_id for form in self.source.forms}
        if not self.include_ledger:
            if self.matching or self.obstructions:
                raise _invariant_error(
                    "a count-only result carries no matching list or ledger"
                )
            return self
        if tuple(self.matching) != tuple(sorted(set(self.matching))):
            raise _invariant_error("matching points are distinct and increasing")
        if any(not self.lower <= point <= self.upper for point in self.matching):
            raise _invariant_error("matching points must lie in the interval")
        rejected = [row.n for row in self.obstructions]
        if rejected != sorted(rejected) or len(set(rejected)) != len(rejected):
            raise _invariant_error("obstructions cover distinct increasing points")
        if any(not self.lower <= point <= self.upper for point in rejected):
            raise _invariant_error("obstructed points must lie in the interval")
        if set(self.matching) & set(rejected):
            raise _invariant_error("accepted and rejected points are disjoint")
        if len(self.matching) + len(rejected) != length:
            raise _invariant_error(
                "accepted and rejected points partition the interval"
            )
        if self.count != len(self.matching):
            raise _invariant_error("the count must equal the matching list")
        for row in self.obstructions:
            if row.form_id not in form_ids:
                raise _invariant_error("obstructions name declared form IDs")
            if row.prime < 2:
                raise _invariant_error("obstruction primes are at least two")
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


def _congruence_classes(
    coefficient: int, constant: int, modulus: int
) -> tuple[int, ...]:
    """Return residues ``n mod modulus`` with ``a*n+b == 0``, ascending."""

    divisor = gcd(coefficient, modulus)
    if constant % divisor != 0:
        return ()
    if divisor == modulus:
        return tuple(range(modulus))
    stride = modulus // divisor
    root = (-(constant // divisor) * pow(coefficient // divisor, -1, stride)) % stride
    step = modulus // divisor
    count = divisor
    return tuple(root + offset * step for offset in range(count))


def interval_count(
    source: SquarefreeAffineFamily,
    lower: int,
    upper: int,
    include_ledger: bool = False,
) -> IntervalCountResult:
    """Count interval points where every affine form value is square-free."""

    magnitude = admit_interval(source, lower, upper)
    if magnitude == 0:
        # Every form value is zero, hence divisible by every square: the
        # recorded obstruction is the least prime.
        zero_obstructions = (
            tuple(
                IntervalObstruction._from_kernel(
                    n=point, form_id=source.forms[0].form_id, prime=2
                )
                for point in range(lower, upper + 1)
            )
            if include_ledger
            else ()
        )
        return IntervalCountResult._from_kernel(
            source=source,
            lower=lower,
            upper=upper,
            include_ledger=include_ledger,
            count=0,
            matching=(),
            obstructions=zero_obstructions,
        )
    # A nonzero value divisible by a square has a prime ``p`` with
    # ``p**2 <= magnitude``, so ``isqrt(magnitude)`` bounds the useful primes.
    # The value ``0`` is divisible by every square, so the least prime must
    # always be probed even when that bound admits none.
    limit = max(isqrt(magnitude), 2)
    primes = primes_up_to(limit)
    # obstruction[n] = (form_index, prime), first hit wins: primes ascend and
    # forms keep source order, so the recorded pair is the least prime and,
    # on ties, the first form.
    obstruction: dict[int, tuple[int, int]] = {}
    for prime in primes:
        modulus = prime * prime
        for form_index, form in enumerate(source.forms):
            for root in _congruence_classes(form.coefficient, form.constant, modulus):
                # Smallest class member at or above the interval lower bound.
                point = root + (-((root - lower) // modulus)) * modulus
                while point <= upper:
                    obstruction.setdefault(point, (form_index, prime))
                    point += modulus
    matching: list[int] = []
    obstructions: list[IntervalObstruction] = []
    for point in range(lower, upper + 1):
        hit = obstruction.get(point)
        if hit is None:
            matching.append(point)
        elif include_ledger:
            form_index, prime = hit
            obstructions.append(
                IntervalObstruction._from_kernel(
                    n=point,
                    form_id=source.forms[form_index].form_id,
                    prime=prime,
                )
            )
    return IntervalCountResult._from_kernel(
        source=source,
        lower=lower,
        upper=upper,
        include_ledger=include_ledger,
        count=len(matching),
        matching=tuple(matching) if include_ledger else (),
        obstructions=tuple(obstructions) if include_ledger else (),
    )


def verify_interval_count(claim: IntervalCountResult) -> bool:
    """Check an interval count by recomputing it within its bounds."""
    try:
        return (
            interval_count(claim.source, claim.lower, claim.upper, claim.include_ledger)
            == claim
        )
    except (OperationDomainValidationError, OperationResourceAdmissionError):
        return False


__all__ = [
    "IntervalCountRequest",
    "IntervalCountResult",
    "IntervalObstruction",
    "interval_count",
    "verify_interval_count",
]
