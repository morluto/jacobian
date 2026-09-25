"""Shared admission and diagnostic conventions for square-free affine forms."""

from __future__ import annotations

from sympy import isprime

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.squarefree_affine_forms.values import (
    MAX_SQUAREFREE_COMPONENT_DIGITS,
    MAX_SQUAREFREE_FORMS,
    SquarefreeAffineFamily,
    SquarefreeAffineForm,
)

MAX_LOCAL_FACTOR_PRIME = 1_000
MAX_LOCAL_FACTOR_WORK = 8_000_000
MAX_LEDGER_ROWS = 8_192
MAX_EULER_PRIMES = 16
MAX_EULER_WORK = 32_000_000
MAX_INFINITE_PRODUCT_CUTOFF = 1_000
MAX_INFINITE_PRODUCT_WORK = 64_000_000
MAX_ADMISSIBILITY_CUTOFF = 1_000
MAX_ADMISSIBILITY_WORK = 64_000_000
MAX_INTERVAL_LENGTH = 20_000
MAX_INTERVAL_VALUE = 10**12
MAX_INTERVAL_SIEVE_RESIDUES = 4_000_000
MAX_INTERVAL_SIEVE_VISITS = 2_000_000

_CODE_PREFIX = "number_theory.squarefree_affine"


def _domain_error(code: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=(), code=f"{_CODE_PREFIX}.{code}", message=message
    )


def _resource_error(code: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=(), code=f"{_CODE_PREFIX}.{code}", message=message
    )


def admit_family(source: SquarefreeAffineFamily) -> None:
    """Re-check caller-supplied family claims against the owner envelope."""

    if not isinstance(source, SquarefreeAffineFamily):
        raise _domain_error(
            "family_source", "family must be a square-free affine family"
        )
    if not isinstance(source.forms, tuple) or any(
        not isinstance(form, SquarefreeAffineForm)
        or type(form.form_id) is not str
        or type(form.coefficient) is not int
        or type(form.constant) is not int
        for form in source.forms
    ):
        raise _domain_error(
            "family_form_source",
            "family forms must be labelled bounded integer affine forms",
        )
    if not 1 <= len(source.forms) <= MAX_SQUAREFREE_FORMS:
        raise _resource_error(
            "family_budget",
            f"family must contain between 1 and {MAX_SQUAREFREE_FORMS} forms",
        )
    form_ids = tuple(form.form_id for form in source.forms)
    if len(set(form_ids)) != len(form_ids):
        raise _domain_error("form_id_unique", "affine form IDs must be unique")
    coefficient_pairs = tuple(
        (form.coefficient, form.constant) for form in source.forms
    )
    if len(set(coefficient_pairs)) != len(coefficient_pairs):
        raise _domain_error("form_duplicate", "affine forms must be pairwise distinct")
    for form in source.forms:
        if form.coefficient == 0 and form.constant == 0:
            raise _domain_error("form_zero", "affine form must not be identically zero")
        if (
            len(str(abs(form.coefficient))) > MAX_SQUAREFREE_COMPONENT_DIGITS
            or len(str(abs(form.constant))) > MAX_SQUAREFREE_COMPONENT_DIGITS
        ):
            raise _resource_error(
                "component_digit_bound",
                "affine coefficient and constant must each have at most "
                f"{MAX_SQUAREFREE_COMPONENT_DIGITS} digits",
            )


def admit_prime(prime: int) -> None:
    """Require one prime inside the bounded p^2 enumeration envelope."""

    if prime < 2 or not isprime(prime):
        raise _domain_error("prime_required", "modulus base must be prime")
    if prime > MAX_LOCAL_FACTOR_PRIME:
        raise _resource_error(
            "prime_budget",
            f"prime must be at most {MAX_LOCAL_FACTOR_PRIME} so that the "
            f"p^2 residue envelope stays bounded ({prime} exceeds it)",
        )


def admit_prime_set(primes: tuple[int, ...]) -> None:
    if primes != tuple(sorted(set(primes))):
        raise _domain_error(
            "prime_order", "primes must be distinct and strictly increasing"
        )
    for prime in primes:
        admit_prime(prime)


def admit_local_factor_work(form_count: int, prime: int) -> None:
    """Preflight p^2 scan work and ledger rows before any residue arithmetic."""

    work = prime * prime * form_count
    if work > MAX_LOCAL_FACTOR_WORK:
        raise _resource_error(
            "work_budget",
            f"local factor needs {work} bounded congruence steps, exceeding "
            f"{MAX_LOCAL_FACTOR_WORK}",
        )
    ledger_rows = form_count * prime
    if ledger_rows > MAX_LEDGER_ROWS:
        raise _resource_error(
            "ledger_budget",
            f"bad-residue ledger may need {ledger_rows} rows, exceeding "
            f"{MAX_LEDGER_ROWS}",
        )


def admit_local_factor(source: SquarefreeAffineFamily, prime: int) -> None:
    admit_family(source)
    admit_prime(prime)
    admit_local_factor_work(source.form_count, prime)


def admit_euler_product(
    source: SquarefreeAffineFamily, primes: tuple[int, ...]
) -> None:
    admit_family(source)
    if not 1 <= len(primes) <= MAX_EULER_PRIMES:
        raise _resource_error(
            "prime_batch_budget",
            f"prime batch must contain between 1 and {MAX_EULER_PRIMES} primes",
        )
    admit_prime_set(primes)
    total_work = sum(prime * prime for prime in primes) * source.form_count
    if total_work > MAX_EULER_WORK:
        raise _resource_error(
            "work_budget",
            f"finite Euler product needs {total_work} bounded congruence "
            f"steps, exceeding {MAX_EULER_WORK}",
        )
    for prime in primes:
        admit_local_factor_work(source.form_count, prime)


def admit_infinite_product(source: SquarefreeAffineFamily, cutoff: int) -> None:
    """Admit a complete prime prefix and its elementary square-sum tail bound."""

    admit_family(source)
    if type(cutoff) is not int or not 1 <= cutoff <= MAX_INFINITE_PRODUCT_CUTOFF:
        raise _resource_error(
            "infinite_product_cutoff_budget",
            f"prime cutoff must be between 1 and {MAX_INFINITE_PRODUCT_CUTOFF}",
        )
    from jacobian.math.number_theory.squarefree_affine_forms._admissibility import (
        admissibility_cutoff,
    )

    theorem_cutoff, _, _ = admissibility_cutoff(source)
    if cutoff < theorem_cutoff:
        raise _domain_error(
            "infinite_product_cutoff_too_small",
            f"prime cutoff {cutoff} is below the family tail-theorem cutoff "
            f"{theorem_cutoff}",
        )
    from jacobian.math.number_theory.squarefree_affine_forms._admissibility import (
        primes_up_to,
    )

    primes = primes_up_to(cutoff)
    work = sum(prime * prime for prime in primes) * source.form_count
    if work > MAX_INFINITE_PRODUCT_WORK:
        raise _resource_error(
            "infinite_product_work_budget",
            f"infinite-product prefix needs {work} congruence steps, exceeding "
            f"{MAX_INFINITE_PRODUCT_WORK}",
        )
    for prime in primes:
        admit_local_factor_work(source.form_count, prime)


def admit_admissibility_cutoff(cutoff: int) -> None:
    """Require a checkable cutoff inside the prime envelope."""

    if cutoff < 1:
        raise _domain_error(
            "cutoff_positive", "the admissibility cutoff is at least one"
        )
    if cutoff > MAX_ADMISSIBILITY_CUTOFF:
        raise _resource_error(
            "cutoff_budget",
            f"the admissibility cutoff exceeds {MAX_ADMISSIBILITY_CUTOFF}, "
            "beyond the checkable prime envelope",
        )


def admit_admissibility_work(form_count: int, primes: tuple[int, ...]) -> None:
    """Preflight the aggregate p^2 work before any factor enumeration."""

    total_work = sum(prime * prime for prime in primes) * form_count
    if total_work > MAX_ADMISSIBILITY_WORK:
        raise _resource_error(
            "work_budget",
            f"admissibility needs {total_work} bounded congruence steps, "
            f"exceeding {MAX_ADMISSIBILITY_WORK}",
        )
    for prime in primes:
        admit_local_factor_work(form_count, prime)


def admit_interval(source: SquarefreeAffineFamily, lower: int, upper: int) -> int:
    """Admit one integer interval and return the governing value magnitude."""

    admit_family(source)
    if type(lower) is not int or type(upper) is not int:
        raise _domain_error("interval_integers", "interval bounds must be integers")
    if lower > upper:
        raise _domain_error(
            "interval_order", "the interval lower bound must not exceed the upper"
        )
    length = upper - lower + 1
    if length > MAX_INTERVAL_LENGTH:
        raise _resource_error(
            "interval_length_budget",
            f"interval length {length} exceeds {MAX_INTERVAL_LENGTH}",
        )
    magnitude = 0
    for form in source.forms:
        for bound in (lower, upper):
            value = abs(form.coefficient * bound + form.constant)
            if value > magnitude:
                magnitude = value
    if magnitude > MAX_INTERVAL_VALUE:
        raise _resource_error(
            "interval_value_budget",
            f"affine values reach {magnitude}, exceeding {MAX_INTERVAL_VALUE}",
        )
    return magnitude


def admit_interval_sieve_residues(work: int) -> None:
    """Preflight congruence-class enumeration before any residue arithmetic."""

    if work > MAX_INTERVAL_SIEVE_RESIDUES:
        raise _resource_error(
            "interval_sieve_budget",
            f"interval sieve would enumerate {work} congruence classes, "
            f"exceeding {MAX_INTERVAL_SIEVE_RESIDUES}",
        )


def admit_interval_sieve_visits(work: int) -> None:
    """Bound interval-point visits along all admitted square-divisor classes."""

    if work > MAX_INTERVAL_SIEVE_VISITS:
        raise _resource_error(
            "interval_sieve_visit_budget",
            f"interval sieve would visit {work} congruent interval points, "
            f"exceeding {MAX_INTERVAL_SIEVE_VISITS}",
        )


__all__ = [
    "MAX_ADMISSIBILITY_CUTOFF",
    "MAX_ADMISSIBILITY_WORK",
    "MAX_EULER_PRIMES",
    "MAX_EULER_WORK",
    "MAX_INFINITE_PRODUCT_CUTOFF",
    "MAX_INFINITE_PRODUCT_WORK",
    "MAX_INTERVAL_LENGTH",
    "MAX_INTERVAL_SIEVE_RESIDUES",
    "MAX_INTERVAL_SIEVE_VISITS",
    "MAX_INTERVAL_VALUE",
    "MAX_LEDGER_ROWS",
    "MAX_LOCAL_FACTOR_PRIME",
    "MAX_LOCAL_FACTOR_WORK",
    "admit_euler_product",
    "admit_family",
    "admit_infinite_product",
    "admit_interval",
    "admit_interval_sieve_residues",
    "admit_interval_sieve_visits",
    "admit_local_factor",
    "admit_prime",
    "admit_prime_set",
]
