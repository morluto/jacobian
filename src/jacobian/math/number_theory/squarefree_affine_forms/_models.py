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
)

MAX_LOCAL_FACTOR_PRIME = 1_000
MAX_LOCAL_FACTOR_WORK = 8_000_000
MAX_LEDGER_ROWS = 8_192
MAX_EULER_PRIMES = 16
MAX_EULER_WORK = 32_000_000
CROSSCHECK_MAX_PRIME = 31

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


__all__ = [
    "CROSSCHECK_MAX_PRIME",
    "MAX_EULER_PRIMES",
    "MAX_EULER_WORK",
    "MAX_LEDGER_ROWS",
    "MAX_LOCAL_FACTOR_PRIME",
    "MAX_LOCAL_FACTOR_WORK",
    "admit_euler_product",
    "admit_family",
    "admit_local_factor",
    "admit_prime",
    "admit_prime_set",
]
