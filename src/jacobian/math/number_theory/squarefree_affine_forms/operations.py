"""Value-based native operations on square-free affine-form families."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.analysis.intervals import ClosedRationalInterval
from jacobian.math.number_theory.squarefree_affine_forms._admissibility import (
    LocalAdmissibilityResult,
)
from jacobian.math.number_theory.squarefree_affine_forms._admissibility import (
    local_admissibility as _local_admissibility_kernel,
)
from jacobian.math.number_theory.squarefree_affine_forms._euler_product import (
    SquarefreeEulerProductResult,
    SquarefreeLocalFactorRow,
)
from jacobian.math.number_theory.squarefree_affine_forms._infinite_product import (
    SquarefreeInfiniteProductEnclosure,
)
from jacobian.math.number_theory.squarefree_affine_forms._interval_count import (
    IntervalCountResult,
)
from jacobian.math.number_theory.squarefree_affine_forms._interval_count import (
    interval_count as _interval_count_kernel,
)
from jacobian.math.number_theory.squarefree_affine_forms._kernel import (
    closed_form_ledger,
)
from jacobian.math.number_theory.squarefree_affine_forms._local_factor import (
    SquarefreeLocalFactorResult,
)
from jacobian.math.number_theory.squarefree_affine_forms._models import (
    admit_euler_product,
    admit_infinite_product,
    admit_local_factor,
)
from jacobian.math.number_theory.squarefree_affine_forms.values import (
    MAX_SQUAREFREE_COMPONENT_DIGITS,
    MAX_SQUAREFREE_FORMS,
    SquarefreeAffineFamily,
)


def verify_squarefree_affine_family(family: object) -> bool:
    """Check a caller-supplied family's bounded-domain claim."""

    try:
        forms = tuple(family.forms)  # type: ignore[attr-defined]
        if not 1 <= len(forms) <= MAX_SQUAREFREE_FORMS:
            return False
        pairs = set()
        identifiers = set()
        for form in forms:
            coefficient = form.coefficient
            constant = form.constant
            identifiers.add(form.form_id)
            if (coefficient, constant) in pairs:
                return False
            pairs.add((coefficient, constant))
            if coefficient == 0 and constant == 0:
                return False
            if (
                len(str(abs(coefficient))) > MAX_SQUAREFREE_COMPONENT_DIGITS
                or len(str(abs(constant))) > MAX_SQUAREFREE_COMPONENT_DIGITS
            ):
                return False
        if len(identifiers) != len(forms):
            return False
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def local_factor(
    source: SquarefreeAffineFamily, prime: int
) -> SquarefreeLocalFactorResult:
    """Return the complete p^2 residue ledger and exact local factor."""

    admit_local_factor(source, prime)
    profiles, ledger, covers_all = closed_form_ledger(source, prime)
    return SquarefreeLocalFactorResult._from_kernel(
        source=source,
        prime=prime,
        profiles=profiles,
        ledger=ledger,
        covers_all=covers_all,
    )


def euler_product(
    source: SquarefreeAffineFamily, primes: tuple[int, ...]
) -> SquarefreeEulerProductResult:
    """Return the exact finite Euler-product prefix over one bounded prime set."""

    admit_euler_product(source, primes)
    rows: list[SquarefreeLocalFactorRow] = []
    for prime in primes:
        _, ledger, covers_all = closed_form_ledger(source, prime)
        modulus = prime * prime
        bad_count = modulus if covers_all else len(ledger)
        valid_count = modulus - bad_count
        rows.append(
            SquarefreeLocalFactorRow(
                prime=prime,
                modulus=modulus,
                bad_count=bad_count,
                valid_count=valid_count,
                local_factor=CanonicalRational.from_fraction(
                    Fraction(valid_count, modulus)
                ),
                has_local_obstruction=valid_count == 0,
            )
        )
    return SquarefreeEulerProductResult._from_kernel(
        source=source, primes=primes, rows=tuple(rows)
    )


def infinite_product_enclosure(
    source: SquarefreeAffineFamily, prime_cutoff: int
) -> SquarefreeInfiniteProductEnclosure:
    """Enclose the convergent square-free Euler product by a rational interval."""

    from jacobian.math.number_theory.squarefree_affine_forms._admissibility import (
        primes_up_to,
    )

    admit_infinite_product(source, prime_cutoff)
    prefix = Fraction(1, 1)
    for prime in primes_up_to(prime_cutoff):
        _, ledger, covers_all = closed_form_ledger(source, prime)
        modulus = prime * prime
        bad_count = modulus if covers_all else len(ledger)
        prefix *= Fraction(modulus - bad_count, modulus)
    tail_lower = max(
        Fraction(0, 1),
        Fraction(prime_cutoff - len(source.forms), prime_cutoff),
    )
    return SquarefreeInfiniteProductEnclosure(
        source=source,
        prime_cutoff=prime_cutoff,
        enclosure=ClosedRationalInterval(
            lower=CanonicalRational.from_fraction(prefix * tail_lower),
            upper=CanonicalRational.from_fraction(prefix),
        ),
    )


def local_admissibility(
    source: SquarefreeAffineFamily,
) -> LocalAdmissibilityResult:
    """Decide local admissibility by finite check plus large-prime proof."""

    return _local_admissibility_kernel(source)


def interval_count(
    source: SquarefreeAffineFamily,
    lower: int,
    upper: int,
    include_ledger: bool = False,
) -> IntervalCountResult:
    """Count interval points where every affine form value is square-free."""

    return _interval_count_kernel(source, lower, upper, include_ledger)


__all__ = [
    "euler_product",
    "infinite_product_enclosure",
    "interval_count",
    "local_admissibility",
    "local_factor",
    "verify_squarefree_affine_family",
]
