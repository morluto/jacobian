"""Replay of Gröbner reduction against a retained basis.

The normal-form operations return their computed Gröbner basis together with
the remainder so result validation can replay the defining reduction relation
without re-running the Gröbner kernel: division of one polynomial by a fixed
Gröbner basis under the declared monomial order has a unique remainder, and
every ideal generator must reduce to zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jacobian.math.polynomials.values import (
        RationalPolynomial,
        RationalPolynomialIdeal,
    )

__all__ = [
    "generators_reduce_to_zero",
    "remainder_matches_claim",
    "replayed_remainder_term_count",
    "retained_basis_in_ideal",
    "retained_basis_is_groebner",
]


def _sparse_ring(variables: tuple[str, ...], order: str) -> Any:
    """Return one SymPy sparse polynomial ring in the declared order."""

    from sympy import QQ
    from sympy.polys.rings import ring

    return ring(", ".join(variables), QQ, order=order)[0]


def _ring_element(ring_context: Any, polynomial: RationalPolynomial) -> Any:
    """Build one sparse-ring element from validated wire data (no parsing)."""

    from sympy import QQ

    if not polynomial.polynomial.terms:
        return ring_context.zero
    return ring_context.from_dict(
        {
            tuple(term.exponents): QQ(*term.coefficient.as_integer_ratio())
            for term in polynomial.polynomial.terms
        }
    )


def _replay_division(
    ideal: RationalPolynomialIdeal,
    groebner_basis: tuple[RationalPolynomial, ...],
    polynomial: RationalPolynomial,
    monomial_order: str,
) -> Any:
    """Return the unique remainder of ``polynomial`` modulo ``groebner_basis``."""

    ring_context = _sparse_ring(ideal.variables, monomial_order)
    divisors = [_ring_element(ring_context, element) for element in groebner_basis]
    dividend = _ring_element(ring_context, polynomial)
    _, remainder = dividend.div(divisors)
    return remainder


def _term_count(element: Any) -> int:
    return len(element.monoms())


def remainder_matches_claim(
    ideal: RationalPolynomialIdeal,
    groebner_basis: tuple[RationalPolynomial, ...],
    polynomial: RationalPolynomial,
    monomial_order: str,
    claimed_remainder: RationalPolynomial,
) -> bool:
    """Check that the claimed wire remainder equals the replayed reduction."""

    ring_context = _sparse_ring(ideal.variables, monomial_order)
    replayed = _replay_division(ideal, groebner_basis, polynomial, monomial_order)
    claimed = _ring_element(ring_context, claimed_remainder)
    return replayed == claimed


def generators_reduce_to_zero(
    ideal: RationalPolynomialIdeal,
    groebner_basis: tuple[RationalPolynomial, ...],
    monomial_order: str,
) -> bool:
    """Check every retained ideal generator reduces to zero modulo the basis."""

    for generator in ideal.generators:
        zero = _sparse_ring(ideal.variables, monomial_order).zero
        replayed = _replay_division(ideal, groebner_basis, generator, monomial_order)
        if replayed != zero:
            return False
    return True


def replayed_remainder_term_count(
    ideal: RationalPolynomialIdeal,
    groebner_basis: tuple[RationalPolynomial, ...],
    polynomial: RationalPolynomial,
    monomial_order: str,
) -> int:
    """Return the replayed remainder's term count without materializing wire data."""

    return _term_count(
        _replay_division(ideal, groebner_basis, polynomial, monomial_order)
    )


def replayed_remainder_exceeds_budget(
    ideal: RationalPolynomialIdeal,
    groebner_basis: tuple[RationalPolynomial, ...],
    polynomial: RationalPolynomial,
    monomial_order: str,
) -> bool:
    """Check whether the replayed remainder exceeds the 1,024-term or 32,768-exponent budget."""

    from jacobian.math.polynomials.values import MAX_POLYNOMIAL_EXPONENT

    remainder = _replay_division(ideal, groebner_basis, polynomial, monomial_order)
    if _term_count(remainder) > 1024:
        return True
    try:
        # Check exponent cap efficiently via monoms
        for monom in remainder.monoms():
            if any(exp > MAX_POLYNOMIAL_EXPONENT for exp in monom):
                return True
    except Exception:
        # If we cannot inspect monoms, conservatively treat as exceeding
        return True
    return False


def retained_basis_in_ideal(
    ideal: RationalPolynomialIdeal,
    groebner_basis: tuple[RationalPolynomial, ...],
    monomial_order: str,
) -> bool:
    """Check every retained basis element belongs to the source ideal.

    Each ``groebner_basis`` element must reduce to zero modulo the Gröbner
    basis of the source ``ideal``.  This catches forgeries such as
    ``<x^2>`` with retained basis ``(1)`` where generators still reduce to
    zero modulo ``(1)`` but ``1`` is not in ``<x^2>``.
    """

    if not groebner_basis:
        return True
    # Compute a Groebner basis for the source ideal once.
    from sympy import QQ, groebner

    from jacobian.math.polynomials._conversions import (
        rational_polynomial_to_sympy,
        symbols_for_variables,
    )

    variables = ideal.variables
    symbols = symbols_for_variables(variables)
    try:
        source_exprs = [
            rational_polynomial_to_sympy(gen).as_expr() for gen in ideal.generators
        ]
        source_gb = groebner(source_exprs, *symbols, order=monomial_order, domain=QQ)
    except Exception:
        return False
    # For the zero ideal, SymPy's Groebner basis is empty; no nonzero
    # polynomial belongs to it, so any non-empty retained basis must be rejected
    # (which the reduction check below will do: reducing a nonzero element
    # modulo an empty basis leaves it unchanged, not zero).
    for element in groebner_basis:
        try:
            expr = rational_polynomial_to_sympy(element).as_expr()
            remainder = source_gb.reduce(expr)[1]
            if remainder != 0:
                return False
        except Exception:
            return False
    return True


def retained_basis_is_groebner(
    ideal: RationalPolynomialIdeal,
    groebner_basis: tuple[RationalPolynomial, ...],
    monomial_order: str,
) -> bool:
    """Check the retained tuple is the reduced Gröbner basis of the ideal.

    Recomputes the exact Gröbner basis from the source generators and
    compares it to the retained wire basis as SymPy ``Poly`` sets.  This
    ensures the retained basis is not merely a generating set but a
    Gröbner basis, so remainder replay via sparse-ring division is the
    unique normal form.
    """

    from sympy import QQ, groebner

    from jacobian.math.polynomials._conversions import (
        rational_polynomial_from_sympy,
        rational_polynomial_to_sympy,
        symbols_for_variables,
    )

    variables = ideal.variables
    symbols = symbols_for_variables(variables)
    try:
        source_exprs = [
            rational_polynomial_to_sympy(gen).as_expr() for gen in ideal.generators
        ]
        source_gb = groebner(source_exprs, *symbols, order=monomial_order, domain=QQ)
        computed_polys = tuple(
            rational_polynomial_from_sympy(
                # ``source_gb.polys`` are ``Poly``; ensure domain QQ
                __import__("sympy").Poly(expr, *symbols, domain=QQ),
                variables,
            )
            for expr in source_gb.exprs
        )
    except Exception:
        return False
    # Empty source ideal must correspond to empty retained basis.
    if not computed_polys and not groebner_basis:
        return True
    if len(computed_polys) != len(groebner_basis):
        return False
    # Compare as unordered sets of wire forms (sorted fingerprint).
    def _key(poly: RationalPolynomial) -> tuple[tuple[tuple[int, ...], str, str], ...]:
        return tuple(
            (term.exponents, term.coefficient.num, term.coefficient.den)
            for term in poly.polynomial.terms
        )

    try:
        return sorted(computed_polys, key=_key) == sorted(groebner_basis, key=_key)
    except Exception:
        return False
