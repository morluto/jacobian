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
    "retained_basis_is_groebner",
]

# Bounded-reduction budget: intermediate reduction work is capped so a
# pathological request cannot expand an unbounded remainder before the
# 1,024-term output boundary is noticed. The intermediate caps are
# conservative multiples of the output term budget; the step cap bounds
# the division loop itself.
_MAX_OUTPUT_TERMS = 1_024
_MAX_INTERMEDIATE_TERMS = 4_096
_MAX_REDUCTION_STEPS = 200_000


def _component_exceeds_limit(numerator: int, denominator: int) -> bool:
    """Check one integer pair against the canonical rational digit limit."""

    from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS

    bound: int = 10**MAX_CANONICAL_RATIONAL_DIGITS
    return abs(numerator) >= bound or abs(denominator) >= bound


def _basis_exceeds_output_budget(
    basis: Any,
    symbols: tuple[Any, ...],
    maximum_terms: int = _MAX_OUTPUT_TERMS,
) -> bool:
    """Check one computed Gröbner basis against the output budgets.

    The aggregate 1,024-term boundary and the representability limits
    (32,768 exponents, canonical coefficient digits) apply only to a
    complete basis: the reduced Gröbner basis of the full generating set
    is unique, so its budget status is a function of the ideal value.
    """

    from sympy import QQ, Poly

    from jacobian.math.polynomials.values import MAX_POLYNOMIAL_EXPONENT

    aggregate_terms = 0
    for expr in basis.exprs:
        poly = Poly(expr, *symbols, domain=QQ)
        terms = poly.terms()
        aggregate_terms += len(terms)
        if aggregate_terms > maximum_terms:
            return True
        for monom, coefficient in terms:
            if any(exp > MAX_POLYNOMIAL_EXPONENT for exp in monom):
                return True
            if _component_exceeds_limit(int(coefficient.p), int(coefficient.q)):
                return True
    return False


def _canonical_generator_order(ideal: RationalPolynomialIdeal) -> tuple[int, ...]:
    """Order generator indices by mathematical content alone.

    The key (ascending maximum total degree, then term count, then full
    term fingerprint) is a deterministic function of the ideal value, so
    the bounded kernel's behavior never depends on the presentation order
    of an equivalent generating set.  Simpler generators first also lets
    collapsing elements — most simply nonzero constants, of degree zero —
    shrink every later prefix instead of arriving after one has grown.
    """

    def sort_key(
        index: int,
    ) -> tuple[int, int, tuple[tuple[tuple[int, ...], str, str], ...]]:
        terms = ideal.generators[index].polynomial.terms
        return (
            max((sum(term.exponents) for term in terms), default=0),
            len(terms),
            tuple(
                sorted(
                    (term.exponents, term.coefficient.num, term.coefficient.den)
                    for term in terms
                )
            ),
        )

    return tuple(sorted(range(len(ideal.generators)), key=sort_key))


def _guarded_incremental(
    exprs: list[Any],
    symbols: tuple[Any, ...],
    monomial_order: str,
) -> tuple[str, Any]:
    """Run the kernel over one generator order under a work bound.

    Returns ``("OK", basis)`` when every prefix stayed representable,
    ``("EXCEEDED", None)`` when the complete generating set's basis leaves
    an output budget, ``("ABORTED", None)`` when a mid-sequence prefix
    left a budget — a work bound, not a conclusion, because later
    generators can still shrink the ideal — and ``("FAILED", None)`` when
    the kernel itself failed, which is likewise no evidence.
    """

    from sympy import QQ, groebner

    current: list[Any] = []
    basis = None
    for position, expr in enumerate(exprs):
        current.append(expr)
        try:
            basis = groebner(current, *symbols, order=monomial_order, domain=QQ)
        except Exception:
            return "FAILED", None
        try:
            exceeded = _basis_exceeds_output_budget(basis, symbols)
        except Exception:
            return "FAILED", None
        if exceeded:
            if position == len(exprs) - 1:
                return "EXCEEDED", None
            return "ABORTED", None
    if basis is None:
        return "FAILED", None
    return "OK", basis


def _finalize_strategy_basis(
    basis: Any,
    symbols: tuple[Any, ...],
    variables: tuple[str, ...],
) -> tuple[str, tuple[RationalPolynomial, ...] | None]:
    """Convert one completed sympy basis to wire form, deciding the budgets.

    Representability failures (canonical term limits, exponent or digit
    bounds) are complete-basis overflow evidence; other failures carry no
    conclusion.
    """

    from pydantic import ValidationError
    from sympy import QQ, Poly

    from jacobian.math.polynomials._conversions import (
        rational_polynomial_from_sympy,
    )

    try:
        polys = [Poly(expr.as_expr(), *symbols, domain=QQ) for expr in basis.exprs]
        if sum(len(poly.terms()) for poly in polys) > _MAX_OUTPUT_TERMS:
            return "EXCEEDED", None
        return "OK", tuple(
            rational_polynomial_from_sympy(poly, variables) for poly in polys
        )
    except ValidationError:
        return "EXCEEDED", None
    except Exception:
        return "FAILED", None


def incremental_source_groebner(
    ideal: RationalPolynomialIdeal,
    monomial_order: str,
) -> tuple[tuple[RationalPolynomial, ...] | None, bool]:
    """Compute the source reduced Gröbner basis in wire form.

    No output-budget decision is ever taken from an intermediate prefix:
    adding a generator grows the ideal but can shrink its reduced basis
    (one trailing coprime generator collapses the ideal to the unit
    ideal), so every prefix property — aggregate term count, exponent,
    coefficient digits — is presentation-dependent evidence that later
    generators may invalidate.  Only the basis of the complete generating
    set, whose reduced form is unique, decides an overflow.

    Each guarded strategy runs the kernel incrementally in a
    content-derived order and aborts at its first mid-sequence budget
    trip purely as a work bound: strategy one uses the canonical
    ascending order, strategy two its reverse.  When both abort or fail,
    a third strategy runs the kernel once over the complete generating
    set inside a killable subprocess with a wall clock, where collapsing
    pairs are visible from the start.  A strategy that cannot conclude
    yields no evidence either way.

    Returns ``(basis, exceeded)``: ``basis`` is the complete reduced
    Gröbner basis when some bounded strategy concluded within budget;
    ``(None, True)`` reports evidenced overflow for the typed budget
    outcome, and ``(None, False)`` reports that no conclusion exists.
    """

    from jacobian.math.polynomials._conversions import (
        rational_polynomial_to_sympy,
        symbols_for_variables,
    )

    variables = ideal.variables
    symbols = symbols_for_variables(variables)
    try:
        exprs = [
            rational_polynomial_to_sympy(gen).as_expr() for gen in ideal.generators
        ]
    except Exception:
        return None, False
    canonical = _canonical_generator_order(ideal)
    for ordered in (canonical, tuple(reversed(canonical))):
        outcome, basis = _guarded_incremental(
            [exprs[index] for index in ordered], symbols, monomial_order
        )
        if outcome == "EXCEEDED":
            return None, True
        if outcome != "OK":
            continue
        concluded, wire_basis = _finalize_strategy_basis(basis, symbols, variables)
        if concluded == "OK":
            return wire_basis, False
        if concluded == "EXCEEDED":
            return None, True
    from jacobian.math.polynomials._groebner_worker import complete_basis_in_worker

    return complete_basis_in_worker(ideal, monomial_order)


def _coefficient_exceeds_canonical_limit(value: Any) -> bool:
    """Check whether one exact QQ coefficient leaves the canonical rational domain."""

    return _component_exceeds_limit(int(value.numerator), int(value.denominator))


def budgeted_reduce(
    ring_context: Any, dividend: Any, divisors: list[Any]
) -> Any | None:
    """Divide ``dividend`` by a fixed Gröbner basis with bounded work.

    This is the standard multivariate long division: each step either
    cancels the current leading monomial against the one basis element
    whose leading monomial divides it (introducing only strictly smaller
    monomials, so the loop terminates), or moves the leading term into
    the remainder. Returns ``None`` when the work or output budget is
    exceeded; the caller then reports a typed budget outcome instead of
    materializing an unbounded intermediate polynomial.
    """

    remainder = ring_context.zero
    work = dividend
    steps = 0
    while work:
        steps += 1
        if steps > _MAX_REDUCTION_STEPS:
            return None
        lead_monom = work.leading_expv()
        reducer = None
        for divisor in divisors:
            lm = divisor.leading_expv()
            if lm is not None and all(lead_monom[k] >= lm[k] for k in range(len(lm))):
                reducer = divisor
                break
        if reducer is not None:
            lm = reducer.leading_expv()
            diff = tuple(a - b for a, b in zip(lead_monom, lm, strict=True))
            scale = ring_context.term_new(diff, work[lead_monom] / reducer[lm])
            work = work - scale * reducer
            if len(work.monoms()) > _MAX_INTERMEDIATE_TERMS:
                return None
        else:
            lead_term = ring_context.term_new(lead_monom, work[lead_monom])
            remainder = remainder + lead_term
            work = work - lead_term
            if len(remainder.monoms()) > _MAX_OUTPUT_TERMS:
                return None
    return remainder


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
) -> Any | None:
    """Return the unique remainder modulo ``groebner_basis``, or ``None`` on budget overflow."""

    ring_context = _sparse_ring(ideal.variables, monomial_order)
    divisors = [_ring_element(ring_context, element) for element in groebner_basis]
    dividend = _ring_element(ring_context, polynomial)
    return budgeted_reduce(ring_context, dividend, divisors)


def _term_count(element: Any) -> int:
    return len(element.monoms())


def remainder_matches_claim(
    ideal: RationalPolynomialIdeal,
    groebner_basis: tuple[RationalPolynomial, ...],
    polynomial: RationalPolynomial,
    monomial_order: str,
    claimed_remainder: RationalPolynomial,
) -> bool:
    """Check that the claimed wire remainder equals the replayed reduction.

    A replay whose bounded reduction overflows cannot corroborate a
    ``COMPUTED`` claim, so it reports a mismatch.
    """

    ring_context = _sparse_ring(ideal.variables, monomial_order)
    replayed = _replay_division(ideal, groebner_basis, polynomial, monomial_order)
    if replayed is None:
        return False
    claimed = _ring_element(ring_context, claimed_remainder)
    return bool(replayed == claimed)


def generators_reduce_to_zero(
    ideal: RationalPolynomialIdeal,
    groebner_basis: tuple[RationalPolynomial, ...],
    monomial_order: str,
) -> bool:
    """Check every retained ideal generator reduces to zero modulo the basis."""

    ring_context = _sparse_ring(ideal.variables, monomial_order)
    zero = ring_context.zero
    for generator in ideal.generators:
        replayed = _replay_division(ideal, groebner_basis, generator, monomial_order)
        # An overflowing reduction cannot be corroborated, so fail closed.
        if replayed is None or replayed != zero:
            return False
    return True


def replayed_remainder_exceeds_budget(
    ideal: RationalPolynomialIdeal,
    groebner_basis: tuple[RationalPolynomial, ...],
    polynomial: RationalPolynomial,
    monomial_order: str,
) -> bool:
    """Check whether the replayed remainder exceeds the 1,024-term, 32,768-exponent, or canonical-coefficient budget."""

    from jacobian.math.polynomials.values import MAX_POLYNOMIAL_EXPONENT

    remainder = _replay_division(ideal, groebner_basis, polynomial, monomial_order)
    if remainder is None:
        # The bounded reduction itself overflowed: the remainder is
        # certainly outside the output budget.
        return True
    try:
        for monom in remainder.monoms():
            if any(exp > MAX_POLYNOMIAL_EXPONENT for exp in monom):
                return True
        # A remainder whose coefficients leave the canonical rational domain is
        # likewise outside the representable result contract: the operation
        # reports BUDGET_EXCEEDED for it, so the replay predicate must agree.
        for _, coefficient in remainder.terms():
            if _coefficient_exceeds_canonical_limit(coefficient):
                return True
    except Exception:
        # If we cannot inspect monoms, conservatively treat as exceeding
        return True
    return False


def retained_source_basis_exceeds_budget(
    ideal: RationalPolynomialIdeal,
    monomial_order: str,
) -> bool:
    """Recompute the source Gröbner basis and check it leaves the budgets.

    This substantiates a ``BUDGET_EXCEEDED`` result that retains no basis:
    such an outcome is accepted only when the recomputation genuinely
    violates the aggregate 1,024-term output boundary or a
    representability limit.  Any kernel failure is not
    evidence and returns ``False`` so authored results stay rejected.
    """

    _basis, exceeded = incremental_source_groebner(ideal, monomial_order)
    return exceeded


def retained_basis_is_groebner(
    ideal: RationalPolynomialIdeal,
    groebner_basis: tuple[RationalPolynomial, ...],
    monomial_order: str,
) -> bool:
    """Check the retained tuple is the reduced Gröbner basis of the ideal.

    Recomputes the exact Gröbner basis from the source generators with
    bounded strategies and compares it to the retained wire basis as
    wire-form sets.  This ensures the retained basis is not merely a
    generating set but a Gröbner basis, so remainder replay via
    sparse-ring division is the unique normal form.
    """

    source_gb, exceeded = incremental_source_groebner(ideal, monomial_order)
    if source_gb is None or exceeded:
        return False
    computed_polys = source_gb
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
        return bool(
            sorted(computed_polys, key=_key) == sorted(groebner_basis, key=_key)
        )
    except Exception:
        return False
