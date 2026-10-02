"""Coefficient-free support closure and exact multivariate division envelopes."""

from dataclasses import dataclass
from heapq import heapify, heappop, heappush

from jacobian.math.polynomials._division_bounds import (
    MAX_DIVISION_ALLOCATION_BITS,
    MAX_DIVISION_RESULT_DIGITS,
    _clear,
    _component,
    _digits,
    _Ledger,
    _reject,
    _source,
    _source_digits,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
)

from ._division import MonomialOrder

_MAX_MONOMIAL_WORK = 32_000_000


@dataclass(frozen=True)
class _DivisionSupport:
    quotient: int
    remainder: int
    nodes: int
    depth: int
    maximum_frontier: int


def _priority(order: MonomialOrder, exponent: tuple[int, ...]) -> tuple[int, ...]:
    """Sign-reversed flattened SymPy order key for the max-order frontier."""
    if order == "lex":
        return tuple(-entry for entry in exponent)
    if order == "grlex":
        return (-sum(exponent), *(-entry for entry in exponent))
    return (-sum(exponent), *reversed(exponent))


def _support(
    left: RationalPolynomial,
    right: RationalPolynomial,
    order: MonomialOrder,
    ledger: _Ledger,
) -> _DivisionSupport:
    """Close source support under the divisor's leading-monomial rewrite.

    Each rewrite e -> e-LM(g)+tail strictly decreases the declared monomial
    order. Visiting the largest pending exponent finalizes its longest path
    depth before it is expanded. Cancellation can only delete reachable nodes.
    The quotient exponent e-LM(g) is injective in divisible nodes; irreducible
    nodes bound remainder support. The live frontier also majorizes every
    intermediate dividend and its leading-term scans in PolyElement.div.
    """
    from sympy.polys.orderings import monomial_key

    key = monomial_key(order)
    exponents = tuple(term.exponents for term in right.polynomial.terms)
    leading = max(exponents, key=key)
    tail = tuple(exponent for exponent in exponents if exponent != leading)
    if not left.polynomial.terms:
        return _DivisionSupport(0, 0, 0, 0, 0)
    if left == right:
        return _DivisionSupport(1, 0, len(exponents), 1, len(exponents))

    depths = {term.exponents: 0 for term in left.polynomial.terms}
    frontier = [(_priority(order, exponent), exponent) for exponent in depths]
    heapify(frontier)
    quotient = remainder = depth = work = 0
    maximum_frontier = len(frontier)
    dimension = len(left.variables)
    while frontier:
        ledger.charge(1)
        # Nondivisible terms cause a second leading_expv scan in the backend;
        # each comparison inspects at most the complete exponent vector.
        work += 2 * len(frontier) * dimension
        if work > _MAX_MONOMIAL_WORK:
            _reject(
                "monomial_work", "division exceeds its monomial-order work envelope"
            )
        _, exponent = heappop(frontier)
        if any(e < lead for e, lead in zip(exponent, leading, strict=True)):
            remainder += 1
            if remainder > MAX_POLYNOMIAL_TERMS:
                _reject(
                    "support",
                    "division remainder support can exceed the canonical term carrier",
                )
            continue
        quotient += 1
        if quotient > MAX_POLYNOMIAL_TERMS:
            _reject(
                "support",
                "division quotient support can exceed the canonical term carrier",
            )
        quotient_exponent = tuple(
            e - lead for e, lead in zip(exponent, leading, strict=True)
        )
        next_depth = depths[exponent] + 1
        depth = max(depth, next_depth)
        work += dimension * (len(tail) + 1)
        if work > _MAX_MONOMIAL_WORK:
            _reject(
                "monomial_work", "division exceeds its support-closure work envelope"
            )
        for term in tail:
            child = tuple(e + t for e, t in zip(quotient_exponent, term, strict=True))
            if any(e > MAX_POLYNOMIAL_EXPONENT for e in child):
                _reject(
                    "exponent", "division can exceed the canonical exponent carrier"
                )
            if child not in depths:
                # Q and R each have at most 4,096 monomials. Stop before
                # allocating a larger union, regardless of the next rewrite.
                if len(depths) >= 2 * MAX_POLYNOMIAL_TERMS:
                    _reject(
                        "support",
                        "division reachable support exceeds its term envelope",
                    )
                depths[child] = next_depth
                heappush(frontier, (_priority(order, child), child))
            else:
                depths[child] = max(depths[child], next_depth)
        maximum_frontier = max(maximum_frontier, len(frontier))
    ledger.charge(work)
    return _DivisionSupport(quotient, remainder, len(depths), depth, maximum_frontier)


def admit_multivariate_division(
    left: RationalPolynomial, right: RationalPolynomial, order: MonomialOrder
) -> None:
    """Admit sparse division, reconstruction, and complete canonical output.

    Let A=L*f and B=M*g be integral, b=|LC(B)|, T=sum|tail(B)|,
    C=b+T, H=sum|A| and D the maximum number of divisible nodes on a rewrite
    path. The path expansion puts every partial coefficient over L*b**D.
    Quotient numerators are bounded by
        M*H*sum(b**(D-1-k)*T**k, k=0..D-1) <= M*H*C**(D-1),
    and remainder/working-dividend numerators by H*C**D. Absolute path sums
    cover arbitrary divisors, merging paths, partial updates and cancellation.
    D=0 leaves f unchanged and Q=0. The maintained kernel and reconstruction
    stay sparse, so no dense degree box or symbolic-expression expansion is
    hidden behind these support bounds.
    """
    ledger = _Ledger()
    a_terms, b_terms = _source(left, ledger), _source(right, ledger)
    support = _support(left, right, order, ledger)
    a = _clear(a_terms, ledger)
    b = a if left == right else _clear(b_terms, ledger)
    from sympy.polys.orderings import monomial_key

    leading_index = max(
        range(len(b_terms)), key=lambda i: monomial_key(order)(b_terms[i].exponents)
    )
    leading = abs(b.coefficients[leading_index])
    growth = 0
    height = 0
    for value in b.coefficients:
        growth = ledger.add(growth, abs(value))
    for value in a.coefficients:
        height = ledger.add(height, abs(value))
    denominator = a.denominator
    qnum = qden = 1
    for _ in range(support.depth):
        qnum = ledger.multiply(b.denominator, height)
        height = ledger.multiply(height, growth)
        denominator = ledger.multiply(denominator, leading)
        for value in (qnum, height, denominator):
            _component(value)
    rnum, rden = height, denominator
    if support.depth:
        qden = denominator
    for value in (qnum, qden, rnum, rden):
        _component(value)

    # Q*g+R uses denominator D*M. A coefficient of Q*g contains at most
    # one contribution per divisor term. The unreduced rational products,
    # additions and quotient-by-leading-coefficient operations are bounded
    # by twice the larger common numerator/denominator width below.
    reconstruction_num = ledger.add(
        ledger.multiply(qnum, growth), ledger.multiply(rnum, b.denominator)
    )
    common_bits = denominator.bit_length() + b.denominator.bit_length()
    source_bits = max(
        a.height.bit_length(),
        b.height.bit_length(),
        a.denominator.bit_length(),
        b.denominator.bit_length(),
    )
    private_bits = (
        2 * max(common_bits, reconstruction_num.bit_length(), source_bits) + 4
    )
    arithmetic = (
        8 * support.quotient * len(b_terms)
        + 8 * support.quotient * len(b_terms)
        + 32 * (len(a_terms) + len(b_terms) + support.quotient + support.remainder)
    )
    ledger.charge(arithmetic, private_bits)
    # Retained sources, backend working dividend/Q/R, reconstruction, and
    # canonical conversion all coexist only within this conservative reserve.
    slots = (
        len(a_terms)
        + len(b_terms)
        + support.maximum_frontier
        + support.quotient
        + support.remainder
        + support.nodes
    )
    if 16 * slots * private_bits > MAX_DIVISION_ALLOCATION_BITS:
        _reject(
            "allocation",
            "division exceeds its aggregate exact-coefficient storage envelope",
        )
    output_digits = (
        _source_digits(a_terms)
        + _source_digits(b_terms)
        + support.quotient * (_digits(qnum) + _digits(qden))
        + support.remainder * (_digits(rnum) + _digits(rden))
    )
    if output_digits > MAX_DIVISION_RESULT_DIGITS:
        _reject(
            "output_size", "division exceeds its retained exact-scalar storage envelope"
        )
