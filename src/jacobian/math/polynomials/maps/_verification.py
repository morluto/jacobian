"""Bounded exact replay of authored generic-fiber evidence over QQ(t)[x].

No Gröbner basis is computed here. Clearing parameter denominators and using
fraction-free reduction keeps every step in a sparse rational polynomial ring.
Nonzero parameter polynomials are units in QQ(t), so these steps preserve zero
remainders, ideals, and source leading monomials. The resource envelope bounds
actual sparse work and growth, including unsuccessful certificate replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from pydantic import BaseModel, ValidationError

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.maps._generic_degree import (
    StandardMonomialLimitError,
    enumerate_standard_monomials,
)
from jacobian.math.polynomials.maps._models import (
    GenericDegreeResult,
    GenericFiberPolynomial,
)
from jacobian.math.polynomials.values import SparseRationalPolynomial

# These are replay limits, independent of both the wire carrier and producer.
# At most 32 basis numerators and their leading-coefficient maps (sharing
# scalar values) plus three source polynomials are retained;
# other polynomial intermediates have constant multiplicity. Every arithmetic
# step precharges a conservative bit allocation forecast as well as bit work.
# Rational normalization can temporarily double scalar widths; those widths
# are admitted before Fraction operations. No dense powers,
# factorization, symbolic cancellation, or unbounded backend calls are used.
_MAX_WORK = 2_000_000
_MAX_BIT_WORK = 1_000_000_000_000
_MAX_ALLOCATED_BITS = 268_435_456
_MAX_BITS = 16_384
_MAX_TERMS = 16_384
_MAX_POLYNOMIAL_BITS = 4_194_304
_MAX_EXPONENT = 1_048_576
_MAX_INPUT_NODES = 262_144

_Monomial = tuple[int, ...]
_Polynomial = dict[_Monomial, Fraction]


def _resource() -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=(),
        code="polynomial.generic_degree_verification_budget",
        message="generic-fiber evidence replay exceeds its sparse work or growth budget",
    )


@dataclass
class _Budget:
    work: int = 0
    bit_work: int = 0
    allocated_bits: int = 0

    def charge(self, work: int, bits: int = 1) -> None:
        self.work += work
        self.bit_work += work * bits * bits
        self.allocated_bits += work * bits
        if (
            self.work > _MAX_WORK
            or bits > _MAX_BITS
            or self.bit_work > _MAX_BIT_WORK
            or self.allocated_bits > _MAX_ALLOCATED_BITS
        ):
            raise _resource()
        request_checkpoint("during generic-fiber evidence replay")


def _plain(value: Any, budget: _Budget, nodes: list[int], depth: int = 0) -> Any:
    """Bound native model copies before structural revalidation or allocation."""
    nodes[0] += 1
    if nodes[0] > _MAX_INPUT_NODES or depth > 16:
        raise _resource()
    budget.charge(1)
    if value is None or type(value) is str:
        if isinstance(value, str) and len(value) > 64:
            raise ValueError("invalid claim string")
        return value
    if type(value) is int:
        budget.charge(1, max(1, value.bit_length()))
        return value
    if type(value) is tuple:
        if len(value) > _MAX_TERMS:
            raise _resource()
        return tuple(_plain(item, budget, nodes, depth + 1) for item in value)
    if isinstance(value, BaseModel):
        return {
            name: _plain(getattr(value, name), budget, nodes, depth + 1)
            for name in type(value).model_fields
        }
    raise ValueError("invalid claim container")


def _canonical_claim(
    claim: GenericDegreeResult, budget: _Budget
) -> GenericDegreeResult | None:
    if type(claim) is not GenericDegreeResult:
        return None
    try:
        return GenericDegreeResult.model_validate(
            _plain(claim, budget, [0]), strict=True
        )
    except OperationResourceAdmissionError:
        raise
    except (AttributeError, ValueError, ValidationError):
        return None


def _bits(value: Fraction) -> int:
    return max(value.numerator.bit_length(), value.denominator.bit_length(), 1)


def _put(
    result: _Polynomial, monomial: _Monomial, value: Fraction, budget: _Budget
) -> None:
    old = result.get(monomial)
    if old is not None:
        # Fraction addition forms cross products and a carry before reduction.
        budget.charge(8, _bits(old) + _bits(value) + 1)
        value += old
    if not value:
        result.pop(monomial, None)
    else:
        if old is None and len(result) >= _MAX_TERMS:
            raise _resource()
        result[monomial] = value


def _storage(polynomial: _Polynomial, budget: _Budget) -> _Polynomial:
    """Cap retained size after precharged, transient arithmetic allocations."""
    budget.charge(len(polynomial))
    if sum(_bits(value) for value in polynomial.values()) > _MAX_POLYNOMIAL_BITS:
        raise _resource()
    return polynomial


def _multiply(a: _Polynomial, b: _Polynomial, budget: _Budget) -> _Polynomial:
    budget.charge(len(a) * len(b))
    result: _Polynomial = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            monomial = tuple(x + y for x, y in zip(ma, mb, strict=True))
            if max(monomial) > _MAX_EXPONENT:
                raise _resource()
            budget.charge(4, _bits(ca) + _bits(cb))
            _put(result, monomial, ca * cb, budget)
    return _storage(result, budget)


def _combine(
    a: _Polynomial, b: _Polynomial, budget: _Budget, *, subtract: bool = True
) -> _Polynomial:
    budget.charge(len(a) + len(b))
    result = dict(a)
    for monomial, value in b.items():
        if subtract:
            budget.charge(1, _bits(value))
            value = -value
        _put(result, monomial, value, budget)
    return _storage(result, budget)


def _shift(a: _Polynomial, shift: _Monomial, budget: _Budget) -> _Polynomial:
    budget.charge(len(a))
    result = {}
    for monomial, value in a.items():
        target = tuple(x + y for x, y in zip(monomial, shift, strict=True))
        if max(target) > _MAX_EXPONENT:
            raise _resource()
        result[target] = value
    return result


def _sparse(
    polynomial: SparseRationalPolynomial,
    prefix: _Monomial,
    suffix: _Monomial,
    budget: _Budget,
) -> _Polynomial:
    result = {}
    for term in polynomial.terms:
        budget.charge(
            8, max(term.coefficient.num.bit_length(), term.coefficient.den.bit_length())
        )
        result[prefix + term.exponents + suffix] = term.coefficient.as_fraction()
    return _storage(result, budget)


def _equal(a: _Polynomial, b: _Polynomial, budget: _Budget) -> bool:
    if len(a) != len(b):
        return False
    budget.charge(len(a))
    for monomial, left in a.items():
        right = b.get(monomial)
        if right is None:
            return False
        budget.charge(1, max(_bits(left), _bits(right)))
        if left != right:
            return False
    return True


def _clear_denominators(
    polynomial: GenericFiberPolynomial, n: int, m: int, budget: _Budget
) -> tuple[_Polynomial, _Polynomial]:
    numerator: _Polynomial = {}
    denominator = {(0,) * (n + m): Fraction(1)}
    for term in polynomial.terms:
        top = _sparse(term.coefficient.numerator, term.source_exponents, (), budget)
        bottom = _sparse(term.coefficient.denominator, (0,) * n, (), budget)
        if not numerator:
            numerator, denominator = top, bottom
        elif _equal(bottom, denominator, budget):
            numerator = _combine(numerator, top, budget, subtract=False)
        else:
            numerator = _combine(
                _multiply(numerator, bottom, budget),
                _multiply(top, denominator, budget),
                budget,
                subtract=False,
            )
            denominator = _multiply(denominator, bottom, budget)
    return numerator, denominator


def _normalize(polynomial: _Polynomial, n: int, budget: _Budget) -> _Polynomial:
    """Remove only QQ(t) units, never a common source monomial."""
    if not polynomial:
        return polynomial
    budget.charge(len(polynomial) * len(next(iter(polynomial))))
    parameter_factor = tuple(
        min(monomial[i] for monomial in polynomial)
        for i in range(n, len(next(iter(polynomial))))
    )
    scale = next(iter(polynomial.values()))
    result = {}
    for monomial, coefficient in polynomial.items():
        budget.charge(4, _bits(coefficient) + _bits(scale))
        result[
            monomial[:n]
            + tuple(x - y for x, y in zip(monomial[n:], parameter_factor, strict=True))
        ] = coefficient / scale
    return _storage(result, budget)


def _leading(
    polynomial: _Polynomial, n: int, budget: _Budget
) -> tuple[_Monomial, _Polynomial]:
    budget.charge(2 * len(polynomial))
    leading = max(polynomial)[:n]
    coefficient = {
        (0,) * n + monomial[n:]: value
        for monomial, value in polynomial.items()
        if monomial[:n] == leading
    }
    return leading, coefficient


def _divides(a: _Monomial, b: _Monomial) -> bool:
    return all(x <= y for x, y in zip(a, b, strict=True))


def _reduces_to_zero(
    polynomial: _Polynomial,
    basis: list[tuple[_Polynomial, _Monomial, _Polynomial]],
    n: int,
    m: int,
    budget: _Budget,
) -> bool:
    while polynomial:
        leading, coefficient = _leading(polynomial, n, budget)
        budget.charge(len(basis) * n)
        divisor = next((g for g in basis if _divides(g[1], leading)), None)
        if divisor is None:
            # Later reductions have smaller source monomials and cannot remove
            # this irreducible leading term from the division remainder.
            return False
        g, lm, lc = divisor
        shift = tuple(a - b for a, b in zip(leading, lm, strict=True)) + (0,) * m
        polynomial = _normalize(
            _combine(
                _multiply(lc, polynomial, budget),
                _multiply(coefficient, _shift(g, shift, budget), budget),
                budget,
            ),
            n,
            budget,
        )
    return True


def _is_groebner(
    basis: list[tuple[_Polynomial, _Monomial, _Polynomial]],
    n: int,
    m: int,
    budget: _Budget,
) -> bool:
    for i, (a, la, ca) in enumerate(basis):
        for b, lb, cb in basis[:i]:
            lcm = tuple(max(x, y) for x, y in zip(la, lb, strict=True))
            sa = tuple(x - y for x, y in zip(lcm, la, strict=True)) + (0,) * m
            sb = tuple(x - y for x, y in zip(lcm, lb, strict=True)) + (0,) * m
            pair = _combine(
                _multiply(cb, _shift(a, sa, budget), budget),
                _multiply(ca, _shift(b, sb, budget), budget),
                budget,
            )
            if not _reduces_to_zero(pair, basis, n, m, budget):
                return False
    return True


def verify_generic_fiber(claim: GenericDegreeResult) -> bool:
    """Prove both ideal inclusions, Buchberger's criterion, and quotient data.

    False means a disproved relation or invalid claim; budget, timeout, and
    cancellation errors propagate without making a mathematical conclusion.
    """
    budget = _Budget()
    budget.charge(0)
    canonical = _canonical_claim(claim, budget)
    if canonical is None or canonical.evidence is None:
        return False
    evidence = canonical.evidence
    n, m = len(evidence.source_variable_order), len(evidence.target_parameters)
    if (
        evidence.source_variable_order != canonical.source.input_variables
        or m != len(canonical.source.output_polynomials)
        or m != len(evidence.basis_from_source)
    ):
        return False
    sources = []
    for i, polynomial in enumerate(canonical.source.output_polynomials):
        source = _sparse(polynomial.polynomial, (), (0,) * m, budget)
        parameter = (0,) * n + tuple(int(j == i) for j in range(m))
        _put(source, parameter, Fraction(-1), budget)
        sources.append(source)
    basis = []
    for j, fiber_polynomial in enumerate(evidence.basis):
        top, bottom = _clear_denominators(fiber_polynomial, n, m, budget)
        expected: _Polynomial = {}
        denominator = {(0,) * (n + m): Fraction(1)}
        for i, source in enumerate(sources):
            multiplier, divisor = _clear_denominators(
                evidence.basis_from_source[i][j], n, m, budget
            )
            term = _multiply(source, multiplier, budget)
            if _equal(divisor, denominator, budget):
                expected = _combine(expected, term, budget, subtract=False)
            else:
                expected = _combine(
                    _multiply(expected, divisor, budget),
                    _multiply(term, denominator, budget),
                    budget,
                    subtract=False,
                )
                denominator = _multiply(denominator, divisor, budget)
        difference = (
            _combine(top, expected, budget)
            if _equal(bottom, denominator, budget)
            else _combine(
                _multiply(top, denominator, budget),
                _multiply(expected, bottom, budget),
                budget,
            )
        )
        if difference:
            return False
        if top:
            top = _normalize(top, n, budget)
            leading, coefficient = _leading(top, n, budget)
            basis.append((top, leading, coefficient))
    if not basis:
        return False  # F_i - t_i is never the zero polynomial.
    if any(not any(leading) for _, leading, _ in basis):
        # The forward identities already show 1 belongs to the source ideal;
        # a unit leading monomial also proves this list is a Gröbner basis.
        return (
            canonical.outcome == "NOT_DOMINANT"
            and canonical.degree is None
            and not evidence.standard_monomials
        )
    if not _is_groebner(basis, n, m, budget):
        return False
    if not all(_reduces_to_zero(source, basis, n, m, budget) for source in sources):
        return False
    # This bounded breadth-first complement visits at most 512 monomials, with
    # at most 3 neighbours and 32 divisibility checks of width 3 per monomial.
    budget.charge(512 * n * len(basis) * n)
    try:
        monomials = enumerate_standard_monomials(tuple(lm for _, lm, _ in basis))
    except StandardMonomialLimitError as exc:
        raise _resource() from exc
    request_checkpoint("after generic-fiber standard-monomial enumeration")
    if monomials is None:
        return (
            canonical.outcome == "DOMINANT_NOT_GENERICALLY_FINITE"
            and canonical.degree is None
            and not evidence.standard_monomials
        )
    return (
        canonical.outcome == "GENERICALLY_FINITE"
        and canonical.degree == len(monomials)
        and evidence.standard_monomials == monomials
    )
