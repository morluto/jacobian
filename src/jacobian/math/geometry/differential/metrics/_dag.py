"""Preflight polynomial DAG with explicitly shared rational denominator factors."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import NoReturn

from jacobian._execution import request_checkpoint
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.rational_functions._bounds import (
    BoundWorkCategory,
    FractionBound,
    PolynomialBound,
    RationalFunctionBoundLimits,
    _add_polynomials,
    _check_raw_polynomial,
    _differentiate_polynomial,
    _fraction_bound,
    _multiply_polynomials,
    _one_polynomial,
    _polynomial_admission_work_units,
    _polynomial_backend_conversion_work_units,
    _polynomial_bound,
    _recognition_work_units,
    _remove_guaranteed_common_monomial,
    _validate_canonical_result_bound,
    _zero_polynomial,
)
from jacobian.math.polynomials.values import RationalFunction, SparseRationalPolynomial


def reject(reason: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("metric",),
        code=f"differential_geometry.curvature.{reason}",
        message=message,
    )


class Ledger:
    limits = RationalFunctionBoundLimits(
        raw_terms=4096,
        raw_digits=4096,
        result_exponent=64,
        result_terms=256,
        result_digits=128,
        label="metric curvature",
        reject=reject,
    )

    def __init__(self) -> None:
        self.work = 0
        self.allocation_bits = 0

    def charge(self, category: BoundWorkCategory, amount: int) -> None:
        request_checkpoint(f"admitting {self.limits.label} {category}")
        self.work += amount
        if self.work > 50_000_000:
            self.limits.reject(
                "work",
                f"complete {self.limits.label} DAG exceeds 50,000,000 work units",
            )


def admit_recognition_work(
    values: tuple[RationalFunction, ...],
    *,
    reject: Callable[[str, str], NoReturn] | None = None,
    label: str | None = None,
) -> None:
    """Charge coprimality work against the shared 50,000,000-unit envelope."""

    ledger = Ledger()
    if reject is not None or label is not None:
        ledger.limits = replace(
            ledger.limits,
            reject=reject or ledger.limits.reject,
            label=label or ledger.limits.label,
        )
    for value in dict.fromkeys(values):
        bound = _fraction_bound(value, ledger)
        ledger.charge("recognition", _recognition_work_units(bound))


@dataclass(frozen=True)
class Expression:
    """Scalar times polynomial-node factors divided by polynomial-node factors."""

    scalar: Fraction = Fraction(1)
    numerator: tuple[int, ...] = ()
    denominator: tuple[int, ...] = ()


ZERO = Expression(Fraction(0))
ONE = Expression()


@dataclass(frozen=True)
class Node:
    operation: str
    arguments: tuple[int, ...]
    bound: PolynomialBound
    source: SparseRationalPolynomial | None = None
    scalar: Fraction = Fraction(1)
    axis: int = 0


class Dag:
    """Plan once without polynomial expansion; executor evaluates these nodes.

    Polynomial identity is used only when identical nodes occur. Fraction
    addition uses a common multiset of denominator factors. Multiplication
    cancels identical numerator/denominator factors before expansion. These
    exact structural reductions retain shared metric denominators across the
    complete inverse/connection/curvature formula.
    """

    def __init__(
        self,
        dimension: int,
        *,
        reject: Callable[[str, str], NoReturn] | None = None,
        label: str | None = None,
    ) -> None:
        self.dimension = dimension
        self.ledger = Ledger()
        if reject is not None or label is not None:
            self.ledger.limits = replace(
                self.ledger.limits,
                reject=reject or self.ledger.limits.reject,
                label=label or self.ledger.limits.label,
            )
        self.nodes = [
            Node("ZERO", (), _zero_polynomial(dimension)),
            Node("ONE", (), _one_polynomial(dimension)),
        ]
        self.keys: dict[tuple[object, ...], int] = {}
        self.guard_keys: dict[int, object] = {}

    def intern(self, key: tuple[object, ...], node: Node) -> int:
        if key in self.keys:
            return self.keys[key]
        _check_raw_polynomial(node.bound, self.ledger.limits)
        content_digits = len(
            format_canonical_integer(abs(node.bound.rational_content.numerator))
        ) + len(format_canonical_integer(node.bound.rational_content.denominator))
        self.ledger.allocation_bits += max(
            node.bound.terms, self.dense(node.bound.degrees)
        ) * (
            16
            + 4 * (node.bound.coefficient_digits + content_digits)
            + 64 * self.dimension
        )
        if len(self.nodes) >= 16384 or self.ledger.allocation_bits > 268_435_456:
            self.ledger.limits.reject(
                "allocation",
                f"{self.ledger.limits.label} DAG exceeds node or coefficient allocation",
            )
        result = len(self.nodes)
        self.nodes.append(node)
        self.keys[key] = result
        return result

    def source(self, polynomial: SparseRationalPolynomial) -> Expression:
        self.ledger.charge(
            "source_conversion",
            _polynomial_admission_work_units(polynomial, self.dimension),
        )
        if not polynomial.terms:
            return ZERO
        if len(polynomial.terms) == 1 and not any(polynomial.terms[0].exponents):
            return Expression(polynomial.terms[0].coefficient.as_fraction())
        key = (
            "SOURCE",
            tuple(
                (term.exponents, term.coefficient.as_fraction())
                for term in polynomial.terms
            ),
        )
        if key in self.keys:
            return Expression(numerator=(self.keys[key],))
        bound = _polynomial_bound(polynomial)
        self.ledger.charge(
            "source_conversion", _polynomial_backend_conversion_work_units(bound)
        )
        index = self.intern(key, Node("SOURCE", (), bound, source=polynomial))
        return Expression(numerator=(index,))

    def fraction(self, value: RationalFunction) -> Expression:
        numerator, denominator = (
            self.source(value.numerator),
            self.source(value.denominator),
        )

        # Recognition sees the authored numerator and denominator separately.
        # Preserve those raw bounds before structural factor cancellation can
        # make a presentation such as x/x look constant. Source conversion
        # is charged again here for the recognition backend; ``source`` owns
        # the separate conversion reservation used by the executor.
        raw_numerator = self.nodes[
            self.polynomial(numerator.numerator, numerator.scalar)
        ].bound
        raw_denominator = self.nodes[
            self.polynomial(denominator.numerator, denominator.scalar)
        ].bound
        raw_bound = FractionBound(raw_numerator, raw_denominator)
        self.ledger.charge("recognition", _recognition_work_units(raw_bound))
        self.ledger.charge(
            "source_conversion",
            _polynomial_backend_conversion_work_units(raw_numerator)
            + _polynomial_backend_conversion_work_units(raw_denominator),
        )
        result = self.multiply(numerator, self.inverse(denominator))
        return result

    @staticmethod
    def inverse(value: Expression) -> Expression:
        if not value.scalar:
            raise ZeroDivisionError("zero expression cannot be inverted")
        return Expression(1 / value.scalar, value.denominator, value.numerator)

    @staticmethod
    def multiply(*values: Expression) -> Expression:
        numerator: Counter[int] = Counter()
        denominator: Counter[int] = Counter()
        scalar = Fraction(1)
        for value in values:
            scalar *= value.scalar
            if not scalar:
                return ZERO
            numerator.update(value.numerator)
            denominator.update(value.denominator)
        common = numerator & denominator
        return Expression(
            scalar,
            tuple(sorted((numerator - common).elements())),
            tuple(sorted((denominator - common).elements())),
        )

    def polynomial(
        self, factors: tuple[int, ...], scalar: Fraction = Fraction(1)
    ) -> int:
        if not scalar:
            return 0
        result = 1
        for factor in factors:
            if result == 1:
                result = factor
            else:
                key: tuple[object, ...] = ("MULTIPLY", result, factor)
                if key in self.keys:
                    result = self.keys[key]
                else:
                    bound = _multiply_polynomials(
                        self.nodes[result].bound, self.nodes[factor].bound, self.ledger
                    )
                    result = self.intern(key, Node("MULTIPLY", (result, factor), bound))
        if scalar != 1:
            key = ("SCALE", result, scalar)
            bound = replace(
                self.nodes[result].bound,
                rational_content=self.nodes[result].bound.rational_content * scalar,
            )
            result = self.intern(key, Node("SCALE", (result,), bound, scalar=scalar))
        return result

    def add(self, *values: Expression) -> Expression:
        grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], Fraction] = {}
        for value in values:
            fraction_key = (value.numerator, value.denominator)
            grouped[fraction_key] = (
                grouped.get(fraction_key, Fraction(0)) + value.scalar
            )
        nonzero = [
            Expression(scalar, *key)
            for key, scalar in sorted(grouped.items())
            if scalar
        ]
        if not nonzero:
            return ZERO
        if len(nonzero) == 1:
            return nonzero[0]
        denominator: Counter[int] = Counter()
        for value in nonzero:
            denominator |= Counter(value.denominator)
        terms = tuple(
            self.polynomial(
                tuple(
                    sorted(
                        (
                            *value.numerator,
                            *(denominator - Counter(value.denominator)).elements(),
                        )
                    )
                ),
                value.scalar,
            )
            for value in nonzero
        )
        key = ("ADD", *terms)
        if key in self.keys:
            result = self.keys[key]
        else:
            bound = self.nodes[terms[0]].bound
            for term in terms[1:]:
                bound = _add_polynomials(bound, self.nodes[term].bound, self.ledger)
            result = self.intern(key, Node("ADD", terms, bound))
        return Expression(
            numerator=(result,), denominator=tuple(sorted(denominator.elements()))
        )

    def derivative(self, value: Expression, axis: int) -> Expression:
        if not value.scalar:
            return ZERO
        numerator = self.polynomial(value.numerator)
        denominator = self.polynomial(value.denominator)
        a, b = self.nodes[numerator].bound, self.nodes[denominator].bound
        if a.degrees[axis] == b.degrees[axis] == 0:
            return ZERO
        key = ("DERIVATIVE", numerator, denominator, axis)
        if key in self.keys:
            result = self.keys[key]
        else:

            def differential(bound: PolynomialBound) -> PolynomialBound:
                return _differentiate_polynomial(
                    bound,
                    axis,
                    self.ledger,
                    active_terms=bound.terms if bound.degrees[axis] else 0,
                    maximum_axis_exponent=max(1, bound.degrees[axis]),
                    minimum_exponents=tuple(
                        max(0, e - int(i == axis))
                        for i, e in enumerate(bound.minimum_exponents)
                    ),
                )

            da, db = differential(a), differential(b)
            self.ledger.charge("differentiation", a.terms + b.terms)
            # The quotient-rule kernel materializes D squared even when
            # later factor cancellation removes it from the result.
            temporary = _multiply_polynomials(b, b, self.ledger)
            temporary_digits = (
                temporary.coefficient_digits
                + len(
                    format_canonical_integer(abs(temporary.rational_content.numerator))
                )
                + len(format_canonical_integer(temporary.rational_content.denominator))
            )
            self.ledger.allocation_bits += self.dense(temporary.degrees) * (
                64 * self.dimension + 4 * temporary_digits + 16
            )
            bound = _add_polynomials(
                _multiply_polynomials(da, b, self.ledger),
                _multiply_polynomials(a, db, self.ledger),
                self.ledger,
            )
            result = self.intern(
                key, Node("DERIVATIVE", (numerator, denominator), bound, axis=axis)
            )
        return Expression(value.scalar, (result,), tuple(sorted(value.denominator * 2)))

    def bound(self, value: Expression) -> FractionBound:
        return FractionBound(
            self.nodes[self.polynomial(value.numerator, value.scalar)].bound,
            self.nodes[self.polynomial(value.denominator)].bound,
        )

    def admit_output(self, value: Expression) -> tuple[int, int, int]:
        """Bound canonical polynomial terms, coefficient bits and coordinate slots."""
        if not value.numerator and not value.denominator:
            # Exact scalar metadata avoids charging a factor-height bound to
            # unchanged constant metric entries or their reciprocals.
            digits = max(
                len(format_canonical_integer(abs(value.scalar.numerator))),
                len(format_canonical_integer(value.scalar.denominator)),
            )
            if digits > 128:
                self.ledger.limits.reject(
                    "result_height", "constant result exceeds 128 coefficient digits"
                )
            self.ledger.charge("normalization", 1)
            terms = 1 + int(bool(value.scalar))
            return terms, 8 * digits * terms, self.dimension * (terms + 2)
        bound = _remove_guaranteed_common_monomial(self.bound(value))
        digits = _validate_canonical_result_bound(bound, self.ledger)
        terms = sum(
            min(256, self.dense(part.degrees))
            for part in (bound.numerator, bound.denominator)
        )
        # Both integers in each rational coefficient have fewer than 4h
        # bits for h decimal digits. Each term retains its exponent vector,
        # and each numerator/denominator polynomial retains its variable axis.
        return terms, 8 * digits * terms, self.dimension * (terms + 2)

    @staticmethod
    def dense(degrees: tuple[int, ...]) -> int:
        result = 1
        for degree in degrees:
            result *= degree + 1
        return result
