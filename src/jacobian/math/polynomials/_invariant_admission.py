"""Sparse integer-ring plans for bounded multivariate invariants.

Berkowitz is division-free. Every intermediate contains at most N matrix-entry
factors, with coefficient l1 norm bounded by (4*N*H)**N. No symbolic expression
or dense multivariate polynomial is constructed by this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, gcd, prod
from typing import Any, NoReturn

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_Exponents = tuple[int, ...]
_Polynomial = dict[_Exponents, int]
_MAX_BITS = (10**MAX_CANONICAL_RATIONAL_DIGITS).bit_length() - 1
_MAX_BIT_WORK = 4_000_000_000_000
_MAX_STORAGE = 128 * 1024 * 1024
_MAX_INTERMEDIATE_TERMS = 16_384


def _refuse(message: str) -> None:
    raise OperationResourceAdmissionError(
        location=(), code="polynomial.invariant_budget", message=message
    )


def require_invariant_source(source: RationalPolynomial) -> None:
    """Bound native shape/scalars before relying on canonical source invariants."""

    def invalid() -> NoReturn:
        raise OperationDomainValidationError(
            location=(),
            code="polynomial.invariant_source",
            message="invariant source must be a canonical sparse QQ polynomial",
        )

    variables = getattr(source, "variables", None)
    sparse = getattr(source, "polynomial", None)
    terms = getattr(sparse, "terms", None)
    if (
        not isinstance(source, RationalPolynomial)
        or getattr(source, "domain", None) != "QQ"
        or not isinstance(variables, tuple)
        or not 1 <= len(variables) <= 8
        or any(not isinstance(v, str) or not 1 <= len(v) <= 32 for v in variables)
        or len(set(variables)) != len(variables)
        or not isinstance(sparse, SparseRationalPolynomial)
        or not isinstance(terms, tuple)
        or len(terms) > 256
    ):
        invalid()
    import re

    if any(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", v) is None for v in variables):
        invalid()
    previous: tuple[int, ...] | None = None
    for term in terms:
        exponents = getattr(term, "exponents", None)
        coefficient = getattr(term, "coefficient", None)
        numerator = getattr(coefficient, "num", None)
        denominator = getattr(coefficient, "den", None)
        if (
            not isinstance(term, RationalPolynomialTerm)
            or not isinstance(exponents, tuple)
            or len(exponents) != len(variables)
            or any(
                type(e) is not int or not 0 <= e <= MAX_POLYNOMIAL_EXPONENT
                for e in exponents
            )
            or not isinstance(coefficient, CanonicalRational)
            or type(numerator) is not int
            or type(denominator) is not int
        ):
            invalid()
        if (
            not numerator
            or denominator <= 0
            or numerator.bit_length() > 1024
            or denominator.bit_length() > 1024
        ):
            invalid()
        if gcd(abs(numerator), denominator) != 1 or (
            previous is not None and previous <= exponents
        ):
            invalid()
        previous = exponents


@dataclass
class _Ledger:
    work: int = 0

    def charge(self, count: int, bits: int) -> None:
        self.work += count * max(1, bits) ** 2
        if self.work > _MAX_BIT_WORK:
            _refuse("invariant exact arithmetic exceeds the work budget")

    def storage(self, cells: int, bits: int) -> None:
        # A mathematical coefficient-storage reserve, not a process RSS bound.
        if cells * (96 + (bits + 7) // 8) > _MAX_STORAGE:
            _refuse("invariant coefficient storage exceeds the workspace budget")


@dataclass(frozen=True)
class _Shape:
    terms: int
    axes: _Exponents
    low_degree: int
    high_degree: int
    norm_bits: int


def _shape(polynomial: _Polynomial, dimensions: int) -> _Shape:
    return _Shape(
        terms=len(polynomial),
        axes=tuple(
            max((e[j] for e in polynomial), default=0) for j in range(dimensions)
        ),
        low_degree=min((sum(e) for e in polynomial), default=0),
        high_degree=max((sum(e) for e in polynomial), default=0),
        norm_bits=sum(abs(c) for c in polynomial.values()).bit_length(),
    )


def _support(axes: _Exponents, low: int, high: int) -> int:
    active = sum(d > 0 for d in axes)
    if not active:
        return 1
    simplex = comb(high + active, active) - (
        comb(low - 1 + active, active) if low else 0
    )
    return min(prod(d + 1 for d in axes), simplex)


def _power_shape(source: _Shape, exponent: int) -> _Shape:
    if not exponent:
        return _Shape(1, (0,) * len(source.axes), 0, 0, 1)
    if not source.terms:
        return _Shape(0, (0,) * len(source.axes), 0, 0, 0)
    axes = tuple(exponent * d for d in source.axes)
    low, high = exponent * source.low_degree, exponent * source.high_degree
    terms = min(
        _support(axes, low, high), comb(exponent + source.terms - 1, source.terms - 1)
    )
    return _Shape(terms, axes, low, high, exponent * source.norm_bits)


def _product_shape(left: _Shape, right: _Shape) -> _Shape:
    axes = tuple(a + b for a, b in zip(left.axes, right.axes, strict=True))
    low, high = left.low_degree + right.low_degree, left.high_degree + right.high_degree
    return _Shape(
        min(left.terms * right.terms, _support(axes, low, high)),
        axes,
        low,
        high,
        left.norm_bits + right.norm_bits,
    )


def _multiply_budget(
    left: _Shape, right: _Shape, result: _Shape, ledger: _Ledger
) -> None:
    if result.terms > _MAX_INTERMEDIATE_TERMS:
        _refuse("invariant intermediate support exceeds the workspace budget")
    ledger.charge(left.terms * right.terms, result.norm_bits + 1)
    ledger.storage(left.terms + right.terms + result.terms, result.norm_bits + 1)


def _power_budget(source: _Shape, exponent: int, ledger: _Ledger) -> _Shape:
    result_degree = 0
    factor_degree = 1
    remaining = exponent
    while remaining:
        if remaining & 1:
            _multiply_budget(
                _power_shape(source, result_degree),
                _power_shape(source, factor_degree),
                _power_shape(source, result_degree + factor_degree),
                ledger,
            )
            result_degree += factor_degree
        remaining >>= 1
        if remaining:
            _multiply_budget(
                _power_shape(source, factor_degree),
                _power_shape(source, factor_degree),
                _power_shape(source, 2 * factor_degree),
                ledger,
            )
            factor_degree *= 2
    return _power_shape(source, exponent)


def _integer_coefficients(
    source: RationalPolynomial, axis: int, ledger: _Ledger
) -> tuple[tuple[_Polynomial, ...], int]:
    terms = source.polynomial.terms
    denominator = 1
    for value in sorted({t.coefficient.den for t in terms}):
        ledger.charge(4, denominator.bit_length() + value.bit_length())
        factor = value // gcd(denominator, value)
        if denominator.bit_length() + factor.bit_length() > _MAX_BITS:
            _refuse("invariant common denominator exceeds the scalar workspace budget")
        denominator *= factor
    rows: list[_Polynomial] = [
        {} for _ in range(max((t.exponents[axis] for t in terms), default=0) + 1)
    ]
    for term in terms:
        bits = abs(term.coefficient.num).bit_length() + denominator.bit_length()
        ledger.charge(4, bits)
        exponent = term.exponents[:axis] + term.exponents[axis + 1 :]
        rows[term.exponents[axis]][exponent] = term.coefficient.num * (
            denominator // term.coefficient.den
        )
    ledger.storage(
        len(terms),
        max((abs(c).bit_length() for row in rows for c in row.values()), default=1),
    )
    return tuple(rows), denominator


@dataclass(frozen=True)
class InvariantPlan:
    variables: tuple[str, ...]
    denominator_factors: tuple[tuple[int, int], ...]
    sign: int
    scalar: int
    factors: tuple[tuple[_Polynomial, int], ...] | None = None
    matrix: tuple[tuple[_Polynomial, ...], ...] | None = None

    def execute(self) -> RationalPolynomial:
        from sympy import ZZ
        from sympy.polys.matrices import DomainMatrix

        domain = ZZ.poly_ring(*self.variables)
        convert = domain.ring.from_dict
        if self.factors is not None:
            value = domain.convert(self.scalar * self.sign)
            for factor, exponent in self.factors:
                value *= _power(convert(factor), exponent, domain.one)
        else:
            if self.matrix is None:
                raise RuntimeError("invariant plan is missing its admitted kernel")
            size = len(self.matrix)
            matrix = DomainMatrix(
                [[convert(entry) for entry in row] for row in self.matrix],
                (size, size),
                domain,
            )
            value = matrix.charpoly_berk()[-1] * ((-1) ** size * self.sign)
        denominator = prod(
            base**exponent for base, exponent in self.denominator_factors
        )
        return RationalPolynomial(
            variables=self.variables,
            polynomial=SparseRationalPolynomial(
                terms=tuple(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational.from_integer_ratio(
                            int(coefficient), denominator
                        ),
                        exponents=exponents,
                    )
                    for exponents, coefficient in sorted(value.items(), reverse=True)
                )
            ),
        )


def _power(value: Any, exponent: int, one: Any) -> Any:
    result = one
    while exponent:
        if exponent & 1:
            result *= value
        exponent >>= 1
        if exponent:
            value *= value
    return result


def _finish(plan: InvariantPlan, shape: _Shape, ledger: _Ledger) -> InvariantPlan:
    if shape.terms > MAX_POLYNOMIAL_TERMS:
        _refuse("invariant output exceeds the 4096-term canonical support budget")
    if any(d > MAX_POLYNOMIAL_EXPONENT for d in shape.axes):
        _refuse("invariant output exceeds the canonical exponent budget")
    denominator_bits = sum(
        e * (d.bit_length() if d != 1 else 0) for d, e in plan.denominator_factors
    )
    if max(shape.norm_bits, denominator_bits) > _MAX_BITS:
        _refuse("invariant output exceeds the canonical scalar budget")
    ledger.charge(
        max(
            1,
            2 * sum(e.bit_length() for _, e in plan.denominator_factors)
            + len(plan.denominator_factors),
        ),
        denominator_bits,
    )
    ledger.charge(4 * shape.terms, shape.norm_bits + denominator_bits + 1)
    ledger.storage(shape.terms, shape.norm_bits + denominator_bits + 1)
    return plan


def _powers(
    variables: tuple[str, ...],
    factors: tuple[tuple[_Polynomial, int], ...],
    denominators: tuple[tuple[int, int], ...],
    sign: int,
    scalar: int,
    ledger: _Ledger,
) -> InvariantPlan:
    shape = _Shape(1, (0,) * len(variables), 0, 0, abs(scalar).bit_length())
    for polynomial, exponent in factors:
        factor = _power_budget(_shape(polynomial, len(variables)), exponent, ledger)
        combined = _product_shape(shape, factor)
        _multiply_budget(shape, factor, combined, ledger)
        shape = combined
    return _finish(
        InvariantPlan(variables, denominators, sign, scalar, factors=factors),
        shape,
        ledger,
    )


def _sylvester(
    left: tuple[_Polynomial, ...], right: tuple[_Polynomial, ...]
) -> list[list[_Polynomial]]:
    m, n = len(left) - 1, len(right) - 1
    rows: list[list[_Polynomial]] = []
    for coefficients, shifts in ((left, n), (right, m)):
        for shift in range(shifts):
            rows.append(
                [{}] * shift
                + list(reversed(coefficients))
                + [{}] * (m + n - shift - len(coefficients))
            )
    return rows


def _matrix_plan(
    variables: tuple[str, ...],
    matrix: list[list[_Polynomial]],
    denominators: tuple[tuple[int, int], ...],
    sign: int,
    ledger: _Ledger,
) -> InvariantPlan:
    size = len(matrix)
    union = {e for row in matrix for entry in row for e in entry}
    union.add((0,) * len(variables))
    axes = tuple(size * max(e[j] for e in union) for j in range(len(variables)))
    total = size * max(map(sum, union))
    support = min(_support(axes, 0, total), comb(size + len(union) - 1, len(union) - 1))
    if support > _MAX_INTERMEDIATE_TERMS:
        _refuse("invariant determinant support exceeds the workspace budget")
    norm = max(
        1, *(sum(abs(c) for c in entry.values()) for row in matrix for entry in row)
    )
    bits = size * (4 * size * norm).bit_length()
    ledger.charge(4 * size**4 * support**2, bits + 1)
    ledger.storage(4 * max(1, size**3) * support, bits + 1)
    return _finish(
        InvariantPlan(
            variables, denominators, sign, 1, matrix=tuple(map(tuple, matrix))
        ),
        _Shape(support, axes, 0, total, bits),
        ledger,
    )


def resultant_plan(
    left: RationalPolynomial, right: RationalPolynomial, axis: int
) -> InvariantPlan:
    variables = left.variables[:axis] + left.variables[axis + 1 :]
    ledger = _Ledger()
    ledger.charge(4 * (len(left.polynomial.terms) + len(right.polynomial.terms)), 1024)
    if not left.polynomial.terms or not right.polynomial.terms:
        return _powers(variables, (({}, 1),), (), 1, 1, ledger)
    f, a = _integer_coefficients(left, axis, ledger)
    g, b = _integer_coefficients(right, axis, ledger)
    m, n = len(f) - 1, len(g) - 1
    denominators = ((a, n), (b, m))
    if sum(bool(row) for row in g) == 1:
        return _powers(
            variables, ((g[n], m), (f[0], n)), denominators, (-1) ** (m * n), 1, ledger
        )
    if sum(bool(row) for row in f) == 1:
        return _powers(variables, ((f[m], n), (g[0], m)), denominators, 1, 1, ledger)
    return _matrix_plan(variables, _sylvester(f, g), denominators, 1, ledger)


def discriminant_plan(source: RationalPolynomial, axis: int) -> InvariantPlan:
    variables = source.variables[:axis] + source.variables[axis + 1 :]
    ledger = _Ledger()
    ledger.charge(4 * len(source.polynomial.terms), 1024)
    f, denominator = _integer_coefficients(source, axis, ledger)
    degree = len(f) - 1
    if degree < 2:
        return _powers(variables, (({}, 1),) if not degree else (), (), 1, 1, ledger)
    sign = (-1) ** (degree * (degree - 1) // 2)
    denominators = ((denominator, 2 * degree - 2),)
    if not any(f[1:-1]):
        return _powers(
            variables,
            ((f[-1], degree - 1), (f[0], degree - 1)),
            denominators,
            sign,
            degree**degree,
            ledger,
        )
    ledger.charge(
        sum(len(row) for row in f),
        max((abs(c).bit_length() for row in f for c in row.values()), default=1)
        + degree.bit_length(),
    )
    derivative = tuple(
        {e: c * k for e, c in f[k].items()} for k in range(1, degree + 1)
    )
    matrix = _sylvester(f, derivative)
    zero = (0,) * len(variables)
    matrix[0][0] = {zero: 1}
    matrix[degree - 1][0] = {zero: degree}
    return _matrix_plan(variables, matrix, denominators, sign, ledger)
