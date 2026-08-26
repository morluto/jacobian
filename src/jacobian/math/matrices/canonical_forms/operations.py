"""Exact canonical-form kernels backed by SymPy polynomial algebra."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

__all__ = [
    "characteristic_polynomial",
    "invariant_factors",
    "minimal_polynomial",
    "primary_decomposition",
]

RationalEntries = Sequence[Sequence[Fraction]]
CoefficientList = tuple[Fraction, ...]


def _integer_decimal_digits(value: int) -> int:
    """Return the decimal width of ``value`` without decimal rendering.

    The conformance probe below can observe large admitted exact components.
    Avoiding ``str(int)`` keeps that private diagnostic independent of
    CPython's configurable decimal-conversion guard.
    """

    magnitude = abs(value)
    if magnitude == 0:
        return 1
    # 0.30103 is an upper rational approximation to log10(2), so this is an
    # upper estimate from the binary width.  A single exact comparison removes
    # the possible one-digit overestimate.
    digits = magnitude.bit_length() * 30_103 // 100_000 + 1
    if magnitude < 10 ** (digits - 1):
        return digits - 1
    return digits


@dataclass
class _HornerEvaluationMetrics:
    """Private kernel measurements for the matrix-polynomial pilot only.

    This is deliberately an optional observer on this owner-local kernel, not
    a cross-operation instrumentation protocol.  It records the published
    dense scalar-product proxy and the widest component of every materialized
    Horner state -- each pre-addition matrix product together with each
    reduced accumulator -- without changing evaluation semantics.
    """

    maximum_component_digits: int = 0
    scalar_product_terms: int = 0
    stored_states: int = 0

    def record_state(self, state: Any) -> None:
        """Record one materialized exact Horner state."""

        state_digits = max(
            (
                max(
                    _integer_decimal_digits(int(entry.p)),
                    _integer_decimal_digits(int(entry.q)),
                )
                for entry in state
            ),
            default=1,
        )
        self.maximum_component_digits = max(self.maximum_component_digits, state_digits)
        self.stored_states += 1


def _square_dimension(entries: RationalEntries) -> int:
    """Return the shared side length of a nonempty square entry matrix."""

    dimension = len(entries)
    if dimension == 0:
        raise ValueError("canonical-form operations require a nonempty square matrix")
    if any(len(row) != dimension for row in entries):
        raise ValueError("canonical-form operations require a square matrix")
    return dimension


def _sympy_matrix(entries: RationalEntries) -> Any:
    from sympy import Matrix, Rational

    return Matrix(
        [
            [Rational(entry.numerator, entry.denominator) for entry in row]
            for row in entries
        ]
    )


def _to_fraction(value: Any) -> Fraction:
    from sympy import Rational

    if not isinstance(value, Rational):
        raise ValueError("canonical-form backend returned a non-rational value")
    return Fraction(int(value.p), int(value.q))


def _coefficients(poly: Any) -> CoefficientList:
    """Return a monic polynomial's increasing-degree rational coefficients."""

    return tuple(
        _to_fraction(coefficient) for coefficient in reversed(poly.all_coeffs())
    )


def characteristic_polynomial(entries: RationalEntries) -> CoefficientList:
    """Return the monic characteristic polynomial coefficients [a_0, ..., a_n]."""

    from sympy import Poly, Symbol

    x = Symbol("x")
    _square_dimension(entries)
    matrix = _sympy_matrix(entries)
    charpoly = matrix.charpoly(x)
    return _coefficients(Poly(charpoly.as_expr(), x))


def _evaluate_polynomial(
    entries: RationalEntries,
    coefficients: Sequence[Fraction],
    *,
    metrics: _HornerEvaluationMetrics | None = None,
) -> tuple[tuple[Fraction, ...], ...]:
    """Return ``f(A)`` for increasing-degree coefficients by exact Horner evaluation."""

    from sympy import Rational, eye, zeros

    dimension = _square_dimension(entries)
    matrix = _sympy_matrix(entries)
    identity = eye(dimension)
    if not coefficients:
        result = zeros(dimension)
        if metrics is not None:
            metrics.record_state(result)
    else:
        leading = coefficients[-1]
        result = Rational(leading.numerator, leading.denominator) * identity
        if metrics is not None:
            metrics.record_state(result)
        for coefficient in reversed(coefficients[:-1]):
            scalar = Rational(coefficient.numerator, coefficient.denominator)
            product = result * matrix
            if metrics is not None:
                metrics.scalar_product_terms += dimension**3
                # The product is materialized before the scalar term can
                # cancel it, so the observed bound must cover this state too.
                metrics.record_state(product)
            result = product + scalar * identity
            if metrics is not None:
                metrics.record_state(result)
    return tuple(
        tuple(_to_fraction(result[row, column]) for column in range(dimension))
        for row in range(dimension)
    )


def minimal_polynomial(entries: RationalEntries) -> CoefficientList:
    """Compute the minimal polynomial via the Krylov/nullspace method.

    Returns the monic minimal polynomial as coefficient list [a_0, ..., a_n].
    """

    from sympy import Matrix, Poly, Symbol, eye

    x = Symbol("x")
    n = _square_dimension(entries)
    matrix = _sympy_matrix(entries)

    powers = [eye(n)]
    for _ in range(n):
        powers.append(powers[-1] * matrix)

    rows = [[mat[i, j] for i in range(n) for j in range(n)] for mat in powers]
    stacked = Matrix(rows).T

    _reduced, pivots = stacked.rref()
    degree = next((index for index in range(n + 1) if index not in pivots), None)
    if degree is None:
        raise ArithmeticError("Krylov subspace exceeded the Cayley-Hamilton bound")
    if degree == 0:
        return (Fraction(1),)

    submatrix = stacked[:, : degree + 1]
    null_vectors = submatrix.nullspace()
    if not null_vectors:
        raise ArithmeticError(
            "Krylov subspace produced no minimal polynomial dependency"
        )

    coefficients = null_vectors[0]
    dependency = sum(coefficients[index] * x**index for index in range(degree + 1))
    return _coefficients(Poly(dependency, x).monic())


def invariant_factors(entries: RationalEntries) -> tuple[CoefficientList, ...]:
    """Compute the non-unit invariant factors over QQ[x].

    Returns a list of monic polynomial coefficient lists, ordered by divisibility:
    f_1 | f_2 | ... | f_s.
    """

    from sympy import QQ, Poly, Symbol, eye
    from sympy.matrices.normalforms import smith_normal_form

    x = Symbol("x")
    n = _square_dimension(entries)
    matrix = _sympy_matrix(entries)
    characteristic_matrix = x * eye(n) - matrix
    smith = smith_normal_form(characteristic_matrix, domain=QQ[x])

    factors: list[CoefficientList] = []
    for index in range(n):
        diagonal = smith[index, index]
        if diagonal == 0:
            continue
        factor = Poly(diagonal, x).monic()
        if factor.degree() >= 1:
            factors.append(_coefficients(factor))
    return tuple(factors)


def primary_decomposition(entries: RationalEntries) -> tuple[CoefficientList, ...]:
    """Decompose the minimal polynomial into irreducible-power components.

    Returns a list of monic polynomial coefficient lists, one for each
    irreducible factor raised to its multiplicity in the minimal polynomial.
    """

    from sympy import Poly, Symbol, factor_list

    x = Symbol("x")
    minimal_coefficients = minimal_polynomial(entries)
    minimal_expression = sum(
        coefficient * x**index for index, coefficient in enumerate(minimal_coefficients)
    )
    _constant, factors = factor_list(minimal_expression, x)

    components: list[CoefficientList] = []
    for factor, power in factors:
        monic = Poly(factor, x).monic()
        components.append(_coefficients(monic**power))
    return tuple(components)
