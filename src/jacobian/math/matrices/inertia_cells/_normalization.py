"""Bounded scalar and singleton-domain reductions before algebraic admission."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.analysis.intervals import ClosedRationalInterval
from jacobian.math.matrices.symbolic.values import RationalPolynomialMatrix
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _budget(message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("interval",),
        code="matrix.inertia_cells.specialization_budget",
        message=message,
    )


def _polynomial(
    variables: tuple[str, ...], coefficients: tuple[Fraction, ...]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(q), exponents=(i,)
                )
                for i, q in reversed(tuple(enumerate(coefficients)))
                if q
            )
        ),
    )


def normalize_affine(
    poly: RationalPolynomial, interval: ClosedRationalInterval
) -> RationalPolynomial:
    """Keep only sign when no root lies in the domain; otherwise primitive affine."""
    coefficients = {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in poly.polynomial.terms
    }
    a, b = coefficients.get(1, Fraction()), coefficients.get(0, Fraction())
    if a:
        root = -b / a
        if interval.lower.as_fraction() <= root <= interval.upper.as_fraction():
            if (
                max(abs(root.numerator).bit_length(), root.denominator.bit_length())
                > 100000
            ):
                raise _budget("rational transition exceeds canonical endpoint height")
            sign = 1 if a > 0 else -1
            return _polynomial(
                poly.variables,
                (Fraction(-sign * root.numerator), Fraction(sign * root.denominator)),
            )
    value = a * interval.lower.as_fraction() + b
    sign = 1 if value > 0 else -1 if value < 0 else 0
    return _polynomial(poly.variables, (Fraction(sign),))


def specialize(
    matrix: RationalPolynomialMatrix, parameter: Fraction
) -> RationalPolynomialMatrix:
    """Evaluate sparse QQ[t] entries after bounding powers and rational sums.

    At 0 only constant terms contribute; at ±1 powers have unit height.
    No root isolation or algebraic carrier is involved in a singleton domain.
    """
    work = 0
    for row in matrix.entries:
        for polynomial in row:
            terms = tuple(
                t
                for t in polynomial.polynomial.terms
                if parameter or not t.exponents[0]
            )
            degree = max((t.exponents[0] for t in terms), default=0)
            denominator_bits = sum(t.coefficient.den.bit_length() for t in terms)
            numerator_bits = max(
                (abs(t.coefficient.num).bit_length() for t in terms), default=1
            )
            power_bits = (
                1
                if parameter in (0, 1, -1)
                else degree
                * max(
                    abs(parameter.numerator).bit_length(),
                    parameter.denominator.bit_length(),
                )
            )
            bits = (
                denominator_bits
                + numerator_bits
                + 2 * power_bits
                + (len(terms) + 1).bit_length()
            )
            if bits > 100000:
                raise _budget(
                    "singleton polynomial evaluation exceeds the rational height budget"
                )
            work += (len(terms) + degree.bit_length() + 1) * bits**2
    if work > 1_000_000_000_000:
        raise _budget("singleton polynomial evaluation exceeds the arithmetic budget")
    entries = []
    for row in matrix.entries:
        request_checkpoint("during admitted singleton specialization")
        values = []
        for polynomial in row:
            value = sum(
                (
                    t.coefficient.as_fraction() * parameter ** t.exponents[0]
                    for t in polynomial.polynomial.terms
                    if parameter or not t.exponents[0]
                ),
                Fraction(),
            )
            values.append(_polynomial(matrix.variables, (value,)))
        entries.append(tuple(values))
    return RationalPolynomialMatrix(
        variables=matrix.variables,
        row_count=matrix.row_count,
        column_count=matrix.column_count,
        entries=tuple(entries),
    )
