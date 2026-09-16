"""Exact relative trace and norm for simple number fields over QQ.

Version 1 scope: the base field is ``QQ`` and the extension is one presented
simple field ``L = QQ(alpha) = QQ[x]/(f)`` with ``f`` primitive irreducible of
degree at most 8.  For ``a`` in ``L`` the kernel builds the ``QQ``-linear
multiplication matrix ``M_a`` in the power basis, reads
``Tr(a) = trace(M_a)`` and ``N(a) = det(M_a)``, and cross-checks both against
the complete embedding family without isolating roots:

* the trace equals the embedding sum ``sum_i g(alpha_i)`` replayed through
  Newton power-sum identities on the monicized defining polynomial;
* the norm equals the embedding product ``prod_i g(alpha_i)`` replayed
  through the exact resultant ``res(f, G) / lc(f)^deg(G)``.

The two reconstructions must agree; disagreement is an internal error, never
a published value.  Irreducibility of ``f`` is established by this
operation's admission (bounded SymPy check), so callers must not pre-certify
it in a model validator.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Any, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.values import RationalMatrix, rational_matrix_from_fractions
from jacobian.math.number_theory.number_fields.values import (
    MAX_SIMPLE_NUMBER_FIELD_COEFFICIENT_DIGITS,
    MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
)

MAX_RELATIVE_TRACE_NORM_DEGREE = 8


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"number_field.relative_trace_norm.{reason}", message)


def _admission_error(
    reason: str, message: str, *, location: tuple[str | int, ...] = ("field",)
) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=location,
        code=f"number_field.relative_trace_norm.{reason}",
        message=message,
    )


def _resource_error(reason: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("field",),
        code=f"number_field.relative_trace_norm.{reason}",
        message=message,
    )


class NumberFieldRelativeTraceNormRequest(StrictModel):
    """One presented field and one of its power-basis elements.

    The degree envelope (at most 8) is part of the wire contract: it bounds
    the multiplication matrix, characteristic polynomial, resultant, and
    power-sum replay before any backend work.  Element/field binding is
    structural; irreducibility and work admission belong to the owner kernel.
    """

    field: SimpleNumberFieldPresentation = Field(
        description=(
            "Primitive presented field QQ(alpha) = QQ[x]/(f); the admitted "
            f"execution envelope requires degree at most {MAX_RELATIVE_TRACE_NORM_DEGREE}."
        )
    )
    element: SimpleNumberFieldElement

    @model_validator(mode="after")
    def require_structural_binding(self) -> Self:
        if self.element.presentation != self.field:
            raise _validation_error(
                "element_field_mismatch",
                "the element must use the request field presentation",
            )
        if self.field.degree > MAX_RELATIVE_TRACE_NORM_DEGREE:
            raise _validation_error(
                "degree_bound",
                "relative trace/norm admits degree at most "
                f"{MAX_RELATIVE_TRACE_NORM_DEGREE}",
            )
        return self


class NumberFieldRelativeTraceNormResult(StrictModel):
    """Exact trace and norm with their multiplication-matrix ledger.

    ``characteristic_polynomial`` holds ascending coefficients of
    ``det(tI - M_a)``.  ``embedding_trace_sum`` and ``embedding_norm_product``
    are the independent Newton-identity and resultant replays of the same
    embedding sum/product; the kernel establishes their equality with
    ``trace``/``norm`` before construction.
    """

    field: SimpleNumberFieldPresentation
    element: SimpleNumberFieldElement
    trace: CanonicalRational
    norm: CanonicalRational
    multiplication_matrix: RationalMatrix
    characteristic_polynomial: tuple[CanonicalRational, ...] = Field(
        min_length=2,
        max_length=MAX_RELATIVE_TRACE_NORM_DEGREE + 1,
    )
    embedding_trace_sum: CanonicalRational
    embedding_norm_product: CanonicalRational

    @model_validator(mode="after")
    def require_result_shape(self) -> Self:
        degree = self.field.degree
        if self.element.presentation != self.field:
            raise _validation_error(
                "element_field_mismatch",
                "the result element must use the result field presentation",
            )
        matrix = self.multiplication_matrix
        if matrix.row_count != degree or matrix.column_count != degree:
            raise _validation_error(
                "matrix_shape",
                "the multiplication matrix must be degree-by-degree",
            )
        if len(self.characteristic_polynomial) != degree + 1:
            raise _validation_error(
                "characteristic_polynomial_degree",
                "the characteristic polynomial must have degree-many coefficients plus one",
            )
        if self.characteristic_polynomial[-1] != CanonicalRational(num=1, den=1):
            raise _validation_error(
                "characteristic_polynomial_monic",
                "the characteristic polynomial must be monic",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        field: SimpleNumberFieldPresentation,
        element: SimpleNumberFieldElement,
        trace: CanonicalRational,
        norm: CanonicalRational,
        multiplication_matrix: RationalMatrix,
        characteristic_polynomial: tuple[CanonicalRational, ...],
        embedding_trace_sum: CanonicalRational,
        embedding_norm_product: CanonicalRational,
    ) -> Self:
        return cls.model_construct(
            field=field,
            element=element,
            trace=trace,
            norm=norm,
            multiplication_matrix=multiplication_matrix,
            characteristic_polynomial=characteristic_polynomial,
            embedding_trace_sum=embedding_trace_sum,
            embedding_norm_product=embedding_norm_product,
        )


def _require_native_types(field: Any, element: Any) -> None:
    if not isinstance(field, SimpleNumberFieldPresentation):
        raise _admission_error(
            "field_type",
            "relative trace/norm requires a SimpleNumberFieldPresentation field",
        )
    if not isinstance(element, SimpleNumberFieldElement):
        raise _admission_error(
            "element_type",
            "relative trace/norm requires a SimpleNumberFieldElement element",
            location=("element",),
        )
    if element.presentation != field:
        raise _admission_error(
            "element_field_mismatch",
            "the element must use the request field presentation",
            location=("element",),
        )


def _admit_relative_trace_norm(field: SimpleNumberFieldPresentation) -> None:
    """Enforce the V1 execution envelope before any matrix/backend work."""
    degree = field.degree
    if degree > MAX_RELATIVE_TRACE_NORM_DEGREE:
        raise _admission_error(
            "degree_bound",
            "relative trace/norm admits degree at most "
            f"{MAX_RELATIVE_TRACE_NORM_DEGREE}",
        )
    # Conservative output reservation: reduction of g*x^j modulo the monicized
    # defining polynomial keeps entry bits within
    # elem_bits + n * (poly_bits + lc_bits + 2); the determinant then fits
    # within n * entry_bits + n * log2(n) bits by Hadamard's inequality.
    poly_bits = max(
        abs(coefficient).bit_length() for coefficient in field.coefficients_descending
    )
    entry_bits = MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS * 4 + degree * (
        poly_bits + MAX_SIMPLE_NUMBER_FIELD_COEFFICIENT_DIGITS * 4 + 2
    )
    determinant_bits = degree * entry_bits + degree * degree.bit_length()
    determinant_digits = determinant_bits * 30103 // 100000 + 2
    from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS

    if determinant_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise _resource_error(
            "output_bound",
            "the predicted trace/norm output exceeds the exact rational digit bound",
        )
    # Irreducibility makes the presented quotient a field; the check runs
    # here (once per invocation, shared by native and catalog paths), not in
    # a request validator.
    from sympy import Poly, Symbol

    polynomial = Poly.from_list(
        list(field.coefficients_descending), Symbol("x"), domain="ZZ"
    )
    if not polynomial.is_irreducible:
        raise _admission_error(
            "not_irreducible",
            "relative trace/norm requires an irreducible defining polynomial",
        )


def _reduce_modulo_field(
    coefficients: list[Fraction], monic: list[Fraction]
) -> list[Fraction]:
    """Reduce a dense ascending polynomial modulo a monic polynomial."""
    degree = len(monic) - 1
    result = list(coefficients)
    while len(result) > degree:
        factor = result.pop()
        if factor:
            base = len(result) - degree
            for index in range(degree):
                result[base + index] -= factor * monic[index]
    while len(result) < degree:
        result.append(Fraction(0))
    return result[:degree]


def _multiplication_matrix(
    field: SimpleNumberFieldPresentation, element: SimpleNumberFieldElement
) -> list[list[Fraction]]:
    """Return M_a with M_a[i][j] = coeff of x^i in (g * x^j mod f)."""
    degree = field.degree
    leading = Fraction(field.coefficients_descending[0])
    # Ascending monicized defining polynomial over QQ.
    monic = [
        Fraction(coefficient) / leading
        for coefficient in reversed(field.coefficients_descending)
    ]
    element_coefficients = [
        value.as_fraction() for value in element.coefficients_ascending
    ]
    matrix = [[Fraction(0)] * degree for _ in range(degree)]
    for column in range(degree):
        shifted = [Fraction(0)] * column + element_coefficients
        reduced = _reduce_modulo_field(shifted, monic)
        for row in range(degree):
            matrix[row][column] = reduced[row]
    return matrix


def _matrix_determinant(matrix: list[list[Fraction]]) -> Fraction:
    """Exact determinant by fraction Gaussian elimination (degree at most 8)."""
    size = len(matrix)
    if size == 1:
        return matrix[0][0]
    work = [list(row) for row in matrix]
    determinant = Fraction(1)
    for pivot in range(size):
        row = next(
            (candidate for candidate in range(pivot, size) if work[candidate][pivot]),
            None,
        )
        if row is None:
            return Fraction(0)
        if row != pivot:
            work[pivot], work[row] = work[row], work[pivot]
            determinant = -determinant
        determinant *= work[pivot][pivot]
        inverse = Fraction(1, 1) / work[pivot][pivot]
        for other in range(pivot + 1, size):
            factor = work[other][pivot] * inverse
            if factor:
                for column in range(pivot, size):
                    work[other][column] -= factor * work[pivot][column]
    return determinant


def _characteristic_polynomial(matrix: list[list[Fraction]]) -> list[Fraction]:
    """Ascending coefficients of det(tI - M) via Faddeeva-LeVerrier."""
    size = len(matrix)

    def multiply(
        left: list[list[Fraction]], right: list[list[Fraction]]
    ) -> list[list[Fraction]]:
        return [
            [
                sum(
                    (left[row][mid] * right[mid][column] for mid in range(size)),
                    Fraction(0),
                )
                for column in range(size)
            ]
            for row in range(size)
        ]

    coefficients = [Fraction(0)] * (size + 1)
    coefficients[size] = Fraction(1)
    auxiliary = [[Fraction(0)] * size for _ in range(size)]
    for step in range(1, size + 1):
        current = multiply(matrix, auxiliary)
        for index in range(size):
            current[index][index] += coefficients[size - step + 1]
        auxiliary = current
        image = multiply(matrix, auxiliary)
        trace = sum((image[row][row] for row in range(size)), Fraction(0))
        coefficients[size - step] = -trace / step
    return coefficients


def _power_sums(monic_descending: list[Fraction]) -> list[Fraction]:
    """Newton identities: p_k = sum of k-th powers of all roots."""
    degree = len(monic_descending) - 1
    # c[0] = 1, c[1..n] descending companions of x^{n-1}..x^0.
    companion = [Fraction(0), *monic_descending[1:]]
    sums: list[Fraction] = []
    for power in range(degree):
        if power == 0:
            sums.append(Fraction(degree))
        else:
            total = Fraction(0)
            for index in range(1, power + 1):
                total += companion[index] * sums[power - index]
            sums.append(-total)
    return sums


def _bareiss_determinant(matrix: list[list[int]]) -> int:
    """Exact integer determinant by the fraction-free Bareiss algorithm."""
    size = len(matrix)
    if size == 1:
        return matrix[0][0]
    work = [list(row) for row in matrix]
    previous = 1
    for pivot in range(size - 1):
        if work[pivot][pivot] == 0:
            swap = next(
                (
                    candidate
                    for candidate in range(pivot + 1, size)
                    if work[candidate][pivot] != 0
                ),
                None,
            )
            if swap is None:
                return 0
            work[pivot], work[swap] = work[swap], work[pivot]
        for row in range(pivot + 1, size):
            for column in range(pivot + 1, size):
                work[row][column] = (
                    work[row][column] * work[pivot][pivot]
                    - work[row][pivot] * work[pivot][column]
                ) // previous
            work[row][pivot] = 0
        previous = work[pivot][pivot]
    return work[size - 1][size - 1]


def _resultant_norm(
    field: SimpleNumberFieldPresentation, element: SimpleNumberFieldElement
) -> Fraction:
    """Embedding-product replay: res(f, G) / lc(f)^deg(G)."""
    degree = field.degree
    ascending_g = [value.as_fraction() for value in element.coefficients_ascending]
    while len(ascending_g) > 1 and ascending_g[-1] == 0:
        ascending_g.pop()
    if all(value == 0 for value in ascending_g):
        return Fraction(0)
    element_degree = len(ascending_g) - 1
    leading = field.coefficients_descending[0]
    # Clear denominators of G: res(f, G) = D^-n res(f, Ghat).
    common = 1
    for value in ascending_g:
        common = common * value.denominator // gcd(common, value.denominator)
    f_ascending = list(reversed(field.coefficients_descending))
    g_hat = [int(value * common) for value in ascending_g]
    size = degree + element_degree
    sylvester = [[0] * size for _ in range(size)]
    for row in range(element_degree):
        for column, coefficient in enumerate(f_ascending):
            sylvester[row][row + column] = coefficient
    for row in range(degree):
        for column, coefficient in enumerate(g_hat):
            sylvester[element_degree + row][row + column] = coefficient
    resultant = _bareiss_determinant(sylvester)
    return Fraction(resultant, (common**degree) * (leading**element_degree))


def relative_trace_norm(
    field: SimpleNumberFieldPresentation,
    element: SimpleNumberFieldElement,
) -> NumberFieldRelativeTraceNormResult:
    """Return exact trace and norm of one field element over QQ.

    This native entry is an admission boundary: it validates strict runtime
    types, shape, degree, output, and irreducibility before any matrix or
    backend work, exactly like ``number_field.relative_trace_norm.compute``.
    """
    _require_native_types(field, element)
    _admit_relative_trace_norm(field)

    degree = field.degree
    matrix = _multiplication_matrix(field, element)
    trace: Fraction = sum(
        (matrix[index][index] for index in range(degree)), Fraction(0)
    )
    determinant = _matrix_determinant(matrix)
    characteristic = _characteristic_polynomial(matrix)

    if characteristic[degree - 1] != -trace or characteristic[0] != (
        determinant if degree % 2 == 0 else -determinant
    ):
        raise RuntimeError(
            "number-field trace/norm kernel produced an inconsistent "
            "characteristic polynomial"
        )

    # Independent embedding-family replays (no root isolation).
    leading = Fraction(field.coefficients_descending[0])
    monic_descending = [
        Fraction(coefficient) / leading for coefficient in field.coefficients_descending
    ]
    sums = _power_sums(monic_descending)
    element_coefficients = [
        value.as_fraction() for value in element.coefficients_ascending
    ]
    embedding_trace: Fraction = sum(
        (
            coefficient * sums[power]
            for power, coefficient in enumerate(element_coefficients)
        ),
        Fraction(0),
    )
    embedding_norm = _resultant_norm(field, element)
    if embedding_trace != trace or embedding_norm != determinant:
        raise RuntimeError(
            "number-field trace/norm kernel disagrees with its embedding replay"
        )

    return NumberFieldRelativeTraceNormResult._from_kernel(
        field=field,
        element=element,
        trace=CanonicalRational.from_fraction(trace),
        norm=CanonicalRational.from_fraction(determinant),
        multiplication_matrix=rational_matrix_from_fractions(
            tuple(tuple(row) for row in matrix)
        ),
        characteristic_polynomial=tuple(
            CanonicalRational.from_fraction(value) for value in characteristic
        ),
        embedding_trace_sum=CanonicalRational.from_fraction(embedding_trace),
        embedding_norm_product=CanonicalRational.from_fraction(embedding_norm),
    )


__all__ = [
    "MAX_RELATIVE_TRACE_NORM_DEGREE",
    "NumberFieldRelativeTraceNormRequest",
    "NumberFieldRelativeTraceNormResult",
    "relative_trace_norm",
]
