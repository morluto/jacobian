"""Exact native tropical arithmetic."""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from typing import Literal

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.tropical._models import AddBranch, InfinityCase
from jacobian.math.polynomials.tropical.values import (
    MAX_TROPICAL_EXPONENT,
    MAX_TROPICAL_POLYNOMIAL_TERMS,
    MAX_TROPICAL_SCALAR_DIGITS,
    TropicalMatrix,
    TropicalPolynomial,
    TropicalPolynomialTerm,
    TropicalScalar,
    TropicalSemiring,
    TropicalVector,
    require_scalar_budget,
)


def _admit_scalar(s: TropicalScalar, semiring: TropicalSemiring) -> None:
    if not isinstance(semiring, TropicalSemiring):
        raise OperationDomainValidationError(
            location=("semiring",),
            code="tropical.semiring_type",
            message="semiring must be a tropical semiring",
        )
    if not isinstance(s, TropicalScalar) or s.semiring != semiring:
        raise OperationDomainValidationError(
            location=("scalar",),
            code="tropical.semiring_mismatch",
            message="scalar must carry the request semiring",
        )
    if s.kind == "FINITE":
        if not isinstance(s.value, CanonicalRational) or (
            semiring.base == "ZZ" and s.value.den != 1
        ):
            raise OperationDomainValidationError(
                location=("scalar",),
                code="tropical.scalar_shape",
                message="finite scalar is not valid for its semiring",
            )
    elif s.value is not None or (s.kind == "POSITIVE_INFINITY") != (
        semiring.convention == "MIN_PLUS"
    ):
        raise OperationDomainValidationError(
            location=("scalar",),
            code="tropical.scalar_shape",
            message="infinity is not licensed by its semiring",
        )
    try:
        require_scalar_budget(s)
    except ValueError as error:
        raise OperationResourceAdmissionError(
            location=("scalar",),
            code="tropical.scalar_output_budget",
            message="scalar exceeds the admitted digit envelope",
        ) from error


def _admit_vector(vector: TropicalVector) -> None:
    if not isinstance(vector, TropicalVector):
        raise OperationDomainValidationError(
            location=("vector",),
            code="tropical.vector_type",
            message="expected a tropical vector",
        )
    if len(vector.axis) != len(vector.entries) or len(set(vector.axis)) != len(
        vector.axis
    ):
        raise OperationDomainValidationError(
            location=("vector",),
            code="tropical.vector_shape",
            message="vector axis and entries must have the same unique labels",
        )
    if len(vector.entries) > 128:
        raise OperationResourceAdmissionError(
            location=("vector",),
            code="tropical.vector_dimension",
            message="vector exceeds the admitted dimension envelope",
        )
    for entry in vector.entries:
        _admit_scalar(entry, vector.semiring)


def _admit_matrix(matrix: TropicalMatrix) -> None:
    if not isinstance(matrix, TropicalMatrix):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="tropical.matrix_type",
            message="expected a tropical matrix",
        )
    if len(matrix.row_axis) != len(matrix.entries) or any(
        len(row) != len(matrix.column_axis) for row in matrix.entries
    ):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="tropical.matrix_shape",
            message="matrix entries must match row and column axes",
        )
    if len(matrix.row_axis) * len(matrix.column_axis) > 4_096:
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="tropical.matrix_cells",
            message="matrix exceeds the admitted cell envelope",
        )
    for row in matrix.entries:
        for entry in row:
            _admit_scalar(entry, matrix.semiring)


def _admit_polynomial(poly: TropicalPolynomial) -> None:
    if not isinstance(poly, TropicalPolynomial):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="tropical.polynomial_type",
            message="expected a tropical polynomial",
        )
    if len(poly.terms) > MAX_TROPICAL_POLYNOMIAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="tropical.polynomial_terms",
            message="polynomial exceeds the admitted term envelope",
        )
    exponents = []
    for term in poly.terms:
        if not isinstance(term, TropicalPolynomialTerm) or len(term.exponents) != len(
            poly.variables
        ):
            raise OperationDomainValidationError(
                location=("polynomial",),
                code="tropical.polynomial_shape",
                message="polynomial terms must match the variable axis",
            )
        if any(
            type(value) is not int or value < 0 or value > MAX_TROPICAL_EXPONENT
            for value in term.exponents
        ):
            raise OperationDomainValidationError(
                location=("polynomial",),
                code="tropical.exponent_bound",
                message="polynomial exponents must be nonnegative and bounded",
            )
        exponents.append(term.exponents)
        if term.coefficient.kind != "FINITE":
            raise OperationDomainValidationError(
                location=("polynomial", "terms"),
                code="tropical.polynomial_infinity",
                message="infinite coefficients are omitted from canonical polynomials",
            )
        _admit_scalar(term.coefficient, poly.semiring)
    if tuple(exponents) != tuple(sorted(exponents)) or len(set(exponents)) != len(
        exponents
    ):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="tropical.polynomial_terms",
            message="polynomial terms must be sorted and unique",
        )


def _finite_value(scalar: TropicalScalar) -> CanonicalRational:
    if scalar.kind != "FINITE" or scalar.value is None:
        raise RuntimeError("admitted finite tropical scalar lost its exact value")
    return scalar.value


def _order(s: TropicalScalar) -> Fraction | None:
    return None if s.kind != "FINITE" else _finite_value(s).as_fraction()


def _digits(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


def _reject_growth(location: tuple[str, ...], message: str) -> None:
    raise OperationResourceAdmissionError(
        location=location,
        code="tropical.scalar_output_budget",
        message=message,
    )


def _check_fraction_sum_growth(
    left: CanonicalRational, right: CanonicalRational
) -> None:
    """Preflight numerator/denominator growth for exact rational addition."""
    numerator_digits = (
        max(
            _digits(left.num) + _digits(right.den),
            _digits(right.num) + _digits(left.den),
        )
        + 1
    )
    denominator_digits = _digits(left.den) + _digits(right.den)
    if max(numerator_digits, denominator_digits) > MAX_TROPICAL_SCALAR_DIGITS:
        _reject_growth(
            ("left", "right"),
            "tropical arithmetic output exceeds the scalar digit envelope",
        )


def _check_scale_growth(value: CanonicalRational, factor: int) -> None:
    if _digits(value.num) + _digits(factor) > MAX_TROPICAL_SCALAR_DIGITS:
        _reject_growth(
            ("scalar",),
            "tropical scaled output exceeds the scalar digit envelope",
        )


def _preflight_scalar_product(left: TropicalScalar, right: TropicalScalar) -> None:
    if left.kind == "FINITE" and right.kind == "FINITE":
        _check_fraction_sum_growth(_finite_value(left), _finite_value(right))


def _sum_fractions_checked(left: Fraction, right: Fraction) -> Fraction:
    result = CanonicalRational.from_fraction(left)
    other = CanonicalRational.from_fraction(right)
    _check_fraction_sum_growth(result, other)
    return left + right


def _scale_fraction_checked(value: Fraction, factor: int) -> Fraction:
    canonical = CanonicalRational.from_fraction(value)
    _check_scale_growth(canonical, factor)
    return value * factor


def tropical_scalar_add(
    semiring: TropicalSemiring, left: TropicalScalar, right: TropicalScalar
) -> tuple[TropicalScalar, AddBranch, InfinityCase]:
    _admit_scalar(left, semiring)
    _admit_scalar(right, semiring)
    lv, rv = _order(left), _order(right)
    if lv is None and rv is None:
        branch: AddBranch = "TIE"
        winner = left
    elif lv is None:
        branch = "RIGHT"
        winner = right
    elif rv is None:
        branch = "LEFT"
        winner = left
    elif lv == rv:
        branch = "TIE"
        winner = left
    elif (semiring.convention == "MIN_PLUS" and lv < rv) or (
        semiring.convention == "MAX_PLUS" and lv > rv
    ):
        branch = "LEFT"
        winner = left
    else:
        branch = "RIGHT"
        winner = right
    case: InfinityCase = (
        "BOTH_INFINITE"
        if left.kind != "FINITE" and right.kind != "FINITE"
        else "LEFT_INFINITE"
        if left.kind != "FINITE"
        else "RIGHT_INFINITE"
        if right.kind != "FINITE"
        else "NONE"
    )
    return (
        TropicalScalar._from_kernel(
            semiring=semiring, kind=winner.kind, value=winner.value
        ),
        branch,
        case,
    )


def tropical_scalar_multiply(
    left: TropicalScalar, right: TropicalScalar
) -> TropicalScalar:
    if not isinstance(left, TropicalScalar) or not isinstance(right, TropicalScalar):
        raise OperationDomainValidationError(
            location=("scalar",),
            code="tropical.scalar_type",
            message="expected tropical scalar operands",
        )
    if left.semiring != right.semiring:
        raise OperationDomainValidationError(
            location=("right",),
            code="tropical.semiring_mismatch",
            message="scalars must share semiring",
        )
    _admit_scalar(left, left.semiring)
    _admit_scalar(right, right.semiring)
    if left.kind != "FINITE" or right.kind != "FINITE":
        return TropicalScalar._from_kernel(
            semiring=left.semiring,
            kind=left.kind if left.kind != "FINITE" else right.kind,
            value=None,
        )
    left_value = _finite_value(left)
    right_value = _finite_value(right)
    _check_fraction_sum_growth(left_value, right_value)
    value = left_value.as_fraction() + right_value.as_fraction()
    return TropicalScalar._from_kernel(
        semiring=left.semiring,
        kind="FINITE",
        value=CanonicalRational.from_fraction(value),
    )


def tropical_scalar_power(scalar: TropicalScalar, exponent: int) -> TropicalScalar:
    if not isinstance(scalar, TropicalScalar):
        raise OperationDomainValidationError(
            location=("scalar",),
            code="tropical.scalar_type",
            message="expected a tropical scalar",
        )
    if type(exponent) is not int or exponent < 0 or exponent > 256:
        raise OperationDomainValidationError(
            location=("exponent",),
            code="tropical.exponent",
            message="exponent must be between 0 and 256",
        )
    _admit_scalar(scalar, scalar.semiring)
    if exponent == 0:
        return TropicalScalar._from_kernel(
            semiring=scalar.semiring,
            kind="FINITE",
            value=CanonicalRational.from_integer_ratio(0, 1),
        )
    if scalar.kind != "FINITE":
        return TropicalScalar._from_kernel(
            semiring=scalar.semiring, kind=scalar.kind, value=None
        )
    value = _finite_value(scalar)
    _check_scale_growth(value, exponent)
    return TropicalScalar._from_kernel(
        semiring=scalar.semiring,
        kind="FINITE",
        value=CanonicalRational.from_fraction(value.as_fraction() * exponent),
    )


def tropical_vector_add(left: TropicalVector, right: TropicalVector) -> TropicalVector:
    _admit_vector(left)
    _admit_vector(right)
    if left.semiring != right.semiring or left.axis != right.axis:
        raise OperationDomainValidationError(
            location=("right",),
            code="tropical.vector_mismatch",
            message="vectors must share semiring and axis",
        )
    return TropicalVector(
        semiring=left.semiring,
        axis=left.axis,
        entries=tuple(
            tropical_scalar_add(left.semiring, a, b)[0]
            for a, b in zip(left.entries, right.entries, strict=True)
        ),
    )


def tropical_vector_scale(
    scalar: TropicalScalar, vector: TropicalVector
) -> TropicalVector:
    if not isinstance(vector, TropicalVector):
        raise OperationDomainValidationError(
            location=("vector",),
            code="tropical.vector_type",
            message="expected a tropical vector",
        )
    _admit_scalar(scalar, vector.semiring)
    _admit_vector(vector)
    if scalar.semiring != vector.semiring:
        raise OperationDomainValidationError(
            location=("scalar",),
            code="tropical.semiring_mismatch",
            message="scalar and vector must share semiring",
        )
    for entry in vector.entries:
        _preflight_scalar_product(scalar, entry)
    return TropicalVector(
        semiring=vector.semiring,
        axis=vector.axis,
        entries=tuple(tropical_scalar_multiply(scalar, e) for e in vector.entries),
    )


def _poly_terms(poly: TropicalPolynomial) -> dict[tuple[int, ...], TropicalScalar]:
    return {term.exponents: term.coefficient for term in poly.terms}


def _poly(
    semiring: TropicalSemiring,
    variables: tuple[str, ...],
    terms: dict[tuple[int, ...], TropicalScalar],
) -> TropicalPolynomial:
    terms = {exp: s for exp, s in terms.items() if s.kind == "FINITE"}
    return TropicalPolynomial(
        semiring=semiring,
        variables=variables,
        terms=tuple(
            TropicalPolynomialTerm(exponents=e, coefficient=s)
            for e, s in sorted(terms.items())
        ),
    )


def tropical_polynomial_add(
    left: TropicalPolynomial, right: TropicalPolynomial
) -> TropicalPolynomial:
    _admit_polynomial(left)
    _admit_polynomial(right)
    if left.semiring != right.semiring or left.variables != right.variables:
        raise OperationDomainValidationError(
            location=("right",),
            code="tropical.polynomial_mismatch",
            message="polynomials must share parent",
        )
    if (
        len(set(_poly_terms(left)) | set(_poly_terms(right)))
        > MAX_TROPICAL_POLYNOMIAL_TERMS
    ):
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="tropical.polynomial_terms",
            message="sum has too many terms",
        )
    out = _poly_terms(left)
    for exp, value in _poly_terms(right).items():
        out[exp] = (
            tropical_scalar_add(left.semiring, out[exp], value)[0]
            if exp in out
            else value
        )
    return _poly(left.semiring, left.variables, out)


def tropical_polynomial_multiply(
    left: TropicalPolynomial, right: TropicalPolynomial
) -> TropicalPolynomial:
    _admit_polynomial(left)
    _admit_polynomial(right)
    if left.semiring != right.semiring or left.variables != right.variables:
        raise OperationDomainValidationError(
            location=("right",),
            code="tropical.polynomial_mismatch",
            message="polynomials must share parent",
        )
    # Establish term and exponent admission before any coefficient convolution.
    left_terms, right_terms = _poly_terms(left), _poly_terms(right)
    predicted_exponents = set()
    for a in left_terms:
        for b in right_terms:
            exp = tuple(x + y for x, y in zip(a, b, strict=True))
            if any(value > MAX_TROPICAL_EXPONENT for value in exp):
                raise OperationResourceAdmissionError(
                    location=("left", "right"),
                    code="tropical.exponent_output_bound",
                    message="polynomial product exponent exceeds the admitted bound",
                )
            predicted_exponents.add(exp)
    if len(predicted_exponents) > MAX_TROPICAL_POLYNOMIAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="tropical.polynomial_terms",
            message="product has too many terms",
        )
    for first in left_terms.values():
        for second in right_terms.values():
            _preflight_scalar_product(first, second)
    out: dict[tuple[int, ...], TropicalScalar] = {}
    for a, ca in left_terms.items():
        for b, cb in _poly_terms(right).items():
            exp = tuple(x + y for x, y in zip(a, b, strict=True))
            value = tropical_scalar_multiply(ca, cb)
            out[exp] = (
                tropical_scalar_add(left.semiring, out[exp], value)[0]
                if exp in out
                else value
            )
    if len(out) > MAX_TROPICAL_POLYNOMIAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="tropical.polynomial_terms",
            message="product has too many terms",
        )
    return _poly(left.semiring, left.variables, out)


def tropical_polynomial_evaluate(
    poly: TropicalPolynomial, point: TropicalVector
) -> tuple[TropicalScalar, tuple[tuple[int, ...], ...]]:
    _admit_polynomial(poly)
    _admit_vector(point)
    if poly.semiring != point.semiring or poly.variables != point.axis:
        raise OperationDomainValidationError(
            location=("point",),
            code="tropical.polynomial_point_mismatch",
            message="point must share polynomial parent",
        )
    values = []
    for term in poly.terms:
        if any(
            point.entries[i].kind != "FINITE" and exp
            for i, exp in enumerate(term.exponents)
        ):
            continue
        total = term.coefficient
        for exp, coord in zip(term.exponents, point.entries, strict=True):
            if exp:
                total = tropical_scalar_multiply(
                    total,
                    TropicalScalar._from_kernel(
                        semiring=poly.semiring,
                        kind="FINITE",
                        value=CanonicalRational.from_fraction(
                            _scale_fraction_checked(
                                _finite_value(coord).as_fraction(), exp
                            )
                        ),
                    ),
                )
        values.append((term.exponents, total))
    if not values:
        return TropicalScalar._from_kernel(
            semiring=poly.semiring,
            kind="POSITIVE_INFINITY"
            if poly.semiring.convention == "MIN_PLUS"
            else "NEGATIVE_INFINITY",
            value=None,
        ), ()
    result = values[0][1]
    for _, value in values[1:]:
        result = tropical_scalar_add(poly.semiring, result, value)[0]
    active = tuple(exp for exp, value in values if value == result)
    return result, active


def tropical_matrix_multiply(
    left: TropicalMatrix, right: TropicalMatrix
) -> TropicalMatrix:
    _admit_matrix(left)
    _admit_matrix(right)
    if left.semiring != right.semiring or left.column_axis != right.row_axis:
        raise OperationDomainValidationError(
            location=("right",),
            code="tropical.matrix_mismatch",
            message="matrix inner axes must match",
        )
    for i in range(len(left.row_axis)):
        for k in range(len(left.column_axis)):
            for j in range(len(right.column_axis)):
                _preflight_scalar_product(left.entries[i][k], right.entries[k][j])
    rows = []
    for i in range(len(left.row_axis)):
        row = []
        for j in range(len(right.column_axis)):
            terms = [
                tropical_scalar_multiply(left.entries[i][k], right.entries[k][j])
                for k in range(len(left.column_axis))
            ]
            value = (
                terms[0]
                if terms
                else TropicalScalar._from_kernel(
                    semiring=left.semiring,
                    kind="POSITIVE_INFINITY"
                    if left.semiring.convention == "MIN_PLUS"
                    else "NEGATIVE_INFINITY",
                    value=None,
                )
            )
            for term in terms[1:]:
                value = tropical_scalar_add(left.semiring, value, term)[0]
            row.append(value)
        rows.append(tuple(row))
    return TropicalMatrix(
        semiring=left.semiring,
        row_axis=left.row_axis,
        column_axis=right.column_axis,
        entries=tuple(rows),
    )


def tropical_matrix_power(matrix: TropicalMatrix, exponent: int) -> TropicalMatrix:
    _admit_matrix(matrix)
    if type(exponent) is not int:
        raise OperationDomainValidationError(
            location=("exponent",),
            code="tropical.exponent",
            message="exponent must be an integer",
        )
    if matrix.row_axis != matrix.column_axis or exponent < 0 or exponent > 64:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="tropical.matrix_power_domain",
            message="power requires a square matrix and exponent 0..64",
        )
    s = matrix.semiring
    n = len(matrix.row_axis)
    zero: Literal["POSITIVE_INFINITY", "NEGATIVE_INFINITY"] = (
        "POSITIVE_INFINITY" if s.convention == "MIN_PLUS" else "NEGATIVE_INFINITY"
    )
    result = TropicalMatrix(
        semiring=s,
        row_axis=matrix.row_axis,
        column_axis=matrix.column_axis,
        entries=tuple(
            tuple(
                TropicalScalar._from_kernel(
                    semiring=s,
                    kind="FINITE",
                    value=CanonicalRational.from_integer_ratio(0, 1),
                )
                if i == j
                else TropicalScalar._from_kernel(semiring=s, kind=zero, value=None)
                for j in range(n)
            )
            for i in range(n)
        ),
    )
    base = matrix
    for _ in range(exponent):
        result = tropical_matrix_multiply(result, base)
    return result


def tropical_matrix_finite_power_sum(
    matrix: TropicalMatrix, max_power: int
) -> tuple[TropicalMatrix, tuple[tuple[tuple[int, ...], ...], ...]]:
    _admit_matrix(matrix)
    if (
        matrix.row_axis != matrix.column_axis
        or type(max_power) is not int
        or max_power < 0
        or max_power > 32
    ):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="tropical.finite_power_sum_domain",
            message="finite power sum requires a square matrix and max_power 0..32",
        )
    powers = [tropical_matrix_power(matrix, 0)]
    for _ in range(max_power):
        powers.append(tropical_matrix_multiply(powers[-1], matrix))
    result = powers[0]
    winners: list[list[tuple[int, ...]]] = [
        [(0,) for _ in matrix.row_axis] for _ in matrix.row_axis
    ]
    for power, value in enumerate(powers[1:], 1):
        rows = []
        for i in range(len(matrix.row_axis)):
            row = []
            for j in range(len(matrix.column_axis)):
                chosen = tropical_scalar_add(
                    matrix.semiring, result.entries[i][j], value.entries[i][j]
                )[0]
                row.append(chosen)
                if chosen == value.entries[i][j] and chosen == result.entries[i][j]:
                    winners[i][j] = (*winners[i][j], power)
                elif chosen == value.entries[i][j]:
                    winners[i][j] = (power,)
            rows.append(tuple(row))
        result = TropicalMatrix(
            semiring=matrix.semiring,
            row_axis=matrix.row_axis,
            column_axis=matrix.column_axis,
            entries=tuple(rows),
        )
    return result, tuple(tuple(tuple(cell) for cell in row) for row in winners)


def tropical_assignment_profile(
    matrix: TropicalMatrix,
) -> tuple[TropicalScalar, tuple[tuple[int, ...], ...]]:
    _admit_matrix(matrix)
    if matrix.row_axis != matrix.column_axis:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="tropical.assignment_square",
            message="assignment requires a square matrix",
        )
    n = len(matrix.row_axis)
    if n > 8:
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="tropical.assignment_bound",
            message="assignment permutation envelope is eight rows",
        )
    scores = []
    for p in permutations(range(n)):
        selected = [matrix.entries[i][p[i]] for i in range(n)]
        if any(entry.kind != "FINITE" for entry in selected):
            value = None
        else:
            value = Fraction(0)
            for entry in selected:
                value = _sum_fractions_checked(
                    value,
                    _finite_value(entry).as_fraction(),
                )
        scores.append((p, value))
    finite = [(p, v) for p, v in scores if v is not None]
    if not finite:
        result = TropicalScalar._from_kernel(
            semiring=matrix.semiring,
            kind="POSITIVE_INFINITY"
            if matrix.semiring.convention == "MIN_PLUS"
            else "NEGATIVE_INFINITY",
            value=None,
        )
        # With no finite assignment, every permutation has the same
        # extended-tropical value. Preserve the complete tied witness rather
        # than silently discarding it.
        return result, tuple(p for p, _ in scores)
    optimum = (
        min(v for _, v in finite)
        if matrix.semiring.convention == "MIN_PLUS"
        else max(v for _, v in finite)
    )
    return TropicalScalar._from_kernel(
        semiring=matrix.semiring,
        kind="FINITE",
        value=CanonicalRational.from_fraction(optimum),
    ), tuple(p for p, v in finite if v == optimum)


__all__ = [
    "tropical_assignment_profile",
    "tropical_matrix_finite_power_sum",
    "tropical_matrix_multiply",
    "tropical_matrix_power",
    "tropical_polynomial_add",
    "tropical_polynomial_evaluate",
    "tropical_polynomial_multiply",
    "tropical_scalar_add",
    "tropical_scalar_multiply",
    "tropical_scalar_power",
    "tropical_vector_add",
    "tropical_vector_scale",
]
