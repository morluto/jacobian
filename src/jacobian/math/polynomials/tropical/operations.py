"""Exact native tropical arithmetic."""

from __future__ import annotations

import unicodedata
from fractions import Fraction
from itertools import pairwise, permutations
from typing import Literal

from jacobian._exact import CanonicalRational
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.tropical._models import (
    AddBranch,
    InfinityCase,
    PolynomialActiveTermsResult,
    ScalarDualResult,
    TropicalActiveTerm,
)
from jacobian.math.polynomials.tropical.values import (
    MAX_TROPICAL_ACTIVE_RESULT_BYTES,
    MAX_TROPICAL_ACTIVE_TERM_WORK,
    MAX_TROPICAL_EXPONENT,
    MAX_TROPICAL_NEWTON_RESULT_BYTES,
    MAX_TROPICAL_POLYNOMIAL_TERMS,
    MAX_TROPICAL_ROOT_CROSSOVER_PAIRS,
    MAX_TROPICAL_ROOT_DIGITS,
    MAX_TROPICAL_ROOT_RESULT_BYTES,
    MAX_TROPICAL_SCALAR_DIGITS,
    TropicalMatrix,
    TropicalNewtonPolygonEdge,
    TropicalNewtonPolygonProfile,
    TropicalNewtonPolygonVertex,
    TropicalPolynomial,
    TropicalPolynomialTerm,
    TropicalRootBreakpoint,
    TropicalRootInterval,
    TropicalScalar,
    TropicalSemiring,
    TropicalUnivariateRootProfile,
    TropicalVector,
    require_scalar_budget,
)

MAX_TROPICAL_SUBSTITUTION_PAIR_WORK = 750_000
MAX_TROPICAL_SUBSTITUTION_COORDINATE_WORK = 10_000_000


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
    if s.kind not in ("FINITE", "POSITIVE_INFINITY", "NEGATIVE_INFINITY"):
        raise OperationDomainValidationError(
            location=("scalar",),
            code="tropical.scalar_shape",
            message="scalar kind must be finite or a declared infinity",
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
    if (
        not isinstance(poly.variables, tuple)
        or len(poly.variables) > 128
        or any(
            not isinstance(label, str)
            or not label
            or label != label.strip()
            or len(label) > 64
            or any(
                unicodedata.category(character) in ("Cc", "Cs") for character in label
            )
            for label in poly.variables
        )
        or len(set(poly.variables)) != len(poly.variables)
    ):
        raise OperationDomainValidationError(
            location=("polynomial", "variables"),
            code="tropical.polynomial_axis",
            message="polynomial variable axis must be bounded, unique, and canonical",
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


def tropical_scalar_dual(scalar: TropicalScalar) -> ScalarDualResult:
    """Map a scalar between min-plus and max-plus by exact negation.

    Negation sends the licensed additive identity to the opposite licensed
    infinity and preserves the multiplicative identity. The result binds both
    semiring identities so callers cannot lose the change of parent.
    """
    if not isinstance(scalar, TropicalScalar):
        raise OperationDomainValidationError(
            location=("scalar",),
            code="tropical.scalar_type",
            message="expected a tropical scalar",
        )
    _admit_scalar(scalar, scalar.semiring)
    if scalar.semiring.convention not in (
        "MIN_PLUS",
        "MAX_PLUS",
    ) or scalar.semiring.base not in ("ZZ", "QQ"):
        raise OperationDomainValidationError(
            location=("scalar", "semiring"),
            code="tropical.semiring_value",
            message="scalar semiring has an unsupported convention or base",
        )
    target_convention = (
        "MAX_PLUS" if scalar.semiring.convention == "MIN_PLUS" else "MIN_PLUS"
    )
    target_semiring = TropicalSemiring(
        convention=target_convention,
        base=scalar.semiring.base,
    )
    if scalar.kind == "FINITE":
        value = _finite_value(scalar)
        result = TropicalScalar._from_kernel(
            semiring=target_semiring,
            kind="FINITE",
            value=CanonicalRational.from_integer_ratio(-value.num, value.den),
        )
    else:
        kind = (
            "NEGATIVE_INFINITY"
            if scalar.kind == "POSITIVE_INFINITY"
            else "POSITIVE_INFINITY"
        )
        result = TropicalScalar._from_kernel(
            semiring=target_semiring,
            kind=kind,
            value=None,
        )
    return ScalarDualResult._from_kernel(
        source=scalar,
        target_semiring=target_semiring,
        result=result,
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


def tropical_vector_projectivize(
    vector: TropicalVector,
) -> tuple[
    Literal["PROJECTIVIZED", "NO_PROJECTIVE_CLASS"],
    TropicalVector | None,
    TropicalScalar | None,
]:
    """Normalize finite support to min 0 (min-plus) or max 0 (max-plus).

    The returned translation is the scalar added to every coordinate. The
    all-infinity vector is the semiring zero and has no projective class.
    """
    _admit_vector(vector)
    finite = [
        entry.value.as_fraction()
        for entry in vector.entries
        if entry.kind == "FINITE" and entry.value is not None
    ]
    if not finite:
        return "NO_PROJECTIVE_CLASS", None, None
    pivot = min(finite) if vector.semiring.convention == "MIN_PLUS" else max(finite)
    shift = -pivot
    translation = TropicalScalar(
        semiring=vector.semiring,
        kind="FINITE",
        value=CanonicalRational.from_fraction(shift),
    )
    representative = tropical_vector_scale(translation, vector)
    return "PROJECTIVIZED", representative, translation


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


def _power_support(
    polynomial: TropicalPolynomial, exponent: int
) -> set[tuple[int, ...]]:
    """Admit exact binary support convolutions before coefficient arithmetic."""
    support: set[tuple[int, ...]] = {term.exponents for term in polynomial.terms}
    result_support: set[tuple[int, ...]] | None = None
    remaining = exponent
    total_pair_work = 0
    total_coordinate_work = 0

    def product_support(
        left: set[tuple[int, ...]], right: set[tuple[int, ...]]
    ) -> set[tuple[int, ...]]:
        nonlocal total_pair_work, total_coordinate_work
        pair_work = len(left) * len(right)
        total_pair_work += pair_work
        total_coordinate_work += pair_work * max(1, len(polynomial.variables))
        if total_pair_work > 750_000 or total_coordinate_work > 10_000_000:
            raise OperationResourceAdmissionError(
                location=("polynomial", "exponent"),
                code="tropical.polynomial_power_work_bound",
                message=(
                    "power exceeds the admitted total sparse convolution or "
                    "coordinate-addition work"
                ),
            )
        output: set[tuple[int, ...]] = set()
        for first in left:
            for second in right:
                summed = tuple(a + b for a, b in zip(first, second, strict=True))
                if any(value > MAX_TROPICAL_EXPONENT for value in summed):
                    raise OperationResourceAdmissionError(
                        location=("polynomial", "exponent"),
                        code="tropical.polynomial_power_exponent_bound",
                        message="a power term exceeds the admitted exponent bound",
                    )
                output.add(summed)
                if len(output) > MAX_TROPICAL_POLYNOMIAL_TERMS:
                    raise OperationResourceAdmissionError(
                        location=("polynomial", "terms"),
                        code="tropical.polynomial_power_term_bound",
                        message="an intermediate power exceeds the admitted term bound",
                    )
        return output

    while remaining:
        if remaining & 1:
            result_support = (
                set(support)
                if result_support is None
                else product_support(result_support, support)
            )
        remaining >>= 1
        if remaining:
            support = product_support(support, support)

    return result_support or set()


def _admit_power_coefficients_and_output(
    polynomial: TropicalPolynomial,
    exponent: int,
    result_support: set[tuple[int, ...]],
) -> None:
    # Repeated rational tropical multiplication adds coefficients. Bound each
    # numerator and denominator by the input digit envelope scaled by the
    # requested power before any exact coefficient arithmetic begins.
    max_input_digits = max(
        (
            max(
                _digits(term.coefficient.value.num), _digits(term.coefficient.value.den)
            )
            for term in polynomial.terms
            if term.coefficient.value is not None
        ),
        default=1,
    )
    coefficient_digit_bound = max_input_digits * exponent + _digits(exponent) + 1
    if coefficient_digit_bound > MAX_TROPICAL_SCALAR_DIGITS:
        raise OperationResourceAdmissionError(
            location=("polynomial", "terms", "coefficient"),
            code="tropical.polynomial_power_scalar_bound",
            message="power coefficient growth may exceed the scalar digit envelope",
        )

    output_count = len(result_support)
    variable_bytes = len(encode_strict_json(list(polynomial.variables)))
    estimated_output_bytes = (
        8_192
        + variable_bytes * 2
        + output_count
        * (128 + 6 * len(polynomial.variables) + 2 * coefficient_digit_bound)
    )
    if estimated_output_bytes > CanonicalLimits().max_output_bytes:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="tropical.polynomial_power_output_bound",
            message="power result may exceed the admitted serialized output size",
        )


def tropical_polynomial_power(
    polynomial: TropicalPolynomial, exponent: int
) -> TropicalPolynomial:
    """Return a bounded formal power, admitting all sparse products first."""
    if type(exponent) is not int or not 0 <= exponent <= 16:
        raise OperationDomainValidationError(
            location=("exponent",),
            code="tropical.polynomial_power_exponent",
            message="polynomial exponent must be an integer between 0 and 16",
        )
    _admit_polynomial(polynomial)

    if exponent == 0:
        zero = TropicalScalar._from_kernel(
            semiring=polynomial.semiring,
            kind="FINITE",
            value=CanonicalRational.from_integer_ratio(0, 1),
        )
        return _poly(
            polynomial.semiring,
            polynomial.variables,
            {tuple(0 for _ in polynomial.variables): zero},
        )

    result_support = _power_support(polynomial, exponent)
    _admit_power_coefficients_and_output(polynomial, exponent, result_support)

    if not polynomial.terms:
        return polynomial
    result: TropicalPolynomial | None = None
    base = polynomial
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = (
                base if result is None else tropical_polynomial_multiply(result, base)
            )
        remaining >>= 1
        if remaining:
            base = tropical_polynomial_multiply(base, base)
    if result is None:
        raise RuntimeError("positive polynomial power produced no result")
    return result


def _substitution_term_is_annihilated(
    exponents: tuple[int, ...], image_supports: list[set[tuple[int, ...]]]
) -> bool:
    return any(
        exponent > 0 and not image_support
        for exponent, image_support in zip(exponents, image_supports, strict=True)
    )


def _substitution_support(  # noqa: C901
    polynomial: TropicalPolynomial,
    images: tuple[TropicalPolynomial, ...],
    target_variables: tuple[str, ...],
) -> tuple[set[tuple[int, ...]], int]:
    """Preflight exact sparse support and all convolution work before coefficients."""
    zero = tuple(0 for _ in target_variables)
    pair_work = 0
    coordinate_work = 0

    def product_support(
        left: set[tuple[int, ...]], right: set[tuple[int, ...]]
    ) -> set[tuple[int, ...]]:
        nonlocal pair_work, coordinate_work
        pairs = len(left) * len(right)
        pair_work += pairs
        coordinate_work += pairs * max(1, len(target_variables))
        if (
            pair_work > MAX_TROPICAL_SUBSTITUTION_PAIR_WORK
            or coordinate_work > MAX_TROPICAL_SUBSTITUTION_COORDINATE_WORK
        ):
            raise OperationResourceAdmissionError(
                location=("polynomial", "images"),
                code="tropical.substitution_work_bound",
                message="substitution exceeds its sparse convolution work envelope",
            )
        output: set[tuple[int, ...]] = set()
        for first in left:
            for second in right:
                exponent = tuple(a + b for a, b in zip(first, second, strict=True))
                if any(value > MAX_TROPICAL_EXPONENT for value in exponent):
                    raise OperationResourceAdmissionError(
                        location=("polynomial", "images"),
                        code="tropical.substitution_exponent_bound",
                        message="a substituted exponent exceeds the admitted bound",
                    )
                output.add(exponent)
                if len(output) > MAX_TROPICAL_POLYNOMIAL_TERMS:
                    raise OperationResourceAdmissionError(
                        location=("polynomial", "images"),
                        code="tropical.substitution_term_bound",
                        message="an intermediate substituted polynomial exceeds the term bound",
                    )
        return output

    image_supports = [{term.exponents for term in image.terms} for image in images]
    result_support: set[tuple[int, ...]] = set()
    for term in polynomial.terms:
        if _substitution_term_is_annihilated(term.exponents, image_supports):
            continue
        monomial_support = {zero}
        for exponent, image_support in zip(term.exponents, image_supports, strict=True):
            if exponent == 0:
                continue
            if not image_support:
                monomial_support.clear()
                break
            power_support = {zero}
            base_support = image_support
            remaining = exponent
            while remaining:
                if remaining & 1:
                    power_support = product_support(power_support, base_support)
                remaining >>= 1
                if remaining:
                    base_support = product_support(base_support, base_support)
            monomial_support = product_support(monomial_support, power_support)
        result_support.update(monomial_support)
        if len(result_support) > MAX_TROPICAL_POLYNOMIAL_TERMS:
            raise OperationResourceAdmissionError(
                location=("polynomial",),
                code="tropical.substitution_term_bound",
                message="substitution result exceeds the polynomial term bound",
            )
    return result_support, pair_work


def _multiply_substitution_coefficients(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
    *,
    semiring: TropicalSemiring,
) -> dict[tuple[int, ...], Fraction]:
    output: dict[tuple[int, ...], Fraction] = {}
    for left_exponent, left_coefficient in left.items():
        for right_exponent, right_coefficient in right.items():
            exponent = tuple(
                a + b for a, b in zip(left_exponent, right_exponent, strict=True)
            )
            coefficient = _sum_fractions_checked(left_coefficient, right_coefficient)
            if exponent in output:
                current = output[exponent]
                coefficient = (
                    min(current, coefficient)
                    if semiring.convention == "MIN_PLUS"
                    else max(current, coefficient)
                )
            output[exponent] = coefficient
    return output


def _validate_substitution_inputs(
    polynomial: TropicalPolynomial,
    target_variables: tuple[str, ...],
    images: tuple[TropicalPolynomial, ...],
) -> None:
    _admit_polynomial(polynomial)
    if not isinstance(target_variables, tuple) or len(target_variables) > 128:
        raise OperationDomainValidationError(
            location=("target_variables",),
            code="tropical.substitution_target_axis",
            message="target variable axis must be a bounded tuple",
        )
    if any(
        not isinstance(label, str)
        or not label
        or label != label.strip()
        or len(label) > 64
        or any(unicodedata.category(character) in ("Cc", "Cs") for character in label)
        for label in target_variables
    ) or len(set(target_variables)) != len(target_variables):
        raise OperationDomainValidationError(
            location=("target_variables",),
            code="tropical.substitution_target_axis",
            message="target labels must be unique canonical opaque labels",
        )
    if not isinstance(images, tuple) or len(images) != len(polynomial.variables):
        raise OperationDomainValidationError(
            location=("images",),
            code="tropical.substitution_arity",
            message="images must contain one polynomial per source variable",
        )
    for image in images:
        _admit_polynomial(image)
        if image.semiring != polynomial.semiring or image.variables != target_variables:
            raise OperationDomainValidationError(
                location=("images",),
                code="tropical.substitution_image_parent",
                message="images must share the source semiring and target variable axis",
            )


def _admit_substitution_output(
    polynomial: TropicalPolynomial,
    target_variables: tuple[str, ...],
    images: tuple[TropicalPolynomial, ...],
    support: set[tuple[int, ...]],
) -> None:
    max_coefficient_digits = 1
    for term in polynomial.terms:
        if term.coefficient.value is None:
            continue
        digit_bound = max(
            _digits(term.coefficient.value.num), _digits(term.coefficient.value.den)
        )
        scalar_additions = 0
        for exponent, image in zip(term.exponents, images, strict=True):
            if exponent == 0:
                continue
            if not image.terms:
                # This source monomial vanishes; later images incur no arithmetic.
                digit_bound = 0
                break
            image_digits = max(
                max(
                    _digits(item.coefficient.value.num),
                    _digits(item.coefficient.value.den),
                )
                for item in image.terms
                if item.coefficient.value is not None
            )
            digit_bound += exponent * image_digits
            scalar_additions += exponent
        digit_bound += scalar_additions
        max_coefficient_digits = max(max_coefficient_digits, digit_bound)
    if max_coefficient_digits > MAX_TROPICAL_SCALAR_DIGITS:
        raise OperationResourceAdmissionError(
            location=("polynomial", "images", "coefficient"),
            code="tropical.substitution_scalar_bound",
            message="substitution coefficient growth may exceed the scalar digit bound",
        )
    estimated_output_bytes = (
        8_192
        + len(encode_strict_json(list(target_variables)))
        + len(support) * (128 + 6 * len(target_variables) + 2 * max_coefficient_digits)
    )
    if estimated_output_bytes > CanonicalLimits().max_output_bytes:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="tropical.substitution_output_bound",
            message="substitution result may exceed the canonical output limit",
        )


def _coefficient_power(
    image: TropicalPolynomial, exponent: int
) -> dict[tuple[int, ...], Fraction]:
    image_terms = {
        term.exponents: _finite_value(term.coefficient).as_fraction()
        for term in image.terms
    }
    result: dict[tuple[int, ...], Fraction] | None = None
    base = image_terms
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = (
                base
                if result is None
                else _multiply_substitution_coefficients(
                    result, base, semiring=image.semiring
                )
            )
        remaining >>= 1
        if remaining:
            base = _multiply_substitution_coefficients(
                base, base, semiring=image.semiring
            )
    if result is None:
        raise RuntimeError("positive substitution exponent had no power")
    return result


def tropical_polynomial_substitute(
    polynomial: TropicalPolynomial,
    target_variables: tuple[str, ...],
    images: tuple[TropicalPolynomial, ...],
) -> TropicalPolynomial:
    """Apply a simultaneous map of source variables to sparse polynomials."""
    _validate_substitution_inputs(polynomial, target_variables, images)
    support, _ = _substitution_support(polynomial, images, target_variables)
    _admit_substitution_output(polynomial, target_variables, images, support)
    zero = tuple(0 for _ in target_variables)
    output: dict[tuple[int, ...], Fraction] = {}
    for source_term in polynomial.terms:
        # An empty factor annihilates the whole tropical product. Check the
        # complete support before expanding any preceding coefficient powers.
        if any(
            exponent > 0 and not image.terms
            for exponent, image in zip(source_term.exponents, images, strict=True)
        ):
            continue
        current = {zero: _finite_value(source_term.coefficient).as_fraction()}
        for exponent, image in zip(source_term.exponents, images, strict=True):
            if exponent == 0:
                continue
            if not image.terms:
                current = {}
                break
            power = _coefficient_power(image, exponent)
            current = _multiply_substitution_coefficients(
                current, power, semiring=polynomial.semiring
            )
        for output_exponents, coefficient in current.items():
            if output_exponents in output:
                coefficient = (
                    min(output[output_exponents], coefficient)
                    if polynomial.semiring.convention == "MIN_PLUS"
                    else max(output[output_exponents], coefficient)
                )
            output[output_exponents] = coefficient

    terms = tuple(
        TropicalPolynomialTerm(
            exponents=exponent,
            coefficient=TropicalScalar._from_kernel(
                semiring=polynomial.semiring,
                kind="FINITE",
                value=CanonicalRational.from_fraction(coefficient),
            ),
        )
        for exponent, coefficient in sorted(output.items())
    )
    return TropicalPolynomial(
        semiring=polynomial.semiring,
        variables=target_variables,
        terms=terms,
    )


def _polynomial_point_plan(
    poly: TropicalPolynomial, point: TropicalVector, *, witness_output: bool
) -> tuple[int | None, ...]:
    _admit_polynomial(poly)
    _admit_vector(point)
    if poly.semiring != point.semiring or poly.variables != point.axis:
        raise OperationDomainValidationError(
            location=("point",),
            code="tropical.polynomial_point_mismatch",
            message="point must share polynomial parent",
        )
    coefficient_bounds: list[int | None] = []
    work = 0
    for term in poly.terms:
        if any(
            coord.kind != "FINITE" and exponent
            for exponent, coord in zip(term.exponents, point.entries, strict=True)
        ):
            coefficient_bounds.append(None)
            continue
        coefficient = _finite_value(term.coefficient)
        numerator_digits = _digits(coefficient.num)
        denominator_digits = _digits(coefficient.den)
        work += numerator_digits * denominator_digits
        for exponent, coord in zip(term.exponents, point.entries, strict=True):
            if not exponent:
                continue
            coordinate = _finite_value(coord)
            scale_num_digits = (
                1
                if coordinate.num == 0
                else _digits(coordinate.num) + _digits(exponent)
            )
            scale_den_digits = _digits(coordinate.den)
            if max(scale_num_digits, scale_den_digits) > MAX_TROPICAL_SCALAR_DIGITS:
                raise OperationResourceAdmissionError(
                    location=("point",),
                    code="tropical.active_term_scalar_growth",
                    message=(
                        "a scaled point coordinate may exceed the exact tropical "
                        "scalar digit envelope"
                    ),
                )
            work += scale_num_digits * _digits(exponent) + scale_den_digits
            previous_numerator_digits = numerator_digits
            previous_denominator_digits = denominator_digits
            numerator_digits = (
                max(
                    numerator_digits + scale_den_digits,
                    scale_num_digits + denominator_digits,
                )
                + 1
            )
            denominator_digits += scale_den_digits
            work += (
                previous_numerator_digits * scale_den_digits
                + scale_num_digits * previous_denominator_digits
                + previous_denominator_digits * scale_den_digits
                + 2 * numerator_digits * denominator_digits
            )
            if max(numerator_digits, denominator_digits) > MAX_TROPICAL_SCALAR_DIGITS:
                raise OperationResourceAdmissionError(
                    location=("polynomial", "terms"),
                    code="tropical.active_term_scalar_growth",
                    message=(
                        "a monomial evaluation may exceed the exact tropical "
                        "scalar digit envelope"
                    ),
                )
            work += max(numerator_digits, denominator_digits)
        coefficient_bounds.append(max(numerator_digits, denominator_digits))
    if work > MAX_TROPICAL_ACTIVE_TERM_WORK:
        raise OperationResourceAdmissionError(
            location=("polynomial", "terms"),
            code="tropical.active_term_work",
            message="aggregate tropical monomial evaluation work exceeds its bound",
        )

    def scalar_bytes(scalar: TropicalScalar) -> int:
        if scalar.value is None:
            return 160
        return _digits(scalar.value.num) + _digits(scalar.value.den) + 160

    def term_bytes(term: TropicalPolynomialTerm) -> int:
        return (
            64
            + sum(_digits(exponent) + 2 for exponent in term.exponents)
            + scalar_bytes(term.coefficient)
        )

    source_bytes = 160 + sum(16 * len(label) + 8 for label in poly.variables)
    source_bytes += sum(term_bytes(term) for term in poly.terms)
    point_bytes = 160 + sum(16 * len(label) + 8 for label in point.axis)
    point_bytes += sum(scalar_bytes(scalar) for scalar in point.entries)
    rows_bytes = sum(
        term_bytes(term) + 2 * (bound or 0) + 240
        for term, bound in zip(poly.terms, coefficient_bounds, strict=True)
        if bound is not None
    )
    maximum_value_bytes = max(
        (2 * bound + 160 for bound in coefficient_bounds if bound is not None),
        default=160,
    )
    if witness_output and (
        source_bytes + point_bytes + rows_bytes + maximum_value_bytes + 512
        > MAX_TROPICAL_ACTIVE_RESULT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="tropical.active_term_output_bytes",
            message="source-bound active-term result may exceed its output-byte bound",
        )
    return tuple(coefficient_bounds)


def _evaluate_polynomial_terms(
    poly: TropicalPolynomial, point: TropicalVector, *, witness_output: bool
) -> tuple[tuple[int, TropicalPolynomialTerm, TropicalScalar], ...]:
    _polynomial_point_plan(poly, point, witness_output=witness_output)
    values = []
    for index, term in enumerate(poly.terms):
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
        values.append((index, term, total))
    return tuple(values)


def _polynomial_extremum(
    poly: TropicalPolynomial,
    values: tuple[tuple[int, TropicalPolynomialTerm, TropicalScalar], ...],
) -> TropicalScalar:
    if not values:
        return TropicalScalar._from_kernel(
            semiring=poly.semiring,
            kind="POSITIVE_INFINITY"
            if poly.semiring.convention == "MIN_PLUS"
            else "NEGATIVE_INFINITY",
            value=None,
        )
    order_values = tuple(_order(value) for _, _, value in values)
    extremal = (
        min(value for value in order_values if value is not None)
        if poly.semiring.convention == "MIN_PLUS"
        else max(value for value in order_values if value is not None)
    )
    return next(value for _, _, value in values if _order(value) == extremal)


def tropical_polynomial_evaluate(
    poly: TropicalPolynomial, point: TropicalVector
) -> tuple[TropicalScalar, tuple[tuple[int, ...], ...]]:
    values = _evaluate_polynomial_terms(poly, point, witness_output=False)
    result = _polynomial_extremum(poly, values)
    active = tuple(term.exponents for _, term, value in values if value == result)
    return result, active


def tropical_polynomial_active_terms(
    poly: TropicalPolynomial, point: TropicalVector
) -> PolynomialActiveTermsResult:
    """Return every source-indexed monomial attaining the exact tropical extremum."""
    values = _evaluate_polynomial_terms(poly, point, witness_output=True)
    result = _polynomial_extremum(poly, values)
    active = tuple(
        TropicalActiveTerm(index=index, term=term, value=value)
        for index, term, value in values
        if value == result
    )
    return PolynomialActiveTermsResult.model_construct(
        polynomial=poly,
        point=point,
        value=result,
        active_terms=active,
    )


def tropical_polynomial_univariate_roots(
    poly: TropicalPolynomial,
) -> TropicalUnivariateRootProfile:
    """Return all exact finite breakpoints of a univariate tropical polynomial.

    Coefficients are line intercepts and exponents are slopes. A min-plus
    root is a corner of the lower envelope; max-plus uses the upper envelope.
    Root multiplicity is the absolute slope jump. The zero polynomial has no
    affine profile or finite roots; a nonzero constant or monomial has one
    unbounded interval and no finite roots.
    """
    _admit_polynomial(poly)
    if len(poly.variables) != 1:
        raise OperationDomainValidationError(
            location=("polynomial", "variables"),
            code="tropical.univariate_required",
            message="univariate root profiles require exactly one polynomial variable",
        )
    if not poly.terms:
        return TropicalUnivariateRootProfile(
            source=poly, kind="ZERO_POLYNOMIAL", intervals=(), roots=()
        )

    term_count = len(poly.terms)
    pair_count = term_count * (term_count - 1) // 2
    if pair_count > MAX_TROPICAL_ROOT_CROSSOVER_PAIRS:
        raise OperationResourceAdmissionError(
            location=("polynomial", "terms"),
            code="tropical.root_crossover_work",
            message="pairwise tropical root crossover work exceeds the admitted bound",
        )

    coefficient_digits = max(
        max(_digits(term.coefficient.value.num), _digits(term.coefficient.value.den))
        for term in poly.terms
        if term.coefficient.value is not None
    )
    # A line intersection subtracts two rationals, so numerator and
    # denominator products are bounded by twice the largest admitted input.
    # Reserve a few digits for subtraction and the (bounded) slope difference.
    root_digit_bound = 2 * coefficient_digits + 8
    if root_digit_bound > MAX_TROPICAL_ROOT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("polynomial", "terms", "coefficient"),
            code="tropical.root_digit_bound",
            message="worst-case exact root size exceeds the admitted root digit bound",
        )
    # The profile has at most n-1 roots; across all corners at most 2n terms
    # can be tied (each affine line meets the lower/upper envelope in at most
    # one interval or point). This conservative JSON-size estimate is checked
    # before any crossover arithmetic.
    estimated_result_bytes = (
        (8 * term_count + 4) * root_digit_bound + 1_024 * term_count + 2_048
    )
    if estimated_result_bytes > MAX_TROPICAL_ROOT_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="tropical.root_result_bytes",
            message="tropical root profile may exceed the admitted result-byte bound",
        )

    is_max = poly.semiring.convention == "MAX_PLUS"
    lines: list[tuple[int, Fraction, int]] = []
    for term in poly.terms:
        exponent = term.exponents[0]
        coefficient = _finite_value(term.coefficient).as_fraction()
        lines.append(
            (
                exponent,
                -coefficient if is_max else coefficient,
                -exponent if is_max else exponent,
            )
        )

    # Negating max-plus lines converts the requested upper envelope to a
    # lower envelope. Distinct exponents make transformed slopes unique.
    lines.sort(key=lambda line: line[2], reverse=True)
    hull: list[tuple[int, Fraction, int]] = []
    starts: list[Fraction | None] = []
    for line in lines:
        crossing: Fraction | None = None
        while hull:
            previous = hull[-1]
            crossing = (previous[1] - line[1]) / (line[2] - previous[2])
            if starts[-1] is None or crossing > starts[-1]:
                break
            hull.pop()
            starts.pop()
        if not hull:
            crossing = None
        hull.append(line)
        starts.append(crossing)

    root_values = [start for start in starts[1:] if start is not None]
    breakpoints: list[TropicalRootBreakpoint] = []
    for index, value in enumerate(root_values):
        left_exponent = hull[index][0]
        right_exponent = hull[index + 1][0]
        transformed_active = min(
            intercept + slope * value for _, intercept, slope in lines
        )
        active = tuple(
            sorted(
                exponent
                for exponent, intercept, slope in lines
                if intercept + slope * value == transformed_active
            )
        )
        root = CanonicalRational.from_fraction(value)
        root_digits = max(_digits(root.num), _digits(root.den))
        if root_digits > MAX_TROPICAL_ROOT_DIGITS:
            raise OperationResourceAdmissionError(
                location=("polynomial", "terms", "coefficient"),
                code="tropical.root_digit_bound",
                message="exact root exceeds the admitted root digit bound",
            )
        left_slope, right_slope = left_exponent, right_exponent
        breakpoints.append(
            TropicalRootBreakpoint(
                value=root,
                multiplicity=abs(right_slope - left_slope),
                left_exponent=left_exponent,
                right_exponent=right_exponent,
                left_slope=left_slope,
                right_slope=right_slope,
                active_exponents=active,
            )
        )

    intervals: list[TropicalRootInterval] = []
    for index, line in enumerate(hull):
        lower = root_values[index - 1] if index > 0 else None
        upper = root_values[index] if index < len(root_values) else None
        intervals.append(
            TropicalRootInterval(
                lower=CanonicalRational.from_fraction(lower)
                if lower is not None
                else None,
                upper=CanonicalRational.from_fraction(upper)
                if upper is not None
                else None,
                active_exponent=line[0],
                slope=line[0],
            )
        )
    return TropicalUnivariateRootProfile(
        source=poly,
        kind="FINITE_PROFILE",
        intervals=tuple(intervals),
        roots=tuple(breakpoints),
    )


def tropical_polynomial_univariate_split_form(
    poly: TropicalPolynomial,
) -> TropicalPolynomial:
    """Return a canonical consecutive-support polynomial with the same function.

    The finite tropical roots, repeated by slope-jump multiplicity, determine
    the coefficients of the split form.  For integer input, the result is
    promoted to QQ exactly when a rational root requires it.
    """
    _admit_polynomial(poly)
    if len(poly.variables) != 1:
        raise OperationDomainValidationError(
            location=("polynomial", "variables"),
            code="tropical.split_form_univariate",
            message="split form requires exactly one polynomial variable",
        )
    if not poly.terms:
        return TropicalPolynomial(
            semiring=poly.semiring, variables=poly.variables, terms=()
        )

    first_exponent = poly.terms[0].exponents[0]
    last_exponent = poly.terms[-1].exponents[0]
    span = last_exponent - first_exponent
    output_term_count = span + 1
    if output_term_count > MAX_TROPICAL_POLYNOMIAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("polynomial", "terms"),
            code="tropical.split_form_terms",
            message="the consecutive split form exceeds the polynomial term envelope",
        )

    # Root construction has its own pair, scalar-height, and output admission.
    # The support expansion bound above is checked first, before that work.
    profile = tropical_polynomial_univariate_roots(poly)
    expanded_roots = tuple(
        root.value.as_fraction()
        for root in profile.roots
        for _ in range(root.multiplicity)
    )
    if len(expanded_roots) != span:
        raise ArithmeticError("tropical root multiplicities do not span the support")

    # For min-plus, coefficient k is c_min minus the k largest roots.  For
    # max-plus it is c_min minus the k smallest roots.  This is the coefficient
    # formula for the tropical product of the corresponding linear factors.
    roots_for_coefficients = tuple(
        sorted(
            expanded_roots,
            reverse=poly.semiring.convention == "MIN_PLUS",
        )
    )
    first_coefficient = _finite_value(poly.terms[0].coefficient).as_fraction()
    coefficients = [first_coefficient]
    for root in roots_for_coefficients:
        left = CanonicalRational.from_fraction(coefficients[-1])
        right = CanonicalRational.from_fraction(-root)
        _check_fraction_sum_growth(left, right)
        coefficients.append(coefficients[-1] - root)

    last_coefficient = _finite_value(poly.terms[-1].coefficient).as_fraction()
    if coefficients[-1] != last_coefficient:
        raise ArithmeticError("tropical split form does not preserve the endpoint term")

    result_base = (
        poly.semiring.base
        if all(value.denominator == 1 for value in coefficients)
        else "QQ"
    )
    result_semiring = TropicalSemiring(
        convention=poly.semiring.convention, base=result_base
    )
    rational_coefficients = tuple(
        CanonicalRational.from_fraction(value) for value in coefficients
    )
    return TropicalPolynomial(
        semiring=result_semiring,
        variables=poly.variables,
        terms=tuple(
            TropicalPolynomialTerm(
                exponents=(first_exponent + index,),
                coefficient=TropicalScalar._from_kernel(
                    semiring=result_semiring,
                    kind="FINITE",
                    value=value,
                ),
            )
            for index, value in enumerate(rational_coefficients)
        ),
    )


def tropical_polynomial_univariate_newton_polygon(
    poly: TropicalPolynomial,
) -> TropicalNewtonPolygonProfile:
    """Return the exact lower/upper coefficient hull and source-face ledger."""
    _admit_polynomial(poly)
    if len(poly.variables) != 1:
        raise OperationDomainValidationError(
            location=("polynomial", "variables"),
            code="tropical.newton_polygon_univariate",
            message="Newton polygon profile requires one variable",
        )
    count = len(poly.terms)
    coefficient_digits = max(
        (
            max(
                _digits(term.coefficient.value.num),
                _digits(term.coefficient.value.den),
            )
            for term in poly.terms
            if term.coefficient.value is not None
        ),
        default=1,
    )
    intermediate_bound = 3 * count * (4 * coefficient_digits + 32)
    result_bound = count * (8 * coefficient_digits + 512)
    if max(intermediate_bound, result_bound) > MAX_TROPICAL_NEWTON_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="tropical.newton_polygon_output_budget",
            message="Newton polygon exact hull or output exceeds the admitted envelope",
        )

    points = [
        (
            term.exponents[0],
            _finite_value(term.coefficient).as_fraction(),
            index,
        )
        for index, term in enumerate(poly.terms)
    ]

    def cross(
        a: tuple[int, Fraction, int],
        b: tuple[int, Fraction, int],
        c: tuple[int, Fraction, int],
    ) -> Fraction:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower = poly.semiring.convention == "MIN_PLUS"
    hull: list[tuple[int, Fraction, int]] = []
    for point in points:
        while len(hull) >= 2:
            turn = cross(hull[-2], hull[-1], point)
            if (lower and turn <= 0) or (not lower and turn >= 0):
                hull.pop()
            else:
                break
        hull.append(point)

    vertices = tuple(
        TropicalNewtonPolygonVertex(
            source_term_index=index,
            exponent=exponent,
            coefficient=poly.terms[index].coefficient,
        )
        for exponent, _, index in hull
    )
    edges = []
    for edge_index, (left, right) in enumerate(pairwise(hull)):
        face_indices = tuple(
            source_index
            for source_index in range(left[2], right[2] + 1)
            if cross(left, right, points[source_index]) == 0
        )
        slope = (right[1] - left[1]) / (right[0] - left[0])
        edges.append(
            TropicalNewtonPolygonEdge(
                left_vertex_index=edge_index,
                right_vertex_index=edge_index + 1,
                source_term_indices=face_indices,
                slope=CanonicalRational.from_fraction(slope),
                tropical_root=CanonicalRational.from_fraction(-slope),
                multiplicity=right[0] - left[0],
            )
        )
    return TropicalNewtonPolygonProfile.model_construct(
        source=poly, hull_vertices=vertices, edges=tuple(edges)
    )


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
    if len(matrix.row_axis) != len(matrix.column_axis):
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
    "tropical_polynomial_active_terms",
    "tropical_polynomial_add",
    "tropical_polynomial_evaluate",
    "tropical_polynomial_multiply",
    "tropical_polynomial_power",
    "tropical_polynomial_substitute",
    "tropical_polynomial_univariate_newton_polygon",
    "tropical_polynomial_univariate_roots",
    "tropical_polynomial_univariate_split_form",
    "tropical_scalar_add",
    "tropical_scalar_dual",
    "tropical_scalar_multiply",
    "tropical_scalar_power",
    "tropical_vector_add",
    "tropical_vector_projectivize",
    "tropical_vector_scale",
]
