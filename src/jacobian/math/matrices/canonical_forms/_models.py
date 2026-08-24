"""Typed wire contracts for exact canonical-form operations over QQ."""

from __future__ import annotations

from collections.abc import Iterable
from math import gcd
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
)
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.matrices.values import (
    MAX_MATRIX_DIMENSION,
    RationalMatrix,
    require_matrix_scalar_digits,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
    require_polynomial_budget,
)

MAX_CANONICAL_FORM_DIMENSION = 16
MAX_CANONICAL_FORM_SCALAR_DIGITS = 256

# A Horner pass performs ``degree`` dense n-by-n products. Ordinary execution
# performs one producer pass and result construction performs one independent
# source-bound replay. This limit therefore admits at most 2,000,000 scalar
# product terms per pass and bounds every accepted exact call before SymPy.
MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS = 4_000_000
MATRIX_POLYNOMIAL_EVALUATION_PASSES = 2
# Couple exact operation count to the largest admitted rational component.
# Reference cases, including result replay: a dense 32x32 degree-61 request
# costs about 33 billion proxy units and 1.5 seconds; a 1x1 degree-327 request
# with a 32,701-digit denominator costs about 700 billion units and one second.
MAX_MATRIX_POLYNOMIAL_DIGIT_WORK = 1_000_000_000_000

# Admission may materialize bounded powers to cross-cancel factors before
# digit budgets are charged. Eight bits per canonical digit sits far above
# log2(10), so the estimate never undercounts a power below the ceiling.
_MAX_ADMISSION_POWER_BITS = 8 * MAX_CANONICAL_RATIONAL_DIGITS
# Total estimated bigint word-work one admission may spend on materialized
# powers and their reductions; exhaustion falls back to the dense capped
# bound. Admits several full-width cancellations per call while bounding
# repeated adversarial reduction work to milliseconds.
_ADMISSION_REDUCTION_BIT_BUDGET = 64 * _MAX_ADMISSION_POWER_BITS

_RESULT_ENTRY_OVERHEAD_BYTES: int = 32
_RESULT_ENVELOPE_RESERVE_BYTES: int = 4_096
_MAX_RESULT_COMPONENT: int = 10**MAX_CANONICAL_RATIONAL_DIGITS - 1


def _capped_multiply(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    if left > _MAX_RESULT_COMPONENT // right:
        return _MAX_RESULT_COMPONENT + 1
    return left * right


def _capped_add(left: int, right: int) -> int:
    if left > _MAX_RESULT_COMPONENT - right:
        return _MAX_RESULT_COMPONENT + 1
    return left + right


def _capped_power(base: int, exponent: int) -> int:
    result = 1
    factor = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _capped_multiply(result, factor)
            if result > _MAX_RESULT_COMPONENT:
                return result
        remaining >>= 1
        if remaining:
            factor = _capped_multiply(factor, factor)
            if factor > _MAX_RESULT_COMPONENT:
                return factor
    return result


def _capped_lcm(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        factor = value // gcd(result, value)
        result = _capped_multiply(result, factor)
        if result > _MAX_RESULT_COMPONENT:
            return result
    return result


def _cancelled_product(
    left_numerator: int,
    left_denominator: int,
    right_numerator: int,
    right_denominator: int,
) -> tuple[int, int]:
    """Return the reduced numerator and denominator of one product of ratios."""

    top = gcd(left_numerator, right_denominator)
    bottom = gcd(right_numerator, left_denominator)
    return (
        _capped_multiply(left_numerator // top, right_numerator // bottom),
        _capped_multiply(left_denominator // bottom, right_denominator // top),
    )


def _decimal_digit_upper_bound(value: int) -> int:
    """Return a conservative decimal digit count without converting huge ints."""

    if value == 0:
        return 1
    # 0.30103 is a strict upper bound for log10(2). Using integer arithmetic
    # keeps this bound deterministic and at most one digit above the truth.
    return (value.bit_length() * 30_103 + 99_999) // 100_000


def _polynomial_degree(polynomial: RationalPolynomial) -> int:
    return max(
        (term.exponents[0] for term in polynomial.polynomial.terms),
        default=0,
    )


def _mathematical_polynomial_degree(
    polynomial: RationalPolynomial,
) -> int | None:
    if not polynomial.polynomial.terms:
        return None
    return polynomial.polynomial.terms[0].exponents[0]


def _coefficient_ratios(
    polynomial: RationalPolynomial,
) -> dict[int, tuple[int, int]]:
    return {
        term.exponents[0]: term.coefficient.as_integer_ratio()
        for term in polynomial.polynomial.terms
    }


def _bounded_rational_sum(terms: Iterable[tuple[int, int]]) -> tuple[int, int]:
    active = tuple(
        (numerator, denominator) for numerator, denominator in terms if numerator
    )
    if not active:
        return 0, 1
    denominator = _capped_lcm(item[1] for item in active)
    if denominator > _MAX_RESULT_COMPONENT:
        return denominator, denominator
    numerator = 0
    for term_numerator, term_denominator in active:
        lifted = _capped_multiply(term_numerator, denominator // term_denominator)
        numerator = _capped_add(numerator, lifted)
        if numerator > _MAX_RESULT_COMPONENT:
            break
    return numerator, denominator


def _materialized_power(base: int, exponent: int) -> tuple[int | None, int]:
    """Return ``base ** exponent`` and its estimated bit size.

    The pair ``(None, bits)`` reports a power whose estimated size exceeds
    ``_MAX_ADMISSION_POWER_BITS``; callers fall back to capped arithmetic.
    """

    if exponent == 0 or base == 1:
        return 1, 0
    bits = exponent * base.bit_length()
    if bits > _MAX_ADMISSION_POWER_BITS:
        return None, bits
    return base**exponent, bits


def _reduced_general_result_bound(
    coefficient_ratios: tuple[tuple[int, int, int], ...],
    cleared_matrix_height: int,
    common_matrix_denominator: int,
    dimension: int,
    longest_walk: int | None,
) -> tuple[int, int] | None:
    """Bound live entries of ``f(A)`` with per-term cancelled fractions.

    Every structurally surviving term contributes the entry bound
    ``|c_k| n^(k-1) h^k / Q^k`` as an exactly reduced fraction: the shared
    factor ``gcd(h, Q)`` is divided out of the cleared height and clearing
    denominator up front (equal exponents make ``gcd(h^k, Q^k) = gcd(h,Q)^k``
    exact), and each term then cancels its coefficient, dimension-, and
    height-power factors against its explicit denominator. The fractions are
    summed with reduced rational arithmetic, so exact cancellation between
    coefficient factors and matrix-power factors is never rejected as growth.

    Returns ``None`` when a required power exceeds the materialization
    ceiling or the reduction budget is exhausted; callers must then fall
    back to the dense capped bound.
    """

    shared = gcd(cleared_matrix_height, common_matrix_denominator)
    height_reduced = cleared_matrix_height // shared
    denominator_reduced = common_matrix_denominator // shared
    budget = _ADMISSION_REDUCTION_BIT_BUDGET
    terms: list[tuple[int, int]] = []
    for exponent, numerator, denominator in coefficient_ratios:
        if numerator == 0:
            continue
        if exponent and longest_walk is not None and exponent > longest_walk:
            continue
        dimension_power, dimension_bits = (
            (1, 0) if exponent == 0 else _materialized_power(dimension, exponent - 1)
        )
        height_power, height_bits = (
            (1, 0) if exponent == 0 else _materialized_power(height_reduced, exponent)
        )
        clearing_power, clearing_bits = (
            (1, 0)
            if exponent == 0
            else _materialized_power(denominator_reduced, exponent)
        )
        if dimension_power is None or height_power is None or clearing_power is None:
            return None
        budget -= dimension_bits + height_bits + clearing_bits
        denominator_total = denominator * clearing_power
        term_numerator = 1
        for factor in (abs(numerator), dimension_power, height_power):
            # A Lehmer gcd costs on the order of (bits / 64) squared word
            # operations; charging a quarter of that square keeps the budget
            # an upper estimate of the reduction work actually performed.
            budget -= (
                min(factor.bit_length(), denominator_total.bit_length()) ** 2 // 4096
            )
            if budget < 0:
                return None
            common = gcd(factor, denominator_total)
            denominator_total //= common
            term_numerator = _capped_multiply(term_numerator, factor // common)
        terms.append((term_numerator, denominator_total))
    return _bounded_rational_sum(terms)


def _acyclic_support_longest_walk(
    ratios: tuple[tuple[int, int], ...],
    dimension: int,
) -> int | None:
    """Return the longest walk length in the entry-support digraph of ``M``.

    Entry ``(i, j)`` of ``M^k`` is nonzero only when the digraph of nonzero
    matrix entries carries a walk of length ``k`` from ``i`` to ``j``, so
    every power longer than the returned bound is identically zero. A support
    cycle keeps arbitrarily long walks alive and yields ``None``.
    """

    indegree = [0] * dimension
    longest = [0] * dimension
    for row_index in range(dimension):
        row_offset = row_index * dimension
        for column_index in range(dimension):
            if ratios[row_offset + column_index][0] != 0:
                indegree[column_index] += 1
    queue = [node for node in range(dimension) if indegree[node] == 0]
    processed = 0
    while processed < len(queue):
        node = queue[processed]
        processed += 1
        row_offset = node * dimension
        for column_index in range(dimension):
            if ratios[row_offset + column_index][0] != 0:
                longest[column_index] = max(longest[column_index], longest[node] + 1)
                indegree[column_index] -= 1
                if indegree[column_index] == 0:
                    queue.append(column_index)
    if processed < dimension:
        return None
    return max(longest)


def _linear_result_component_bounds(
    matrix: RationalMatrix,
    coefficients: dict[int, tuple[int, int]],
) -> tuple[tuple[tuple[int, int], ...], int]:
    """Bound every entry of ``c_1 A + c_0 I`` independently.

    Products are reduced before digit budgets are charged so exact
    cancellations between coefficients and matrix entries are not rejected as
    growth; the SymPy kernel keeps every intermediate reduced as well.
    """

    linear_numerator, linear_denominator = coefficients.get(1, (0, 1))
    constant_numerator, constant_denominator = coefficients.get(0, (0, 1))
    bounds: list[tuple[int, int]] = []
    for row_index, row in enumerate(matrix.entries):
        for column_index, entry in enumerate(row):
            entry_numerator, entry_denominator = entry.as_integer_ratio()
            product_numerator, product_denominator = _cancelled_product(
                abs(linear_numerator),
                linear_denominator,
                abs(entry_numerator),
                entry_denominator,
            )
            terms = [(product_numerator, product_denominator)]
            if row_index == column_index:
                terms.append((abs(constant_numerator), constant_denominator))
            numerator, denominator = _bounded_rational_sum(terms)
            bounds.append(
                (
                    _decimal_digit_upper_bound(numerator),
                    _decimal_digit_upper_bound(denominator),
                )
            )
    return (
        tuple(bounds),
        max(max(numerator, denominator) for numerator, denominator in bounds),
    )


def _general_result_component_bounds(
    matrix: RationalMatrix,
    polynomial: RationalPolynomial,
    degree: int,
) -> tuple[tuple[tuple[int, int], ...], int]:
    """Bound ``f(A)`` after clearing one exact global denominator.

    If ``Q`` clears every entry denominator and ``M = Q A``, then with ``V``
    clearing the polynomial coefficients, ``V Q^d f(A)`` is the integer matrix

    ``sum_k w_k Q^(d-k) M^k``.

    Entry ``(i, j)`` of ``M^k`` is nonzero only when the support digraph of
    the cleared integer entries carries a walk of length ``k`` from ``i`` to
    ``j``, so an acyclic support digraph with longest walk ``L`` makes every
    exponent beyond ``L`` identically zero; structurally vanishing powers are
    dropped from the result bound. When no nonconstant power survives,
    ``f(A) = c(0) I`` reduces exactly to the constant coefficient and the
    ``Q^d`` denominator factor is dropped with them.

    The surviving-powers result bound charges reduced factors rather than
    unreduced products: each live term contributes ``|c_k| n^(k-1) h^k / Q^k``
    as an exactly reduced fraction (see
    ``_reduced_general_result_bound``), so cancellations between coefficient
    factors and matrix-power factors are not rejected as growth. When a
    required power exceeds the admission materialization ceiling, the dense
    capped entry bound takes over: for ``h = max |M_ij|``, each entry of
    ``M^k`` is bounded by ``n^(k-1) h^k``, which reduction can only shrink.
    The zero-matrix case is handled separately because its intermediates are
    individual input coefficients.

    The dense sum over every surviving exponent is retained as a second,
    never-structural work bound: every Horner intermediate equals one of its
    sub-sums over surviving powers, since structurally vanishing powers
    contribute identically zero matrices, with denominator at most ``V Q^d``,
    so its digit size drives the coupled digit-work estimate independently of
    how small the exact result is.
    """

    matrix_ratios = tuple(
        entry.as_integer_ratio() for row in matrix.entries for entry in row
    )
    dimension = len(matrix.entries)
    longest_walk = _acyclic_support_longest_walk(matrix_ratios, dimension)
    common_matrix_denominator = _capped_lcm(
        denominator for _numerator, denominator in matrix_ratios
    )
    coefficient_ratios = tuple(
        (
            term.exponents[0],
            *term.coefficient.as_integer_ratio(),
        )
        for term in polynomial.polynomial.terms
    )
    common_coefficient_denominator = _capped_lcm(
        denominator for _exponent, _numerator, denominator in coefficient_ratios
    )
    if (
        common_matrix_denominator > _MAX_RESULT_COMPONENT
        or common_coefficient_denominator > _MAX_RESULT_COMPONENT
    ):
        raise ValueError(
            "matrix polynomial denominator growth exceeds the canonical "
            f"{MAX_CANONICAL_RATIONAL_DIGITS:,}-digit result bound"
        )

    work_denominator_bound = _capped_multiply(
        common_coefficient_denominator,
        _capped_power(common_matrix_denominator, degree),
    )
    # |numerator| * (Q // denominator) is at most the square of two canonical
    # components, so the exact cleared height is always safe to materialize.
    cleared_matrix_height = max(
        abs(numerator) * (common_matrix_denominator // denominator)
        for numerator, denominator in matrix_ratios
    )
    numerator_bound = 0
    work_numerator_bound = 0
    live_nonconstant_power = False
    for exponent, numerator, denominator in coefficient_ratios:
        coefficient_height = _capped_multiply(
            abs(numerator), common_coefficient_denominator // denominator
        )
        matrix_power_height = 1
        if exponent:
            matrix_power_height = _capped_multiply(
                _capped_power(cleared_matrix_height, exponent),
                _capped_power(dimension, exponent - 1),
            )
        term_height = _capped_multiply(
            coefficient_height,
            _capped_power(common_matrix_denominator, degree - exponent),
        )
        term_height = _capped_multiply(term_height, matrix_power_height)
        if exponent and longest_walk is not None and exponent > longest_walk:
            # Structurally vanishing powers are identically zero, so neither
            # the exact result nor any Horner intermediate contains them.
            continue
        if exponent:
            live_nonconstant_power = True
        work_numerator_bound = _capped_add(work_numerator_bound, term_height)
        numerator_bound = _capped_add(numerator_bound, term_height)

    if not live_nonconstant_power:
        # Every nonconstant power vanishes structurally, so f(A) reduces to
        # c(0) I exactly; no Q^d clearing factor is needed.
        numerator_bound, denominator_bound = next(
            (
                (abs(numerator), denominator)
                for exponent, numerator, denominator in coefficient_ratios
                if exponent == 0
            ),
            (0, 1),
        )
    else:
        reduced_bound = _reduced_general_result_bound(
            coefficient_ratios,
            cleared_matrix_height,
            common_matrix_denominator,
            dimension,
            longest_walk,
        )
        if reduced_bound is not None:
            numerator_bound, denominator_bound = reduced_bound
        else:
            denominator_bound = work_denominator_bound

    maximum_work_digits = max(
        _decimal_digit_upper_bound(work_numerator_bound),
        _decimal_digit_upper_bound(work_denominator_bound),
    )
    if numerator_bound == 0:
        component_bound = (1, 1)
    else:
        if (
            numerator_bound > _MAX_RESULT_COMPONENT
            or denominator_bound > _MAX_RESULT_COMPONENT
        ):
            raise ValueError(
                "matrix polynomial coefficient growth exceeds the canonical "
                f"{MAX_CANONICAL_RATIONAL_DIGITS:,}-digit result bound"
            )
        component_bound = (
            _decimal_digit_upper_bound(numerator_bound),
            _decimal_digit_upper_bound(denominator_bound),
        )
    return (
        (component_bound,) * (dimension * dimension),
        maximum_work_digits,
    )


def _require_matrix_polynomial_output_budget(
    matrix: RationalMatrix,
    polynomial: RationalPolynomial,
    degree: int,
) -> int:
    coefficients = _coefficient_ratios(polynomial)
    matrix_is_zero = all(entry.num == "0" for row in matrix.entries for entry in row)
    if degree <= 1 or matrix_is_zero:
        component_bounds, maximum_arithmetic_digits = _linear_result_component_bounds(
            matrix, coefficients
        )
    else:
        component_bounds, maximum_arithmetic_digits = _general_result_component_bounds(
            matrix, polynomial, degree
        )
    if any(
        numerator_digits > MAX_CANONICAL_RATIONAL_DIGITS
        or denominator_digits > MAX_CANONICAL_RATIONAL_DIGITS
        for numerator_digits, denominator_digits in component_bounds
    ):
        raise ValueError(
            "matrix polynomial evaluation can exceed the canonical "
            f"{MAX_CANONICAL_RATIONAL_DIGITS:,}-digit rational result bound"
        )

    source_bytes = len(encode_strict_json(matrix.model_dump(mode="json")))
    polynomial_bytes = len(encode_strict_json(polynomial.model_dump(mode="json")))
    value_bytes = sum(
        numerator_digits + denominator_digits + _RESULT_ENTRY_OVERHEAD_BYTES
        for numerator_digits, denominator_digits in component_bounds
    )
    estimated_result_bytes = (
        source_bytes + polynomial_bytes + value_bytes + _RESULT_ENVELOPE_RESERVE_BYTES
    )
    output_limit = CanonicalLimits().max_output_bytes
    if estimated_result_bytes > output_limit:
        raise ValueError(
            "the retained matrix, polynomial, and exact value can exceed the "
            f"{output_limit}-byte canonical output limit"
        )
    arithmetic_component_digits = [
        len(component.lstrip("-"))
        for row in matrix.entries
        for entry in row
        for component in (entry.num, entry.den)
    ]
    arithmetic_component_digits.extend(
        len(component.lstrip("-"))
        for term in polynomial.polynomial.terms
        for component in (term.coefficient.num, term.coefficient.den)
    )
    arithmetic_component_digits.append(maximum_arithmetic_digits)
    return max(arithmetic_component_digits)


class MatrixPolynomialEvaluationRequest(StrictModel):
    """Evaluate one exact univariate rational polynomial at a square matrix."""

    matrix: RationalMatrix = Field(
        description=(
            "Nonempty square matrix over QQ through order "
            f"{MAX_MATRIX_DIMENSION}. Matrix and polynomial coefficients "
            "share the exact rational field."
        )
    )
    polynomial: RationalPolynomial = Field(
        description=(
            "Sparse polynomial over QQ in exactly one declared variable; terms "
            "use the canonical descending exponent order of RationalPolynomial. "
            "Exact admission couples the ordinary degree to the matrix order: "
            f"{MATRIX_POLYNOMIAL_EVALUATION_PASSES} * degree * order^3 must "
            f"stay within {MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS:,} scalar "
            "products across both Horner passes, so the largest admitted "
            "ordinary degree at matrix order 32 is "
            f"{MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS // (MATRIX_POLYNOMIAL_EVALUATION_PASSES * MAX_MATRIX_DIMENSION**3)}. "
            "Exact admission additionally multiplies the total "
            "(2 * degree * order^3) scalar products by the square of the "
            "largest decimal-digit component among the matrix entries, the "
            "polynomial coefficients, and the predicted exact-result "
            "components; that digit-work product must stay within "
            f"{MAX_MATRIX_POLYNOMIAL_DIGIT_WORK:,} units."
        )
    )

    @model_validator(mode="after")
    def require_square_univariate_bounded_evaluation(self) -> Self:
        dimension = len(self.matrix.entries)
        if len(self.matrix.entries[0]) != dimension:
            raise ValueError("matrix polynomial evaluation requires a square matrix")
        if len(self.polynomial.variables) != 1:
            raise ValueError(
                "matrix polynomial evaluation requires exactly one polynomial variable"
            )
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=MAX_POLYNOMIAL_TERMS,
            maximum_exponent=MAX_POLYNOMIAL_EXPONENT,
            maximum_coefficient_digits=MAX_CANONICAL_RATIONAL_DIGITS,
            label="matrix polynomial",
        )
        degree = _polynomial_degree(self.polynomial)
        scalar_products_per_pass = degree * dimension**3
        total_scalar_products = (
            MATRIX_POLYNOMIAL_EVALUATION_PASSES * scalar_products_per_pass
        )
        if total_scalar_products > MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS:
            raise ValueError(
                "matrix polynomial Horner evaluation and source-bound replay "
                f"exceed the {MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS:,}-scalar-product "
                "work bound"
            )
        maximum_arithmetic_digits = _require_matrix_polynomial_output_budget(
            self.matrix,
            self.polynomial,
            degree,
        )
        digit_work = total_scalar_products * maximum_arithmetic_digits**2
        if digit_work > MAX_MATRIX_POLYNOMIAL_DIGIT_WORK:
            raise ValueError(
                "matrix polynomial exact-arithmetic work exceeds the coupled "
                f"{MAX_MATRIX_POLYNOMIAL_DIGIT_WORK:,}-unit digit-work bound"
            )
        return self


class MatrixPolynomialEvaluationResult(StrictModel):
    """Source-bound exact value of one polynomial at one rational matrix."""

    source_matrix: RationalMatrix
    polynomial: RationalPolynomial
    value: RationalMatrix
    polynomial_degree: int | None = Field(
        default=None,
        ge=0,
        le=MAX_POLYNOMIAL_EXPONENT,
        description="The ordinary degree for a nonzero polynomial; null for zero.",
    )
    matrix_multiplications: int = Field(ge=0, le=MAX_POLYNOMIAL_EXPONENT)
    scalar_product_terms: int = Field(
        ge=0,
        le=MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS // MATRIX_POLYNOMIAL_EVALUATION_PASSES,
    )
    method: Literal["HORNER_OVER_QQ"] = "HORNER_OVER_QQ"

    @model_validator(mode="after")
    def bind_exact_evaluation(self) -> Self:
        request = MatrixPolynomialEvaluationRequest(
            matrix=self.source_matrix,
            polynomial=self.polynomial,
        )
        expected_degree = _mathematical_polynomial_degree(request.polynomial)
        if self.polynomial_degree != expected_degree:
            raise ValueError(
                "matrix polynomial result degree does not match its source"
            )
        expected_multiplications = expected_degree or 0
        if self.matrix_multiplications != expected_multiplications:
            raise ValueError(
                "Horner matrix multiplication count must equal the polynomial degree"
            )
        dimension = len(request.matrix.entries)
        if self.scalar_product_terms != expected_multiplications * dimension**3:
            raise ValueError(
                "Horner scalar-product count must equal degree times matrix order cubed"
            )
        from jacobian.math.matrices.canonical_forms._operations import (
            evaluate_matrix_polynomial_value,
        )

        expected = evaluate_matrix_polynomial_value(request)
        if self.value != expected:
            raise ValueError(
                "matrix polynomial value does not equal the retained exact evaluation"
            )
        return self


class SquareMatrixRequest(StrictModel):
    """One square rational matrix bounded for canonical-form computation."""

    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_bounded_square(self) -> Self:
        rows = len(self.matrix.entries)
        columns = len(self.matrix.entries[0])
        if rows != columns:
            raise ValueError("canonical-form operations require a square matrix")
        if rows > MAX_CANONICAL_FORM_DIMENSION:
            raise ValueError(
                "canonical-form operations are bounded to 16 x 16 matrices"
            )
        require_matrix_scalar_digits(
            self.matrix.entries,
            maximum=MAX_CANONICAL_FORM_SCALAR_DIGITS,
            label="canonical-form matrix",
        )
        return self


class MonicPolynomial(StrictModel):
    """One monic univariate polynomial over QQ, as increasing-degree coefficients."""

    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_CANONICAL_FORM_DIMENSION + 1,
    )

    @model_validator(mode="after")
    def require_monic(self) -> Self:
        if self.coefficients[-1].as_fraction() != 1:
            raise ValueError("polynomial must be monic (leading coefficient = 1)")
        return self


class MinimalPolynomialResult(StrictModel):
    """Exact minimal polynomial of a square rational matrix."""

    minimal_polynomial: MonicPolynomial
    characteristic_polynomial: MonicPolynomial
    degree: int = Field(ge=1, le=MAX_CANONICAL_FORM_DIMENSION)
    method: Literal["KRYLOV_NULLSPACE"] = "KRYLOV_NULLSPACE"


class InvariantFactorEntry(StrictModel):
    """One monic invariant factor from the rational canonical form."""

    factor: MonicPolynomial
    block_size: int = Field(ge=1, le=MAX_CANONICAL_FORM_DIMENSION)


class RationalCanonicalFormResult(StrictModel):
    """Exact rational (Frobenius) canonical form of a square rational matrix."""

    invariant_factors: tuple[InvariantFactorEntry, ...] = Field(min_length=1)
    characteristic_polynomial: MonicPolynomial
    minimal_polynomial: MonicPolynomial
    total_block_size: int = Field(ge=1, le=MAX_CANONICAL_FORM_DIMENSION)
    method: Literal["SMITH_NORMAL_FORM"] = "SMITH_NORMAL_FORM"


class PrimaryDecompositionResult(StrictModel):
    """Primary decomposition of the minimal polynomial into irreducible-power components."""

    components: tuple[MonicPolynomial, ...] = Field(min_length=1)
    minimal_polynomial: MonicPolynomial
    method: Literal["FACTOR_LCM"] = "FACTOR_LCM"


__all__ = [
    "MATRIX_POLYNOMIAL_EVALUATION_PASSES",
    "MAX_CANONICAL_FORM_DIMENSION",
    "MAX_CANONICAL_FORM_SCALAR_DIGITS",
    "MAX_MATRIX_POLYNOMIAL_DIGIT_WORK",
    "MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS",
    "InvariantFactorEntry",
    "MatrixPolynomialEvaluationRequest",
    "MatrixPolynomialEvaluationResult",
    "MinimalPolynomialResult",
    "MonicPolynomial",
    "PrimaryDecompositionResult",
    "RationalCanonicalFormResult",
    "SquareMatrixRequest",
]
