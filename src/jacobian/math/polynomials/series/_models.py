"""Wire contracts for exact truncated formal power series operations."""

from __future__ import annotations

from bisect import bisect_left
from math import comb, lcm
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math._rational_height import RationalHeight, sum_heights
from jacobian.math.polynomials.values import PolynomialVariable, RationalPolynomial

# ---------------------------------------------------------------------------
# Public bounds
# ---------------------------------------------------------------------------

MAX_TRUNCATION_ORDER = 512
MAX_MULTIPLY_INCIDENCES = MAX_TRUNCATION_ORDER * (MAX_TRUNCATION_ORDER + 1) // 2
MAX_RATIONAL_DIGITS = 256
MAX_RESULT_RATIONAL_DIGITS = 4_096
MAX_REVERSION_INTERMEDIATE_DIGITS = 16_384
MAX_REVERSION_BACKEND_WORK = 1_000_000_000
MAX_POWER_EXPONENT = 1_000

# Truncation sources are admitted through the widest carrier canonical
# values can carry: formal-series results keep the 512-order input
# envelope, and level-one modular q-expansion replay admits E4/E6
# expansions up to order 25280 under its own 4,000,000 work-term budget
# (p * isqrt(p)).  Request admission still materializes and
# height-validates every source coefficient before the prefix is read,
# so this ceiling bounds that linear admission work while
# producer-to-truncate composition stays closed over every representable
# expansion.
MAX_TRUNCATE_SOURCE_ORDER = 25_280

CoefficientHeight = RationalHeight | None


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by formal-series contracts."""

    return PydanticCustomError(f"formal_power_series.{reason}", message)


def _resource_error(reason: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=(), code=f"formal_power_series.{reason}", message=message
    )


def _has_degree_at_most(series: TruncatedSeries, degree: int) -> bool:
    return series.truncation_order <= MAX_TRUNCATE_SOURCE_ORDER and not any(
        value.num for value in series.coefficients[degree + 1 :]
    )


def _height(value: CanonicalRational) -> RationalHeight:
    return RationalHeight.from_canonical(value)


def _coefficient_height(value: CanonicalRational) -> CoefficientHeight:
    return None if value.num == 0 else _height(value)


def _add_height(left: CoefficientHeight, right: CoefficientHeight) -> CoefficientHeight:
    if left is None:
        return right
    if right is None:
        return left
    return sum_heights((left, right))


def _height_vector(
    coefficients: tuple[CanonicalRational, ...],
) -> tuple[CoefficientHeight, ...]:
    return tuple(_coefficient_height(value) for value in coefficients)


def _require_height_vector(
    coefficients: tuple[CoefficientHeight, ...], operation: str
) -> None:
    if any(
        height is not None and height.exceeds(MAX_RESULT_RATIONAL_DIGITS)
        for height in coefficients
    ):
        raise _resource_error(
            f"{operation}_coefficient_growth",
            f"{operation} coefficient growth exceeds the "
            f"{MAX_RESULT_RATIONAL_DIGITS}-digit result bound",
        )


def _convolve_height_vectors(
    left: tuple[CoefficientHeight, ...],
    right: tuple[CoefficientHeight, ...],
    order: int,
    operation: str,
) -> tuple[CoefficientHeight, ...]:
    result: list[CoefficientHeight] = []
    for degree in range(order):
        terms: list[RationalHeight] = []
        for index in range(degree + 1):
            if index >= len(left) or degree - index >= len(right):
                continue
            left_height = left[index]
            right_height = right[degree - index]
            if left_height is not None and right_height is not None:
                terms.append(left_height.product(right_height))
        result.append(sum_heights(terms) if terms else None)
    coefficients = tuple(result)
    _require_height_vector(coefficients, operation)
    return coefficients


def _composition_height_vector(
    outer: tuple[CoefficientHeight, ...],
    inner: tuple[CoefficientHeight, ...],
    order: int,
    operation: str,
) -> tuple[CoefficientHeight, ...]:
    powers: tuple[CoefficientHeight, ...] = (
        RationalHeight(1, 1),
        *([None] * (order - 1)),
    )
    result: list[CoefficientHeight] = [None] * order
    for outer_degree in range(order):
        coefficient = outer[outer_degree]
        if coefficient is not None:
            for degree, power in enumerate(powers):
                if power is not None:
                    result[degree] = _add_height(
                        result[degree], coefficient.product(power)
                    )
            _require_height_vector(tuple(result), operation)
        if outer_degree + 1 < order:
            powers = _convolve_height_vectors(powers, inner, order, operation)
    return tuple(result)


def _max_height(values: tuple[CanonicalRational, ...]) -> RationalHeight:
    heights = tuple(_height(value) for value in values)
    return RationalHeight(
        max(value.numerator_digits for value in heights),
        max(value.denominator_digits for value in heights),
    )


def _convolution_height(
    left: RationalHeight, right: RationalHeight, term_count: int
) -> RationalHeight:
    term = left.product(right)
    return sum_heights(term for _ in range(term_count))


def _require_height(height: RationalHeight, operation: str) -> None:
    if height.exceeds(MAX_RESULT_RATIONAL_DIGITS):
        raise _resource_error(
            f"{operation}_coefficient_growth",
            f"{operation} coefficient growth exceeds the "
            f"{MAX_RESULT_RATIONAL_DIGITS}-digit result bound",
        )


def _require_zero_residual(
    coefficients: tuple[CanonicalRational, ...], order: int, operation: str
) -> None:
    if len(coefficients) != order:
        raise _validation_error(
            f"{operation}_residual_length",
            f"{operation} residual must contain exactly {order} coefficients",
        )
    if any(value.num != 0 for value in coefficients):
        raise _validation_error(
            f"{operation}_residual_nonzero",
            f"{operation} residual must be identically zero",
        )


def _require_binary_height(numerator: int, denominator: int, operation: str) -> None:
    # If |p| <= 2**b then p has at most floor(b / 3) + 1 decimal
    # digits, since 2**3 < 10. No large power or floating log is needed.
    _require_height(RationalHeight(numerator // 3 + 1, denominator // 3 + 1), operation)


def _binary_decimal_digits(bits: int) -> int:
    """Return a safe decimal digit bound for an integer below ``2**bits``."""
    return (bits * 30_103) // 100_000 + 1


_RESULT_BOUND_CAP: int = 10**MAX_RESULT_RATIONAL_DIGITS


def _result_bound_product(left: int, right: int) -> int:
    return min(_RESULT_BOUND_CAP, left * right)


def _result_bound_power(value: int, exponent: int) -> int:
    result = 1
    factor = min(_RESULT_BOUND_CAP, value)
    while exponent:
        if exponent & 1:
            result = _result_bound_product(result, factor)
        exponent >>= 1
        if exponent:
            factor = _result_bound_product(factor, factor)
    return result


def _result_bound_sum(left: int, right: int) -> int:
    return min(_RESULT_BOUND_CAP, left + right)


def _result_bound_digits(value: int) -> int:
    if value >= _RESULT_BOUND_CAP:
        return MAX_RESULT_RATIONAL_DIGITS + 1
    return _bounded_integer_digits(value)


_REVERSION_BOUND_CAP: int = 10 ** (MAX_REVERSION_INTERMEDIATE_DIGITS + 1)


def _bounded_integer_product(left: int, right: int) -> int:
    if not left or not right:
        return 0
    if left >= _REVERSION_BOUND_CAP or right >= _REVERSION_BOUND_CAP:
        return _REVERSION_BOUND_CAP
    return min(_REVERSION_BOUND_CAP, left * right)


def _bounded_integer_power(value: int, exponent: int) -> int:
    result = 1
    factor = min(_REVERSION_BOUND_CAP, abs(value))
    while exponent:
        if exponent & 1:
            result = _bounded_integer_product(result, factor)
        exponent >>= 1
        if exponent:
            factor = _bounded_integer_product(factor, factor)
    return result


def _bounded_integer_sum(values: tuple[int, ...]) -> int:
    result = 0
    for value in values:
        result = min(_REVERSION_BOUND_CAP, result + value)
        if result >= _REVERSION_BOUND_CAP:
            return _REVERSION_BOUND_CAP
    return result


def _bounded_integer_digits(value: int) -> int:
    if value >= _REVERSION_BOUND_CAP:
        return MAX_REVERSION_INTERMEDIATE_DIGITS + 1
    # 0.30103 is an upper decimal-logarithm bound for binary integers.  The
    # resulting estimate is conservative and avoids converting an admitted
    # bound to a string larger than Python's integer-string safety limit.
    return (max(1, value).bit_length() * 30_103) // 100_000 + 1


def _reversion_lagrange_height_vector(
    series: TruncatedSeries,
) -> tuple[tuple[CoefficientHeight, ...], int, int]:
    """Bound reversion coefficients with the Lagrange majorant.

    For ``F(x) = a_1*x*(1 + H(x))`` and ``G = F^(-1)``, Lagrange inversion
    gives ``[x^k]G = [x^(k-1)](1 + H)^(-k) / (k*a_1**k)``.  If
    ``H = B/E`` coefficientwise, the absolute value of the degree ``k-1``
    coefficient is bounded by
    ``sum_m binom(k+m-1,m) * B**m * E**(k-1-m)``.  This is a majorant only:
    no result coefficients or backend arithmetic are evaluated during
    admission.
    """

    linear = series.coefficients[1].as_fraction()
    normalized = tuple(
        coefficient.as_fraction() / linear for coefficient in series.coefficients[2:]
    )
    common_denominator = 1
    for coefficient in normalized:
        common_denominator = lcm(common_denominator, coefficient.denominator)
    numerators = tuple(
        int(coefficient * common_denominator) for coefficient in normalized
    )
    coefficient_sum = sum(abs(value) for value in numerators)
    p = abs(linear.numerator)
    q = linear.denominator
    result: list[CoefficientHeight] = [
        None,
        RationalHeight(_bounded_integer_digits(q), _bounded_integer_digits(p)),
    ]
    denominator_bounds = [1, p]
    for degree in range(2, series.truncation_order):
        target_degree = degree - 1
        # Every negative-binomial coefficient is at most the final one, so
        # the composition sum is bounded without replaying its O(k) terms.
        coefficient_bound = _bounded_integer_product(
            comb(2 * degree - 2, target_degree),
            _bounded_integer_power(
                _bounded_integer_sum((coefficient_sum, common_denominator)),
                target_degree,
            ),
        )
        numerator_bound = _bounded_integer_product(
            coefficient_bound, _bounded_integer_power(q, degree)
        )
        denominator_bound = _bounded_integer_product(
            _bounded_integer_power(common_denominator, target_degree),
            _bounded_integer_product(degree, _bounded_integer_power(p, degree)),
        )
        result.append(
            RationalHeight(
                _bounded_integer_digits(numerator_bound),
                _bounded_integer_digits(denominator_bound),
            )
        )
        denominator_bounds.append(denominator_bound)
    # Each ``denominator_bound`` is the explicit common multiple
    # ``E**(k-1) * k * p**k`` supplied by the Lagrange formula, not merely an
    # upper bound on the actual denominator.  Its LCM is therefore a sound
    # common denominator for all bounded result coefficients.
    common_result_denominator = 1
    for denominator in denominator_bounds:
        common_result_denominator = lcm(common_result_denominator, denominator)
        if common_result_denominator >= _REVERSION_BOUND_CAP:
            common_result_denominator = _REVERSION_BOUND_CAP
            break
    common_result_numerator = _REVERSION_BOUND_CAP
    if common_result_denominator < _REVERSION_BOUND_CAP:
        common_result_numerator = 0
        for _degree, height in enumerate(result):
            if height is None:
                continue
            common_result_numerator = min(
                _REVERSION_BOUND_CAP,
                common_result_numerator
                + _bounded_integer_product(
                    _bounded_integer_power(10, height.numerator_digits),
                    common_result_denominator,
                ),
            )
    return (
        tuple(result),
        common_result_denominator,
        common_result_numerator,
    )


def _common_series_bound(
    coefficients: tuple[CanonicalRational, ...],
) -> tuple[int, int, tuple[int, ...]]:
    """Return a bounded common denominator, L1 numerator, and terms."""

    denominator = 1
    for coefficient in coefficients:
        denominator = lcm(denominator, coefficient.den)
        if denominator >= _REVERSION_BOUND_CAP:
            return _REVERSION_BOUND_CAP, _REVERSION_BOUND_CAP, ()
    terms = tuple(
        abs(coefficient.num) * (denominator // coefficient.den)
        for coefficient in coefficients
    )
    return denominator, _bounded_integer_sum(terms), terms


def _result_series_bound(
    heights: tuple[CoefficientHeight, ...], denominator: int
) -> tuple[int, int, tuple[int, ...]]:
    if denominator >= _REVERSION_BOUND_CAP:
        return _REVERSION_BOUND_CAP, _REVERSION_BOUND_CAP, ()
    terms = tuple(
        0
        if height is None
        else _bounded_integer_product(
            _bounded_integer_power(10, height.numerator_digits), denominator
        )
        for height in heights
    )
    return denominator, _bounded_integer_sum(terms), terms


def _bound_product(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return (
        _bounded_integer_product(left[0], right[0]),
        _bounded_integer_product(left[1], right[1]),
    )


def _bound_composition(
    outer_denominator: int,
    outer_terms: tuple[int, ...],
    inner_denominator: int,
    inner_numerator: int,
    order: int,
) -> tuple[int, int]:
    degree = min(
        max((index for index, value in enumerate(outer_terms) if value), default=0),
        order - 1,
    )
    denominator = _bounded_integer_product(
        outer_denominator, _bounded_integer_power(inner_denominator, degree)
    )
    numerator = _bounded_integer_sum(
        tuple(
            _bounded_integer_product(
                term,
                _bounded_integer_product(
                    _bounded_integer_power(inner_numerator, index),
                    _bounded_integer_power(inner_denominator, degree - index),
                ),
            )
            for index, term in enumerate(outer_terms[: degree + 1])
        )
    )
    return denominator, numerator


def _bound_derivative(
    denominator: int, terms: tuple[int, ...]
) -> tuple[int, tuple[int, ...]]:
    return denominator, tuple(
        _bounded_integer_product(index, value) for index, value in enumerate(terms)
    )[1:]


def _reversion_backend_bound(
    series: TruncatedSeries,
    result_vector: tuple[CoefficientHeight, ...],
    result_denominator: int,
) -> int:
    """Bound non-cancelling FLINT intermediates independently of zero residuals."""

    order = series.truncation_order
    source_denominator, source_numerator, source_terms = _common_series_bound(
        series.coefficients
    )
    result_denominator, result_numerator, result_terms = _result_series_bound(
        result_vector, result_denominator
    )
    if _REVERSION_BOUND_CAP in (
        source_denominator,
        source_numerator,
        result_denominator,
        result_numerator,
    ):
        return MAX_REVERSION_INTERMEDIATE_DIGITS + 1

    source_derivative_denominator, source_derivative_terms = _bound_derivative(
        source_denominator, source_terms
    )
    result_derivative_denominator, result_derivative_terms = _bound_derivative(
        result_denominator, result_terms
    )
    forward = _bound_composition(
        source_denominator,
        source_terms,
        result_denominator,
        result_numerator,
        order,
    )
    reverse = _bound_composition(
        result_denominator,
        result_terms,
        source_denominator,
        source_numerator,
        order,
    )
    derivative_forward = _bound_composition(
        source_derivative_denominator,
        source_derivative_terms,
        result_denominator,
        result_numerator,
        order,
    )
    derivative_reverse = _bound_composition(
        result_derivative_denominator,
        result_derivative_terms,
        source_denominator,
        source_numerator,
        order,
    )
    source_result = _bound_product(
        (source_denominator, source_numerator),
        (result_denominator, result_numerator),
    )
    derivative_product = _bound_product(derivative_forward, derivative_reverse)
    correction = (
        source_result[0],
        _bounded_integer_sum(
            (_bounded_integer_product(2, source_result[0]), source_result[1])
        ),
    )
    newton_product = _bound_product((result_denominator, result_numerator), correction)
    derivative_correction = (
        derivative_product[0],
        _bounded_integer_sum(
            (_bounded_integer_product(2, derivative_product[0]), derivative_product[1])
        ),
    )
    derivative_newton_product = _bound_product(
        derivative_reverse, derivative_correction
    )
    residual_product = _bound_product(forward, derivative_reverse)
    bounds = (
        forward,
        reverse,
        derivative_forward,
        derivative_reverse,
        source_result,
        derivative_product,
        derivative_newton_product,
        newton_product,
        residual_product,
    )
    return max(
        max(_bounded_integer_digits(denominator), _bounded_integer_digits(numerator))
        for denominator, numerator in bounds
    )


def _cleared_series(
    series: TruncatedSeries, operation: str = "inverse"
) -> tuple[int, tuple[int, ...]]:
    """Bound denominator clearing before building P in A=P/D.

    At most N lcms are formed. Each temporary integer adds at most one
    admitted input denominator to the previous bounded lcm; no series
    coefficients are computed during admission.
    """
    denominator = 1
    for value in series.coefficients:
        denominator = lcm(denominator, value.den)
        _require_binary_height(0, denominator.bit_length(), operation)
    return denominator, tuple(
        value.num * (denominator // value.den) for value in series.coefficients
    )


def _inverse_height(series: TruncatedSeries) -> tuple[int, int, int, int]:
    """Return shared numerator/denominator bit bounds and source norm bounds.

    Write A=P/D, p=|P[0]|, C=p+sum_{i>0}|P[i]|. The finite geometric
    expansion gives a common denominator p**N and numerator magnitude at
    most D*C**(N-1) for every inverse coefficient. Indeed the degree-k
    numerator over p**(k+1) is bounded by D*C**k by induction in the
    triangular recurrence. Homogenizing to p**N preserves that bound.
    Constant sources only need denominator p. This also bounds every
    partial recurrence sum after multiplication by a source coefficient.

    With N<=512 the existing kernel uses O(N**2) Fraction operations.
    Reduced terms and partial sums fit the checked common-denominator
    bounds; Fraction's unreduced cross-products have at most twice their
    component bit bounds. Thus intermediates are bounded before execution.
    """
    denominator, coefficients = _cleared_series(series)
    constant = abs(coefficients[0])
    tail = sum(abs(value) for value in coefficients[1:])
    order = series.truncation_order if tail else 1
    source_norm = (constant + tail).bit_length()
    source_denominator = denominator.bit_length()
    numerator = source_denominator + (order - 1) * source_norm
    inverse_denominator = order * (constant - 1).bit_length()
    _require_binary_height(numerator, inverse_denominator, "inverse")
    # Product residuals and partial recurrence sums share D*p**N.
    _require_binary_height(
        numerator + source_norm + 1,
        inverse_denominator + source_denominator,
        "inverse residual",
    )
    return numerator, inverse_denominator, source_norm, source_denominator


Variable = PolynomialVariable


# ---------------------------------------------------------------------------
# Shared value type
# ---------------------------------------------------------------------------


class TruncatedSeries(StrictModel):
    """One immutable element of QQ[[x]]/(x^N).

    The coefficient tuple has exactly ``truncation_order`` entries in
    ascending-power order.  Two series are equal iff they share the same
    variable, truncation order, and coefficient tuple.
    """

    variable: Variable = Field(description="The single formal variable.")
    truncation_order: StrictInt = Field(
        ge=1,
        description="Truncation order N (coefficients a_0..a_{N-1}).",
    )
    coefficients: tuple[CanonicalRational, ...] = Field(
        description="Exactly N rational coefficients in ascending powers.",
    )

    @model_validator(mode="after")
    def require_dense_tuple(self) -> Self:
        if len(self.coefficients) != self.truncation_order:
            raise _validation_error(
                "coefficient_count_mismatch",
                "coefficient tuple must have exactly truncation_order entries",
            )
        return self


# Native callers already hold the canonical carrier.  Keep their admission
# separate from the JSON request models: rebuilding a TruncatedSeries or
# an operation Request here would turn the native API into a wire adapter and
# would make its public exceptions and coercions depend on Pydantic.
def _require_native_scalar(value: int, label: str) -> None:
    if type(value) is not int:  # bool is not a semantic integer scalar here.
        raise _validation_error(f"{label}_type", f"{label} must be an integer scalar")


def _require_native_input_series(
    series: TruncatedSeries, *, maximum_order: int = MAX_TRUNCATION_ORDER
) -> None:
    if series.truncation_order > maximum_order:
        raise _resource_error(
            "input_order", f"truncation order exceeds {maximum_order}"
        )
    try:
        for value in series.coefficients:
            require_bounded_rational(
                value, max_digits=MAX_RATIONAL_DIGITS, label="input coefficient"
            )
    except ValueError as exc:
        raise _resource_error("input_coefficient", str(exc)) from exc


def _require_native_pair(
    left: TruncatedSeries,
    right: TruncatedSeries,
    *,
    maximum_order: int = MAX_TRUNCATION_ORDER,
) -> None:
    _require_native_input_series(left, maximum_order=maximum_order)
    _require_native_input_series(right, maximum_order=maximum_order)
    if left.variable != right.variable:
        raise _validation_error(
            "operand_variable_mismatch", "operands must share the same variable"
        )
    if left.truncation_order != right.truncation_order:
        raise _validation_error(
            "operand_order_mismatch", "operands must share the same truncation order"
        )


def admit_native_add_subtract(left: TruncatedSeries, right: TruncatedSeries) -> None:
    _require_native_pair(left, right)
    for left_value, right_value in zip(
        left.coefficients, right.coefficients, strict=True
    ):
        _require_height(sum_heights((_height(left_value), _height(right_value))), "sum")


def admit_native_multiply(left: TruncatedSeries, right: TruncatedSeries) -> None:
    _require_native_pair(left, right, maximum_order=MAX_TRUNCATE_SOURCE_ORDER)
    order = left.truncation_order
    left_support = [i for i, value in enumerate(left.coefficients) if value.num]
    right_support = [i for i, value in enumerate(right.coefficients) if value.num]
    # Count before allocating incidences; pairs outside the retained prefix
    # perform no rational arithmetic in the sparse convolution kernel.
    incidences = sum(bisect_left(right_support, order - i) for i in left_support)
    if incidences > MAX_MULTIPLY_INCIDENCES:
        raise _resource_error(
            "multiplication_work",
            f"retained convolution incidences exceed {MAX_MULTIPLY_INCIDENCES}",
        )
    left_denominators = {left.coefficients[i].den for i in left_support}
    right_denominators = {right.coefficients[i].den for i in right_support}
    if len(left_denominators) == len(right_denominators) == 1:
        left_numerator = max(abs(left.coefficients[i].num) for i in left_support)
        right_numerator = max(abs(right.coefficients[i].num) for i in right_support)
        denominators = left_denominators.pop() * right_denominators.pop()
        for degree in range(order):
            count = min(degree + 1, incidences)
            if count:
                numerator = count * left_numerator * right_numerator
                _require_height(
                    RationalHeight(
                        _bounded_integer_digits(numerator),
                        _bounded_integer_digits(denominators),
                    ),
                    "multiplication",
                )
        return
    terms: dict[int, list[RationalHeight]] = {}
    for i in left_support:
        left_height = _height(left.coefficients[i])
        for j in right_support:
            if i + j >= order:
                break
            terms.setdefault(i + j, []).append(
                left_height.product(_height(right.coefficients[j]))
            )
    for coefficient_terms in terms.values():
        _require_height(sum_heights(coefficient_terms), "multiplication")


def admit_native_scalar_multiply(
    series: TruncatedSeries, scalar: CanonicalRational
) -> None:
    _require_native_input_series(series)
    _require_height(
        _max_height(series.coefficients).product(_height(scalar)),
        "scalar multiplication",
    )


def admit_native_power(series: TruncatedSeries, exponent: int) -> None:
    _require_native_input_series(series)
    _require_native_scalar(exponent, "exponent")
    if not 0 <= exponent <= MAX_POWER_EXPONENT:
        raise _validation_error(
            "power_exponent", f"exponent must be between 0 and {MAX_POWER_EXPONENT}"
        )
    if exponent == 0:
        return
    denominator, coefficients = _cleared_series(series, "power")
    # Write f=A/D. The l1 norm is submultiplicative, even under
    # truncation: every coefficient and partial convolution sum of A^j
    # has magnitude <= max(1, ||A||_1)^e for 0 <= j <= e. Every
    # denominator divides D^j. Binary powering never exceeds exponent e.
    norm_bits = max(1, sum(abs(c) for c in coefficients)).bit_length()
    denominator_bits = 0 if denominator == 1 else denominator.bit_length()
    _require_height(
        RationalHeight(
            _binary_decimal_digits(exponent * norm_bits),
            _binary_decimal_digits(exponent * denominator_bits),
        ),
        "power",
    )


def admit_native_inverse(series: TruncatedSeries) -> None:
    constant = _has_degree_at_most(series, 0)
    _require_native_input_series(
        series,
        maximum_order=MAX_TRUNCATE_SOURCE_ORDER if constant else MAX_TRUNCATION_ORDER,
    )
    if series.coefficients[0].as_fraction() == 0:
        raise _validation_error(
            "inverse_zero_constant", "inverse requires a nonzero constant term"
        )
    if not constant:
        _inverse_height(series)


def admit_native_divide(
    numerator: TruncatedSeries, denominator: TruncatedSeries
) -> None:
    constant = _has_degree_at_most(denominator, 0)
    _require_native_pair(
        numerator,
        denominator,
        maximum_order=MAX_TRUNCATE_SOURCE_ORDER if constant else MAX_TRUNCATION_ORDER,
    )
    if denominator.coefficients[0].as_fraction() == 0:
        raise _validation_error(
            "denominator_zero_constant", "denominator must have a nonzero constant term"
        )
    if constant:
        # A coefficient quotient has at most twice the input component bound.
        _require_height(
            RationalHeight(2 * MAX_RATIONAL_DIGITS, 2 * MAX_RATIONAL_DIGITS), "division"
        )
        return
    inv_num, inv_den, source_norm, source_den = _inverse_height(denominator)
    common_denominator, coefficients = _cleared_series(numerator)
    norm = sum(abs(value) for value in coefficients).bit_length()
    common_den = common_denominator.bit_length()
    # For numerator Q/E, all quotient partial sums have denominator
    # E*p**N and numerator bounded by ||Q||_1 times the inverse bound.
    quotient_num = norm + inv_num
    quotient_den = common_den + inv_den
    _require_binary_height(quotient_num, quotient_den, "division")
    # B*(Q/B)-Q has common denominator D*E*p**N. Include both
    # convolution partial sums and the final source subtraction.
    _require_binary_height(
        max(source_norm + quotient_num, norm + source_den + inv_den) + 1,
        source_den + quotient_den,
        "division residual",
    )


def admit_native_compose(outer: TruncatedSeries, inner: TruncatedSeries) -> None:
    _require_native_pair(outer, inner)
    if inner.coefficients[0].as_fraction() != 0:
        raise _validation_error(
            "composition_nonzero_inner_constant",
            "inner series must have zero constant term for composition with a finite prefix",
        )
    outer_denominators = {value.den for value in outer.coefficients if value.num}
    inner_denominators = {value.den for value in inner.coefficients if value.num}
    if len(outer_denominators) <= 1 and len(inner_denominators) <= 1:
        from math import comb

        outer_denominator = next(iter(outer_denominators), 1)
        inner_denominator = next(iter(inner_denominators), 1)
        outer_numerators = tuple(value.num for value in outer.coefficients)
        inner_norm = sum(abs(value.num) for value in inner.coefficients)
        for degree in range(outer.truncation_order):
            bound = 0
            for power_degree in range(degree + 1):
                if not outer_numerators[power_degree] or (
                    power_degree == 0 and degree != 0
                ):
                    continue
                term = abs(outer_numerators[power_degree])
                if power_degree:
                    term = _result_bound_product(
                        term, comb(degree - 1, power_degree - 1)
                    )
                    term = _result_bound_product(
                        term, _result_bound_power(inner_norm, power_degree)
                    )
                    term = _result_bound_product(
                        term,
                        _result_bound_power(inner_denominator, degree - power_degree),
                    )
                bound = _result_bound_sum(bound, term)
            denominator = _result_bound_product(
                outer_denominator,
                _result_bound_power(inner_denominator, degree),
            )
            if (
                _result_bound_digits(bound) > MAX_RESULT_RATIONAL_DIGITS
                or _result_bound_digits(denominator) > MAX_RESULT_RATIONAL_DIGITS
            ):
                _require_height(
                    RationalHeight(
                        _result_bound_digits(bound),
                        _result_bound_digits(denominator),
                    ),
                    "composition",
                )
        return
    _composition_height_vector(
        _height_vector(outer.coefficients),
        _height_vector(inner.coefficients),
        outer.truncation_order,
        "composition",
    )


def admit_native_reversion(series: TruncatedSeries) -> None:
    linear_source = _has_degree_at_most(series, 1)
    _require_native_input_series(
        series,
        maximum_order=MAX_TRUNCATE_SOURCE_ORDER
        if linear_source
        else MAX_TRUNCATION_ORDER,
    )
    if series.truncation_order < 2:
        raise _validation_error(
            "reversion_order", "reversion requires truncation order >= 2"
        )
    if series.coefficients[0].as_fraction() != 0:
        raise _validation_error(
            "reversion_nonzero_constant", "reversion requires zero constant term"
        )
    if series.coefficients[1].as_fraction() == 0:
        raise _validation_error(
            "reversion_zero_linear_coefficient",
            "reversion requires nonzero linear coefficient",
        )
    if linear_source:
        return
    result_vector, common_denominator, common_numerator = (
        _reversion_lagrange_height_vector(series)
    )
    _require_height_vector(result_vector, "reversion")

    # Residual ledgers are mathematically zero and are established by the
    # exact backend.  Admission therefore bounds the backend's finite
    # operand/intermediate arithmetic separately instead of charging a
    # non-cancelling expansion of those zero residuals against the result
    # carrier.
    intermediate_digits = _reversion_backend_bound(
        series, result_vector, common_denominator
    )
    if common_numerator >= _REVERSION_BOUND_CAP:
        intermediate_digits = MAX_REVERSION_INTERMEDIATE_DIGITS + 1
    if intermediate_digits > MAX_REVERSION_INTERMEDIATE_DIGITS:
        raise _resource_error(
            "reversion_intermediate_growth",
            "reversion backend intermediate growth exceeds the "
            f"{MAX_REVERSION_INTERMEDIATE_DIGITS}-digit bound",
        )
    backend_work = series.truncation_order**2 * max(1, intermediate_digits)
    if backend_work > MAX_REVERSION_BACKEND_WORK:
        raise _resource_error(
            "reversion_backend_work",
            "reversion backend arithmetic exceeds the bounded work limit",
        )


def admit_native_integral(series: TruncatedSeries, output_order: int) -> None:
    _require_native_input_series(series)
    _require_native_scalar(output_order, "output_order")
    if not 1 <= output_order <= MAX_TRUNCATION_ORDER:
        raise _validation_error(
            "integral_output_order",
            f"output_order must be between 1 and {MAX_TRUNCATION_ORDER}",
        )
    if output_order > series.truncation_order + 1:
        raise _validation_error(
            "integral_output_order_exceeds_source",
            "output_order must not exceed source_order + 1",
        )
    integer = RationalHeight(len(str(max(1, output_order - 1))), 1)
    _require_height(_max_height(series.coefficients).quotient(integer), "integration")


def admit_native_truncate(series: TruncatedSeries, target_order: int) -> None:
    _require_native_scalar(target_order, "target_order")
    if series.truncation_order > MAX_TRUNCATE_SOURCE_ORDER:
        raise _validation_error(
            "truncate_source_order",
            f"source truncation order exceeds {MAX_TRUNCATE_SOURCE_ORDER}",
        )
    if target_order > series.truncation_order:
        raise _validation_error(
            "truncate_target_exceeds_source",
            "target_order must not exceed source truncation order",
        )
    if not 1 <= target_order <= MAX_TRUNCATION_ORDER:
        raise _validation_error(
            "truncate_target_exceeds_public_bound",
            "target_order exceeds the public bound",
        )


def admit_native_identity_check(left: TruncatedSeries, right: TruncatedSeries) -> None:
    _require_native_pair(left, right)
    for left_value, right_value in zip(
        left.coefficients, right.coefficients, strict=True
    ):
        _require_height(
            sum_heights((_height(left_value), _height(right_value))), "difference"
        )


def admit_native_from_polynomial(series: TruncatedSeries) -> None:
    _require_native_input_series(series)


# ---------------------------------------------------------------------------
# Pair / single-series request helpers
# ---------------------------------------------------------------------------


class _SeriesPairRequest(StrictModel):
    """Base request with two series that must share variable and order."""

    left: TruncatedSeries
    right: TruncatedSeries


class _SeriesAddSubtractRequest(_SeriesPairRequest):
    """Pair request admitted through coefficientwise arithmetic bounds."""


class _SeriesMultiplyRequest(_SeriesPairRequest):
    """Pair request admitted through Cauchy-product growth bounds."""


class _SeriesIdentityCheckRequest(_SeriesPairRequest):
    """Admit one identity check through its own linear work envelope.

    The kernel compares N coefficient pairs and emits at most one pairwise
    difference ``a_i - b_i``, so admission charges the coefficientwise
    comparison and bounds that difference height; it never charges the
    unrelated Cauchy-convolution growth that multiplication must preflight.
    """


class SeriesDivideRequest(_SeriesPairRequest):
    """Divide two series when the denominator is a unit."""


# ---------------------------------------------------------------------------
# Arithmetic: add / subtract / multiply / scalar multiply
# ---------------------------------------------------------------------------


class SeriesArithmeticResult(StrictModel):
    result: TruncatedSeries
    residual_congruence: Literal["EXACT_MOD_X_TO_N"] = "EXACT_MOD_X_TO_N"


class SeriesMultiplyResult(StrictModel):
    left: TruncatedSeries
    right: TruncatedSeries
    result: TruncatedSeries
    convolution_ledger: tuple[CanonicalRational, ...] = Field(
        description="Per-degree Cauchy convolution sums c_n = sum_{i=0}^n a_i b_{n-i}.",
    )
    residual_congruence: Literal["EXACT_MOD_X_TO_N"] = "EXACT_MOD_X_TO_N"

    @model_validator(mode="after")
    def require_structural_ledger(self) -> Self:
        if (
            self.left.variable != self.right.variable
            or self.left.variable != self.result.variable
            or self.left.truncation_order != self.right.truncation_order
            or self.left.truncation_order != self.result.truncation_order
        ):
            raise _validation_error(
                "source_context_mismatch",
                "multiplication series must share variable and truncation order",
            )
        if len(self.convolution_ledger) != self.result.truncation_order:
            raise _validation_error(
                "convolution_ledger_length",
                "convolution ledger must contain one coefficient per result degree",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        left: TruncatedSeries,
        right: TruncatedSeries,
        result: TruncatedSeries,
        convolution_ledger: tuple[CanonicalRational, ...],
    ) -> Self:
        return cls.model_construct(
            left=left,
            right=right,
            result=result,
            convolution_ledger=convolution_ledger,
        )


class SeriesScalarMultiplyRequest(StrictModel):
    series: TruncatedSeries
    scalar: CanonicalRational


class SeriesScalarMultiplyResult(StrictModel):
    result: TruncatedSeries


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


class SeriesPowerRequest(StrictModel):
    series: TruncatedSeries
    exponent: StrictInt = Field(ge=0)


class SeriesPowerResult(StrictModel):
    result: TruncatedSeries
    multiplication_count: StrictInt
    residual_congruence: Literal["EXACT_MOD_X_TO_N"] = "EXACT_MOD_X_TO_N"


# ---------------------------------------------------------------------------
# Inverse
# ---------------------------------------------------------------------------


class SeriesInverseRequest(StrictModel):
    """Invert a truncated series that is a unit (nonzero constant term)."""

    variable: Variable = Field(description="The single formal variable.")
    truncation_order: StrictInt = Field(
        ge=1,
        description=(
            "Truncation order N; the inverse growth budget must fit every "
            "returned coefficient in the 4096-digit result bound."
        ),
    )
    coefficients: tuple[CanonicalRational, ...] = Field(
        description="Exactly N rational coefficients with a nonzero constant term.",
    )

    def as_series(self) -> TruncatedSeries:
        return TruncatedSeries(
            variable=self.variable,
            truncation_order=self.truncation_order,
            coefficients=self.coefficients,
        )


class SeriesInverseResult(StrictModel):
    source: TruncatedSeries
    result: TruncatedSeries
    residual_congruence: Literal["PRODUCT_IS_ONE_MOD_X_TO_N"] = (
        "PRODUCT_IS_ONE_MOD_X_TO_N"
    )
    residual_coefficients: tuple[CanonicalRational, ...] = Field(
        description="A(x) * B(x) - 1 coefficients (must all be zero).",
    )

    @model_validator(mode="after")
    def require_structural_residual(self) -> Self:
        if (
            self.source.variable != self.result.variable
            or self.source.truncation_order != self.result.truncation_order
        ):
            raise _validation_error(
                "source_context_mismatch",
                "inverse source and result must share variable and truncation order",
            )
        _require_zero_residual(
            self.residual_coefficients,
            self.result.truncation_order,
            "inverse",
        )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: TruncatedSeries,
        result: TruncatedSeries,
        residual_coefficients: tuple[CanonicalRational, ...],
    ) -> Self:
        return cls.model_construct(
            source=source,
            result=result,
            residual_coefficients=residual_coefficients,
        )


# ---------------------------------------------------------------------------
# Divide
# ---------------------------------------------------------------------------


class SeriesDivideResult(StrictModel):
    numerator: TruncatedSeries
    denominator: TruncatedSeries
    quotient: TruncatedSeries
    residual_congruence: Literal[
        "DENOMINATOR_TIMES_QUOTIENT_MINUS_NUMERATOR_IS_ZERO_MOD_X_TO_N"
    ] = "DENOMINATOR_TIMES_QUOTIENT_MINUS_NUMERATOR_IS_ZERO_MOD_X_TO_N"
    residual_coefficients: tuple[CanonicalRational, ...] = Field(
        description="B(x) Q(x) - A(x) coefficients (must all be zero).",
    )

    @model_validator(mode="after")
    def require_structural_residual(self) -> Self:
        if (
            self.numerator.variable != self.denominator.variable
            or self.numerator.variable != self.quotient.variable
            or self.numerator.truncation_order != self.denominator.truncation_order
            or self.numerator.truncation_order != self.quotient.truncation_order
        ):
            raise _validation_error(
                "source_context_mismatch",
                "division series must share variable and truncation order",
            )
        _require_zero_residual(
            self.residual_coefficients,
            self.quotient.truncation_order,
            "division",
        )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        numerator: TruncatedSeries,
        denominator: TruncatedSeries,
        quotient: TruncatedSeries,
        residual_coefficients: tuple[CanonicalRational, ...],
    ) -> Self:
        return cls.model_construct(
            numerator=numerator,
            denominator=denominator,
            quotient=quotient,
            residual_coefficients=residual_coefficients,
        )


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class SeriesComposeRequest(StrictModel):
    outer: TruncatedSeries
    inner: TruncatedSeries


class SeriesComposeResult(StrictModel):
    result: TruncatedSeries
    residual_congruence: Literal["EXACT_MOD_X_TO_N"] = "EXACT_MOD_X_TO_N"


# ---------------------------------------------------------------------------
# Reversion
# ---------------------------------------------------------------------------


class SeriesReversionRequest(StrictModel):
    """Compositional inverse of a series with F(0)=0 and F'(0) != 0."""

    variable: Variable
    truncation_order: StrictInt = Field(ge=2)
    coefficients: tuple[CanonicalRational, ...]

    def as_series(self) -> TruncatedSeries:
        return TruncatedSeries(
            variable=self.variable,
            truncation_order=self.truncation_order,
            coefficients=self.coefficients,
        )


class SeriesReversionResult(StrictModel):
    source: TruncatedSeries
    result: TruncatedSeries
    left_identity: Literal["F_OF_G_IS_X_MOD_X_TO_N"] = "F_OF_G_IS_X_MOD_X_TO_N"
    right_identity: Literal["G_OF_F_IS_X_MOD_X_TO_N"] = "G_OF_F_IS_X_MOD_X_TO_N"
    left_residual: tuple[CanonicalRational, ...]
    right_residual: tuple[CanonicalRational, ...]

    @model_validator(mode="after")
    def require_structural_residuals(self) -> Self:
        order = self.result.truncation_order
        if (
            self.source.variable != self.result.variable
            or self.source.truncation_order != order
        ):
            raise _validation_error(
                "source_context_mismatch",
                "reversion source and result must share variable and truncation order",
            )
        _require_zero_residual(self.left_residual, order, "left reversion")
        _require_zero_residual(self.right_residual, order, "right reversion")
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: TruncatedSeries,
        result: TruncatedSeries,
        left_residual: tuple[CanonicalRational, ...],
        right_residual: tuple[CanonicalRational, ...],
    ) -> Self:
        return cls.model_construct(
            source=source,
            result=result,
            left_residual=left_residual,
            right_residual=right_residual,
        )


# ---------------------------------------------------------------------------
# Derivative / integral
# ---------------------------------------------------------------------------


class SeriesDerivativeResult(StrictModel):
    result: TruncatedSeries
    output_order_convention: Literal["MAX_N_MINUS_1_AT_LEAST_1"] = (
        "MAX_N_MINUS_1_AT_LEAST_1"
    )


class SeriesIntegralRequest(StrictModel):
    series: TruncatedSeries
    output_order: StrictInt = Field(ge=1)


class SeriesIntegralResult(StrictModel):
    result: TruncatedSeries


# ---------------------------------------------------------------------------
# Truncate
# ---------------------------------------------------------------------------


class SeriesTruncateRequest(StrictModel):
    """Extract a prefix of at most the public order bound from one series."""

    series: TruncatedSeries = Field(
        description=(
            "Source series whose every coefficient request admission "
            "validates before only the first target_order entries are read."
        ),
    )
    target_order: StrictInt = Field(ge=1)


class SeriesTruncateResult(StrictModel):
    result: TruncatedSeries


# ---------------------------------------------------------------------------
# Identity check
# ---------------------------------------------------------------------------


class SeriesIdentityCheckResult(StrictModel):
    status: Literal["EQUAL_MOD_X_TO_N", "NOT_EQUAL"]
    first_differing_index: StrictInt | None = None
    exact_difference: CanonicalRational | None = None

    @model_validator(mode="after")
    def require_consistent_diff(self) -> Self:
        if self.status == "EQUAL_MOD_X_TO_N":
            if (
                self.first_differing_index is not None
                or self.exact_difference is not None
            ):
                raise _validation_error(
                    "equal_carries_difference", "EQUAL must not carry a difference"
                )
        else:
            if self.first_differing_index is None or self.exact_difference is None:
                raise _validation_error(
                    "not_equal_missing_difference",
                    "NOT_EQUAL must carry a difference",
                )
        return self


# ---------------------------------------------------------------------------
# Polynomial conversions
# ---------------------------------------------------------------------------


class SeriesFromPolynomialRequest(StrictModel):
    """The explicit QQ[x] to QQ[[x]]/(x^N) projection."""

    polynomial: RationalPolynomial
    truncation_order: StrictInt = Field(ge=1)


class SeriesFromPolynomialResult(StrictModel):
    source: RationalPolynomial
    result: TruncatedSeries

    @model_validator(mode="after")
    def require_same_variable(self) -> Self:
        if self.source.variables != (self.result.variable,):
            raise _validation_error(
                "conversion_variable", "projection must retain the polynomial variable"
            )
        return self


class SeriesToPolynomialResult(StrictModel):
    source: TruncatedSeries
    result: RationalPolynomial
    polynomial_label: Literal["TRUNCATED_POLYNOMIAL_REPRESENTATIVE"] = (
        "TRUNCATED_POLYNOMIAL_REPRESENTATIVE"
    )

    @model_validator(mode="after")
    def require_same_variable(self) -> Self:
        if self.result.variables != (self.source.variable,):
            raise _validation_error(
                "conversion_variable", "representative must retain the series variable"
            )
        return self
