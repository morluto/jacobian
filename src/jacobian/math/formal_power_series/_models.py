"""Wire contracts for exact truncated formal power series operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math._rational_height import RationalHeight, sum_heights

# ---------------------------------------------------------------------------
# Public bounds
# ---------------------------------------------------------------------------

MAX_TRUNCATION_ORDER = 512
MAX_RATIONAL_DIGITS = 256
MAX_RESULT_RATIONAL_DIGITS = 4_096
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

# Replaying result validators recompute their defining invariants during
# deserialization at quadratic or cubic cost in the claimed order.  Every
# producer of these results emits order equal to its admitted request
# inputs, which cap at MAX_TRUNCATION_ORDER, so payloads claiming larger
# orders are outside the representable result envelope and must be
# rejected before any replay work starts.
MAX_RESULT_REPLAY_ORDER = MAX_TRUNCATION_ORDER

CoefficientHeight = RationalHeight | None


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by formal-series contracts."""

    return PydanticCustomError(f"formal_power_series.{reason}", message)


def _height(value: CanonicalRational) -> RationalHeight:
    return RationalHeight.from_canonical(value)


def _coefficient_height(value: CanonicalRational) -> CoefficientHeight:
    return None if value.num == "0" else _height(value)


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
        raise _validation_error(
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


def _merge_max(left: RationalHeight, right: RationalHeight) -> RationalHeight:
    return RationalHeight(
        max(left.numerator_digits, right.numerator_digits),
        max(left.denominator_digits, right.denominator_digits),
    )


def _convolution_height(
    left: RationalHeight, right: RationalHeight, term_count: int
) -> RationalHeight:
    term = left.product(right)
    return sum_heights(term for _ in range(term_count))


def _require_height(height: RationalHeight, operation: str) -> None:
    if height.exceeds(MAX_RESULT_RATIONAL_DIGITS):
        raise _validation_error(
            f"{operation}_coefficient_growth",
            f"{operation} coefficient growth exceeds the "
            f"{MAX_RESULT_RATIONAL_DIGITS}-digit result bound",
        )


def _require_replay_order(order: int, operation: str) -> None:
    if order > MAX_RESULT_REPLAY_ORDER:
        raise _validation_error(
            f"{operation}_replay_order",
            f"{operation} result replay exceeds the "
            f"{MAX_RESULT_REPLAY_ORDER}-order replay envelope",
        )


def _require_zero_residual(
    coefficients: tuple[CanonicalRational, ...], order: int, operation: str
) -> None:
    if len(coefficients) != order:
        raise _validation_error(
            f"{operation}_residual_length",
            f"{operation} residual must contain exactly {order} coefficients",
        )
    if any(value.num != "0" for value in coefficients):
        raise _validation_error(
            f"{operation}_residual_nonzero",
            f"{operation} residual must be identically zero",
        )


def _fraction_coefficients(series: TruncatedSeries) -> tuple[Fraction, ...]:
    return tuple(value.as_fraction() for value in series.coefficients)


def _convolution_coefficients(
    left: TruncatedSeries, right: TruncatedSeries
) -> tuple[Fraction, ...]:
    if (
        left.variable != right.variable
        or left.truncation_order != right.truncation_order
    ):
        raise _validation_error(
            "source_context_mismatch",
            "source series must share variable and truncation order",
        )
    left_values = _fraction_coefficients(left)
    right_values = _fraction_coefficients(right)
    return tuple(
        sum(
            (
                left_values[index] * right_values[degree - index]
                for index in range(degree + 1)
            ),
            start=Fraction(0),
        )
        for degree in range(left.truncation_order)
    )


def _composition_coefficients(
    outer: TruncatedSeries, inner: TruncatedSeries
) -> tuple[Fraction, ...]:
    if (
        outer.variable != inner.variable
        or outer.truncation_order != inner.truncation_order
    ):
        raise _validation_error(
            "source_context_mismatch",
            "source series must share variable and truncation order",
        )
    order = outer.truncation_order
    outer_values = _fraction_coefficients(outer)
    inner_values = _fraction_coefficients(inner)
    power = (Fraction(1), *([Fraction(0)] * (order - 1)))
    result = [Fraction(0)] * order
    for outer_degree in range(order):
        for degree in range(order):
            result[degree] += outer_values[outer_degree] * power[degree]
        if outer_degree + 1 < order:
            power = tuple(
                sum(
                    (
                        power[index] * inner_values[degree - index]
                        for index in range(degree + 1)
                    ),
                    start=Fraction(0),
                )
                for degree in range(order)
            )
    return tuple(result)


def _inverse_height(series: InputTruncatedSeries) -> RationalHeight:
    source = _max_height(series.coefficients)
    reciprocal = RationalHeight(1, 1).quotient(_height(series.coefficients[0]))
    _require_height(reciprocal, "inverse")
    result = reciprocal
    for degree in range(1, series.truncation_order):
        recurrence_sum = sum_heights(source.product(result) for _ in range(degree))
        coefficient = reciprocal.product(recurrence_sum)
        _require_height(coefficient, "inverse")
        result = _merge_max(result, coefficient)
    return result


def _composition_height(
    outer: RationalHeight, inner: RationalHeight, order: int, operation: str
) -> RationalHeight:
    power = RationalHeight(1, 1)
    result = outer.product(power)
    for _ in range(1, order):
        power = _convolution_height(power, inner, order)
        _require_height(power, operation)
        result = sum_heights((result, outer.product(power)))
        _require_height(result, operation)
    return result


Variable = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-z][a-z0-9]*$",
        max_length=16,
        strict=True,
    ),
]


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
        for value in self.coefficients:
            require_bounded_rational(
                value,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label="coefficient",
            )
        return self


class InputTruncatedSeries(TruncatedSeries):
    """A truncated series admitted as an operation input.

    Operation kernels do work that grows with the truncation order, so the
    input envelope keeps the shared order ceiling that the pure value carrier
    does not impose.
    """

    truncation_order: StrictInt = Field(
        ge=1,
        le=MAX_TRUNCATION_ORDER,
        description=(
            "Truncation order N (coefficients a_0..a_{N-1}); bounded because "
            "operation work scales with N."
        ),
    )

    @model_validator(mode="after")
    def require_input_digit_bound(self) -> Self:
        for value in self.coefficients:
            require_bounded_rational(
                value,
                max_digits=MAX_RATIONAL_DIGITS,
                label="input coefficient",
            )
        return self


def as_input_series(series: TruncatedSeries) -> InputTruncatedSeries:
    """Re-admit one carrier value through the bounded operation input envelope.

    Native callers pass the shared carrier directly, so the native execution
    path proves the same truncation-order and input-digit admission that wire
    requests prove before any kernel work starts.
    """
    return InputTruncatedSeries(
        variable=series.variable,
        truncation_order=series.truncation_order,
        coefficients=series.coefficients,
    )


class TruncateSourceSeries(TruncatedSeries):
    """A truncated series admitted as a truncation source.

    Request admission materializes and height-validates every source
    coefficient before only the requested prefix is read, so admission work
    and memory scale with the source order.  The order ceiling admits every
    carrier current producers emit while bounding that admission; the kernel
    itself still touches only the first ``target_order`` coefficients.
    """

    truncation_order: StrictInt = Field(
        ge=1,
        le=MAX_TRUNCATE_SOURCE_ORDER,
        description=(
            "Source truncation order N (coefficients a_0..a_{N-1}); bounded "
            "because request admission validates all N coefficients before "
            "the prefix is read."
        ),
    )


# ---------------------------------------------------------------------------
# Pair / single-series request helpers
# ---------------------------------------------------------------------------


class _SeriesPairRequest(StrictModel):
    """Base request with two series that must share variable and order."""

    left: InputTruncatedSeries
    right: InputTruncatedSeries

    @model_validator(mode="after")
    def require_matching_context(self) -> Self:
        if self.left.variable != self.right.variable:
            raise _validation_error(
                "operand_variable_mismatch", "operands must share the same variable"
            )
        if self.left.truncation_order != self.right.truncation_order:
            raise _validation_error(
                "operand_order_mismatch",
                "operands must share the same truncation order",
            )
        return self


class _SeriesAddSubtractRequest(_SeriesPairRequest):
    @model_validator(mode="after")
    def require_bounded_result_height(self) -> Self:
        for left, right in zip(
            self.left.coefficients, self.right.coefficients, strict=True
        ):
            _require_height(sum_heights((_height(left), _height(right))), "sum")
        return self


class _SeriesMultiplyRequest(_SeriesPairRequest):
    @model_validator(mode="after")
    def require_bounded_result_height(self) -> Self:
        left = _max_height(self.left.coefficients)
        right = _max_height(self.right.coefficients)
        _require_height(
            _convolution_height(left, right, self.left.truncation_order),
            "multiplication",
        )
        return self


class _SeriesIdentityCheckRequest(_SeriesPairRequest):
    """Admit one identity check through its own linear work envelope.

    The kernel compares N coefficient pairs and emits at most one pairwise
    difference ``a_i - b_i``, so admission charges the coefficientwise
    comparison and bounds that difference height; it never charges the
    unrelated Cauchy-convolution growth that multiplication must preflight.
    """

    @model_validator(mode="after")
    def require_bounded_difference_height(self) -> Self:
        for left, right in zip(
            self.left.coefficients, self.right.coefficients, strict=True
        ):
            _require_height(
                sum_heights((_height(left), _height(right))),
                "difference",
            )
        return self


class SeriesDivideRequest(_SeriesPairRequest):
    """Divide two series when the denominator is a unit."""

    @model_validator(mode="after")
    def require_unit_denominator(self) -> Self:
        if self.right.coefficients[0].as_fraction() == 0:
            raise _validation_error(
                "denominator_zero_constant",
                "denominator must have a nonzero constant term",
            )
        return self

    @model_validator(mode="after")
    def require_bounded_result_height(self) -> Self:
        inverse = _inverse_height(self.right)
        quotient = _convolution_height(
            _max_height(self.left.coefficients),
            inverse,
            self.left.truncation_order,
        )
        _require_height(quotient, "division")
        residual = _convolution_height(
            _max_height(self.right.coefficients),
            quotient,
            self.left.truncation_order,
        )
        _require_height(
            sum_heights((residual, _max_height(self.left.coefficients))),
            "division residual",
        )
        return self


# ---------------------------------------------------------------------------
# Arithmetic: add / subtract / multiply / scalar multiply
# ---------------------------------------------------------------------------


class SeriesArithmeticResult(StrictModel):
    result: TruncatedSeries
    residual_congruence: Literal["EXACT_MOD_X_TO_N"] = "EXACT_MOD_X_TO_N"
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"


class SeriesMultiplyResult(StrictModel):
    left: TruncatedSeries
    right: TruncatedSeries
    result: TruncatedSeries
    convolution_ledger: tuple[CanonicalRational, ...] = Field(
        description="Per-degree Cauchy convolution sums c_n = sum_{i=0}^n a_i b_{n-i}.",
    )
    residual_congruence: Literal["EXACT_MOD_X_TO_N"] = "EXACT_MOD_X_TO_N"
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"

    @model_validator(mode="after")
    def require_exact_convolution_ledger(self) -> Self:
        _require_replay_order(self.result.truncation_order, "multiplication")
        expected = _convolution_coefficients(self.left, self.right)
        if _fraction_coefficients(self.result) != expected:
            raise _validation_error(
                "convolution_result_mismatch",
                "result must equal the source convolution",
            )
        if tuple(value.as_fraction() for value in self.convolution_ledger) != expected:
            raise _validation_error(
                "convolution_ledger_mismatch",
                "convolution ledger must equal the source convolution",
            )
        return self


class SeriesScalarMultiplyRequest(StrictModel):
    series: InputTruncatedSeries
    scalar: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_result_height(self) -> Self:
        _require_height(
            _max_height(self.series.coefficients).product(_height(self.scalar)),
            "scalar multiplication",
        )
        return self


class SeriesScalarMultiplyResult(StrictModel):
    result: TruncatedSeries
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


class SeriesPowerRequest(StrictModel):
    series: InputTruncatedSeries
    exponent: StrictInt = Field(ge=0, le=MAX_POWER_EXPONENT)

    @model_validator(mode="after")
    def require_result_digit_budget(self) -> Self:
        order = self.series.truncation_order
        result = RationalHeight(1, 1)
        base = _max_height(self.series.coefficients)
        exponent = self.exponent
        while exponent > 0:
            if exponent & 1:
                result = _convolution_height(result, base, order)
                _require_height(result, "power")
            exponent >>= 1
            if exponent:
                base = _convolution_height(base, base, order)
                _require_height(base, "power")
        return self


class SeriesPowerResult(StrictModel):
    result: TruncatedSeries
    multiplication_count: StrictInt
    residual_congruence: Literal["EXACT_MOD_X_TO_N"] = "EXACT_MOD_X_TO_N"
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"


# ---------------------------------------------------------------------------
# Inverse
# ---------------------------------------------------------------------------


class SeriesInverseRequest(StrictModel):
    """Invert a truncated series that is a unit (nonzero constant term)."""

    variable: Variable = Field(description="The single formal variable.")
    truncation_order: StrictInt = Field(
        ge=1,
        le=MAX_TRUNCATION_ORDER,
        description=(
            "Truncation order N; the inverse growth budget must fit every "
            "returned coefficient in the 4096-digit result bound."
        ),
    )
    coefficients: tuple[CanonicalRational, ...] = Field(
        description="Exactly N rational coefficients with a nonzero constant term.",
    )

    @model_validator(mode="after")
    def require_unit_constant(self) -> Self:
        series = InputTruncatedSeries(
            variable=self.variable,
            truncation_order=self.truncation_order,
            coefficients=self.coefficients,
        )
        if series.coefficients[0].as_fraction() == 0:
            raise _validation_error(
                "inverse_zero_constant", "inverse requires a nonzero constant term"
            )
        inverse = _inverse_height(series)
        residual = _convolution_height(
            _max_height(series.coefficients), inverse, self.truncation_order
        )
        _require_height(residual, "inverse residual")
        return self

    def as_series(self) -> InputTruncatedSeries:
        return InputTruncatedSeries(
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
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"

    @model_validator(mode="after")
    def require_inverse_identity(self) -> Self:
        _require_replay_order(self.result.truncation_order, "inverse")
        _require_zero_residual(
            self.residual_coefficients,
            self.result.truncation_order,
            "inverse",
        )
        product = list(_convolution_coefficients(self.source, self.result))
        product[0] -= 1
        residual = tuple(value.as_fraction() for value in self.residual_coefficients)
        if tuple(product) != residual:
            raise _validation_error(
                "inverse_residual_mismatch",
                "inverse residual must equal source times result minus one",
            )
        return self


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
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"

    @model_validator(mode="after")
    def require_division_identity(self) -> Self:
        _require_replay_order(self.quotient.truncation_order, "division")
        _require_zero_residual(
            self.residual_coefficients,
            self.quotient.truncation_order,
            "division",
        )
        product = _convolution_coefficients(self.denominator, self.quotient)
        numerator = _fraction_coefficients(self.numerator)
        expected = tuple(
            product[index] - numerator[index]
            for index in range(self.quotient.truncation_order)
        )
        residual = tuple(value.as_fraction() for value in self.residual_coefficients)
        if expected != residual:
            raise _validation_error(
                "division_residual_mismatch",
                "division residual must equal denominator times quotient minus numerator",
            )
        return self


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class SeriesComposeRequest(StrictModel):
    outer: InputTruncatedSeries
    inner: InputTruncatedSeries

    @model_validator(mode="after")
    def require_matching_variable_and_zero_inner_constant(self) -> Self:
        if self.outer.variable != self.inner.variable:
            raise _validation_error(
                "composition_variable_mismatch",
                "outer and inner series must share the same variable",
            )
        if self.outer.truncation_order != self.inner.truncation_order:
            raise _validation_error(
                "composition_order_mismatch",
                "outer and inner series must share the same truncation order",
            )
        if self.inner.coefficients[0].as_fraction() != 0:
            raise _validation_error(
                "composition_nonzero_inner_constant",
                "inner series must have zero constant term for composition with a finite prefix",
            )
        _composition_height_vector(
            _height_vector(self.outer.coefficients),
            _height_vector(self.inner.coefficients),
            self.outer.truncation_order,
            "composition",
        )
        return self


class SeriesComposeResult(StrictModel):
    result: TruncatedSeries
    residual_congruence: Literal["EXACT_MOD_X_TO_N"] = "EXACT_MOD_X_TO_N"
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"


# ---------------------------------------------------------------------------
# Reversion
# ---------------------------------------------------------------------------


class SeriesReversionRequest(StrictModel):
    """Compositional inverse of a series with F(0)=0 and F'(0) != 0."""

    variable: Variable
    truncation_order: StrictInt = Field(ge=2, le=MAX_TRUNCATION_ORDER)
    coefficients: tuple[CanonicalRational, ...]

    @model_validator(mode="after")
    def require_reversion_hypotheses(self) -> Self:
        series = InputTruncatedSeries(
            variable=self.variable,
            truncation_order=self.truncation_order,
            coefficients=self.coefficients,
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
        source = _height_vector(series.coefficients)
        linear = _height(series.coefficients[1])
        result: list[CoefficientHeight] = [
            None,
            RationalHeight(1, 1).quotient(linear),
        ]
        _require_height_vector(tuple(result), "reversion")
        for degree in range(2, self.truncation_order):
            padded = (*result, None)
            power = padded
            terms: list[RationalHeight] = []
            for source_degree in range(2, degree + 1):
                power = _convolve_height_vectors(power, padded, degree + 1, "reversion")
                source_height = source[source_degree]
                power_height = power[degree]
                if source_height is not None and power_height is not None:
                    terms.append(source_height.product(power_height))
            known = sum_heights(terms)
            coefficient = known.quotient(linear)
            _require_height(coefficient, "reversion")
            result.append(coefficient)
        result_vector = tuple(result)
        _composition_height_vector(
            source, result_vector, self.truncation_order, "reversion residual"
        )
        _composition_height_vector(
            result_vector, source, self.truncation_order, "reversion residual"
        )
        return self

    def as_series(self) -> InputTruncatedSeries:
        return InputTruncatedSeries(
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
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"

    @model_validator(mode="after")
    def require_zero_residuals(self) -> Self:
        order = self.result.truncation_order
        _require_replay_order(order, "reversion")
        _require_zero_residual(self.left_residual, order, "left reversion")
        _require_zero_residual(self.right_residual, order, "right reversion")
        identity = tuple(Fraction(int(index == 1)) for index in range(order))
        left = _composition_coefficients(self.source, self.result)
        right = _composition_coefficients(self.result, self.source)
        left_residual = tuple(value.as_fraction() for value in self.left_residual)
        right_residual = tuple(value.as_fraction() for value in self.right_residual)
        if (
            tuple(left[index] - identity[index] for index in range(order))
            != left_residual
        ):
            raise _validation_error(
                "reversion_left_residual_mismatch",
                "left residual must equal source composed with result minus x",
            )
        if (
            tuple(right[index] - identity[index] for index in range(order))
            != right_residual
        ):
            raise _validation_error(
                "reversion_right_residual_mismatch",
                "right residual must equal result composed with source minus x",
            )
        return self


# ---------------------------------------------------------------------------
# Derivative / integral
# ---------------------------------------------------------------------------


class SeriesDerivativeResult(StrictModel):
    result: TruncatedSeries
    output_order_convention: Literal["MAX_N_MINUS_1_AT_LEAST_1"] = (
        "MAX_N_MINUS_1_AT_LEAST_1"
    )
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"


class SeriesIntegralRequest(StrictModel):
    series: InputTruncatedSeries
    output_order: StrictInt = Field(ge=1, le=MAX_TRUNCATION_ORDER)

    @model_validator(mode="after")
    def require_output_order_in_range(self) -> Self:
        if self.output_order > self.series.truncation_order + 1:
            raise _validation_error(
                "integral_output_order_exceeds_source",
                "output_order must not exceed source_order + 1",
            )
        integer = RationalHeight(len(str(max(1, self.output_order - 1))), 1)
        _require_height(
            _max_height(self.series.coefficients).quotient(integer), "integration"
        )
        return self


class SeriesIntegralResult(StrictModel):
    result: TruncatedSeries
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"


# ---------------------------------------------------------------------------
# Truncate
# ---------------------------------------------------------------------------


class SeriesTruncateRequest(StrictModel):
    """Extract a prefix of at most the public order bound from one series."""

    series: TruncateSourceSeries = Field(
        description=(
            "Source series whose every coefficient request admission "
            "validates before only the first target_order entries are read."
        ),
    )
    target_order: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def require_target_le_source(self) -> Self:
        if self.target_order > self.series.truncation_order:
            raise _validation_error(
                "truncate_target_exceeds_source",
                "target_order must not exceed source truncation order",
            )
        if self.target_order > MAX_TRUNCATION_ORDER:
            raise _validation_error(
                "truncate_target_exceeds_public_bound",
                "target_order exceeds the public bound",
            )
        return self


class SeriesTruncateResult(StrictModel):
    result: TruncatedSeries
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"


# ---------------------------------------------------------------------------
# Identity check
# ---------------------------------------------------------------------------


class SeriesIdentityCheckResult(StrictModel):
    status: Literal["EQUAL_MOD_X_TO_N", "NOT_EQUAL"]
    first_differing_index: StrictInt | None = None
    exact_difference: CanonicalRational | None = None
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"

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
    """Convert a dense rational polynomial coefficient prefix into a series."""

    variable: Variable
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_TRUNCATION_ORDER,
    )
    truncation_order: StrictInt = Field(ge=1, le=MAX_TRUNCATION_ORDER)

    @model_validator(mode="after")
    def require_dense_tuple(self) -> Self:
        if len(self.coefficients) != self.truncation_order:
            raise _validation_error(
                "input_coefficient_count_mismatch",
                "input coefficients must match truncation_order exactly",
            )
        for value in self.coefficients:
            require_bounded_rational(
                value,
                max_digits=MAX_RATIONAL_DIGITS,
                label="input coefficient",
            )
        return self


class SeriesFromPolynomialResult(StrictModel):
    result: TruncatedSeries
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"


class SeriesToPolynomialResult(StrictModel):
    result: TruncatedSeries
    polynomial_label: Literal["TRUNCATED_POLYNOMIAL_REPRESENTATIVE"] = (
        "TRUNCATED_POLYNOMIAL_REPRESENTATIVE"
    )
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
