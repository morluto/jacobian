"""Wire contracts for exact truncated formal power series operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

# ---------------------------------------------------------------------------
# Public bounds
# ---------------------------------------------------------------------------

MAX_TRUNCATION_ORDER = 512
MAX_RATIONAL_DIGITS = 256
MAX_RESULT_RATIONAL_DIGITS = 4_096
MAX_RESULT_BYTES = 10 * 1024 * 1024
MAX_POWER_EXPONENT = 1_000

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
        le=MAX_TRUNCATION_ORDER,
        description="Truncation order N (coefficients a_0..a_{N-1}).",
    )
    coefficients: tuple[CanonicalRational, ...] = Field(
        description="Exactly N rational coefficients in ascending powers.",
    )

    @model_validator(mode="after")
    def require_dense_tuple(self) -> Self:
        if len(self.coefficients) != self.truncation_order:
            raise ValueError(
                "coefficient tuple must have exactly truncation_order entries"
            )
        for value in self.coefficients:
            require_bounded_rational(
                value,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label="coefficient",
            )
        return self


class InputTruncatedSeries(TruncatedSeries):
    """A truncated series admitted as an operation input."""

    @model_validator(mode="after")
    def require_input_digit_bound(self) -> Self:
        for value in self.coefficients:
            require_bounded_rational(
                value,
                max_digits=MAX_RATIONAL_DIGITS,
                label="input coefficient",
            )
        return self


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
            raise ValueError("operands must share the same variable")
        if self.left.truncation_order != self.right.truncation_order:
            raise ValueError("operands must share the same truncation order")
        return self


class SeriesDivideRequest(_SeriesPairRequest):
    """Divide two series when the denominator is a unit."""

    @model_validator(mode="after")
    def require_unit_denominator(self) -> Self:
        if self.right.coefficients[0].as_fraction() == 0:
            raise ValueError("denominator must have a nonzero constant term")
        return self


# ---------------------------------------------------------------------------
# Arithmetic: add / subtract / multiply / scalar multiply
# ---------------------------------------------------------------------------


class SeriesArithmeticResult(StrictModel):
    result: TruncatedSeries
    residual_congruence: Literal["EXACT_MOD_X_TO_N"] = "EXACT_MOD_X_TO_N"
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"


class SeriesMultiplyResult(StrictModel):
    result: TruncatedSeries
    convolution_ledger: tuple[CanonicalRational, ...] = Field(
        description="Per-degree Cauchy convolution sums c_n = sum_{i=0}^n a_i b_{n-i}.",
    )
    residual_congruence: Literal["EXACT_MOD_X_TO_N"] = "EXACT_MOD_X_TO_N"
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"


class SeriesScalarMultiplyRequest(StrictModel):
    series: InputTruncatedSeries
    scalar: CanonicalRational


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
        if self.exponent == 0:
            return self
        for coefficient in self.series.coefficients:
            digits = max(len(coefficient.num.lstrip("-")), len(coefficient.den))
            if digits * self.exponent > MAX_RESULT_RATIONAL_DIGITS:
                raise ValueError(
                    "power would exceed the 4096-digit result coefficient bound"
                )
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
            raise ValueError("inverse requires a nonzero constant term")
        # In the recurrence b_n = -a_0^-1 * sum(a_i * b_{n-i}), each path
        # contributes at most one reciprocal and one input coefficient per
        # degree.  Two input digit widths cover numerator and denominator
        # growth; one extra digit per degree covers the recurrence sum.
        maximum_input_digits = max(
            max(len(coefficient.num.lstrip("-")), len(coefficient.den))
            for coefficient in series.coefficients
        )
        if (2 * maximum_input_digits + 1) * self.truncation_order > (
            MAX_RESULT_RATIONAL_DIGITS
        ):
            raise ValueError(
                "inverse coefficient growth would exceed the "
                f"{MAX_RESULT_RATIONAL_DIGITS}-digit result bound"
            )
        return self

    def as_series(self) -> InputTruncatedSeries:
        return InputTruncatedSeries(
            variable=self.variable,
            truncation_order=self.truncation_order,
            coefficients=self.coefficients,
        )


class SeriesInverseResult(StrictModel):
    result: TruncatedSeries
    residual_congruence: Literal["PRODUCT_IS_ONE_MOD_X_TO_N"] = (
        "PRODUCT_IS_ONE_MOD_X_TO_N"
    )
    residual_coefficients: tuple[CanonicalRational, ...] = Field(
        description="A(x) * B(x) - 1 coefficients (must all be zero).",
    )
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"


# ---------------------------------------------------------------------------
# Divide
# ---------------------------------------------------------------------------


class SeriesDivideResult(StrictModel):
    quotient: TruncatedSeries
    residual_congruence: Literal[
        "DENOMINATOR_TIMES_QUOTIENT_MINUS_NUMERATOR_IS_ZERO_MOD_X_TO_N"
    ] = "DENOMINATOR_TIMES_QUOTIENT_MINUS_NUMERATOR_IS_ZERO_MOD_X_TO_N"
    residual_coefficients: tuple[CanonicalRational, ...] = Field(
        description="B(x) Q(x) - A(x) coefficients (must all be zero).",
    )
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class SeriesComposeRequest(StrictModel):
    outer: InputTruncatedSeries
    inner: InputTruncatedSeries

    @model_validator(mode="after")
    def require_matching_variable_and_zero_inner_constant(self) -> Self:
        if self.outer.variable != self.inner.variable:
            raise ValueError("outer and inner series must share the same variable")
        if self.outer.truncation_order != self.inner.truncation_order:
            raise ValueError(
                "outer and inner series must share the same truncation order"
            )
        if self.inner.coefficients[0].as_fraction() != 0:
            raise ValueError(
                "inner series must have zero constant term for composition with a finite prefix"
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
            raise ValueError("reversion requires zero constant term")
        if series.coefficients[1].as_fraction() == 0:
            raise ValueError("reversion requires nonzero linear coefficient")
        return self

    def as_series(self) -> InputTruncatedSeries:
        return InputTruncatedSeries(
            variable=self.variable,
            truncation_order=self.truncation_order,
            coefficients=self.coefficients,
        )


class SeriesReversionResult(StrictModel):
    result: TruncatedSeries
    left_identity: Literal["F_OF_G_IS_X_MOD_X_TO_N"] = "F_OF_G_IS_X_MOD_X_TO_N"
    right_identity: Literal["G_OF_F_IS_X_MOD_X_TO_N"] = "G_OF_F_IS_X_MOD_X_TO_N"
    left_residual: tuple[CanonicalRational, ...]
    right_residual: tuple[CanonicalRational, ...]
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"


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
            raise ValueError("output_order must not exceed source_order + 1")
        return self


class SeriesIntegralResult(StrictModel):
    result: TruncatedSeries
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"


# ---------------------------------------------------------------------------
# Truncate
# ---------------------------------------------------------------------------


class SeriesTruncateRequest(StrictModel):
    series: InputTruncatedSeries
    target_order: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def require_target_le_source(self) -> Self:
        if self.target_order > self.series.truncation_order:
            raise ValueError("target_order must not exceed source truncation order")
        if self.target_order > MAX_TRUNCATION_ORDER:
            raise ValueError("target_order exceeds the public bound")
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
                raise ValueError("EQUAL must not carry a difference")
        else:
            if self.first_differing_index is None or self.exact_difference is None:
                raise ValueError("NOT_EQUAL must carry a difference")
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
            raise ValueError("input coefficients must match truncation_order exactly")
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
