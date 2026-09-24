"""Typed wire contracts for exact shift Ore operators over QQ(n)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.number_theory.sequences.core._models import FiniteRationalSequence
from jacobian.math.polynomials.values import RationalFunction

MAX_SHIFT_ORDER = 16
MAX_SHIFT_TERMS = 16
MAX_SHIFT_COEFFICIENT_TERMS = 64
MAX_SHIFT_COEFFICIENT_DEGREE = 64
MAX_SHIFT_COEFFICIENT_DIGITS = 64
MAX_SHIFT_RESULT_ORDER = 32
MAX_SHIFT_RESULT_DEGREE = 128
MAX_SHIFT_RESULT_DIGITS = 256
MAX_SHIFT_LEDGER_ROWS = 256
MAX_DIFFERENTIAL_ORDER = 16
MAX_DIFFERENTIAL_TERMS = 32
MAX_DIFFERENTIAL_VARIABLE = "x"
MAX_DIFFERENTIAL_ADDITIVE_WORK_CELLS = 100_000_000
MAX_DIFFERENTIAL_ADDITIVE_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_DFINITE_PREFIX_COUNT = 100_000
MAX_DFINITE_PREFIX_WORK_UNITS = 100_000_000
MAX_DFINITE_PREFIX_OUTPUT_BYTES = 12 * 1024 * 1024
MAX_DFINITE_PREFIX_SCALAR_BITS = 131_072
MAX_SHIFT_PREFIX_INDEX = 100_000
MAX_SHIFT_PREFIX_EVALUATION_CELLS = 2_000_000
MAX_SHIFT_PREFIX_WORK_UNITS = 100_000_000
MAX_SHIFT_PREFIX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_SHIFT_ADDITIVE_WORK_CELLS = 4_096
MAX_SHIFT_ADDITIVE_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_SHIFT_POWER_EXPONENT = 16
MAX_SHIFT_POWER_WORK_CELLS = 4_096
MAX_RECURRENCE_PREFIX_STEPS = 512
MAX_RECURRENCE_PREFIX_INDEX = 100_000
MAX_RECURRENCE_PREFIX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_RECURRENCE_PREFIX_WORK_CELLS = 1_000_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by Ore-algebra contracts."""

    return PydanticCustomError(f"ore_algebra.{reason}", message)


def _is_polynomial_coefficient(value: RationalFunction) -> bool:
    """Return whether a QQ(n) value lies in the polynomial subring QQ[n]."""
    return (
        value.variables == ("n",)
        and len(value.denominator.terms) == 1
        and value.denominator.terms[0].exponents == (0,)
        and value.denominator.terms[0].coefficient.as_fraction() == 1
    )


class DifferentialOreTerm(StrictModel):
    """One left-coefficient monomial a(x) D^order."""

    order: StrictInt = Field(ge=0, le=MAX_DIFFERENTIAL_ORDER)
    coefficient: RationalFunction

    @model_validator(mode="after")
    def require_axis(self) -> Self:
        if (
            self.coefficient.variables != (MAX_DIFFERENTIAL_VARIABLE,)
            or not self.coefficient.numerator.terms
        ):
            raise _validation_error(
                "differential_coefficient",
                "differential coefficients must be nonzero rational functions in QQ(x)",
            )
        return self


class DifferentialOreOperator(StrictModel):
    variable: Literal["x"] = "x"
    terms: tuple[DifferentialOreTerm, ...] = Field(
        default=(), max_length=MAX_DIFFERENTIAL_TERMS
    )

    @model_validator(mode="after")
    def require_order(self) -> Self:
        orders = tuple(term.order for term in self.terms)
        if orders != tuple(sorted(orders)) or len(set(orders)) != len(orders):
            raise _validation_error(
                "differential_term_order",
                "differential terms must use strictly increasing orders",
            )
        return self

    @property
    def order(self) -> int:
        return max((term.order for term in self.terms), default=-1)


class DFinitePowerSeries(StrictModel):
    """A formal power series fixed by an ODE at the ordinary point x=0."""

    operator: DifferentialOreOperator
    initial_derivatives: FiniteRationalSequence = Field(
        description=(
            "Exact values f(0), f'(0), ..., f^(r-1)(0), where r is the "
            "differential-operator order."
        )
    )
    center: Literal[0] = 0

    @model_validator(mode="after")
    def require_ordinary_initial_value_problem(self) -> Self:
        if not self.operator.terms:
            raise _validation_error(
                "zero_differential_operator",
                "a D-finite series requires a nonzero differential operator",
            )
        if len(self.initial_derivatives.values) != self.operator.order:
            raise _validation_error(
                "differential_initial_data",
                "initial_derivatives must contain exactly order(operator) values",
            )
        leading = next(
            term.coefficient
            for term in self.operator.terms
            if term.order == self.operator.order
        )
        for term in self.operator.terms:
            denominator_at_zero = sum(
                (
                    coefficient.coefficient.as_fraction()
                    for coefficient in term.coefficient.denominator.terms
                    if coefficient.exponents == (0,)
                ),
                start=0,
            )
            if denominator_at_zero == 0:
                raise _validation_error(
                    "differential_coefficient_pole",
                    "every differential coefficient must be regular at x=0",
                )
        leading_at_zero = sum(
            (
                coefficient.coefficient.as_fraction()
                for coefficient in leading.numerator.terms
                if coefficient.exponents == (0,)
            ),
            start=0,
        ) / sum(
            (
                coefficient.coefficient.as_fraction()
                for coefficient in leading.denominator.terms
                if coefficient.exponents == (0,)
            ),
            start=0,
        )
        if leading_at_zero == 0:
            raise _validation_error(
                "differential_singular_center",
                "the leading differential coefficient must be nonzero at x=0",
            )
        return self


class DFinitePowerSeriesRequest(StrictModel):
    """Input data for an ordinary-point D-finite formal-series value."""

    operator: DifferentialOreOperator
    initial_derivatives: FiniteRationalSequence
    center: Literal[0] = 0


class DFinitePowerSeriesPrefixRequest(StrictModel):
    """Request the initial Taylor coefficients of an ordinary-point series."""

    series: DFinitePowerSeries
    count: StrictInt = Field(ge=0, le=MAX_DFINITE_PREFIX_COUNT)


class DifferentialOperatorMultiplyRequest(StrictModel):
    left: DifferentialOreOperator
    right: DifferentialOreOperator


class DifferentialOperatorAddRequest(StrictModel):
    """Two differential operators in the same QQ(x) algebra."""

    left: DifferentialOreOperator
    right: DifferentialOreOperator


class DifferentialOperatorAddResult(StrictModel):
    """Exact coefficientwise sum, in canonical sparse order."""

    left: DifferentialOreOperator
    right: DifferentialOreOperator
    sum: DifferentialOreOperator

    @classmethod
    def _from_kernel(
        cls,
        left: DifferentialOreOperator,
        right: DifferentialOreOperator,
        result: DifferentialOreOperator,
    ) -> Self:
        return cls.model_construct(left=left, right=right, sum=result)


class DifferentialOperatorNormalizeRequest(StrictModel):
    """Extract rational content from polynomial-coefficient differential operators."""

    operator: DifferentialOreOperator


class DifferentialOperatorNormalizeResult(StrictModel):
    operator: DifferentialOreOperator
    normalized: DifferentialOreOperator
    scale: RationalFunction

    @classmethod
    def _from_kernel(
        cls,
        operator: DifferentialOreOperator,
        normalized: DifferentialOreOperator,
        scale: RationalFunction,
    ) -> Self:
        return cls.model_construct(
            operator=operator, normalized=normalized, scale=scale
        )


class DifferentialOperatorApplyRequest(StrictModel):
    operator: DifferentialOreOperator
    function: RationalFunction

    @model_validator(mode="after")
    def require_axis(self) -> Self:
        if self.function.variables != (MAX_DIFFERENTIAL_VARIABLE,):
            raise _validation_error(
                "differential_function", "the applied function must use the QQ(x) axis"
            )
        return self


class DifferentialOperatorApplyResult(StrictModel):
    operator: DifferentialOreOperator
    function: RationalFunction
    result: RationalFunction

    @classmethod
    def _from_kernel(
        cls,
        operator: DifferentialOreOperator,
        function: RationalFunction,
        result: RationalFunction,
    ) -> Self:
        return cls.model_construct(operator=operator, function=function, result=result)


class DifferentialOperatorMultiplyResult(StrictModel):
    left: DifferentialOreOperator
    right: DifferentialOreOperator
    product: DifferentialOreOperator

    @classmethod
    def _from_kernel(
        cls,
        left: DifferentialOreOperator,
        right: DifferentialOreOperator,
        product: DifferentialOreOperator,
    ) -> Self:
        return cls.model_construct(left=left, right=right, product=product)


class ShiftOreTerm(StrictModel):
    """One nonzero left-coefficient monomial p(n) S^exponent."""

    exponent: StrictInt = Field(ge=0, le=MAX_SHIFT_ORDER)
    coefficient: RationalFunction = Field(
        description=(
            "Exact coefficient in QQ(n) with variable axis ('n',); "
            "it must be a nonzero reduced presentation."
        )
    )

    @model_validator(mode="after")
    def require_shift_coefficient_shape(self) -> Self:
        if self.coefficient.variables != ("n",):
            raise _validation_error(
                "coefficient_axis",
                "shift-operator coefficients must use the variable axis ('n',)",
            )
        if not self.coefficient.numerator.terms:
            raise _validation_error(
                "zero_coefficient", "operator terms must have nonzero coefficients"
            )
        return self


class ShiftOreOperator(StrictModel):
    """One shift operator in left-coefficient normal form sum_i p_i(n) S^i."""

    variable: Literal["n"] = "n"
    terms: tuple[ShiftOreTerm, ...] = Field(
        min_length=0,
        max_length=MAX_SHIFT_TERMS,
        description=(
            "Sparse operator terms in strictly increasing exponent order; "
            "the empty family is the canonical zero operator."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_term_order(self) -> Self:
        exponents = tuple(term.exponent for term in self.terms)
        if tuple(sorted(exponents)) != exponents or len(set(exponents)) != len(
            exponents
        ):
            raise _validation_error(
                "term_order",
                "operator terms must use strictly increasing exponent order",
            )
        return self

    @property
    def order(self) -> int:
        """Highest shift exponent, or -1 for the zero operator."""

        return max((term.exponent for term in self.terms), default=-1)


class ShiftOperatorMultiplyRequest(StrictModel):
    """Two shift operators over QQ(n) to multiply noncommutatively."""

    left: ShiftOreOperator
    right: ShiftOreOperator


class ShiftOperatorAddRequest(StrictModel):
    """Add shift operators in the polynomial-coefficient subalgebra QQ[n]<S>."""

    left: ShiftOreOperator = Field(
        description="Shift operator whose coefficients are polynomials in QQ[n]."
    )
    right: ShiftOreOperator = Field(
        description="Shift operator whose coefficients are polynomials in QQ[n]."
    )

    @model_validator(mode="after")
    def require_polynomial_coefficients(self) -> Self:
        if any(
            not _is_polynomial_coefficient(term.coefficient)
            for operator in (self.left, self.right)
            for term in operator.terms
        ):
            raise _validation_error(
                "polynomial_operator_coefficients",
                "shift addition currently accepts polynomial coefficients in QQ[n]",
            )
        return self


class ShiftOperatorAddResult(StrictModel):
    left: ShiftOreOperator
    right: ShiftOreOperator
    sum: ShiftOreOperator

    @classmethod
    def _from_kernel(
        cls,
        left: ShiftOreOperator,
        right: ShiftOreOperator,
        result: ShiftOreOperator,
    ) -> Self:
        return cls.model_construct(left=left, right=right, sum=result)


class ShiftOperatorScalarMultiplyRequest(StrictModel):
    """Left-scale a polynomial-coefficient shift operator by one value in QQ."""

    scalar: RationalFunction = Field(
        description="A canonical rational constant, represented on the QQ(n) axis."
    )
    operator: ShiftOreOperator = Field(
        description="Shift operator whose coefficients are polynomials in QQ[n]."
    )

    @model_validator(mode="after")
    def require_polynomial_operator_and_constant(self) -> Self:
        if (
            self.scalar.variables != ("n",)
            or any(term.exponents != (0,) for term in self.scalar.numerator.terms)
            or not _is_polynomial_coefficient(self.scalar)
            or any(
                not _is_polynomial_coefficient(term.coefficient)
                for term in self.operator.terms
            )
        ):
            raise _validation_error(
                "rational_constant_scalar",
                "left scaling requires a rational constant and polynomial QQ[n] operator coefficients",
            )
        return self


class ShiftOperatorScalarMultiplyResult(StrictModel):
    scalar: RationalFunction
    operator: ShiftOreOperator
    product: ShiftOreOperator

    @classmethod
    def _from_kernel(
        cls,
        scalar: RationalFunction,
        operator: ShiftOreOperator,
        result: ShiftOreOperator,
    ) -> Self:
        return cls.model_construct(scalar=scalar, operator=operator, product=result)


class ShiftOperatorNormalizeRequest(StrictModel):
    """Normalize rational scalar content in polynomial shift coefficients."""

    operator: ShiftOreOperator = Field(
        description="Shift operator whose coefficients are polynomials in QQ[n]."
    )

    @model_validator(mode="after")
    def require_polynomial_operator(self) -> Self:
        if any(
            not _is_polynomial_coefficient(term.coefficient)
            for term in self.operator.terms
        ):
            raise _validation_error(
                "polynomial_operator_coefficients",
                "normalization currently accepts polynomial coefficients in QQ[n]",
            )
        return self


class ShiftOperatorNormalizeResult(StrictModel):
    operator: ShiftOreOperator
    normalized: ShiftOreOperator
    scale: RationalFunction

    @classmethod
    def _from_kernel(
        cls,
        operator: ShiftOreOperator,
        normalized: ShiftOreOperator,
        scale: RationalFunction,
    ) -> Self:
        return cls.model_construct(
            operator=operator, normalized=normalized, scale=scale
        )


class ShiftOperatorPrefixRequest(StrictModel):
    """Evaluate finite residuals on values indexed from an explicit origin."""

    operator: ShiftOreOperator
    start_index: StrictInt
    sequence: FiniteRationalSequence


class ShiftOperatorPrefixContribution(StrictModel):
    exponent: StrictInt = Field(ge=0, le=MAX_SHIFT_ORDER)
    sequence_index: StrictInt
    coefficient: CanonicalRational
    sequence_value: CanonicalRational
    value: CanonicalRational


class ShiftOperatorPrefixResidual(StrictModel):
    index: StrictInt
    contributions: tuple[ShiftOperatorPrefixContribution, ...] = Field(
        max_length=MAX_SHIFT_TERMS
    )
    residual: CanonicalRational


class ShiftOperatorPrefixPoleExclusion(StrictModel):
    index: StrictInt
    exponents: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_SHIFT_TERMS)


class ShiftOperatorPrefixResult(StrictModel):
    operator: ShiftOreOperator
    start_index: StrictInt
    sequence: FiniteRationalSequence
    residuals: tuple[ShiftOperatorPrefixResidual, ...] = Field(max_length=100_000)
    coefficient_poles: tuple[ShiftOperatorPrefixPoleExclusion, ...] = Field(
        max_length=100_000
    )
    right_boundary_indices: tuple[StrictInt, ...] = Field(max_length=100_000)

    @classmethod
    def _from_kernel(
        cls,
        operator: ShiftOreOperator,
        start_index: int,
        sequence: FiniteRationalSequence,
        residuals: tuple[ShiftOperatorPrefixResidual, ...],
        coefficient_poles: tuple[ShiftOperatorPrefixPoleExclusion, ...],
        right_boundary_indices: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            operator=operator,
            start_index=start_index,
            sequence=sequence,
            residuals=residuals,
            coefficient_poles=coefficient_poles,
            right_boundary_indices=right_boundary_indices,
        )


class PolynomialRecurrencePrefixRequest(StrictModel):
    """Solve a polynomial recurrence on one explicitly finite index interval."""

    operator: ShiftOreOperator = Field(
        description=(
            "Polynomial-coefficient operator sum p_i(n) S^i. The coefficient "
            "of its greatest shift exponent must be nonzero at every index "
            "generated; the recurrence is asserted only on that finite range."
        )
    )
    start_index: StrictInt = Field(
        ge=-MAX_RECURRENCE_PREFIX_INDEX, le=MAX_RECURRENCE_PREFIX_INDEX
    )
    initial_values: FiniteRationalSequence = Field(
        description="Exactly order(operator) consecutive values starting at start_index."
    )
    steps: StrictInt = Field(ge=0, le=MAX_RECURRENCE_PREFIX_STEPS)

    @model_validator(mode="after")
    def require_recurrence_shape(self) -> Self:
        if not self.operator.terms:
            raise _validation_error(
                "zero_recurrence", "the zero operator does not define a recurrence"
            )
        if self.operator.order < 1:
            raise _validation_error(
                "positive_order",
                "finite recurrence generation requires positive shift order",
            )
        if any(
            not _is_polynomial_coefficient(term.coefficient)
            for term in self.operator.terms
        ):
            raise _validation_error(
                "polynomial_coefficients",
                "finite recurrence generation requires coefficients in QQ[n]",
            )
        order = self.operator.order
        if len(self.initial_values.values) != order:
            raise _validation_error(
                "initial_value_count",
                "initial_values must contain exactly order(operator) consecutive values",
            )
        if abs(self.start_index + self.steps + order) > MAX_RECURRENCE_PREFIX_INDEX:
            raise _validation_error(
                "index_range",
                "the generated recurrence interval exceeds its index envelope",
            )
        return self


class PolynomialRecurrencePrefix(StrictModel):
    """Finite sequence whose recurrence equations hold on the declared interval."""

    operator: ShiftOreOperator
    start_index: StrictInt
    recurrence_indices: tuple[StrictInt, ...]
    values: FiniteRationalSequence

    @classmethod
    def _from_kernel(
        cls,
        operator: ShiftOreOperator,
        start_index: int,
        recurrence_indices: tuple[int, ...],
        values: FiniteRationalSequence,
    ) -> Self:
        return cls.model_construct(
            operator=operator,
            start_index=start_index,
            recurrence_indices=recurrence_indices,
            values=values,
        )


class ShiftMultiplyLedgerRow(StrictModel):
    """One exact (i, j) contribution of the shift product rule."""

    left_exponent: StrictInt = Field(ge=0, le=MAX_SHIFT_ORDER)
    right_exponent: StrictInt = Field(ge=0, le=MAX_SHIFT_ORDER)
    result_exponent: StrictInt = Field(ge=0, le=MAX_SHIFT_RESULT_ORDER)
    shifted_coefficient: RationalFunction = Field(
        description="Exact sigma^left_exponent applied to the right coefficient."
    )
    contribution: RationalFunction = Field(
        description="Exact left coefficient times the shifted right coefficient."
    )

    @model_validator(mode="after")
    def require_exponent_transport(self) -> Self:
        if self.result_exponent != self.left_exponent + self.right_exponent:
            raise _validation_error(
                "exponent_transport",
                "ledger result exponents must add the consumed exponents",
            )
        return self


class ShiftOperatorMultiplyResult(StrictModel):
    """The exact noncommutative shift-operator product with its ledger."""

    left: ShiftOreOperator
    right: ShiftOreOperator
    product: ShiftOreOperator
    ledger: tuple[ShiftMultiplyLedgerRow, ...] = Field(
        max_length=MAX_SHIFT_LEDGER_ROWS,
        description=(
            "One row per nonzero (left, right) term pair, in left-then-right "
            "exponent order; rows sum exactly to the product."
        ),
    )

    @model_validator(mode="after")
    def require_ledger_coverage(self) -> Self:
        expected = tuple(
            (left.exponent, right.exponent)
            for left in self.left.terms
            for right in self.right.terms
        )
        observed = tuple((row.left_exponent, row.right_exponent) for row in self.ledger)
        if observed != expected:
            raise _validation_error(
                "ledger_coverage",
                "the ledger must cover every nonzero term pair exactly once",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        left: ShiftOreOperator,
        right: ShiftOreOperator,
        *,
        product: ShiftOreOperator,
        ledger: tuple[ShiftMultiplyLedgerRow, ...],
    ) -> Self:
        return cls.model_construct(
            left=left,
            right=right,
            product=product,
            ledger=ledger,
        )


class ShiftOperatorPowerRequest(StrictModel):
    """Power an operator; exponents above one currently require ``ZZ[n]`` coefficients.

    Exponents zero and one are exact projections and accept the full ``QQ(n)``
    coefficient carrier. Higher powers use a whole-stage growth bound that is
    currently proved only for integer-polynomial coefficients.
    """

    operator: ShiftOreOperator
    exponent: StrictInt = Field(ge=0, le=MAX_SHIFT_POWER_EXPONENT)


class ShiftOperatorPowerResult(StrictModel):
    operator: ShiftOreOperator
    exponent: StrictInt = Field(ge=0, le=MAX_SHIFT_POWER_EXPONENT)
    power: ShiftOreOperator

    @classmethod
    def _from_kernel(
        cls,
        operator: ShiftOreOperator,
        exponent: int,
        power: ShiftOreOperator,
    ) -> Self:
        return cls.model_construct(operator=operator, exponent=exponent, power=power)
