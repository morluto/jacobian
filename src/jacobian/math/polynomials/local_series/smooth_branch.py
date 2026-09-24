"""First-order formal lifting of a simple rational local branch."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, log10

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.local_series.newton_polygon import (
    LocalPolynomialCoefficient,
    LocalPolynomialInSeries,
)
from jacobian.math.polynomials.local_series.values import (
    MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
    TruncatedLaurentWindow,
)

MAX_SMOOTH_BRANCH_ROWS = 17
MAX_SMOOTH_BRANCH_DEGREE = 16
MAX_SMOOTH_BRANCH_SERIES_SLOTS = 512
MAX_SMOOTH_BRANCH_SCALAR_DIGITS = 64
MAX_SMOOTH_BRANCH_INPUT_DIGITS = 256
MAX_SMOOTH_BRANCH_WORK = 1_000


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"local_series.smooth_branch.{reason}", message)


class SmoothBranchFirstJetRequest(StrictModel):
    """Lift a supplied simple rational root through first order in the parameter."""

    polynomial: LocalPolynomialInSeries
    initial_root: CanonicalRational


class SmoothBranchFirstJetResult(StrictModel):
    """The unique series jet ``y(t)=c+a*t+O(t^2)`` at a simple root."""

    source: LocalPolynomialInSeries
    initial_root: CanonicalRational
    series: TruncatedLaurentWindow
    residual_precision: StrictInt = Field(
        ge=2,
        description="The source polynomial vanishes modulo t^2 after substitution.",
    )

    @model_validator(mode="after")
    def require_matching_parent(self) -> SmoothBranchFirstJetResult:
        if (
            self.series.variable,
            self.series.place,
            self.series.center,
            self.series.precision,
            self.series.valuation_lower,
        ) != (
            self.source.variable,
            self.source.place,
            self.source.center,
            self.residual_precision,
            0,
        ):
            raise _error(
                "result_parent",
                "branch jet must retain the source parameter and declared precision",
            )
        if self.residual_precision != 2 or not self.series.coefficients:
            raise _error("result_precision", "a first jet has precision exactly two")
        if self.series.coefficients[0].as_fraction() != self.initial_root.as_fraction():
            raise _error(
                "result_root", "the branch constant must equal its supplied root"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: LocalPolynomialInSeries,
        initial_root: CanonicalRational,
        series: TruncatedLaurentWindow,
    ) -> SmoothBranchFirstJetResult:
        return cls.model_construct(
            source=source,
            initial_root=initial_root,
            series=series,
            residual_precision=2,
        )


def _coefficient(series: TruncatedLaurentWindow, exponent: int) -> Fraction:
    if exponent < series.valuation_lower:
        return Fraction(0)
    if exponent >= series.precision:
        raise AssertionError("admission must establish the requested input precision")
    return series.coefficients[exponent - series.valuation_lower].as_fraction()


def _fraction_digits(value: Fraction) -> int:
    return max(len(str(abs(value.numerator))), len(str(value.denominator)))


def _horner_pair(
    coefficients: tuple[Fraction, ...], point: Fraction
) -> tuple[Fraction, Fraction]:
    """Evaluate a polynomial and its derivative with one bounded Horner pass."""
    value = Fraction(0)
    derivative = Fraction(0)
    for coefficient in reversed(coefficients):
        derivative = derivative * point + value
        value = value * point + coefficient
    return value, derivative


@dataclass(frozen=True)
class _SmoothBranchPlan:
    root: Fraction
    constant_coefficients: tuple[Fraction, ...]
    linear_coefficients: tuple[Fraction, ...]


def _coefficient_inputs(
    source: LocalPolynomialInSeries, max_degree: int
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], int, int]:
    constant_coefficients = [Fraction(0)] * (max_degree + 1)
    linear_coefficients = [Fraction(0)] * (max_degree + 1)
    constant_digit_sum = 0
    linear_digit_sum = 0
    for row in source.coefficients:
        series = row.series
        if series is None:
            continue
        if (series.variable, series.place, series.center) != (
            source.variable,
            source.place,
            source.center,
        ):
            raise OperationDomainValidationError(
                location=("polynomial", "coefficients", row.y_degree),
                code="local_series.smooth_branch.parent",
                message="coefficient series must share the polynomial's local parent",
            )
        for exponent, value in enumerate(
            series.coefficients, start=series.valuation_lower
        ):
            coefficient = value.as_fraction()
            if _fraction_digits(coefficient) > MAX_SMOOTH_BRANCH_INPUT_DIGITS:
                raise OperationResourceAdmissionError(
                    location=("polynomial", "coefficients", row.y_degree),
                    code="local_series.smooth_branch.input_scalar_budget",
                    message=(
                        "source coefficients exceed the "
                        f"{MAX_SMOOTH_BRANCH_INPUT_DIGITS}-digit branch-lift bound"
                    ),
                )
            if exponent < 0 and coefficient:
                raise OperationDomainValidationError(
                    location=("polynomial", "coefficients", row.y_degree),
                    code="local_series.smooth_branch.pole",
                    message="smooth branch lifting requires power-series coefficients without poles",
                )
        constant = _coefficient(series, 0)
        linear = _coefficient(series, 1)
        constant_coefficients[row.y_degree] = constant
        linear_coefficients[row.y_degree] = linear
        if constant:
            constant_digit_sum += _fraction_digits(constant)
        if linear:
            linear_digit_sum += _fraction_digits(linear)
    return (
        tuple(constant_coefficients),
        tuple(linear_coefficients),
        constant_digit_sum,
        linear_digit_sum,
    )


def _admit_growth(
    *,
    root: Fraction,
    max_degree: int,
    rows: int,
    constant_digit_sum: int,
    linear_digit_sum: int,
) -> None:
    root_digits = _fraction_digits(root)
    if root_digits > MAX_SMOOTH_BRANCH_SCALAR_DIGITS:
        raise OperationResourceAdmissionError(
            location=("initial_root",),
            code="local_series.smooth_branch.scalar_budget",
            message=(
                "the initial root exceeds the "
                f"{MAX_SMOOTH_BRANCH_SCALAR_DIGITS}-digit branch-lift bound"
            ),
        )
    count_digits = ceil(log10(rows + 1))
    degree_digits = ceil(log10(max_degree + 1)) if max_degree else 1
    root_power_digits = max_degree * root_digits
    # For a common denominator of sum_j a_j*c^j, multiply the input
    # coefficient denominators and the largest possible power of c's
    # denominator. The corresponding numerator bound sums coefficient heights,
    # root powers, and at most log10(rows) digits for adding the terms.
    constant_height = constant_digit_sum + root_power_digits + count_digits + 4
    derivative_height = (
        constant_digit_sum + root_power_digits + count_digits + degree_digits + 4
    )
    linear_height = linear_digit_sum + root_power_digits + count_digits + 4
    slope_height = linear_height + derivative_height + 1
    if max(constant_height, derivative_height, linear_height, slope_height) > 4096:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="local_series.smooth_branch.growth_budget",
            message="the first-jet rational intermediate bound exceeds 4096 digits",
        )


def _admit_request(request: SmoothBranchFirstJetRequest) -> _SmoothBranchPlan:
    if not isinstance(request, SmoothBranchFirstJetRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="local_series.smooth_branch.request_type",
            message="request must supply a local polynomial and rational simple root",
        )
    source = request.polynomial
    if not isinstance(source, LocalPolynomialInSeries) or any(
        not isinstance(row, LocalPolynomialCoefficient)
        or (
            row.series is not None
            and not isinstance(row.series, TruncatedLaurentWindow)
        )
        for row in source.coefficients
    ):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="local_series.smooth_branch.polynomial_type",
            message="polynomial must be a canonical local polynomial in series",
        )
    if not isinstance(request.initial_root, CanonicalRational):
        raise OperationDomainValidationError(
            location=("initial_root",),
            code="local_series.smooth_branch.root_type",
            message="initial_root must be an exact rational value",
        )
    max_degree, rows = _admit_source(source)
    root = request.initial_root.as_fraction()
    if _fraction_digits(root) > MAX_SMOOTH_BRANCH_SCALAR_DIGITS:
        raise OperationResourceAdmissionError(
            location=("initial_root",),
            code="local_series.smooth_branch.scalar_budget",
            message=(
                "the initial root exceeds the "
                f"{MAX_SMOOTH_BRANCH_SCALAR_DIGITS}-digit branch-lift bound"
            ),
        )
    constant, linear, constant_digits, linear_digits = _coefficient_inputs(
        source, max_degree
    )
    _admit_growth(
        root=root,
        max_degree=max_degree,
        rows=rows,
        constant_digit_sum=constant_digits,
        linear_digit_sum=linear_digits,
    )
    return _SmoothBranchPlan(root, constant, linear)


def _admit_source(source: LocalPolynomialInSeries) -> tuple[int, int]:
    """Bound shape, work, and serialized source/output before rational evaluation."""
    if len(source.coefficients) > MAX_SMOOTH_BRANCH_ROWS:
        raise OperationResourceAdmissionError(
            location=("polynomial", "coefficients"),
            code="local_series.smooth_branch.row_budget",
            message=f"smooth branch lifting admits at most {MAX_SMOOTH_BRANCH_ROWS} coefficient rows",
        )
    slots = sum(
        len(row.series.coefficients)
        for row in source.coefficients
        if row.series is not None
    )
    if slots > MAX_SMOOTH_BRANCH_SERIES_SLOTS:
        raise OperationResourceAdmissionError(
            location=("polynomial", "coefficients"),
            code="local_series.smooth_branch.input_window_budget",
            message=(
                "smooth branch lifting admits at most "
                f"{MAX_SMOOTH_BRANCH_SERIES_SLOTS} source series coefficients"
            ),
        )
    max_degree = max((row.y_degree for row in source.coefficients), default=0)
    if max_degree > MAX_SMOOTH_BRANCH_DEGREE:
        raise OperationResourceAdmissionError(
            location=("polynomial", "coefficients"),
            code="local_series.smooth_branch.degree_budget",
            message=f"smooth branch lifting admits y-degree at most {MAX_SMOOTH_BRANCH_DEGREE}",
        )
    if source.place != "FINITE":
        raise OperationDomainValidationError(
            location=("polynomial", "place"),
            code="local_series.smooth_branch.place",
            message="smooth branch lifting requires a finite local parameter",
        )
    if any(
        row.series is not None and row.series.precision < 2
        for row in source.coefficients
    ):
        raise OperationDomainValidationError(
            location=("polynomial", "coefficients"),
            code="local_series.smooth_branch.input_precision",
            message="each coefficient series must be known through exponent one",
        )
    work_estimate = slots + 3 * (max_degree + 1) + len(source.coefficients)
    if work_estimate > MAX_SMOOTH_BRANCH_WORK:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="local_series.smooth_branch.work_budget",
            message="first-jet input scan and exact Horner work exceeds its admitted bound",
        )
    center_digits = _fraction_digits(source.center.as_fraction())
    output_bytes = (
        512
        + len(source.coefficients) * 192
        + slots * (2 * MAX_SMOOTH_BRANCH_INPUT_DIGITS + 96)
        + 2 * (len(source.coefficients) + 1) * center_digits
        + 4 * MAX_LOCAL_SERIES_COEFFICIENT_DIGITS
        + 512
        + len(source.variable)
    )
    if output_bytes > CanonicalLimits().max_output_bytes:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="local_series.smooth_branch.output_budget",
            message="source-bound first-jet output exceeds the canonical byte limit",
        )
    return max_degree, max(1, len(source.coefficients))


def smooth_branch_first_jet(
    request: SmoothBranchFirstJetRequest,
) -> SmoothBranchFirstJetResult:
    """Return the exact first-order jet at an explicitly supplied simple root.

    For ``F(t,y)=sum_j a_j(t)y^j``, the request provides ``c`` with
    ``F(0,c)=0`` and ``F_y(0,c)!=0``. The returned slope is
    ``-F_t(0,c)/F_y(0,c)``. This computes a smooth unramified first jet only.
    """
    plan = _admit_request(request)
    source = request.polynomial
    root = plan.root

    constant_value, derivative_value = _horner_pair(plan.constant_coefficients, root)
    if constant_value != 0:
        raise OperationDomainValidationError(
            location=("initial_root",),
            code="local_series.smooth_branch.not_root",
            message="the supplied rational value is not a root of F(0,y)",
        )
    if derivative_value == 0:
        raise OperationDomainValidationError(
            location=("initial_root",),
            code="local_series.smooth_branch.not_simple",
            message="the supplied root must be simple: F_y(0,c) must be nonzero",
        )
    linear_value, _ = _horner_pair(plan.linear_coefficients, root)
    slope = -linear_value / derivative_value
    slope_digits = _fraction_digits(slope)
    if slope_digits > 4096:
        raise AssertionError("admitted slope height must fit the Laurent carrier")
    branch = TruncatedLaurentWindow(
        variable=source.variable,
        place=source.place,
        center=source.center,
        valuation_lower=0,
        precision=2,
        coefficients=(
            CanonicalRational.from_fraction(root),
            CanonicalRational.from_fraction(slope),
        ),
    )
    return SmoothBranchFirstJetResult._from_kernel(
        source=source,
        initial_root=request.initial_root,
        series=branch,
    )


__all__ = [
    "MAX_SMOOTH_BRANCH_DEGREE",
    "MAX_SMOOTH_BRANCH_INPUT_DIGITS",
    "MAX_SMOOTH_BRANCH_ROWS",
    "MAX_SMOOTH_BRANCH_SCALAR_DIGITS",
    "MAX_SMOOTH_BRANCH_SERIES_SLOTS",
    "MAX_SMOOTH_BRANCH_WORK",
    "SmoothBranchFirstJetRequest",
    "SmoothBranchFirstJetResult",
    "smooth_branch_first_jet",
]
