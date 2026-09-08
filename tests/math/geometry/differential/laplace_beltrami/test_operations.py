"""Exact identities for the rational Laplace--Beltrami operation."""

from fractions import Fraction
from time import monotonic

import pytest
from sympy import cancel, symbols

from jacobian._exact import CanonicalRational
from jacobian._execution import OperationExecutionTimeoutError, request_execution
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.differential.laplace_beltrami import (
    RationalLaplaceBeltramiResult,
    laplace_beltrami,
)
from jacobian.math.geometry.differential.metrics import RationalCoordinateMetric
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    canonical_locus_guards,
)
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _metric(
    values: tuple[object, ...], axis: tuple[str, ...]
) -> RationalCoordinateMetric:
    return RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=tuple(
                rational_function_from_sympy(value, axis) for value in values
            ),
        )
    )


def _scalar(value: object, axis: tuple[str, ...]) -> RationalFunction:
    return rational_function_from_sympy(value, axis)


def test_polar_metric_matches_exact_coordinate_values() -> None:
    r, theta = symbols("r theta")
    metric = _metric((1, 0, 0, r**2), ("r", "theta"))

    assert (
        rational_function_to_sympy(
            laplace_beltrami(metric, _scalar(r**2, ("r", "theta"))).value
        )
        == 4
    )
    assert (
        rational_function_to_sympy(
            laplace_beltrami(metric, _scalar(theta, ("r", "theta"))).value
        )
        == 0
    )


def test_euclidean_and_constant_metrics_match_independent_formula() -> None:
    x, y = symbols("x y")
    axis = ("x", "y")
    scalar = x**3 + x * y**2 + 1 / (x + 2)
    result = laplace_beltrami(_metric((1, 0, 0, 1), axis), _scalar(scalar, axis))
    expected = scalar.diff(x, 2) + scalar.diff(y, 2)
    assert cancel(rational_function_to_sympy(result.value) - expected) == 0

    weighted = laplace_beltrami(_metric((2, 0, 0, 4), axis), _scalar(x**2 + y**2, axis))
    assert rational_function_to_sympy(weighted.value) == Fraction(3, 2)


def test_conformal_metric_agrees_with_divergence_formula() -> None:
    x, y = symbols("x y")
    axis = ("x", "y")
    conformal = 1 + x**2 + y**2
    metric = _metric((conformal, 0, 0, conformal), axis)
    scalar = x**2 * y + 1 / (x + y + 3)
    result = laplace_beltrami(metric, _scalar(scalar, axis))
    # In two dimensions sqrt(det(g)) = conformal and
    # sqrt(det(g)) * g^ij = delta^ij.
    expected = sum(scalar.diff(coordinate, 2) for coordinate in (x, y)) / conformal
    assert cancel(rational_function_to_sympy(result.value) - expected) == 0


def test_zero_constant_serialization_and_axis_validation() -> None:
    x, _y = symbols("x y")
    axis = ("x", "y")
    metric = _metric((1, 0, 0, 1), axis)
    result = laplace_beltrami(metric, _scalar(7, axis))
    assert rational_function_to_sympy(result.value) == 0
    assert (
        RationalCoordinateTensor.model_validate_json(metric.tensor.model_dump_json())
        == metric.tensor
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result
    with pytest.raises(OperationDomainValidationError, match="same coordinate axis"):
        laplace_beltrami(metric, _scalar(x, ("y", "x")))


def test_shared_deadline_is_honored() -> None:
    x = symbols("x")
    metric = _metric((1,), ("x",))
    with (
        request_execution(monotonic() - 200),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        laplace_beltrami(metric, _scalar(x**2, ("x",)))


def test_identically_singular_metric_rejects_before_cancellation() -> None:
    x = symbols("x")
    metric = _metric((1, x, x, x**2), ("x", "y"))
    with pytest.raises(
        OperationDomainValidationError, match="identically zero"
    ) as rejected:
        laplace_beltrami(metric, _scalar(x, ("x", "y")))
    assert rejected.value.errors()[0]["type"].endswith(
        "laplace_beltrami.singular_metric"
    )


def test_structurally_zero_metric_uses_the_laplace_domain_code() -> None:
    metric = _metric((0,), ("x",))
    with pytest.raises(
        OperationDomainValidationError, match="identically zero"
    ) as rejected:
        laplace_beltrami(metric, _scalar(1, ("x",)))
    assert rejected.value.errors()[0]["type"].endswith(
        "laplace_beltrami.singular_metric"
    )


def test_result_denominator_reuses_inherited_monomial_guards() -> None:
    x = symbols("x")
    axis = ("x",)
    x_poly = _scalar(x, axis).numerator
    x3_poly = _scalar(x**3, axis).numerator
    fillers = tuple(_scalar(x + offset, axis).numerator for offset in range(1, 767))
    guards = canonical_locus_guards((x_poly, x3_poly, *fillers), variable_count=1)
    assert len(guards) == 768
    metric = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=_metric((1,), axis).tensor.components,
            retained_nonzero_denominators=guards,
        )
    )
    result = laplace_beltrami(metric, _scalar(1 / x, axis))
    assert len(result.retained_nonzero_denominators) == 768
    assert rational_function_to_sympy(result.value) == 2 / x**3


def test_powered_binomial_result_denominator_is_reserved() -> None:
    x = symbols("x")
    axis = ("x",)
    linear = _scalar(x + 1, axis).numerator
    fillers = tuple(_scalar(x + offset, axis).numerator for offset in range(2, 769))
    guards = canonical_locus_guards((linear, *fillers), variable_count=1)
    assert len(guards) == 768
    metric = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=_metric((1,), axis).tensor.components,
            retained_nonzero_denominators=guards,
        )
    )
    with pytest.raises(OperationResourceAdmissionError, match="768 guards"):
        laplace_beltrami(metric, _scalar(1 / (x + 1), axis))


def test_nonreduced_scalar_is_a_domain_error_before_admission() -> None:
    x = symbols("x")
    metric = _metric((1,), ("x",))
    scalar = RationalFunction(
        variables=("x",),
        numerator=_scalar(x**64 - 1, ("x",)).numerator,
        denominator=_scalar(x**32 - 1, ("x",)).numerator,
    )
    with pytest.raises(OperationDomainValidationError) as rejected:
        laplace_beltrami(metric, scalar)
    error = rejected.value.errors()[0]
    assert error["type"].endswith("noncanonical_source")
    assert error["loc"] == ("scalar",)


def test_nonreduced_metric_component_keeps_its_request_path() -> None:
    x = symbols("x")
    unreduced = RationalFunction(
        variables=("x",),
        numerator=_scalar(x**64 - 1, ("x",)).numerator,
        denominator=_scalar(x**32 - 1, ("x",)).numerator,
    )
    metric = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=("x",),
            variance=("COVARIANT", "COVARIANT"),
            components=(unreduced,),
            retained_nonzero_denominators=(unreduced.denominator,),
        )
    )
    with pytest.raises(OperationDomainValidationError) as rejected:
        laplace_beltrami(metric, _scalar(1, ("x",)))
    error = rejected.value.errors()[0]
    assert error["type"].endswith("noncanonical_source")
    assert error["loc"] == ("metric", "tensor", "components", 0)


def test_monic_powered_result_denominator_reuses_inherited_guards() -> None:
    x = symbols("x")
    axis = ("x",)
    x_poly = _scalar(x, axis).numerator
    x2_poly = _scalar(x**2, axis).numerator
    fillers = tuple(_scalar(x + offset, axis).numerator for offset in range(1, 767))
    guards = canonical_locus_guards((x_poly, x2_poly, *fillers), variable_count=1)
    assert len(guards) == 768
    metric = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=_metric((2 * x,), axis).tensor.components,
            retained_nonzero_denominators=guards,
        )
    )
    result = laplace_beltrami(metric, _scalar(x, axis))
    assert len(result.retained_nonzero_denominators) == 768
    assert rational_function_to_sympy(result.value) == -1 / (4 * x**2)


def test_recognition_work_is_rejected_before_the_gcd_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axis = ("x", "y", "z", "w")
    high = SparseRationalPolynomial(
        terms=(
            RationalPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=1),
                exponents=(64, 64, 64, 64),
            ),
            RationalPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=1),
                exponents=(0, 0, 0, 0),
            ),
        )
    )
    scalar = RationalFunction(variables=axis, numerator=high, denominator=high)
    metric = _metric(
        tuple(1 if i == j else 0 for i in range(4) for j in range(4)), axis
    )
    monkeypatch.setattr(
        "jacobian.math.geometry.differential.laplace_beltrami.operations.recognize_canonical_rational_functions",
        lambda *args, **kwargs: pytest.fail(
            "recognition worker must follow work admission"
        ),
    )
    with pytest.raises(OperationResourceAdmissionError, match="work") as rejected:
        laplace_beltrami(metric, scalar)
    error = rejected.value.errors()[0]
    assert error["type"].endswith(".work")
    assert error["loc"] == ("scalar",)


def test_scalar_derivative_bounds_report_the_scalar_field() -> None:
    x = symbols("x")
    metric = _metric((1,), ("x",))
    with pytest.raises(OperationResourceAdmissionError) as rejected:
        laplace_beltrami(metric, _scalar(1 / (x**64 + 1), ("x",)))
    error = rejected.value.errors()[0]
    assert error["loc"] == ("scalar",)
    assert error["type"].endswith("result_exponent")


def test_result_locus_guard_budget_is_bounded() -> None:
    x = symbols("x")
    axis = ("x",)
    metric = _metric((1,), axis)
    one = _scalar(1, axis)
    guards = tuple(_scalar(x + offset, axis).numerator for offset in range(1, 770))

    with pytest.raises(ValueError, match="at most 768"):
        RationalLaplaceBeltramiResult(
            metric=metric,
            scalar=one,
            value=one,
            retained_nonzero_denominators=guards,
        )
