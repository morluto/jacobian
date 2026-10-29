"""Denominator growth and retained-value closure of the circle kernel."""

from __future__ import annotations

import json
import math
import random
import sys
import threading
import time
from fractions import Fraction
from itertools import pairwise
from types import FrameType
from typing import Any

import pytest
import sympy
from tests.fixtures.accounting import assert_charged_work_parity

from jacobian._execution import (
    OperationExecutionCancelledError,
    current_request_execution,
    request_execution,
)
from jacobian.canonical import decimal_digit_width
from jacobian.math._root_isolation import strict_root_count
from jacobian.math.polynomials.unit_circle import _sup_norm as owner
from jacobian.math.polynomials.unit_circle._sup_norm_bounds import sup_norm_bounds
from jacobian.math.polynomials.unit_circle._sup_norm_models import (
    MAX_SUP_NORM_ENCLOSURE_COMPONENT_DIGITS,
    MAX_SUP_NORM_PARAMETER_COMPONENT_DIGITS,
    UnitCircleSupNormSquaredRequest,
    UnitCircleSupNormSquaredResult,
)


def _rational_request() -> UnitCircleSupNormSquaredRequest:
    rng = random.Random(2768)
    terms = []
    for exponent in range(8, -1, -1):
        components = []
        for _ in range(2):
            value = Fraction(
                rng.randrange(10**31, 10**32), rng.randrange(10**31, 10**32)
            )
            components.append(
                {"num": str(value.numerator), "den": str(value.denominator)}
            )
        terms.append(
            {
                "exponent": exponent,
                "coefficient": {"real": components[0], "imaginary": components[1]},
            }
        )
    return UnitCircleSupNormSquaredRequest.model_validate_json(
        json.dumps({"polynomial": {"terms": terms}})
    )


def _width(value: Fraction) -> int:
    return max(
        decimal_digit_width(value.numerator), decimal_digit_width(value.denominator)
    )


def _evaluate(values: tuple[Fraction, ...], at: Fraction) -> Fraction:
    result = Fraction(0)
    for value in reversed(values):
        result = result * at + value
    return result


def _poly_digits(polynomial: Any) -> int:
    return max(
        decimal_digit_width(int(part))
        for coefficient in polynomial.all_coeffs()
        for part in coefficient.as_numer_denom()
    )


def _limbs(digits: int) -> int:
    return (4 * digits + 63) // 64


def _fraction_widths(value: Any) -> list[int]:
    if isinstance(value, Fraction):
        return [_width(value)]
    if isinstance(value, (list, tuple)):
        return [width for item in value for width in _fraction_widths(item)]
    return []


def _retained_digits(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_retained_digits(item) for item in value.values())
    if isinstance(value, list):
        return sum(_retained_digits(item) for item in value)
    if isinstance(value, str) and value.lstrip("-").isdecimal():
        return len(value.lstrip("-"))
    if isinstance(value, int):
        return decimal_digit_width(value)
    return 0


def _polynomial_sign(coefficients: tuple[int, ...], at: Fraction) -> int:
    """Independent division-free rational evaluation, scaled by q**degree."""
    numerator = coefficients[0]
    denominator_power = 1
    for coefficient in coefficients[1:]:
        denominator_power *= at.denominator
        numerator = numerator * at.numerator + coefficient * denominator_power
    return (numerator > 0) - (numerator < 0)


def _direct_circle_value(
    coefficients: list[tuple[Fraction, Fraction]], parameter: Fraction
) -> Fraction:
    # Independent Gaussian-rational Horner evaluation of the source P(z).
    z_real = (1 - parameter**2) / (1 + parameter**2)
    z_imaginary = 2 * parameter / (1 + parameter**2)
    real = imaginary = Fraction(0)
    for a, b in reversed(coefficients):
        real, imaginary = (
            real * z_real - imaginary * z_imaginary + a,
            real * z_imaginary + imaginary * z_real + b,
        )
    return real**2 + imaginary**2


def _run_measured_supremum(
    request: UnitCircleSupNormSquaredRequest,
) -> tuple[UnitCircleSupNormSquaredResult, dict[str, int]]:
    """Observe real primitive inputs/results without replacing mathematical work."""
    executed = dict.fromkeys(
        (
            "transform",
            "derivative_isolation",
            "resultant",
            "factor",
            "value_analysis",
            "postprocessing",
        ),
        0,
    )
    interval_code = sympy.Poly.intervals.__code__

    def observe(frame: FrameType, event: str, returned: Any) -> None:
        if event != "return":
            return
        is_interval = frame.f_code is interval_code
        if not is_interval and frame.f_globals.get("__name__") != owner.__name__:
            return
        local = frame.f_locals
        name = frame.f_code.co_name
        if is_interval:
            polynomial = local["f"]
            degree = int(polynomial.degree())
            height = _poly_digits(polynomial)
            endpoints = [
                int(part)
                for interval, _ in returned
                for endpoint in interval
                for part in endpoint.as_numer_denom()
            ]
            endpoint_digits = max(map(decimal_digit_width, endpoints), default=1)
            if str(polynomial.gen) == "t":
                executed["derivative_isolation"] += (degree + 2) ** 3 * (
                    height + endpoint_digits + 64
                )
            else:
                executed["value_analysis"] += (degree + 1) ** 4 * _limbs(
                    height + endpoint_digits
                ) ** 2
        elif frame.f_globals.get("__name__") == owner.__name__:
            if name == "_transformed_numerator":
                execution = current_request_execution()
                assert execution is not None and execution.deadline is not None
                degree = len(local["coefficients"]) - 1
                executed["transform"] += (2 * degree + 1) ** 3 * (
                    max(_fraction_widths(returned)) + 64
                )
            elif name == "_flint_resultant_and_factors_impl":
                resultant, _factors = returned
                degree = local["degree"]
                executed["resultant"] += (
                    (4 * degree + 1) ** 3
                    * (2 * degree + 1) ** 2
                    * _limbs(_poly_digits(resultant)) ** 2
                )
                executed["factor"] += (int(resultant.degree()) + 1) ** 6 * _limbs(
                    _poly_digits(resultant)
                ) ** 3
            elif name == "rational_level_gcd":
                polynomial = local["derivative_polynomial"]
                numerator = sympy.Poly(local["numerator_expression"], local["symbol"])
                height = (
                    _poly_digits(polynomial)
                    + _poly_digits(numerator)
                    + _width(local["value"])
                )
                executed["value_analysis"] += (
                    int(polynomial.degree()) + 1
                ) ** 4 * _limbs(height) ** 2
            elif name in (
                "_compare_rational_to_value_roots",
                "_sturm_root_index_below",
            ):
                polynomial = local.get("polynomial", local.get("factor"))
                assert polynomial is not None
                degree = int(polynomial.degree())
                height = max(
                    [_poly_digits(polynomial), *_fraction_widths(tuple(local.values()))]
                )
                executed["value_analysis"] += (degree + 1) ** 4 * _limbs(height) ** 2
            elif name == "_bounded_parameter_interval" and "scale" in local:
                polynomial = local["polynomial"]
                height = _poly_digits(polynomial)
                endpoints = _fraction_widths(returned)
                executed["derivative_isolation"] += (
                    int(polynomial.degree()) + 2
                ) ** 2 * (height + max(endpoints) + 64)
            elif name in (
                "_value_enclosure",
                "_circle_image_enclosure",
                "_reported_interval",
                "_sqrt_bounds",
                "_interval",
            ):
                widths = [
                    *_fraction_widths(tuple(local.values())),
                    *_fraction_widths(returned),
                ]
                executed["postprocessing"] += (
                    32 * 17 * _limbs(max(widths, default=1)) ** 2
                )

    previous_profile = sys.getprofile()
    sys.setprofile(observe)
    try:
        result = owner.unit_circle_sup_norm_squared(request.polynomial)
    finally:
        sys.setprofile(previous_profile)
    assert current_request_execution() is None
    return result, executed


def test_degree_eight_independent_denominators_retain_exact_certificate() -> None:
    request = _rational_request()
    coefficients = owner._admit_sup_norm_polynomial(request.polynomial)
    bounds = sup_norm_bounds(coefficients)
    result, executed = _run_measured_supremum(request)
    assert_charged_work_parity(
        charged={
            "transform": bounds.transform_work,
            "derivative_isolation": bounds.isolation_work,
            "resultant": bounds.resultant_work,
            "factor": bounds.factor_work,
            "value_analysis": bounds.value_analysis_work,
            "postprocessing": bounds.postprocessing_work,
            "retained_scalars": bounds.certificate_scalar_digits,
        },
        executed={
            **executed,
            "retained_scalars": _retained_digits(result.model_dump(mode="json")),
        },
    )
    assert all(executed.values())

    assert len(result.critical_points) == 14
    assert result.sup_norm_squared_exact is None  # Factor exceeds the shared carrier.
    numerator = tuple(value.as_fraction() for value in result.transformed_numerator)
    derivative = tuple(value.as_fraction() for value in result.derivative_numerator)
    assert max(map(_width, numerator)) == 1118
    assert max(map(_width, derivative)) == 565
    assert max(map(_width, numerator)) <= bounds.numerator_digits
    assert max(map(_width, derivative)) <= bounds.derivative_digits
    for parameter in map(Fraction, (-2, -1, 0, 1, 2)):
        assert _evaluate(numerator, parameter) / (1 + parameter**2) ** 8 == (
            _direct_circle_value(coefficients, parameter)
        )
        # Differentiate the retained defining identity independently.
        numerator_prime = tuple(i * numerator[i] for i in range(1, len(numerator)))
        assert _evaluate(derivative, parameter) == (
            _evaluate(numerator_prime, parameter) * (1 + parameter**2)
            - 16 * parameter * _evaluate(numerator, parameter)
        )
    t = sympy.Symbol("t")
    derivative_poly = sympy.Poly.from_list(
        [sympy.Rational(v.numerator, v.denominator) for v in reversed(derivative)], t
    )
    # One independent Sturm sequence certifies the whole emitted family,
    # without repeating a large sequence construction for each endpoint.
    sturm = [
        tuple(map(int, poly.clear_denoms()[1].primitive()[1].all_coeffs()))
        for poly in derivative_poly.sqf_part().sturm()
    ]

    def variations(signs: list[int]) -> int:
        nonzero = [sign for sign in signs if sign]
        return sum(left != right for left, right in pairwise(nonzero))

    assert (
        variations(
            [
                ((poly[0] > 0) - (poly[0] < 0)) * (-1) ** (len(poly) - 1)
                for poly in sturm
            ]
        )
        - variations([(poly[0] > 0) - (poly[0] < 0) for poly in sturm])
    ) == len(result.critical_points)
    previous_upper: Fraction | None = None
    for point in result.critical_points:
        lower = point.parameter.lower.as_fraction()
        upper = point.parameter.upper.as_fraction()
        assert (
            _polynomial_sign(sturm[0], lower) != 0 != _polynomial_sign(sturm[0], upper)
        )
        assert (
            variations([_polynomial_sign(poly, lower) for poly in sturm])
            - variations([_polynomial_sign(poly, upper) for poly in sturm])
            == 1
        )
        if previous_upper is not None:
            assert previous_upper <= lower
        previous_upper = upper
        sample = _direct_circle_value(coefficients, (lower + upper) / 2)
        assert (
            point.value.lower.as_fraction() <= sample <= point.value.upper.as_fraction()
        )
        for interval in (point.value, point.z_real, point.z_imaginary):
            if interval.lower != interval.upper:
                assert max(
                    _width(v.as_fraction()) for v in (interval.lower, interval.upper)
                ) <= (MAX_SUP_NORM_ENCLOSURE_COMPONENT_DIGITS)
    squared = result.sup_norm_squared_enclosure
    assert (
        result.sup_norm_enclosure.lower.as_fraction() ** 2
        <= squared.lower.as_fraction()
    )
    assert (
        result.sup_norm_enclosure.upper.as_fraction() ** 2
        >= squared.upper.as_fraction()
    )
    decoded = UnitCircleSupNormSquaredResult.model_validate_json(
        result.model_dump_json()
    )
    assert decoded == result
    # The actual public claim-checking consumer accepts this unchanged certificate.
    assert owner.verify_unit_circle_sup_norm_squared(decoded)


@pytest.mark.parametrize(
    "coefficients",
    [
        [(Fraction(0), Fraction(0))],
        [(Fraction(1, 97), Fraction(2, 101))],
        [(Fraction(1, 97), Fraction(2, 97)), (Fraction(-3, 97), Fraction(4, 97))],
        [(Fraction(1, 97), Fraction(2, 101)), (Fraction(-3, 103), Fraction(4, 107))],
    ],
)
def test_denominator_bounds_cover_actual_cayley_coefficients(
    coefficients: list[tuple[Fraction, Fraction]],
) -> None:
    bounds = sup_norm_bounds(coefficients)
    numerator, derivative = owner._transformed_numerator(coefficients)
    assert max(map(_width, numerator)) <= bounds.numerator_digits
    assert max(map(_width, derivative)) <= bounds.derivative_digits


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (Fraction(-5, 13), Fraction(-1, 7)),
        (Fraction(-1, 7), Fraction(5, 13)),
        (Fraction(5, 13), Fraction(5, 13)),
        (Fraction(1, 10**500), Fraction(1, 10**499)),
    ],
)
def test_reported_grid_rounds_outward_and_preserves_singletons(
    lower: Fraction, upper: Fraction
) -> None:
    reported = owner._reported_interval(lower, upper)
    assert (
        reported.lower.as_fraction() <= lower <= upper <= reported.upper.as_fraction()
    )
    if lower == upper:
        assert reported.lower.as_fraction() == reported.upper.as_fraction() == lower
    else:
        assert (
            max(_width(v.as_fraction()) for v in (reported.lower, reported.upper))
            <= 256
        )


def test_oversized_parameter_encoding_is_compacted_without_changing_its_root() -> None:
    # A genuine, extremely narrow isolator of sqrt(2), rather than a fake backend.
    scale = 10**33_000
    lower_integer = math.isqrt(2 * scale**2)
    lower, upper = Fraction(lower_integer, scale), Fraction(lower_integer + 1, scale)
    t = sympy.Symbol("t")
    polynomial = sympy.Poly((t**2 - 2) ** 2, t, domain=sympy.QQ)
    compact_lower, compact_upper = owner._bounded_parameter_interval(
        polynomial, lower, upper
    )
    assert compact_lower <= lower < upper <= compact_upper
    assert strict_root_count(polynomial, compact_lower, compact_upper) == 1
    assert polynomial.eval(compact_lower) != 0 != polynomial.eval(compact_upper)
    assert (
        max(_width(compact_lower), _width(compact_upper))
        <= MAX_SUP_NORM_PARAMETER_COMPONENT_DIGITS
    )
    assert owner._bounded_parameter_interval(polynomial, Fraction(1), Fraction(2)) == (
        Fraction(1),
        Fraction(2),
    )


def test_rational_exact_value_respects_shared_carrier_height() -> None:
    value = Fraction(10**999 + 1, 10**999 + 3)
    assert owner._rational_algebraic_value(value) is not None
    assert owner._rational_algebraic_value(Fraction(10**1000 + 1)) is None


def test_zero_critical_values_at_irrational_parameters_are_exact_singletons() -> None:
    request = UnitCircleSupNormSquaredRequest.model_validate_json(
        json.dumps(
            {
                "polynomial": {
                    "terms": [
                        {
                            "exponent": exponent,
                            "coefficient": {
                                "real": {"num": "1", "den": "1"},
                                "imaginary": {"num": "0", "den": "1"},
                            },
                        }
                        for exponent in (2, 1, 0)
                    ]
                }
            }
        )
    )
    result = owner.unit_circle_sup_norm_squared(request.polynomial)
    assert result.sup_norm_squared_enclosure.lower.as_fraction() == 9
    assert result.sup_norm_squared_enclosure.upper.as_fraction() == 9
    assert len(result.critical_points) == 3
    zero_points = [point for point in result.critical_points if not point.is_maximizer]
    assert len(zero_points) == 2
    for point in zero_points:
        assert point.parameter.interval_type == "OPEN"  # t=+-sqrt(3)
        assert point.value.interval_type == "SINGLETON"
        assert point.value.lower.as_fraction() == point.value.upper.as_fraction() == 0


@pytest.mark.parametrize("exponents", [(), (1, 0)])
def test_cancellation_after_real_result_construction_stays_operational(
    exponents: tuple[int, ...],
) -> None:
    request = UnitCircleSupNormSquaredRequest.model_validate_json(
        json.dumps(
            {
                "polynomial": {
                    "terms": [
                        {
                            "exponent": exponent,
                            "coefficient": {
                                "real": {"num": "1", "den": "1"},
                                "imaginary": {"num": "0", "den": "1"},
                            },
                        }
                        for exponent in exponents
                    ]
                }
            }
        )
    )
    cancelled = threading.Event()
    construction_code = UnitCircleSupNormSquaredResult.__dict__[
        "_from_kernel"
    ].__func__.__code__

    def cancel_after_construction(frame: FrameType, event: str, _returned: Any) -> None:
        if event == "return" and frame.f_code is construction_code:
            cancelled.set()

    previous_profile = sys.getprofile()
    try:
        with request_execution(time.monotonic(), cancellation_signal=cancelled):
            sys.setprofile(cancel_after_construction)
            with pytest.raises(OperationExecutionCancelledError):
                owner.unit_circle_sup_norm_squared(request.polynomial)
    finally:
        sys.setprofile(previous_profile)
    assert cancelled.is_set()
    assert current_request_execution() is None
