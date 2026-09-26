"""Tests for truncated formal power series operations."""

import json
from fractions import Fraction
from typing import Any, cast

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.series import (
    add,
    compose,
    derivative,
    identity_check,
    integral_zero_constant,
    inverse,
    multiply,
    power,
    reversion,
    to_polynomial,
    truncate,
)
from jacobian.math.polynomials.series._models import (
    MAX_RATIONAL_DIGITS,
    MAX_TRUNCATE_SOURCE_ORDER,
    MAX_TRUNCATION_ORDER,
    SeriesInverseRequest,
    SeriesPowerRequest,
    SeriesTruncateRequest,
    TruncatedSeries,
)


def _coeff(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational(num=num, den=den)


def _ascending(order: int) -> TruncatedSeries:
    return TruncatedSeries(
        variable="q",
        truncation_order=order,
        coefficients=tuple(_coeff(index + 1) for index in range(order)),
    )


def test_derivative_of_order_one_is_zero() -> None:
    series = TruncatedSeries(
        variable="x",
        truncation_order=1,
        coefficients=(_coeff(7),),
    )
    result = derivative(series)
    assert result.result.truncation_order == 1
    assert result.result.coefficients[0].as_fraction() == 0


def test_power_rejects_result_digit_overflow() -> None:
    import pytest

    huge = 10 ** (MAX_RATIONAL_DIGITS - 1)
    request = SeriesPowerRequest(
        series=TruncatedSeries(
            variable="x",
            truncation_order=1,
            coefficients=(_coeff(huge),),
        ),
        exponent=1000,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        power(request.series, request.exponent)
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.power_coefficient_growth"
    )


def test_reversion_rejects_nonzero_constant() -> None:
    import pytest

    from jacobian.math.polynomials.series._models import SeriesReversionRequest

    request = SeriesReversionRequest(
        variable="x",
        truncation_order=2,
        coefficients=(_coeff(1), _coeff(1)),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        reversion(request.as_series())
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.reversion_nonzero_constant"
    )


def test_integral_rejects_oversized_output_order() -> None:
    import pytest

    from jacobian.math.polynomials.series._models import SeriesIntegralRequest

    request = SeriesIntegralRequest(
        series=TruncatedSeries(
            variable="x",
            truncation_order=2,
            coefficients=(_coeff(1), _coeff(0)),
        ),
        output_order=4,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        integral_zero_constant(request.series, request.output_order)
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.integral_output_order_exceeds_source"
    )


def test_inverse_rejects_zero_constant() -> None:
    import pytest

    from jacobian.math.polynomials.series._models import SeriesInverseRequest

    request = SeriesInverseRequest(
        variable="x",
        truncation_order=2,
        coefficients=(_coeff(0), _coeff(1)),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        inverse(request.as_series())
    assert (
        error.value.errors()[0]["type"] == "formal_power_series.inverse_zero_constant"
    )


def test_inverse_rejects_result_coefficient_growth() -> None:
    import pytest

    huge = 10 ** (MAX_RATIONAL_DIGITS - 1)
    request = SeriesInverseRequest(
        variable="x",
        truncation_order=20,
        coefficients=(
            _coeff(1),
            _coeff(-huge),
            *(_coeff(0) for _ in range(18)),
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        inverse(request.as_series())
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.inverse_coefficient_growth"
    )


def test_input_series_rejects_oversized_coefficients() -> None:
    import pytest

    huge = 10**MAX_RATIONAL_DIGITS
    oversized = TruncatedSeries(
        variable="x",
        truncation_order=1,
        coefficients=(_coeff(huge),),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        derivative(oversized)
    assert error.value.errors()[0]["type"] == "formal_power_series.input_coefficient"


def test_product_can_exceed_input_digit_bound() -> None:
    large = 10 ** (MAX_RATIONAL_DIGITS - 1)
    left = TruncatedSeries(
        variable="x",
        truncation_order=1,
        coefficients=(_coeff(large),),
    )
    right = TruncatedSeries(
        variable="x",
        truncation_order=1,
        coefficients=(_coeff(large),),
    )
    result = multiply(left, right)
    value = result.result.coefficients[0]
    assert abs(value.num) >= 10**MAX_RATIONAL_DIGITS
    assert abs(value.num) < 10**4096


def test_native_exports_admit_inputs_before_kernel_work() -> None:
    import pytest

    wide = _ascending(513)
    with pytest.raises(OperationDomainValidationError) as error:
        multiply(wide, wide)
    assert error.value.errors()[0]["type"] == "formal_power_series.multiplication_work"
    with pytest.raises(OperationDomainValidationError) as error:
        power(wide, 2)
    assert error.value.errors()[0]["type"] == "formal_power_series.input_order"
    with pytest.raises(OperationDomainValidationError) as error:
        compose(wide, wide)
    assert error.value.errors()[0]["type"] == "formal_power_series.input_order"
    with pytest.raises(OperationDomainValidationError) as error:
        to_polynomial(wide)
    assert error.value.errors()[0]["type"] == "formal_power_series.input_order"

    tall = 10**MAX_RATIONAL_DIGITS
    oversized = TruncatedSeries(
        variable="x",
        truncation_order=1,
        coefficients=(_coeff(tall),),
    )
    with pytest.raises(OperationDomainValidationError) as overflow_error:
        multiply(oversized, oversized)
    assert str(overflow_error.value) == "input coefficient exceeds the 256-digit bound"


def test_native_and_wire_operations_return_the_same_canonical_values() -> None:
    """Native canonical calls and catalog wire adapters share each kernel."""
    from jacobian.math.polynomials.series import (
        divide,
        inverse,
        reversion,
    )
    from jacobian.math.polynomials.series._tools import TOOLS

    def run_wire(operation_id: str, payload: dict[str, object]) -> StrictModel:
        tool = cast(
            MathTool[Any, StrictModel],
            next(tool for tool in TOOLS if tool.operation_id == operation_id),
        )
        return tool.run(tool.request_type.model_validate(payload))

    series = TruncatedSeries(
        variable="x",
        truncation_order=3,
        coefficients=(_coeff(1), _coeff(2), _coeff(0)),
    )
    inner = TruncatedSeries(
        variable="x",
        truncation_order=3,
        coefficients=(_coeff(0), _coeff(1), _coeff(0)),
    )
    reversible = TruncatedSeries(
        variable="x",
        truncation_order=3,
        coefficients=(_coeff(0), _coeff(1), _coeff(1)),
    )
    cases = (
        (
            "formal_series.rational.multiply.compute",
            {"left": series, "right": series},
            multiply(series, series),
        ),
        (
            "formal_series.rational.power.compute",
            {"series": series, "exponent": 2},
            power(series, 2),
        ),
        (
            "formal_series.rational.inverse.compute",
            series.model_dump(),
            inverse(series),
        ),
        (
            "formal_series.rational.divide.compute",
            {"left": series, "right": series},
            divide(series, series),
        ),
        (
            "formal_series.rational.compose.compute",
            {"outer": series, "inner": inner},
            compose(series, inner),
        ),
        (
            "formal_series.rational.reversion.compute",
            reversible.model_dump(),
            reversion(reversible),
        ),
    )
    for operation_id, payload, native in cases:
        normalized = {
            key: value.model_dump() if isinstance(value, TruncatedSeries) else value
            for key, value in payload.items()
        }
        assert native.model_dump() == run_wire(operation_id, normalized).model_dump()


def test_native_and_wire_boundaries_reject_the_same_oversized_series() -> None:
    import pytest

    from jacobian.math.polynomials.series._tools import TOOLS

    wide = _ascending(MAX_TRUNCATION_ORDER + 1)
    with pytest.raises(OperationDomainValidationError) as native:
        power(wide, 2)
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "formal_series.rational.power.compute"
    )
    power_tool = cast(MathTool[SeriesPowerRequest, StrictModel], tool)
    request = power_tool.request_type.model_validate(
        {"series": wide.model_dump(), "exponent": 2}
    )
    with pytest.raises(OperationDomainValidationError) as wire:
        power_tool.run(request)
    assert native.value.errors()[0]["type"] == "formal_power_series.input_order"
    assert wire.value.errors()[0]["type"] == "formal_power_series.input_order"


def test_native_exports_still_admit_the_wire_boundary_order() -> None:
    edge = _ascending(MAX_TRUNCATION_ORDER)
    assert power(edge, 0).result.truncation_order == MAX_TRUNCATION_ORDER
    assert identity_check(edge, edge).status == "EQUAL_MOD_X_TO_N"
    polynomial = to_polynomial(edge).result
    assert polynomial.variables == ("q",)
    assert {
        (term.exponents[0], term.coefficient.as_fraction())
        for term in polynomial.polynomial.terms
    } == {(degree, Fraction(degree + 1)) for degree in range(MAX_TRUNCATION_ORDER)}


def test_identity_check_admits_bounded_inputs_whose_product_would_overflow() -> None:
    import pytest

    from jacobian.math.polynomials.series._models import _SeriesIdentityCheckRequest

    tall = tuple(_coeff(1, 2**800) for _ in range(20))
    left = TruncatedSeries(variable="x", truncation_order=20, coefficients=tall)
    right = TruncatedSeries(variable="x", truncation_order=20, coefficients=tall)

    verdict = identity_check(left, right)
    assert verdict.status == "EQUAL_MOD_X_TO_N"
    assert verdict.first_differing_index is None

    differing = TruncatedSeries(
        variable="x",
        truncation_order=20,
        coefficients=(*tall[:7], _coeff(3, 2**800), *tall[8:]),
    )
    mismatch = identity_check(left, differing)
    assert mismatch.status == "NOT_EQUAL"
    assert mismatch.first_differing_index == 7
    assert mismatch.exact_difference is not None
    assert mismatch.exact_difference.as_fraction() == -2 / 2**800

    payload = {
        "left": left.model_dump(),
        "right": differing.model_dump(),
    }
    admitted = _SeriesIdentityCheckRequest.model_validate(payload)
    assert admitted.right.coefficients == differing.coefficients

    mismatched_request = _SeriesIdentityCheckRequest.model_validate(
        {
            "left": left.model_dump(),
            "right": TruncatedSeries(
                variable="x", truncation_order=19, coefficients=tall[:19]
            ).model_dump(),
        }
    )
    with pytest.raises(OperationDomainValidationError) as error:
        identity_check(
            mismatched_request.left,
            mismatched_request.right,
        )
    assert error.value.errors()[0]["type"] == (
        "formal_power_series.operand_order_mismatch"
    )


def test_truncate_accepts_widened_carrier_orders_and_replays_the_prefix() -> None:
    source = _ascending(1477)
    request = SeriesTruncateRequest.model_validate(
        {"series": source.model_dump(), "target_order": MAX_TRUNCATION_ORDER}
    )
    result = truncate(request.series, request.target_order)
    assert result.result.truncation_order == MAX_TRUNCATION_ORDER
    assert result.result.coefficients == source.coefficients[:MAX_TRUNCATION_ORDER]

    native = truncate(source, 3)
    assert native.result.coefficients == source.coefficients[:3]

    import pytest

    request = SeriesTruncateRequest.model_validate(
        {
            "series": source.model_dump(),
            "target_order": MAX_TRUNCATION_ORDER + 1,
        }
    )
    with pytest.raises(OperationDomainValidationError) as error:
        truncate(request.series, request.target_order)
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.truncate_target_exceeds_public_bound"
    )


def test_truncate_source_order_admission_bounds_runtime_work() -> None:
    import pytest

    edge = _ascending(MAX_TRUNCATE_SOURCE_ORDER)
    request = SeriesTruncateRequest.model_validate(
        {"series": edge.model_dump(), "target_order": 1}
    )
    assert (
        truncate(request.series, request.target_order).result.coefficients
        == edge.coefficients[:1]
    )

    oversized = _ascending(MAX_TRUNCATE_SOURCE_ORDER + 1)
    request = SeriesTruncateRequest.model_validate(
        {"series": oversized.model_dump(), "target_order": 1}
    )
    with pytest.raises(OperationDomainValidationError) as error:
        truncate(request.series, request.target_order)
    assert (
        error.value.errors()[0]["type"] == "formal_power_series.truncate_source_order"
    )


def test_truncate_source_order_bound_is_schema_visible() -> None:
    schema = SeriesTruncateRequest.model_json_schema()
    source_property = schema["$defs"]["TruncatedSeries"]["properties"][
        "truncation_order"
    ]
    assert "maximum" not in source_property


def test_level_one_q_expansion_results_are_consumable_through_truncate() -> None:
    from jacobian.math.number_theory.modular_forms.operations import (
        level_one_named_q_expansion,
    )

    e4 = level_one_named_q_expansion("E4", 3_000).q_expansion
    prefix = truncate(e4, MAX_TRUNCATION_ORDER)
    assert prefix.result.truncation_order == MAX_TRUNCATION_ORDER
    assert prefix.result.coefficients == e4.coefficients[:MAX_TRUNCATION_ORDER]


def test_all_formal_series_results_retain_canonical_types_after_json() -> None:
    from jacobian.math.polynomials.series._tools import TOOLS

    for tool in cast(tuple[MathTool[Any, Any], ...], TOOLS):
        request = tool.request_type.model_validate_json(
            json.dumps(tool.examples[0].input)
        )
        result = tool.run(request)
        assert (
            tool.result_type.model_validate_json(result.model_dump_json()) == result
        ), tool.operation_id


def _fractions(series: TruncatedSeries) -> tuple[Fraction, ...]:
    return tuple(coefficient.as_fraction() for coefficient in series.coefficients)


def _series_of(values: tuple[Fraction, ...]) -> TruncatedSeries:
    return TruncatedSeries(
        variable="x",
        truncation_order=len(values),
        coefficients=tuple(CanonicalRational.from_fraction(value) for value in values),
    )


def test_multiply_matches_cauchy_convolution_oracle() -> None:
    from fractions import Fraction

    left = _series_of((Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(4, 5)))
    right = _series_of((Fraction(1), Fraction(-1), Fraction(1, 3), Fraction(0)))
    result = multiply(left, right)
    left_f, right_f = _fractions(left), _fractions(right)
    expected = tuple(
        sum(left_f[i] * right_f[k - i] for i in range(k + 1))
        for k in range(left.truncation_order)
    )
    assert _fractions(result.result) == expected


def test_add_is_coefficientwise_and_power_is_repeated_multiplication() -> None:
    from fractions import Fraction

    base = _series_of((Fraction(1), Fraction(2), Fraction(3)))
    total = add(base, base)
    assert _fractions(total.result) == (Fraction(2), Fraction(4), Fraction(6))
    cubed = power(base, 3)
    twice = multiply(base, base)
    assert _fractions(cubed.result) == _fractions(multiply(twice.result, base).result)
    assert _fractions(cubed.result)[0] == Fraction(1)


def test_derivative_integral_round_trip_termwise() -> None:
    from fractions import Fraction

    series = _series_of((Fraction(5), Fraction(1, 2), Fraction(-3, 4), Fraction(2)))
    derived = derivative(series)
    assert _fractions(derived.result) == (
        Fraction(1, 2),
        Fraction(-3, 2),
        Fraction(6),
    )
    recovered = integral_zero_constant(derived.result, 4)
    assert _fractions(recovered.result) == (
        Fraction(0),
        Fraction(1, 2),
        Fraction(-3, 4),
        Fraction(2),
    )


def _compose_coefficients(
    outer: tuple[Fraction, ...], inner: tuple[Fraction, ...], order: int
) -> tuple[Fraction, ...]:
    """Direct coefficient composition, independent of the compose kernel."""
    powers = [[Fraction(1)] + [Fraction(0)] * (order - 1)]
    for _ in range(1, order):
        prior = powers[-1]
        powers.append(
            [
                sum(
                    (prior[i] * inner[k - i] for i in range(k + 1)),
                    Fraction(0),
                )
                for k in range(order)
            ]
        )
    return tuple(
        sum(
            (outer[j] * powers[j][k] for j in range(min(len(outer), order))),
            Fraction(0),
        )
        for k in range(order)
    )


def test_compose_matches_nested_evaluation_oracle() -> None:
    from fractions import Fraction

    outer = _series_of((Fraction(1), Fraction(1), Fraction(1, 2)))
    inner = _series_of((Fraction(0), Fraction(1, 3), Fraction(-1, 4)))
    result = compose(outer, inner)
    expected = _compose_coefficients(_fractions(outer), _fractions(inner), 3)
    assert _fractions(result.result) == expected
    # Composing with the zero series yields the constant outer term.
    zero_inner = _series_of((Fraction(0), Fraction(0), Fraction(0)))
    constant = compose(outer, zero_inner)
    assert _fractions(constant.result) == (Fraction(1), Fraction(0), Fraction(0))


def test_truncate_replays_the_source_prefix() -> None:
    from fractions import Fraction

    series = _series_of(
        (Fraction(3), Fraction(1, 7), Fraction(-2), Fraction(9, 5), Fraction(0))
    )
    prefix = truncate(series, 3)
    assert _fractions(prefix.result) == _fractions(series)[:3]
    full = truncate(series, 5)
    assert _fractions(full.result) == _fractions(series)
