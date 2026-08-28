"""Tests for truncated formal power series operations."""

from typing import Any, cast

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.polynomials.series import (
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
    InputTruncatedSeries,
    SeriesInverseRequest,
    SeriesPowerRequest,
    SeriesTruncateRequest,
    TruncatedSeries,
)


def _coeff(num: str, den: str = "1") -> CanonicalRational:
    return CanonicalRational(num=num, den=den)


def _ascending(order: int) -> TruncatedSeries:
    return TruncatedSeries(
        variable="q",
        truncation_order=order,
        coefficients=tuple(_coeff(str(index + 1)) for index in range(order)),
    )


def test_derivative_of_order_one_is_zero() -> None:
    series = TruncatedSeries(
        variable="x",
        truncation_order=1,
        coefficients=(_coeff("7"),),
    )
    result = derivative(series)
    assert result.result.truncation_order == 1
    assert result.result.coefficients[0].as_fraction() == 0


def test_native_exports_call_the_shared_typed_kernels() -> None:
    series = TruncatedSeries(
        variable="x",
        truncation_order=2,
        coefficients=(_coeff("1"), _coeff("2")),
    )

    assert derivative(series) == derivative(series)
    assert multiply(series, series) == multiply(series, series)
    assert to_polynomial(series) == to_polynomial(series)


def test_power_rejects_result_digit_overflow() -> None:
    import pytest

    huge = "1" + "0" * (MAX_RATIONAL_DIGITS - 1)
    request = SeriesPowerRequest(
        series=InputTruncatedSeries(
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
        coefficients=(_coeff("1"), _coeff("1")),
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
        series=InputTruncatedSeries(
            variable="x",
            truncation_order=2,
            coefficients=(_coeff("1"), _coeff("0")),
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
        coefficients=(_coeff("0"), _coeff("1")),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        inverse(request.as_series())
    assert (
        error.value.errors()[0]["type"] == "formal_power_series.inverse_zero_constant"
    )


def test_inverse_rejects_result_coefficient_growth() -> None:
    import pytest

    huge = "1" + "0" * (MAX_RATIONAL_DIGITS - 1)
    request = SeriesInverseRequest(
        variable="x",
        truncation_order=20,
        coefficients=(
            _coeff("1"),
            _coeff("-" + huge),
            *(_coeff("0") for _ in range(18)),
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

    huge = "1" + "0" * MAX_RATIONAL_DIGITS
    oversized = InputTruncatedSeries(
        variable="x",
        truncation_order=1,
        coefficients=(_coeff(huge),),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        derivative(oversized)
    assert error.value.errors()[0]["type"] == "formal_power_series.admission"


def test_product_can_exceed_input_digit_bound() -> None:
    large = "1" + "0" * (MAX_RATIONAL_DIGITS - 1)
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
    assert len(value.num.lstrip("-")) > MAX_RATIONAL_DIGITS
    assert len(value.num.lstrip("-")) <= 4096


def test_native_exports_admit_inputs_before_kernel_work() -> None:
    import pytest

    wide = _ascending(513)
    with pytest.raises(OperationDomainValidationError) as error:
        multiply(wide, wide)
    assert error.value.errors()[0]["type"] == "formal_power_series.input_order"
    with pytest.raises(OperationDomainValidationError) as error:
        power(wide, 2)
    assert error.value.errors()[0]["type"] == "formal_power_series.input_order"
    with pytest.raises(OperationDomainValidationError) as error:
        compose(wide, wide)
    assert error.value.errors()[0]["type"] == "formal_power_series.input_order"
    with pytest.raises(OperationDomainValidationError) as error:
        to_polynomial(wide)
    assert error.value.errors()[0]["type"] == "formal_power_series.input_order"

    tall = "1" + "0" * MAX_RATIONAL_DIGITS
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
        coefficients=(_coeff("1"), _coeff("2"), _coeff("0")),
    )
    inner = TruncatedSeries(
        variable="x",
        truncation_order=3,
        coefficients=(_coeff("0"), _coeff("1"), _coeff("0")),
    )
    reversible = TruncatedSeries(
        variable="x",
        truncation_order=3,
        coefficients=(_coeff("0"), _coeff("1"), _coeff("1")),
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
    assert to_polynomial(edge) == to_polynomial(edge)


def test_identity_check_admits_bounded_inputs_whose_product_would_overflow() -> None:
    import pytest

    from jacobian.math.polynomials.series._models import _SeriesIdentityCheckRequest

    tall = tuple(_coeff("1", str(2**800)) for _ in range(20))
    left = TruncatedSeries(variable="x", truncation_order=20, coefficients=tall)
    right = TruncatedSeries(variable="x", truncation_order=20, coefficients=tall)

    verdict = identity_check(left, right)
    assert verdict.status == "EQUAL_MOD_X_TO_N"
    assert verdict.first_differing_index is None

    differing = TruncatedSeries(
        variable="x",
        truncation_order=20,
        coefficients=(*tall[:7], _coeff("3", str(2**800)), *tall[8:]),
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


def test_relaxed_identity_check_request_is_versioned_as_version_two() -> None:
    from jacobian.math.polynomials.series._tools import TOOLS

    {tool.operation_id: tool for tool in TOOLS}


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


def test_truncate_source_admission_bounds_the_request_before_parsing() -> None:
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
    source_property = schema["$defs"]["TruncateSourceSeries"]["properties"][
        "truncation_order"
    ]
    assert "maximum" not in source_property


def test_level_one_q_expansion_results_are_consumable_through_truncate() -> None:
    from jacobian.math.number_theory.modular_forms.operations import (
        level_one_named_q_expansion,
    )

    e4 = level_one_named_q_expansion("E4", 1477).q_expansion
    prefix = truncate(e4, MAX_TRUNCATION_ORDER)
    assert prefix.result.truncation_order == MAX_TRUNCATION_ORDER
    assert prefix.result.coefficients == e4.coefficients[:MAX_TRUNCATION_ORDER]


def test_widened_truncate_request_is_versioned_as_version_three() -> None:
    from jacobian.math.polynomials.series._tools import TOOLS

    {tool.operation_id: tool for tool in TOOLS}


def test_truncate_accepts_a_large_canonical_modular_series() -> None:
    from jacobian.math.number_theory.modular_forms.kernel import (
        eisenstein_coefficients,
        metadata,
    )
    from jacobian.math.number_theory.modular_forms.values import (
        LevelOneModularQExpansion,
    )

    weight, space_kind, normalization = metadata("E4")
    coefficients = eisenstein_coefficients("E4", 3_000)
    value = LevelOneModularQExpansion.model_validate(
        {
            "form": "E4",
            "weight": weight,
            "space_kind": space_kind,
            "normalization": normalization,
            "q_expansion": TruncatedSeries(
                variable="q",
                truncation_order=3_000,
                coefficients=tuple(
                    _coeff(str(term.numerator), str(term.denominator))
                    for term in coefficients
                ),
            ).model_dump(),
        }
    )
    prefix = truncate(value.q_expansion, MAX_TRUNCATION_ORDER)
    assert prefix.result.truncation_order == MAX_TRUNCATION_ORDER
    assert (
        prefix.result.coefficients[-1].as_fraction()
        == eisenstein_coefficients("E4", MAX_TRUNCATION_ORDER)[-1]
    )
