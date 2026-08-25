"""Tests for truncated formal power series operations."""

from jacobian.math.formal_power_series import (
    compose,
    derivative,
    identity_check,
    multiply,
    power,
    to_polynomial,
    truncate,
)
from jacobian.math.formal_power_series._models import (
    MAX_RATIONAL_DIGITS,
    MAX_TRUNCATE_SOURCE_ORDER,
    MAX_TRUNCATION_ORDER,
    InputTruncatedSeries,
    SeriesInverseRequest,
    SeriesTruncateRequest,
    TruncatedSeries,
)
from jacobian.math.formal_power_series._operations import (
    compute_derivative,
    compute_multiply,
    compute_to_polynomial,
    compute_truncate,
)


def _coeff(num: str, den: str = "1") -> dict[str, str]:
    return {"num": num, "den": den}


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
    result = compute_derivative(series)
    assert result.result.truncation_order == 1
    assert result.result.coefficients[0].as_fraction() == 0


def test_native_projection_aliases_call_the_shared_typed_kernels() -> None:
    series = TruncatedSeries(
        variable="x",
        truncation_order=2,
        coefficients=(_coeff("1"), _coeff("2")),
    )

    assert derivative(series) == compute_derivative(series)
    assert multiply(series, series) == compute_multiply(series, series)
    assert to_polynomial(series) == compute_to_polynomial(series)


def test_power_rejects_result_digit_overflow() -> None:
    import pytest
    from pydantic import ValidationError

    from jacobian.math.formal_power_series._models import SeriesPowerRequest

    huge = "1" + "0" * (MAX_RATIONAL_DIGITS - 1)
    with pytest.raises(ValidationError) as error:
        SeriesPowerRequest(
            series=InputTruncatedSeries(
                variable="x",
                truncation_order=1,
                coefficients=(_coeff(huge),),
            ),
            exponent=1000,
        )
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.power_coefficient_growth"
    )


def test_reversion_rejects_nonzero_constant() -> None:
    import pytest
    from pydantic import ValidationError

    from jacobian.math.formal_power_series._models import SeriesReversionRequest

    with pytest.raises(ValidationError) as error:
        SeriesReversionRequest(
            variable="x",
            truncation_order=2,
            coefficients=(_coeff("1"), _coeff("1")),
        )
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.reversion_nonzero_constant"
    )


def test_integral_rejects_oversized_output_order() -> None:
    import pytest
    from pydantic import ValidationError

    from jacobian.math.formal_power_series._models import SeriesIntegralRequest

    with pytest.raises(ValidationError) as error:
        SeriesIntegralRequest(
            series=InputTruncatedSeries(
                variable="x",
                truncation_order=2,
                coefficients=(_coeff("1"), _coeff("0")),
            ),
            output_order=4,
        )
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.integral_output_order_exceeds_source"
    )


def test_inverse_rejects_zero_constant() -> None:
    import pytest
    from pydantic import ValidationError

    from jacobian.math.formal_power_series._models import SeriesInverseRequest

    with pytest.raises(ValidationError) as error:
        SeriesInverseRequest(
            variable="x",
            truncation_order=2,
            coefficients=(_coeff("0"), _coeff("1")),
        )
    assert (
        error.value.errors()[0]["type"] == "formal_power_series.inverse_zero_constant"
    )


def test_inverse_rejects_result_coefficient_growth() -> None:
    import pytest
    from pydantic import ValidationError

    huge = "1" + "0" * (MAX_RATIONAL_DIGITS - 1)
    with pytest.raises(ValidationError) as error:
        SeriesInverseRequest(
            variable="x",
            truncation_order=20,
            coefficients=(
                _coeff("1"),
                _coeff("-" + huge),
                *(_coeff("0") for _ in range(18)),
            ),
        )
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.inverse_coefficient_growth"
    )


def test_input_series_rejects_oversized_coefficients() -> None:
    import pytest
    from pydantic import ValidationError

    huge = "1" + "0" * MAX_RATIONAL_DIGITS
    with pytest.raises(ValidationError) as error:
        InputTruncatedSeries(
            variable="x",
            truncation_order=1,
            coefficients=(_coeff(huge),),
        )
    assert error.value.errors()[0]["type"] == "value_error"


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
    result = compute_multiply(left, right)
    value = result.result.coefficients[0]
    assert len(value.num.lstrip("-")) > MAX_RATIONAL_DIGITS
    assert len(value.num.lstrip("-")) <= 4096


def test_native_exports_admit_inputs_before_kernel_work() -> None:
    import pytest
    from pydantic import ValidationError

    wide = _ascending(513)
    with pytest.raises(ValidationError) as error:
        multiply(wide, wide)
    assert error.value.errors()[0]["type"] == "less_than_equal"
    with pytest.raises(ValidationError) as error:
        power(wide, 2)
    assert error.value.errors()[0]["type"] == "less_than_equal"
    with pytest.raises(ValidationError) as error:
        compose(wide, wide)
    assert error.value.errors()[0]["type"] == "less_than_equal"
    with pytest.raises(ValidationError) as error:
        to_polynomial(wide)
    assert error.value.errors()[0]["type"] == "less_than_equal"

    tall = "1" + "0" * MAX_RATIONAL_DIGITS
    oversized = TruncatedSeries(
        variable="x",
        truncation_order=1,
        coefficients=(_coeff(tall),),
    )
    with pytest.raises(ValidationError) as error:
        multiply(oversized, oversized)
    assert error.value.errors()[0]["type"] == "value_error"


def test_native_exports_still_admit_the_wire_boundary_order() -> None:
    edge = _ascending(MAX_TRUNCATION_ORDER)
    assert power(edge, 0).result.truncation_order == MAX_TRUNCATION_ORDER
    assert identity_check(edge, edge).status == "EQUAL_MOD_X_TO_N"
    assert to_polynomial(edge) == compute_to_polynomial(edge)


def test_identity_check_admits_bounded_inputs_whose_product_would_overflow() -> None:
    import pytest
    from pydantic import ValidationError

    from jacobian.math.formal_power_series._models import _SeriesIdentityCheckRequest

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
    assert mismatch.exact_difference.as_fraction() == -2 / 2**800

    payload = {
        "left": left.model_dump(),
        "right": differing.model_dump(),
    }
    admitted = _SeriesIdentityCheckRequest.model_validate(payload)
    assert admitted.right.coefficients == differing.coefficients

    with pytest.raises(ValidationError) as error:
        _SeriesIdentityCheckRequest(
            left=left.model_dump(),
            right=TruncatedSeries(
                variable="x", truncation_order=19, coefficients=tall[:19]
            ).model_dump(),
        )
    assert (
        error.value.errors()[0]["type"] == "formal_power_series.operand_order_mismatch"
    )


def test_relaxed_identity_check_request_is_versioned_as_version_two() -> None:
    from jacobian.math.formal_power_series._tools import TOOLS

    {tool.operation_id: tool for tool in TOOLS}


def test_truncate_accepts_widened_carrier_orders_and_replays_the_prefix() -> None:
    source = _ascending(1477)
    request = SeriesTruncateRequest(
        series=source.model_dump(), target_order=MAX_TRUNCATION_ORDER
    )
    result = compute_truncate(request.series, request.target_order)
    assert result.result.truncation_order == MAX_TRUNCATION_ORDER
    assert result.result.coefficients == source.coefficients[:MAX_TRUNCATION_ORDER]

    native = truncate(source, 3)
    assert native.result.coefficients == source.coefficients[:3]

    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as error:
        SeriesTruncateRequest(
            series=source.model_dump(), target_order=MAX_TRUNCATION_ORDER + 1
        )
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.truncate_target_exceeds_public_bound"
    )


def test_truncate_source_admission_bounds_the_request_before_parsing() -> None:
    import pytest
    from pydantic import ValidationError

    edge = _ascending(MAX_TRUNCATE_SOURCE_ORDER)
    request = SeriesTruncateRequest(series=edge.model_dump(), target_order=1)
    assert (
        compute_truncate(request.series, request.target_order).result.coefficients
        == edge.coefficients[:1]
    )

    oversized = _ascending(MAX_TRUNCATE_SOURCE_ORDER + 1)
    with pytest.raises(ValidationError) as error:
        SeriesTruncateRequest(series=oversized.model_dump(), target_order=1)
    assert error.value.errors()[0]["type"] == "less_than_equal"


def test_truncate_source_order_bound_is_schema_visible() -> None:
    schema = SeriesTruncateRequest.model_json_schema()
    source_property = schema["$defs"]["TruncateSourceSeries"]["properties"][
        "truncation_order"
    ]
    assert source_property["maximum"] == MAX_TRUNCATE_SOURCE_ORDER


def test_level_one_q_expansion_results_are_consumable_through_truncate() -> None:
    from jacobian.math.modular_forms.operations import level_one_named_q_expansion

    e4 = level_one_named_q_expansion("E4", 1477).q_expansion
    prefix = truncate(e4, MAX_TRUNCATION_ORDER)
    assert prefix.result.truncation_order == MAX_TRUNCATION_ORDER
    assert prefix.result.coefficients == e4.coefficients[:MAX_TRUNCATION_ORDER]


def test_widened_truncate_request_is_versioned_as_version_three() -> None:
    from jacobian.math.formal_power_series._tools import TOOLS

    {tool.operation_id: tool for tool in TOOLS}


def test_truncate_source_ceiling_covers_the_level_one_replay_envelope() -> None:
    from jacobian.math.modular_forms.kernel import (
        eisenstein_coefficients,
        metadata,
        require_level_one_replay,
    )
    from jacobian.math.modular_forms.values import LevelOneModularQExpansion

    assert require_level_one_replay("E4", MAX_TRUNCATE_SOURCE_ORDER) is None
    widest, lo, hi = 1, 1, MAX_TRUNCATE_SOURCE_ORDER + 1
    while lo < hi:
        middle = (lo + hi + 1) // 2
        try:
            require_level_one_replay("E4", middle)
        except ValueError:
            hi = middle - 1
        else:
            widest = middle
            lo = middle
    assert widest == MAX_TRUNCATE_SOURCE_ORDER

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
