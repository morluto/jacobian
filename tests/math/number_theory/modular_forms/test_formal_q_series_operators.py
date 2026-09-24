"""Formal q-series U/V coefficient maps do not assert modularity."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.modular_forms import (
    formal_q_series_u_operator,
    formal_q_series_v_operator,
)
from jacobian.math.polynomials.series._models import TruncatedSeries
from jacobian.math.polynomials.series.operations import scalar_multiply


def _series(values: tuple[int, ...], *, variable: str = "q") -> TruncatedSeries:
    return TruncatedSeries(
        variable=variable,
        truncation_order=len(values),
        coefficients=tuple(CanonicalRational(num=value, den=1) for value in values),
    )


def _integers(series: TruncatedSeries) -> tuple[int, ...]:
    return tuple(int(value.as_fraction()) for value in series.coefficients)


def test_formal_u_and_v_follow_their_coefficient_maps_exactly() -> None:
    source = _series((1, 2, 3, 4, 5, 6, 7))

    u_image = formal_q_series_u_operator(source, 2, 4)
    v_image = formal_q_series_v_operator(source, 2, 7)

    assert type(u_image) is TruncatedSeries
    assert type(v_image) is TruncatedSeries
    assert _integers(u_image) == (1, 3, 5, 7)
    assert _integers(v_image) == (1, 0, 2, 0, 3, 0, 4)
    assert u_image.variable == v_image.variable == "q"
    assert u_image.truncation_order == 4
    assert v_image.truncation_order == 7
    assert not hasattr(u_image, "space")
    assert not hasattr(v_image, "space")


def test_u_v_keep_the_zero_and_order_one_degenerate_prefixes() -> None:
    zero = _series((0,))
    assert _integers(formal_q_series_u_operator(zero, 2, 1)) == (0,)
    assert _integers(formal_q_series_v_operator(zero, 2, 1)) == (0,)

    constant = _series((7,))
    assert _integers(formal_q_series_v_operator(constant, 5, 4)) == (7, 0, 0, 0)


def test_u_v_accept_exact_source_precision_and_reject_one_term_short() -> None:
    assert _integers(formal_q_series_u_operator(_series((1, 2, 3, 4, 5)), 2, 3)) == (
        1,
        3,
        5,
    )
    assert _integers(formal_q_series_v_operator(_series((1, 2, 3)), 2, 5)) == (
        1,
        0,
        2,
        0,
        3,
    )

    with pytest.raises(OperationDomainValidationError) as u_error:
        formal_q_series_u_operator(_series((1, 2, 3, 4)), 2, 3)
    assert u_error.value.errors()[0]["type"] == "formal_q_series.insufficient_precision"

    with pytest.raises(OperationDomainValidationError) as v_error:
        formal_q_series_v_operator(_series((1, 2)), 2, 5)
    assert v_error.value.errors()[0]["type"] == "formal_q_series.insufficient_precision"


def test_formal_prefix_requests_enforce_prime_variable_and_source_envelope() -> None:
    with pytest.raises(OperationDomainValidationError) as prime_error:
        formal_q_series_u_operator(_series((1, 2, 3, 4)), 4, 2)
    assert prime_error.value.errors()[0]["type"] == "formal_q_series.operator_prime"

    with pytest.raises(OperationDomainValidationError) as variable_error:
        formal_q_series_v_operator(_series((1, 2), variable="x"), 2, 2)
    assert variable_error.value.errors()[0]["type"] == "formal_q_series.variable"

    with pytest.raises(OperationDomainValidationError) as precision_error:
        formal_q_series_u_operator(_series((1,)), 2, 0)
    assert (
        precision_error.value.errors()[0]["type"] == "formal_q_series.transform_bounds"
    )

    with pytest.raises(OperationResourceAdmissionError) as source_error:
        formal_q_series_u_operator(_series((1,)), 9_973, 4)
    assert (
        source_error.value.errors()[0]["type"]
        == "formal_q_series.required_source_precision_bound"
    )


def test_selected_coefficient_digits_and_aggregate_output_bytes_are_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.number_theory.modular_forms.transforms as transforms

    oversized = TruncatedSeries(
        variable="q",
        truncation_order=1,
        coefficients=(CanonicalRational(num=10**4096, den=1),),
    )
    with pytest.raises(OperationResourceAdmissionError) as digits_error:
        formal_q_series_v_operator(oversized, 2, 1)
    assert (
        digits_error.value.errors()[0]["type"]
        == "formal_q_series.source_coefficient_bound"
    )

    monkeypatch.setattr(transforms, "MAX_FORMAL_Q_SERIES_OPERATOR_OUTPUT_BYTES", 1)
    with pytest.raises(OperationResourceAdmissionError) as bytes_error:
        formal_q_series_u_operator(_series((1, 2, 3)), 2, 2)
    assert bytes_error.value.errors()[0]["type"] == "formal_q_series.output_bytes_bound"


def test_formal_results_roundtrip_and_compose_with_series_arithmetic() -> None:
    source = _series((1, 2, 3, 4, 5))
    result = formal_q_series_v_operator(source, 2, 5)
    restored = TruncatedSeries.model_validate_json(result.model_dump_json())
    assert restored == result

    doubled = scalar_multiply(restored, CanonicalRational(num=2, den=1)).result
    assert tuple(value.as_fraction() for value in doubled.coefficients) == (
        Fraction(2),
        Fraction(0),
        Fraction(4),
        Fraction(0),
        Fraction(6),
    )


def test_catalog_requests_are_formal_and_return_only_truncated_series() -> None:
    from jacobian.catalog.catalog import Catalog

    catalog = Catalog.open()
    u = catalog.operation("modular_form.formal_q_series.u_operator.compute")
    v = catalog.operation("modular_form.formal_q_series.v_operator.compute")
    assert u is not None and v is not None

    u_result = u.run(
        u.request_type.model_validate_json(
            encode_strict_json(u.examples[0].input), strict=True
        )
    )
    v_result = v.run(
        v.request_type.model_validate_json(
            encode_strict_json(v.examples[0].input), strict=True
        )
    )
    assert type(u_result) is type(v_result) is TruncatedSeries
    assert _integers(u_result) == (1, 3, 5)
    assert _integers(v_result) == (1, 0, 2, 0, 3)
