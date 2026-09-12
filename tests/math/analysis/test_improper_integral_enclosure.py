from __future__ import annotations

from fractions import Fraction
from time import sleep

import pytest
from pydantic import ValidationError

from jacobian._execution import OperationExecutionTimeoutError
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.analysis import _improper_integral_enclosure as improper_module
from jacobian.math.analysis._definite_integral_enclosure import (
    DefiniteIntegralDomainUnproven,
)
from jacobian.math.analysis._improper_integral_enclosure import (
    EndpointLogImproperIntegralRequest,
    EndpointLogImproperIntegralResult,
    enclose_endpoint_log_improper_integral,
)
from jacobian.math.analysis._models import MAX_DYADIC_EXPONENT


def _request(**updates: object) -> EndpointLogImproperIntegralRequest:
    payload: dict[str, object] = {
        "smooth_expression": {"op": "const", "value": {"num": 1, "den": 1}},
        "interval": {
            "lower": {"num": 0, "den": 1},
            "upper": {"num": 1, "den": 1},
        },
        "variable": "x",
        "left_log_power": 1,
        "target_width": {"mantissa": 1, "exponent": -2},
        "max_leaves": 128,
    }
    payload.update(updates)
    return EndpointLogImproperIntegralRequest.model_validate(payload)


def _combined_bounds(
    result: EndpointLogImproperIntegralResult,
) -> tuple[Fraction, Fraction]:
    quadratures = (result.left_quadrature, result.right_quadrature)
    lower = Fraction()
    upper = Fraction()
    for quadrature in quadratures:
        outcome = quadrature.outcome
        assert not isinstance(outcome, DefiniteIntegralDomainUnproven)
        lower += outcome.enclosure.lower.as_fraction()
        upper += outcome.enclosure.upper.as_fraction()
    lower += result.left_tail.enclosure.lower.as_fraction()
    lower += result.right_tail.enclosure.lower.as_fraction()
    upper += result.left_tail.enclosure.upper.as_fraction()
    upper += result.right_tail.enclosure.upper.as_fraction()
    return lower, upper


def test_left_logarithm_encloses_its_exact_unit_integral() -> None:
    result = enclose_endpoint_log_improper_integral(_request())
    lower, upper = _combined_bounds(result)
    assert lower <= 1 <= upper
    assert result.left_quadrature.outcome.status == "TARGET_MET"
    assert result.right_quadrature.outcome.status == "TARGET_MET"
    assert result.enclosure.lower.as_fraction() == lower
    assert result.enclosure.upper.as_fraction() == upper
    assert result.model_validate_json(result.model_dump_json()) == result


def test_squared_left_logarithm_encloses_its_exact_integral() -> None:
    result = enclose_endpoint_log_improper_integral(
        _request(left_log_power=2, target_width={"mantissa": 1, "exponent": -1})
    )

    assert result.enclosure.lower.as_fraction() <= 2
    assert result.enclosure.upper.as_fraction() >= 2


@pytest.mark.parametrize(
    ("left_log_power", "right_log_power", "exact"),
    ((1, 0, Fraction(33, 4)), (0, 1, Fraction(51, 4))),
)
def test_endpoint_log_orientation_on_an_asymmetric_interval(
    left_log_power: int, right_log_power: int, exact: Fraction
) -> None:
    result = enclose_endpoint_log_improper_integral(
        _request(
            smooth_expression={"op": "var", "variable": "x"},
            interval={
                "lower": {"num": 2, "den": 1},
                "upper": {"num": 5, "den": 1},
            },
            left_log_power=left_log_power,
            right_log_power=right_log_power,
            target_width={"mantissa": 1, "exponent": -3},
        )
    )

    assert (
        result.enclosure.lower.as_fraction()
        <= exact
        <= (result.enclosure.upper.as_fraction())
    )


def test_both_endpoint_log_factors_use_the_same_truncation() -> None:
    result = enclose_endpoint_log_improper_integral(
        _request(right_log_power=1, target_width={"mantissa": 1, "exponent": -1})
    )
    assert result.left_tail.truncation == result.right_tail.truncation
    assert result.left_tail.enclosure.lower.as_fraction() <= 0
    assert result.right_tail.enclosure.upper.as_fraction() >= 0


def test_refining_the_target_returns_a_narrower_valid_enclosure() -> None:
    coarse = enclose_endpoint_log_improper_integral(
        _request(target_width={"mantissa": 1, "exponent": -2})
    ).enclosure
    fine = enclose_endpoint_log_improper_integral(
        _request(target_width={"mantissa": 1, "exponent": -5})
    ).enclosure

    coarse_width = coarse.upper.as_fraction() - coarse.lower.as_fraction()
    fine_width = fine.upper.as_fraction() - fine.lower.as_fraction()
    assert coarse.lower.as_fraction() <= 1 <= coarse.upper.as_fraction()
    assert fine.lower.as_fraction() <= 1 <= fine.upper.as_fraction()
    assert fine_width < coarse_width


def test_symmetric_endpoint_logs_enclose_an_exact_cancellation() -> None:
    result = enclose_endpoint_log_improper_integral(
        _request(
            smooth_expression={
                "op": "sub",
                "children": [
                    {
                        "op": "mul",
                        "children": [
                            {"op": "const", "value": {"num": 2, "den": 1}},
                            {"op": "var", "variable": "x"},
                        ],
                    },
                    {"op": "const", "value": {"num": 1, "den": 1}},
                ],
            },
            left_log_power=1,
            right_log_power=1,
            target_width={"mantissa": 1, "exponent": -2},
        )
    )

    assert result.enclosure.lower.as_fraction() <= 0
    assert result.enclosure.upper.as_fraction() >= 0


def test_closed_box_unsafe_smooth_factor_is_rejected_before_quadrature() -> None:
    with pytest.raises(OperationDomainValidationError):
        enclose_endpoint_log_improper_integral(
            _request(
                smooth_expression={
                    "op": "log",
                    "children": [{"op": "var", "variable": "x"}],
                }
            )
        )


def test_divergent_smooth_factor_is_outside_the_proved_class() -> None:
    with pytest.raises(OperationDomainValidationError):
        enclose_endpoint_log_improper_integral(
            _request(
                smooth_expression={
                    "op": "div",
                    "children": [
                        {"op": "const", "value": {"num": 1, "den": 1}},
                        {"op": "var", "variable": "x"},
                    ],
                }
            )
        )


def test_smooth_factor_intermediate_overflow_is_typed_admission() -> None:
    with pytest.raises(
        OperationDomainValidationError,
        match="smooth factor exceeds the admitted exact preflight bound",
    ):
        enclose_endpoint_log_improper_integral(
            _request(
                smooth_expression={
                    "op": "exp",
                    "children": [{"op": "var", "variable": "x"}],
                },
                interval={
                    "lower": {"num": 10_000, "den": 1},
                    "upper": {"num": 10_001, "den": 1},
                },
            )
        )


def test_large_positive_target_width_does_not_require_a_large_fraction() -> None:
    result = enclose_endpoint_log_improper_integral(
        _request(target_width={"mantissa": 1, "exponent": MAX_DYADIC_EXPONENT})
    )

    assert result.left_tail.truncation == 1
    assert result.right_tail.truncation == 1


def test_direct_native_call_checks_deadline_after_combined_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_compute = improper_module.__dict__["_compute_definite_integral_enclosure"]
    calls = 0

    def delayed_second_quadrature(request: object) -> object:
        nonlocal calls
        result = original_compute(request)
        calls += 1
        if calls == 2:
            sleep(1.05)
        return result

    monkeypatch.setattr(
        improper_module,
        "_compute_definite_integral_enclosure",
        delayed_second_quadrature,
    )
    with pytest.raises(OperationExecutionTimeoutError):
        enclose_endpoint_log_improper_integral(_request(wall_seconds=1))


def test_tail_target_beyond_the_admitted_truncation_is_rejected() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        enclose_endpoint_log_improper_integral(
            _request(
                left_log_power=12,
                target_width={"mantissa": 1, "exponent": -100},
            )
        )


def test_transformed_expression_growth_is_typed_operation_admission() -> None:
    expression: dict[str, object] = {"op": "var", "variable": "x"}
    for _ in range(7):
        expression = {
            "op": "add",
            "children": [expression, {"op": "var", "variable": "x"}],
        }
    request = _request(smooth_expression=expression)

    with pytest.raises(
        OperationResourceAdmissionError,
        match="endpoint substitution exceeds",
    ):
        enclose_endpoint_log_improper_integral(request)


def test_forged_tail_does_not_round_trip() -> None:
    result = enclose_endpoint_log_improper_integral(_request())
    forged = result.model_dump()
    forged["left_tail"]["enclosure"]["upper"] = {"num": 0, "den": 1}
    with pytest.raises(ValidationError):
        EndpointLogImproperIntegralResult.model_validate(forged)


def test_result_decoding_does_not_replay_combined_enclosure_arithmetic() -> None:
    result = enclose_endpoint_log_improper_integral(_request())
    forged = result.model_dump()
    forged["enclosure"] = {
        "lower": {"num": -99, "den": 1},
        "upper": {"num": 99, "den": 1},
    }

    decoded = EndpointLogImproperIntegralResult.model_validate(forged)

    assert decoded.enclosure.lower.as_fraction() == -99
    assert decoded.enclosure.upper.as_fraction() == 99


def test_malformed_direct_native_request_is_typed() -> None:
    with pytest.raises(OperationDomainValidationError, match="validated endpoint-log"):
        enclose_endpoint_log_improper_integral("not a request")  # type: ignore[arg-type]
