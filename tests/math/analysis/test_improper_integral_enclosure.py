from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.analysis._definite_integral_enclosure import (
    DefiniteIntegralDomainUnproven,
)
from jacobian.math.analysis._improper_integral_enclosure import (
    EndpointLogImproperIntegralRequest,
    EndpointLogImproperIntegralResult,
    enclose_endpoint_log_improper_integral,
)


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


def test_both_endpoint_log_factors_use_the_same_truncation() -> None:
    result = enclose_endpoint_log_improper_integral(
        _request(right_log_power=1, target_width={"mantissa": 1, "exponent": -1})
    )
    assert result.left_tail.truncation == result.right_tail.truncation
    assert result.left_tail.enclosure.lower.as_fraction() <= 0
    assert result.right_tail.enclosure.upper.as_fraction() >= 0


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


def test_tail_target_beyond_the_admitted_truncation_is_rejected() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        enclose_endpoint_log_improper_integral(
            _request(
                left_log_power=12,
                target_width={"mantissa": 1, "exponent": -100},
            )
        )


def test_forged_tail_does_not_round_trip() -> None:
    result = enclose_endpoint_log_improper_integral(_request())
    forged = result.model_dump()
    forged["left_tail"]["enclosure"]["upper"] = {"num": 0, "den": 1}
    with pytest.raises(ValidationError):
        EndpointLogImproperIntegralResult.model_validate(forged)


def test_forged_combined_enclosure_does_not_round_trip() -> None:
    result = enclose_endpoint_log_improper_integral(_request())
    forged = result.model_dump()
    forged["enclosure"]["upper"] = {"num": 0, "den": 1}
    with pytest.raises(ValidationError):
        EndpointLogImproperIntegralResult.model_validate(forged)
