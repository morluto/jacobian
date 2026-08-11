from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.domains.analysis.protocol import (
    ArbEnclosedWorkerResponse,
    parse_arb_worker_request,
    parse_arb_worker_response,
)


def test_arb_request_parses_the_complete_public_request_before_execution() -> None:
    request = parse_arb_worker_request(
        {
            "protocol": "jacobian.analysis.arb-point-enclosure/v1",
            "request": {
                "function": "SQRT",
                "argument": {"num": "4", "den": "1"},
                "precision_bits": 32,
                "wall_seconds": 1,
            },
        }
    )
    assert request.request.argument.as_integer_ratio() == (4, 1)

    with pytest.raises(ValidationError):
        parse_arb_worker_request(
            {
                "protocol": "jacobian.analysis.arb-point-enclosure/v1",
                "request": {
                    "function": "SQRT",
                    "argument": {"num": "4", "den": "0"},
                    "precision_bits": 32,
                },
            }
        )


def test_arb_response_status_selects_one_closed_payload_shape() -> None:
    enclosed = parse_arb_worker_response(
        {
            "protocol": "jacobian.analysis.arb-point-enclosure/v1",
            "status": "ENCLOSED",
            "lower": {"mantissa": "1", "exponent": 0},
            "upper": {"mantissa": "1", "exponent": 0},
            "relative_accuracy_bits": None,
            "exact": True,
        }
    )
    assert isinstance(enclosed, ArbEnclosedWorkerResponse)

    with pytest.raises(ValueError, match="invalid Arb"):
        parse_arb_worker_response(
            {
                "protocol": "jacobian.analysis.arb-point-enclosure/v1",
                "status": "NONFINITE",
                "lower": {"mantissa": "1", "exponent": 0},
            }
        )

    with pytest.raises(ValueError, match="invalid Arb"):
        parse_arb_worker_response(
            {
                "protocol": "jacobian.analysis.arb-point-enclosure/v1",
                "status": "ENCLOSED",
                "lower": {"mantissa": "1", "exponent": 0},
                "upper": {"mantissa": "1", "exponent": 0},
                "relative_accuracy_bits": None,
                "exact": "true",
            }
        )
