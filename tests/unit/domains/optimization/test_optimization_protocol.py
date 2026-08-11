from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.domains.optimization.protocol import (
    parse_optimization_worker_request,
    parse_optimization_worker_response,
)


def _program() -> dict[str, object]:
    return {
        "variables": ["x"],
        "objective": [{"num": "1", "den": "1"}],
        "coefficients": [[{"num": "1", "den": "1"}]],
        "rhs": [{"num": "1", "den": "1"}],
    }


def test_optimization_request_parses_before_worker_execution() -> None:
    request = parse_optimization_worker_request(
        {
            "protocol": "jacobian.optimization.rational-linear/v1",
            "request": {"program": _program(), "wall_seconds": 1},
        }
    )
    assert request.request.program.variables == ("x",)

    invalid_program = _program()
    invalid_program["objective"] = []
    with pytest.raises(ValidationError):
        parse_optimization_worker_request(
            {
                "protocol": "jacobian.optimization.rational-linear/v1",
                "request": {"program": invalid_program, "wall_seconds": 1},
            }
        )


def test_optimization_response_is_a_closed_typed_result() -> None:
    response = parse_optimization_worker_response(
        {
            "protocol": "jacobian.optimization.rational-linear/v1",
            "result": {
                "status": "NO_CERTIFICATE",
                "detail": "the solver produced no exact certificate",
            },
        }
    )
    assert response.result.status == "NO_CERTIFICATE"

    with pytest.raises(ValueError, match="invalid rational optimization"):
        parse_optimization_worker_response(
            {
                "protocol": "jacobian.optimization.rational-linear/v1",
                "result": {
                    "status": "NO_CERTIFICATE",
                    "detail": "invalid extra candidate",
                    "primal_candidate": [{"num": "1", "den": "1"}],
                },
            }
        )
