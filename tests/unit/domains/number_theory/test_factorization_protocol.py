from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.domains.number_theory.factorization_protocol import (
    DivisorsWorkerResponse,
    parse_factorization_worker_request,
    parse_factorization_worker_response,
)


def test_request_operation_selects_its_exact_request_contract() -> None:
    with pytest.raises(ValidationError):
        parse_factorization_worker_request(
            {
                "protocol": "jacobian.number-theory.factorization.sympy.v1",
                "operation": "powerful",
                "request": {"n": 72},
            }
        )


def test_response_operation_selects_and_binds_its_exact_result_contract() -> None:
    response = {
        "protocol": "jacobian.number-theory.factorization.sympy.v1",
        "operation": "divisors",
        "result": {"divisors": ["1", "2", "4"]},
    }
    parsed = parse_factorization_worker_response(
        response,
        expected_operation="divisors",
    )
    assert isinstance(parsed, DivisorsWorkerResponse)

    with pytest.raises(ValueError, match="does not match"):
        parse_factorization_worker_response(
            response,
            expected_operation="proper_divisors",
        )
    with pytest.raises(ValueError, match="invalid factorization worker response"):
        parse_factorization_worker_response(
            {**response, "result": {"holds": True}},
            expected_operation="divisors",
        )
