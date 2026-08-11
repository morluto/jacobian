from __future__ import annotations

import pytest

from jacobian.domains.polynomial.groebner_protocol import (
    GroebnerWorkerResultLimitExceeded,
    parse_groebner_worker_response,
)


def test_groebner_worker_limit_response_has_no_result_variant() -> None:
    response = parse_groebner_worker_response(
        {
            "protocol": "jacobian.polynomial.groebner.sympy.v1",
            "error": {
                "code": "POLYNOMIAL_GROEBNER_RESULT_LIMIT_EXCEEDED",
                "message": "basis output exceeds the bound",
            },
        }
    )

    assert isinstance(response, GroebnerWorkerResultLimitExceeded)


def test_groebner_worker_response_rejects_mixed_result_and_error() -> None:
    with pytest.raises(ValueError, match="invalid Gröbner worker response"):
        parse_groebner_worker_response(
            {
                "protocol": "jacobian.polynomial.groebner.sympy.v1",
                "result": {},
                "error": {
                    "code": "POLYNOMIAL_GROEBNER_RESULT_LIMIT_EXCEEDED",
                    "message": "basis output exceeds the bound",
                },
            }
        )
