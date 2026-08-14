"""Independent SymPy normalization checker and certificate-binding tests."""

from __future__ import annotations

from typing import Any

from tests.support.operations import invoke_operation as _invoke
from tests.support.rationals import rational_payload as _q

from jacobian.contracts.results import ExecutionStatus
from jacobian.process_policy import ProcessResult, ProcessTermination


def _variable(name: str) -> dict[str, Any]:
    return {"kind": "variable", "name": name}


def _expression(
    node: dict[str, Any], *, variables: list[str] | None = None
) -> dict[str, Any]:
    return {"variables": variables or ["x", "y"], "expression": node}


def _difference_of_squares_plus_half_x() -> dict[str, Any]:
    return _expression(
        {
            "kind": "add",
            "operands": [
                {
                    "kind": "multiply",
                    "operands": [
                        {"kind": "add", "operands": [_variable("x"), _variable("y")]},
                        {
                            "kind": "add",
                            "operands": [
                                _variable("x"),
                                {"kind": "negate", "operand": _variable("y")},
                            ],
                        },
                    ],
                },
                {
                    "kind": "multiply",
                    "operands": [
                        {"kind": "rational", "value": _q(1, 2)},
                        _variable("x"),
                    ],
                },
            ],
        }
    )


def test_independent_checker_verifies_full_ast_relation(
    authorized_polynomial_normalization_services,
) -> None:
    runtime = authorized_polynomial_normalization_services
    computed = _invoke(
        runtime,
        "polynomial.expression.normalize",
        {"expression": _difference_of_squares_plus_half_x()},
    )
    verified = _invoke(
        runtime,
        "polynomial.expression_normalization.verify",
        {"normalization_uri": computed.output["normalization_uri"]},
    )
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED_NORMALIZATION"
    assert verified.output["conclusion"] == "TRUE"
    assert verified.output["verification_record_uri"].startswith("artifact://sha256/")
    assert verified.verification_record_uri is not None


def test_independent_checker_rejects_wrong_bound_coefficients(
    authorized_polynomial_normalization_services,
) -> None:
    runtime = authorized_polynomial_normalization_services
    expression_uri = runtime.core.polynomial_expressions.put_expression(
        _expression(_variable("x"), variables=["x"])
    ).artifact_uri
    candidate = runtime.core.polynomial_expressions.put_normalization(
        expression_uri=expression_uri,
        normalized={"terms": []},
        producer=runtime.producer,
        resource_budget={"wall_seconds": 5},
    )
    rejected = _invoke(
        runtime,
        "polynomial.expression_normalization.verify",
        {"normalization_uri": candidate.artifact_uri},
    )
    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_normalization_checker_timeout_is_operational(
    authorized_polynomial_normalization_services,
    monkeypatch,
) -> None:
    runtime = authorized_polynomial_normalization_services
    computed = _invoke(
        runtime,
        "polynomial.expression.normalize",
        {"expression": _expression(_variable("x"), variables=["x"])},
    )
    monkeypatch.setattr(
        "jacobian.verification.executor.execute_process",
        lambda *_args, **_kwargs: ProcessResult(
            termination=ProcessTermination.TIMED_OUT,
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        ),
    )
    result = _invoke(
        runtime,
        "polynomial.expression_normalization.verify",
        {"normalization_uri": computed.output["normalization_uri"]},
    )
    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.output == {}
    assert result.verification_record_uri is None
