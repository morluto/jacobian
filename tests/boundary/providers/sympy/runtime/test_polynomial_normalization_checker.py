"""Independent SymPy normalization checker and certificate-binding tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.support.capabilities import invoke_capability as _invoke
from tests.support.rationals import rational_payload as _q

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.polynomial_expression_capabilities import (
    install_polynomial_expression_checker,
)
from jacobian.process_policy import ProcessResult, ProcessTermination
from jacobian.runtime import create_runtime
from jacobian.runtime.model import JacobianRuntime


def _variable(name: str) -> dict[str, Any]:
    return {"kind": "variable", "name": name}


def _expression(
    node: dict[str, Any], *, variables: list[str] | None = None
) -> dict[str, Any]:
    return {"variables": variables or ["x", "y"], "expression": node}


def _runtime_with_checker(root: Path) -> JacobianRuntime:
    runtime = create_runtime(root)
    adapter, _installation = install_polynomial_expression_checker(
        runtime.core.store,
        runtime.core.schemas,
        runtime.core.artifacts,
        runtime.core.polynomial_expressions,
        runtime.services.verification,
        runtime.core.checkers,
        authorize_checker=True,
    )
    assert adapter is not None
    runtime.core.capabilities.register(adapter)
    return runtime


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


def test_independent_checker_verifies_full_ast_relation(tmp_path: Path) -> None:
    runtime = _runtime_with_checker(tmp_path)
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
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert verified.completeness.assurance_level is CapabilityAssuranceLevel.VERIFIED
    assert (
        verified.completeness.verification_record_uri
        == verified.assurance.verification_record_uri
    )
    assert verified.relationships[0].status.value == "VERIFIED"
    assert (
        verified.relationships[0].verification_record_uri
        == verified.assurance.verification_record_uri
    )


def test_independent_checker_rejects_wrong_bound_coefficients(tmp_path: Path) -> None:
    runtime = _runtime_with_checker(tmp_path)
    expression_uri = runtime.core.polynomial_expressions.put_expression(
        _expression(_variable("x"), variables=["x"])
    ).artifact_uri
    candidate = runtime.core.polynomial_expressions.put_normalization(
        expression_uri=expression_uri,
        normalized={"terms": []},
        producer=runtime.portfolio.sympy_polynomial_normalization_runtime,
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
    assert rejected.assurance.level is not CapabilityAssuranceLevel.VERIFIED


def test_normalization_checker_timeout_is_operational(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime_with_checker(tmp_path)
    computed = _invoke(
        runtime,
        "polynomial.expression.normalize",
        {"expression": _expression(_variable("x"), variables=["x"])},
    )
    monkeypatch.setattr(
        "jacobian.verification.service.execute_process",
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
    assert result.output["status"] == "TIMEOUT"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["verification_record_uri"] is None
    assert result.assurance.level is not CapabilityAssuranceLevel.VERIFIED
