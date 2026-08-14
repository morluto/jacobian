"""SymPy normalization runtime identity and worker-protocol tests."""

from __future__ import annotations

import os
from typing import Any

from tests.support.operations import invoke_operation as _invoke
from tests.support.rationals import rational_payload as _q

import jacobian.providers.sympy_runtime as sympy_runtime
from jacobian.canonical import canonicalize_json
from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderInstallTier,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.process_policy import ProcessResult, ProcessTermination
from jacobian.providers.sympy_runtime import (
    sympy_polynomial_normalization_provider_runtime,
)


def _variable(name: str) -> dict[str, Any]:
    return {"kind": "variable", "name": name}


def _expression(
    node: dict[str, Any], *, variables: list[str] | None = None
) -> dict[str, Any]:
    return {"variables": variables or ["x", "y"], "expression": node}


def test_sympy_normalization_runtime_has_exact_profile() -> None:
    runtime = sympy_polynomial_normalization_provider_runtime()
    assert runtime.availability is ProviderAvailability.AVAILABLE
    assert runtime.version == "1.14.0"
    assert runtime.install_tier is ProviderInstallTier.T0
    assert runtime.digest is not None and runtime.digest.startswith("sha256:")
    assert runtime.configuration == {
        "distribution": "sympy",
        "domain": "QQ",
        "operation": "Poly(expression, *variables, domain=QQ).terms()",
        "expression_schema_version": "1",
        "maximum_variables": 4,
        "maximum_nodes": 128,
        "maximum_depth": 16,
        "maximum_expanded_terms": 1024,
        "maximum_exponent_per_variable": 127,
        "maximum_coefficient_digit_budget": 4096,
    }


def test_sympy_normalization_runtime_rejects_unpinned_version(monkeypatch) -> None:
    available = sympy_polynomial_normalization_provider_runtime()
    wrong = available.model_copy(update={"version": "1.13.3"})
    monkeypatch.setattr(
        sympy_runtime,
        "python_distribution_provider_runtime",
        lambda *_args, **_kwargs: wrong,
    )
    rejected = sympy_polynomial_normalization_provider_runtime(refresh=True)
    assert rejected.availability is ProviderAvailability.UNAVAILABLE
    assert rejected.version is None
    assert rejected.digest is None
    assert "pinned 1.14.0" in rejected.diagnostic


def test_sympy_normalization_timeout_is_operational(
    polynomial_normalization_services, monkeypatch
) -> None:
    monkeypatch.setattr(
        "jacobian.sympy_polynomial_normalization.execute_process",
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
        polynomial_normalization_services,
        "polynomial.expression.normalize",
        {"expression": _expression(_variable("x"), variables=["x"])},
    )
    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.output == {}
    assert result.verification_record_uri is None


def test_sympy_worker_gets_only_fixed_environment_and_budget(
    polynomial_normalization_services, monkeypatch
) -> None:
    monkeypatch.setenv("JACOBIAN_SYMPY_SECRET", "must-not-propagate")
    observed: dict[str, Any] = {}

    def fake_worker(request: Any) -> ProcessResult:
        observed["timeout_seconds"] = request.timeout_seconds
        observed["environment"] = dict(request.environment)
        stdout = (
            canonicalize_json(
                {
                    "protocol": "jacobian.sympy-polynomial-normalization/v1",
                    "status": "NORMALIZATION_PRODUCED",
                    "backend_version": "1.14.0",
                    "normalized": {"terms": [{"coefficient": _q(1), "exponents": [1]}]},
                }
            )
            + b"\n"
        )
        return ProcessResult(
            termination=ProcessTermination.EXITED,
            returncode=0,
            stdout=stdout,
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        )

    monkeypatch.setattr(
        "jacobian.sympy_polynomial_normalization.execute_process", fake_worker
    )
    result = _invoke(
        polynomial_normalization_services,
        "polynomial.expression.normalize",
        {
            "expression": _expression(_variable("x"), variables=["x"]),
            "resource_budget": {"wall_seconds": 7},
        },
    )
    assert result.output["status"] == "NORMALIZATION_PRODUCED"
    assert observed["timeout_seconds"] == 7.0
    assert observed["environment"] == {
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    assert "JACOBIAN_SYMPY_SECRET" in os.environ
    assert "JACOBIAN_SYMPY_SECRET" not in observed["environment"]


def test_invalid_worker_protocol_retains_no_normalization_evidence(
    polynomial_normalization_services, monkeypatch
) -> None:
    monkeypatch.setattr(
        "jacobian.sympy_polynomial_normalization.execute_process",
        lambda *_args, **_kwargs: ProcessResult(
            termination=ProcessTermination.EXITED,
            returncode=0,
            stdout=b'{"status":"NORMALIZATION_PRODUCED"}\n',
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        ),
    )
    result = _invoke(
        polynomial_normalization_services,
        "polynomial.expression.normalize",
        {"expression": _expression(_variable("x"), variables=["x"])},
    )
    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output == {}
