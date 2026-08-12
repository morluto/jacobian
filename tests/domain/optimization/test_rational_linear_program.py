from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.support.rationals import rational_payload as _rational
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.contracts.validated_analysis import RationalLinearProgramObligation
from jacobian.domains.optimization import build_rational_optimization_bundle


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state", build_rational_optimization_bundle()
    ) as services:
        yield services


def test_rational_lp_produces_inspectable_primal_dual_certificate(
    domain_services: DomainTestServices,
) -> None:
    runtime = domain_services

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="optimization.linear.rational_optimum.compute",
            input={
                "program": {
                    "variables": ["x", "y"],
                    "objective": [_rational(1), _rational(2)],
                    "coefficients": [[_rational(1), _rational(1)]],
                    "rhs": [_rational(1)],
                },
                "wall_seconds": 10,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    output = result.output["result"]
    assert output["status"] == "CERTIFICATE_PRODUCED"
    assert output["primal_candidate"] == [_rational(1), _rational(0)]
    assert output["dual_candidate"] == [_rational(1)]
    assert output["primal_objective"] == _rational(1)
    assert output["dual_objective"] == _rational(1)
    assert output["primal_residuals"] == [_rational(0)]
    assert output["dual_slacks"] == [_rational(0), _rational(1)]


def test_rational_lp_dual_variables_are_unrestricted_and_dimension_bound(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="optimization.linear.rational_optimum.compute",
            input={
                "program": {
                    "variables": ["x", "y"],
                    "objective": [_rational(-1), _rational(3)],
                    "coefficients": [
                        [_rational(1), _rational(0)],
                        [_rational(0), _rational(1)],
                    ],
                    "rhs": [_rational(1), _rational(2)],
                },
                "wall_seconds": 10,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    output = result.output["result"]
    assert output["status"] == "CERTIFICATE_PRODUCED"
    assert output["primal_candidate"] == [_rational(1), _rational(2)]
    assert output["dual_candidate"] == [_rational(-1), _rational(3)]
    assert output["primal_objective"] == _rational(5)
    assert output["dual_objective"] == _rational(5)
    assert output["dual_slacks"] == [_rational(0), _rational(0)]


def test_rational_lp_obligation_rejects_wrong_candidate_dimensions() -> None:
    program = {
        "variables": ["x", "y"],
        "objective": [_rational(1), _rational(1)],
        "coefficients": [[_rational(1), _rational(1)]],
        "rhs": [_rational(1)],
    }

    with pytest.raises(ValidationError, match="primal candidate length"):
        RationalLinearProgramObligation.model_validate(
            {
                "program": program,
                "status": "CERTIFICATE_PRODUCED",
                "primal_candidate": [_rational(1)],
                "dual_candidate": [_rational(1)],
            }
        )
    with pytest.raises(ValidationError, match="dual candidate length"):
        RationalLinearProgramObligation.model_validate(
            {
                "program": program,
                "status": "CERTIFICATE_PRODUCED",
                "primal_candidate": [_rational(1), _rational(0)],
                "dual_candidate": [_rational(1), _rational(0)],
            }
        )


def test_invalid_rational_lp_never_reaches_backend_worker(
    domain_services: DomainTestServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.domains.optimization import operations

    def unexpected_worker(_request: object) -> object:
        raise AssertionError("worker unexpectedly called")

    monkeypatch.setattr(operations, "_run_worker", unexpected_worker)
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="optimization.linear.rational_optimum.compute",
            input={
                "program": {
                    "variables": ["x", "y"],
                    "objective": [_rational(1), _rational(2)],
                    "coefficients": [[_rational(1)]],
                    "rhs": [_rational(1)],
                },
                "wall_seconds": 10,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_RATIONAL_OPTIMIZATION_REQUEST"
    assert result.artifact_uris == ()
