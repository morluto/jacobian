from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.rationals import rational_payload
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.analysis import build_real_analysis_bundle


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state", build_real_analysis_bundle()
    ) as services:
        yield services


def test_arb_point_enclosure_returns_exact_dyadics(
    domain_services: DomainTestServices,
) -> None:
    runtime = domain_services

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="analysis.real_function.point_enclosure.compute",
            input={
                "function": "EXP",
                "argument": rational_payload(1, 3),
                "precision_bits": 128,
                "wall_seconds": 10,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    output = result.output["result"]
    assert output["status"] == "ENCLOSED"
    assert output["conclusion"] == "UNKNOWN"
    assert output["lower"]["mantissa"]
    assert output["upper"]["mantissa"]
    assert output["relative_accuracy_bits"] >= 120
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.artifact_uris == ()
    assert result.obligations == ()


def test_arb_nonfinite_and_timeout_are_non_conclusions(
    domain_services: DomainTestServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = domain_services
    nonfinite = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="analysis.real_function.point_enclosure.compute",
            input={
                "function": "LOG",
                "argument": rational_payload(-1),
                "wall_seconds": 10,
            },
        )
    )

    output = nonfinite.output["result"]
    assert output["status"] == "NONFINITE"
    assert output["lower"] is None
    assert output["upper"] is None
    assert output["conclusion"] == "UNKNOWN"

    from jacobian.domains.analysis import operations

    def timeout(_request: object) -> object:
        raise TimeoutError

    monkeypatch.setattr(operations, "_run_worker", timeout)
    timed_out = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="analysis.real_function.point_enclosure.compute",
            input={
                "function": "SIN",
                "argument": rational_payload(1),
                "wall_seconds": 1,
            },
        )
    )

    assert timed_out.execution.status is ExecutionStatus.TIMEOUT
    assert timed_out.output["error"]["code"] == "ARB_POINT_ENCLOSURE_TIMEOUT"
    assert timed_out.diagnostics[0].code == "ARB_POINT_ENCLOSURE_TIMEOUT"
    assert timed_out.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert timed_out.obligations == ()
