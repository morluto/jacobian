from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityObligationStatus,
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


def _rational(num: int, den: int = 1) -> dict[str, str]:
    return {"num": str(num), "den": str(den)}


def test_arb_point_enclosure_materializes_exact_dyadics_and_obligation(
    domain_services: DomainTestServices,
) -> None:
    runtime = domain_services

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="analysis.real_function.point_enclosure.compute",
            input={
                "function": "EXP",
                "argument": _rational(1, 3),
                "precision_bits": 128,
                "wall_seconds": 10,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "ENCLOSED"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["lower"]["mantissa"]
    assert result.output["upper"]["mantissa"]
    assert result.output["relative_accuracy_bits"] >= 120
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert len(result.artifact_uris) == 3
    assert result.obligations[0].status is CapabilityObligationStatus.OPEN
    obligation = runtime.core.store.get(result.obligations[0].obligation_uri)
    assert obligation.payload["required_checker"] == (
        "AUTHORIZED_INDEPENDENT_BALL_ARITHMETIC"
    )
    assert set(obligation.manifest.parents) == set(result.artifact_uris[:2])


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
                "argument": _rational(-1),
                "wall_seconds": 10,
            },
        )
    )

    assert nonfinite.output["status"] == "NONFINITE"
    assert nonfinite.output["lower"] is None
    assert nonfinite.output["upper"] is None
    assert nonfinite.output["conclusion"] == "UNKNOWN"
    assert nonfinite.completeness.status is CapabilityCompletenessStatus.UNKNOWN
    assert nonfinite.obligations[0].status is CapabilityObligationStatus.OPEN

    from jacobian.domains.analysis import operations

    def timeout(
        _payload: dict[str, object],
        *,
        wall_seconds: int,
    ) -> dict[str, object]:
        raise TimeoutError

    monkeypatch.setattr(operations, "_run_worker", timeout)
    timed_out = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="analysis.real_function.point_enclosure.compute",
            input={
                "function": "SIN",
                "argument": _rational(1),
                "wall_seconds": 1,
            },
        )
    )

    assert timed_out.execution.status is ExecutionStatus.TIMEOUT
    assert timed_out.output["status"] == "TIMEOUT"
    assert timed_out.output["conclusion"] == "UNKNOWN"
    assert timed_out.diagnostics[0].code == "ARB_POINT_ENCLOSURE_TIMEOUT"
    assert timed_out.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert timed_out.completeness.status is CapabilityCompletenessStatus.UNKNOWN
    assert timed_out.obligations[0].status is CapabilityObligationStatus.OPEN
