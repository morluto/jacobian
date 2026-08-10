"""Composition-owned finite partition verification-boundary edges."""

from __future__ import annotations

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityObligationStatus,
    CapabilityRequest,
)
from jacobian.contracts.checkers import CheckerDecision
from jacobian.contracts.results import Arithmetic, Conclusion, Coverage, Method


def _request(*, verify: bool = True) -> CapabilityRequest:
    return CapabilityRequest(
        capability_id=(
            "case.partition.finite.verify" if verify else "case.partition.finite"
        ),
        input={
            "universe": ["0", "1", "2", "3", "4", "5"],
            "cases": [
                {"case_id": "even", "members": ["0", "2", "4"]},
                {"case_id": "odd", "members": ["1", "3", "5"]},
            ],
            "require_disjoint": True,
        },
    )


def test_verification_rejects_checker_obligation_outside_request(
    authorized_complete_runtime,
    monkeypatch,
) -> None:
    runtime = authorized_complete_runtime

    def accept_with_unbound_obligation(
        *,
        request: dict[str, object],
        **_: object,
    ) -> CheckerDecision:
        scope = request["scope"]
        candidate = request["candidate"]
        assert isinstance(scope, dict)
        assert isinstance(candidate, dict)
        return CheckerDecision(
            accepted=True,
            conclusion=Conclusion.TRUE,
            arithmetic=Arithmetic.EXACT_INTEGER,
            method=Method.EXHAUSTIVE_FINITE,
            coverage=Coverage.EXHAUSTIVE,
            relation_id="case.relation.partitions",
            relationship_source_artifact_uris=(str(scope["artifact_uri"]),),
            relationship_target_artifact_uris=(str(candidate["artifact_uri"]),),
            obligation_uri="artifact://sha256/" + "9" * 64,
        )

    monkeypatch.setattr(
        runtime.services.verification,
        "_run_checker",
        accept_with_unbound_obligation,
    )

    result = runtime.core.capabilities.invoke(_request(verify=True))

    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.output["verification_record_uri"] is None
    assert result.obligations[0].status is CapabilityObligationStatus.OPEN
