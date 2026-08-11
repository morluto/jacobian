from __future__ import annotations

import pytest
from tests.support.provider_lean import (
    PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
    pinned_mathlib_runtime_available,
)

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityRequest,
)

pytestmark = [
    pytest.mark.skipif(
        not pinned_mathlib_runtime_available(),
        reason=PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
    ),
]


def test_exact_proof_edit_is_bound_to_authorized_lean_check(
    authorized_complete_runtime,
) -> None:

    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.proof_edit.validate",
            input={
                "environment": "CORE",
                "statement": "True",
                "original_proof": "by\n  exact True.intro",
                "edited_proof": "by\n  trivial",
            },
        )
    )

    assert result.output["accepted"] is True
    assert result.output["baseline_accepted"] is True
    assert result.output["baseline_verification_record_uri"] in result.artifact_uris
    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert (
        result.assurance.verification_record_uri
        == (result.output["verification_record_uri"])
    )
    assert result.output["verification_record_uri"] in result.artifact_uris
    assert result.completeness.status is CapabilityCompletenessStatus.NOT_APPLICABLE
    assert result.output["proof_edit_uri"] in result.artifact_uris
    edit = authorized_complete_runtime.core.store.get(result.output["proof_edit_uri"])
    assert edit.payload["edited_proof"] == "by\n  trivial"
    assert set(edit.manifest.parents) == {
        result.output["claim_uri"],
        result.output["baseline_candidate_uri"],
        result.output["baseline_certificate_uri"],
        result.output["candidate_uri"],
        result.output["certificate_uri"],
    }


def test_proof_edit_rejects_holes_before_checker_invocation(
    authorized_complete_runtime,
) -> None:

    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.proof_edit.validate",
            input={
                "environment": "CORE",
                "statement": "True",
                "original_proof": "by\n  trivial",
                "edited_proof": "by\n  sorry",
            },
        )
    )

    assert result.execution.status.value == "ERROR"
    assert result.output["error"]["code"] == "INVALID_LEAN_PROOF_EDIT_REQUEST"


def test_rejected_edit_keeps_checker_evidence_without_becoming_accepted(
    authorized_complete_runtime,
) -> None:

    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.proof_edit.validate",
            input={
                "environment": "CORE",
                "statement": "True",
                "original_proof": "by\n  trivial",
                "edited_proof": "by\n  exact False.elim (by trivial)",
            },
        )
    )

    assert result.output["accepted"] is False
    assert result.output["verification_record_uri"] is None
    assert result.output["certificate_uri"] in result.artifact_uris
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert result.assurance.verification_record_uri is None


def test_valid_edit_is_not_accepted_when_original_baseline_is_invalid(
    authorized_complete_runtime,
) -> None:

    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.proof_edit.validate",
            input={
                "environment": "CORE",
                "statement": "True",
                "original_proof": "by\n  exact False.elim (by trivial)",
                "edited_proof": "by\n  trivial",
            },
        )
    )

    assert result.output["baseline_accepted"] is False
    assert result.output["accepted"] is False
    assert result.output["verification_record_uri"] is not None
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert result.assurance.verification_record_uri is None
