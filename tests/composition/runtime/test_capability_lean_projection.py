"""Tests for Lean capability projection: verified result and repairable diagnostics."""

from __future__ import annotations

import pytest
from tests.support.provider_lean import (
    PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
    pinned_mathlib_runtime_available,
)

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.runtime.model import JacobianRuntime


@pytest.mark.skipif(
    not pinned_mathlib_runtime_available(),
    reason=PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
)
def test_lean_capability_returns_bound_verified_result(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    runtime = authorized_complete_runtime

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.check",
            input={
                "statement": "1 + 1 = 2",
                "proof": "rfl",
                "environment": "CORE",
            },
        )
    )

    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert result.assurance.verification_record_uri is not None
    assert result.assurance.verification_record_uri in result.artifact_uris
    assert result.output["conclusion"] == "TRUE"


@pytest.mark.skipif(
    not pinned_mathlib_runtime_available(),
    reason=PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
)
def test_lean_capability_projects_repairable_checker_diagnostics(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    runtime = authorized_complete_runtime

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.check",
            input={
                "statement": "1 + 1 = 2",
                "proof": "sorry",
                "environment": "CORE",
            },
        )
    )

    assert result.output["input"]["status"] == "REJECTED"
    assert "forbidden Lean command" in result.output["input"]["errors"][0]
    assert result.output["diagnostics"] == result.output["input"]["errors"]
    assert result.assurance.verification_record_uri is None
