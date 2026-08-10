"""Composition-owned finite coverage authority edge."""

from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus


def _request(
    scope: list[str | int],
    pages: list[list[str | int]],
    *,
    canonicalizer_id: str = "finite.string.nfc@1",
) -> CapabilityRequest:
    return CapabilityRequest(
        capability_id="finite.coverage.verify",
        input={
            "canonicalizer_id": canonicalizer_id,
            "scope_items": scope,
            "pages": [{"items": items} for items in pages],
        },
    )


def test_finite_coverage_is_unavailable_without_authorized_checker(
    attached_complete_runtime,
) -> None:
    result = attached_complete_runtime.core.capabilities.invoke(
        _request(["alpha"], [["alpha"]])
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output["error"]["code"] == "UNKNOWN_CAPABILITY"
