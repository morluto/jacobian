from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus


def test_advertised_inverse_example_uses_cold_worker_budget(
    authorized_polynomial_services,
) -> None:
    descriptor = next(
        item
        for item in authorized_polynomial_services.core.capabilities.catalog().capabilities
        if item.capability_id == "polynomial.map.inverse.candidate_synthesize"
    )
    example = descriptor.invocation_examples[0]
    assert example.input["limits"]["timeout_ms"] == 30000

    result = authorized_polynomial_services.core.capabilities.invoke(
        CapabilityRequest(capability_id=descriptor.capability_id, input=example.input)
    )
    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "FOUND"
