from __future__ import annotations

from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ExecutionStatus


def test_advertised_inverse_example_uses_cold_worker_budget(
    authorized_polynomial_services,
) -> None:
    descriptor = next(
        item
        for item in authorized_polynomial_services.core.operations.snapshot().operations
        if item.operation_id == "polynomial.map.inverse.candidate_synthesize"
    )
    example = descriptor.examples[0]
    assert example.input["limits"]["timeout_ms"] == 30000

    result = authorized_polynomial_services.core.operations.invoke(
        OperationRequest(operation_id=descriptor.operation_id, input=example.input)
    )
    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "FOUND"
