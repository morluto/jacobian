"""Graph-owned coloring encodings and independent replay tests."""

from __future__ import annotations

from jacobian.contracts.capabilities import (
    CapabilityMode,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.runtime.model import JacobianRuntime


def _encode(runtime: JacobianRuntime) -> CapabilityResult:
    return runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.coloring.encode_k_cnf",
            input={
                "graph": {
                    "vertices": ["c", "a", "b"],
                    "edges": [["b", "a"], ["c", "b"], ["a", "c"]],
                },
                "colors": 3,
            },
        )
    )


def test_graph_coloring_encoding_is_canonical_and_inspectable(
    attached_complete_runtime,
) -> None:
    runtime = attached_complete_runtime

    result = _encode(runtime)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["graph"] == {
        "graph_schema_version": "1",
        "vertices": ["a", "b", "c"],
        "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
    }
    assert result.output["variable_count"] == 9
    assert result.output["clause_count"] == 21
    assert result.output["checker_id"] is None
    assert len(result.artifact_uris) == 5


def test_graph_coloring_encoding_replays_through_generic_certificate_verifier(
    authorized_complete_runtime,
) -> None:
    encoded = _encode(authorized_complete_runtime)

    assert encoded.output["checker_id"] is not None
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="certificate.verify",
            mode=CapabilityMode.VERIFY,
            input={
                "certificate_uri": encoded.output["certificate_uri"],
                "checker_id": encoded.output["checker_id"],
            },
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["conclusion"] == "TRUE"
    assert verified.output["assurance"]["verification"] == "VERIFIED"
    assert verified.output["verification_record_uri"] is not None
