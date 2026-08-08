from __future__ import annotations

from copy import deepcopy
from typing import Any

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus


def _cycle_request() -> dict[str, object]:
    return {
        "graph": {
            "vertices": ["a", "b", "c", "d"],
            "edges": [
                ["a", "b"],
                ["a", "d"],
                ["b", "c"],
                ["c", "d"],
            ],
        },
        "generators": [
            {
                "generator_id": "quarter_turn",
                "mapping": {
                    "a": "b",
                    "b": "c",
                    "c": "d",
                    "d": "a",
                },
            }
        ],
    }


def _result_payload(runtime: Any, computed: Any) -> dict[str, Any]:
    del runtime
    return computed.output["result"]


def test_graph_symmetry_orbits_are_independently_replayed(
    authorized_complete_runtime,
) -> None:
    payload = _cycle_request()
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.symmetry.generator_orbits.compute",
            input=payload,
        )
    )
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.symmetry.generator_orbits.verify",
            mode=CapabilityMode.VERIFY,
            input={
                "input": payload,
                "candidate": computed.output["result"],
            },
        )
    )

    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == (
        "graph.symmetry.generator_orbits.compute"
    )
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["verification_record_uri"] is not None


def test_graph_symmetry_checker_rejects_forged_orbit_partition(
    authorized_complete_runtime,
) -> None:
    payload = _cycle_request()
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.symmetry.generator_orbits.compute",
            input=payload,
        )
    )
    forged_candidate = deepcopy(_result_payload(authorized_complete_runtime, computed))
    forged_candidate["vertex_orbits"] = [
        {
            "orbit_index": 0,
            "representative": "a",
            "members": ["a", "b"],
        },
        {
            "orbit_index": 1,
            "representative": "c",
            "members": ["c", "d"],
        },
    ]
    forged_candidate["vertex_orbit_count"] = 2

    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.symmetry.generator_orbits.verify",
            mode=CapabilityMode.VERIFY,
            input={
                "input": payload,
                "candidate": forged_candidate,
            },
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED
