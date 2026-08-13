from __future__ import annotations

from copy import deepcopy
from typing import Any

from jacobian.contracts.operations import (
    OperationRequest,
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


def _result_payload(computed: Any) -> dict[str, Any]:
    return computed.output["result"]


def test_graph_symmetry_orbits_are_independently_replayed(
    graph_symmetry_services,
) -> None:
    payload = _cycle_request()
    computed = graph_symmetry_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.symmetry.generator_orbits.compute",
            input=payload,
        )
    )
    verified = graph_symmetry_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.symmetry.generator_orbits.verify",
            input={
                "input": payload,
                "candidate": computed.output["result"],
            },
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == (
        "graph.symmetry.generator_orbits.compute"
    )
    assert verified.output["verification_record_uri"] is not None


def test_graph_symmetry_checker_rejects_forged_orbit_partition(
    graph_symmetry_services,
) -> None:
    payload = _cycle_request()
    computed = graph_symmetry_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.symmetry.generator_orbits.compute",
            input=payload,
        )
    )
    forged_candidate = deepcopy(_result_payload(computed))
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

    rejected = graph_symmetry_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.symmetry.generator_orbits.verify",
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
