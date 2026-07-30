from __future__ import annotations

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


def test_graph_symmetry_orbits_are_independently_replayed(
    authorized_complete_runtime,
) -> None:
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.symmetry.generator_orbits.compute",
            input=_cycle_request(),
        )
    )
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.symmetry.generator_orbits.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
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
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.symmetry.generator_orbits.compute",
            input=_cycle_request(),
        )
    )
    result_artifact = authorized_complete_runtime.core.store.get(
        computed.output["result_uri"]
    )
    false_payload = dict(result_artifact.payload)
    false_payload["vertex_orbits"] = [
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
    false_payload["vertex_orbit_count"] = 2
    false_result = authorized_complete_runtime.core.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload=false_payload,
        summary="adversarial false graph-symmetry orbit partition",
    )

    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.symmetry.generator_orbits.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED
