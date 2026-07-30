from __future__ import annotations

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus


def _compute(authorized_complete_runtime):
    return authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.certified.compute",
            input={
                "matrix": {
                    "row_count": 2,
                    "column_count": 3,
                    "entries": [["2", "4", "4"], ["6", "6", "12"]],
                }
            },
        )
    )


def test_certified_smith_result_is_independently_verified(
    authorized_complete_runtime,
) -> None:
    computed = _compute(authorized_complete_runtime)
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.certified.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["verification_record_uri"] in verified.artifact_uris


def test_certified_smith_checker_rejects_a_forged_relation(
    authorized_complete_runtime,
) -> None:
    computed = _compute(authorized_complete_runtime)
    artifact = authorized_complete_runtime.core.store.get(computed.output["result_uri"])
    payload = dict(artifact.payload)
    certificate = dict(payload["certificate"])
    left = dict(certificate["left_transformation"])
    entries = [list(row) for row in left["entries"]]
    entries[0][0] = str(int(entries[0][0]) + 1)
    left["entries"] = entries
    certificate["left_transformation"] = left
    payload["certificate"] = certificate
    forged = authorized_complete_runtime.core.artifacts.put(
        schema_uri=artifact.manifest.schema_uri,
        semantics_uri=artifact.manifest.semantics_uri,
        parents=artifact.manifest.parents,
        payload=payload,
        summary="adversarial rebound Smith transformation",
    )

    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.certified.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": forged.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED
