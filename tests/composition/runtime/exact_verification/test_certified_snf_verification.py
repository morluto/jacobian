from __future__ import annotations

from copy import deepcopy

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus


def _matrix_payload() -> dict[str, object]:
    return {
        "matrix": {
            "row_count": 2,
            "column_count": 3,
            "entries": [["2", "4", "4"], ["6", "6", "12"]],
        }
    }


def _compute(authorized_complete_runtime):
    return authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.certified.compute",
            input=_matrix_payload(),
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
            input={
                "input": _matrix_payload(),
                "candidate": computed.output["result"],
            },
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
    forged_candidate = deepcopy(computed.output["result"])
    certificate = dict(forged_candidate["certificate"])
    left = dict(certificate["left_transformation"])
    entries = [list(row) for row in left["entries"]]
    entries[0][0] = str(int(entries[0][0]) + 1)
    left["entries"] = entries
    certificate["left_transformation"] = left
    forged_candidate["certificate"] = certificate

    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.certified.verify",
            mode=CapabilityMode.VERIFY,
            input={"input": _matrix_payload(), "candidate": forged_candidate},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED
