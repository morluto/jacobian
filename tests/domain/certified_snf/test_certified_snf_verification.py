from __future__ import annotations

from copy import deepcopy
from typing import Any

from tests.support.services import DomainTestServices

from jacobian.contracts.capabilities import (
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


def _compute(certified_snf_services: DomainTestServices):
    return certified_snf_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.certified.compute",
            input=_matrix_payload(),
        )
    )


def _result_payload(runtime: Any, computed: Any) -> dict[str, Any]:
    return runtime.core.store.get(computed.output["result_uri"]).payload


def _forged_result_uri(runtime: Any, computed: Any, payload: dict[str, Any]) -> str:
    source = runtime.core.store.get(computed.output["result_uri"])
    return runtime.core.store.put(
        schema_uri=source.manifest.schema_uri,
        semantics_uri=source.manifest.semantics_uri,
        payload=payload,
        parents=source.manifest.parents,
        summary="forged certified Smith result",
    ).artifact_uri


def test_certified_smith_result_is_independently_verified(
    certified_snf_services: DomainTestServices,
) -> None:
    computed = _compute(certified_snf_services)
    verified = certified_snf_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.certified.verify",
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.verification_record_uri is not None
    assert verified.output["verification_record_uri"] in verified.artifact_uris


def test_certified_smith_checker_rejects_a_forged_relation(
    certified_snf_services: DomainTestServices,
) -> None:
    computed = _compute(certified_snf_services)
    forged_candidate = deepcopy(_result_payload(certified_snf_services, computed))
    certificate = dict(forged_candidate["certificate"])
    left = dict(certificate["left_transformation"])
    entries = [list(row) for row in left["entries"]]
    entries[0][0] = str(int(entries[0][0]) + 1)
    left["entries"] = entries
    certificate["left_transformation"] = left
    forged_candidate["certificate"] = certificate

    rejected = certified_snf_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.certified.verify",
            input={
                "result_uri": _forged_result_uri(
                    certified_snf_services, computed, forged_candidate
                )
            },
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
