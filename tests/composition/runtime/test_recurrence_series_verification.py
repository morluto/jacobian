from __future__ import annotations

from copy import deepcopy

import pytest

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus

_RECURRENCE_CONVENTION = "A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"


def _q(numerator: int, denominator: int = 1) -> dict[str, str]:
    return {"num": str(numerator), "den": str(denominator)}


_CASES = (
    (
        "combinatorics.recurrence.linear.evaluate",
        {
            "coefficients": [_q(1), _q(1)],
            "initial_values": [_q(0), _q(1)],
            "coefficient_convention": _RECURRENCE_CONVENTION,
            "scope": "PREFIX",
            "term_count": 8,
            "indices": [],
        },
    ),
    (
        "combinatorics.generating_function.coefficients.compute",
        {
            "numerator": [_q(1)],
            "denominator": [_q(1), _q(-1), _q(-1)],
            "coefficient_convention": "ASCENDING_POWERS_OF_X",
            "expansion_point": "0",
            "truncation_order": 8,
        },
    ),
)


@pytest.mark.parametrize(("capability_id", "payload"), _CASES)
def test_recurrence_and_series_results_are_independently_verified(
    authorized_complete_runtime,
    capability_id: str,
    payload: dict[str, object],
) -> None:
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(capability_id=capability_id, input=payload)
    )
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )
    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == capability_id
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


@pytest.mark.parametrize(("capability_id", "payload"), _CASES)
def test_checker_rejects_contract_valid_false_results(
    authorized_complete_runtime,
    capability_id: str,
    payload: dict[str, object],
) -> None:
    runtime = authorized_complete_runtime
    computed = runtime.core.capabilities.invoke(
        CapabilityRequest(capability_id=capability_id, input=payload)
    )
    stored = runtime.core.store.get(computed.output["result_uri"])
    forged_payload = deepcopy(stored.payload)
    if capability_id == "combinatorics.recurrence.linear.evaluate":
        forged_payload["replay_prefix"][7] = _q(14)
        forged_payload["values"][7]["value"] = _q(14)
    else:
        forged_payload["coefficients"][7] = _q(22)
    forged = runtime.core.artifacts.put(
        schema_uri=stored.manifest.schema_uri,
        semantics_uri=stored.manifest.semantics_uri,
        parents=stored.manifest.parents,
        payload=forged_payload,
        summary="adversarial contract-valid exact combinatorics result",
    )
    rejected = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": forged.artifact_uri},
        )
    )
    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_checker_runtime_binds_only_independent_source(
    authorized_complete_runtime,
) -> None:
    descriptor = next(
        item
        for item in authorized_complete_runtime.core.capabilities.catalog().capabilities
        if item.capability_id == "combinatorics.result.verify"
    )
    assert descriptor.provider_runtime is not None
    assert {
        component["provider"]
        for component in descriptor.provider_runtime.configuration["components"]
    } == {"jacobian.combinatorics-exact-checker-source"}
