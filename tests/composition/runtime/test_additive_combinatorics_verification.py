from __future__ import annotations

import pytest

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus

_BASE = ["1", "2", "4", "8", "13"]
_INLINE_CASES = (
    (
        "combinatorics.integer_set.sidon.decide",
        "combinatorics.integer_set.sidon.verify",
        {"elements": _BASE},
    ),
    (
        "combinatorics.cyclic_difference_set.perfect.decide",
        "combinatorics.cyclic_difference_set.perfect.verify",
        {"modulus": 7, "residues": [0, 1, 3]},
    ),
)


@pytest.mark.parametrize(("producer_id", "verifier_id", "payload"), _INLINE_CASES)
def test_additive_decisions_are_independently_verified(
    authorized_complete_runtime,
    producer_id: str,
    verifier_id: str,
    payload: dict[str, object],
) -> None:
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(capability_id=producer_id, input=payload)
    )
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=verifier_id,
            mode=CapabilityMode.VERIFY,
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )

    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == producer_id
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


def test_fixed_order_negative_result_is_verified_inline(
    authorized_complete_runtime,
) -> None:
    producer_id = "combinatorics.cyclic_difference_set.extension.decide"
    verifier_id = "combinatorics.cyclic_difference_set.extension.verify"
    payload = {"base_elements": _BASE, "target_order": 7}
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=producer_id,
            input=payload,
        )
    )
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=verifier_id,
            mode=CapabilityMode.VERIFY,
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == producer_id
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert computed.artifact_uris == ()


def test_fixed_order_positive_witness_is_independently_verified(
    authorized_complete_runtime,
) -> None:
    payload = {"base_elements": ["0", "1"], "target_order": 3}
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.cyclic_difference_set.extension.decide",
            input=payload,
        )
    )
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.cyclic_difference_set.extension.verify",
            mode=CapabilityMode.VERIFY,
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )

    assert computed.output["result"]["extension"] == [0, 1, 3]
    assert verified.output["status"] == "VERIFIED"
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


def test_inline_checker_rejects_a_contract_valid_false_sidon_result(
    authorized_complete_runtime,
) -> None:
    payload = {"elements": _BASE}
    different = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.integer_set.sidon.decide",
            input={"elements": ["0", "1", "2"]},
        )
    )

    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.integer_set.sidon.verify",
            mode=CapabilityMode.VERIFY,
            input={"input": payload, "candidate": different.output["result"]},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_additive_checker_runtime_binds_both_independent_sources(
    authorized_complete_runtime,
) -> None:
    descriptor = next(
        item
        for item in authorized_complete_runtime.core.capabilities.catalog().capabilities
        if item.capability_id == "combinatorics.integer_set.sidon.verify"
    )
    assert descriptor.provider_runtime is not None
    assert {
        component["provider"]
        for component in descriptor.provider_runtime.configuration["components"]
    } == {
        "jacobian.additive-combinatorics-checker-source",
        "jacobian.combinatorics-exact-checker-source",
    }
