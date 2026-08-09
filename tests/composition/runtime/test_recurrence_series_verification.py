from __future__ import annotations

from copy import deepcopy

import pytest

from jacobian.checker_operations import derive_verification_capability_id
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus

_RECURRENCE_CONVENTION = "A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"
_P_RECURSIVE_CONVENTION = "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"


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
        "combinatorics.recurrence.p_recursive.evaluate",
        {
            "coefficient_polynomials": [[_q(1)], [_q(0), _q(-1)]],
            "initial_values": [_q(1)],
            "coefficient_convention": _P_RECURSIVE_CONVENTION,
            "polynomial_convention": "ASCENDING_POWERS_OF_N",
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
            capability_id=derive_verification_capability_id(capability_id),
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
    forged_candidate = deepcopy(computed.output["result"])
    if capability_id == "combinatorics.recurrence.linear.evaluate":
        forged_candidate["replay_prefix"][7] = _q(14)
        forged_candidate["values"][7]["value"] = _q(14)
    elif capability_id == "combinatorics.recurrence.p_recursive.evaluate":
        forged_candidate["replay_prefix"][7] = _q(5039)
        forged_candidate["values"][7]["value"] = _q(5039)
    else:
        forged_candidate["coefficients"][7] = _q(22)
    rejected = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=derive_verification_capability_id(capability_id),
            mode=CapabilityMode.VERIFY,
            input={"input": payload, "candidate": forged_candidate},
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
        if item.capability_id == "combinatorics.recurrence.linear.verify"
    )
    assert descriptor.provider_runtime is not None
    assert {
        component["provider"]
        for component in descriptor.provider_runtime.configuration["components"]
    } == {
        "jacobian.additive-combinatorics-checker-source",
        "jacobian.combinatorics-exact-checker-source",
    }


def test_checker_replays_a_result_above_python_default_integer_digit_limit(
    authorized_complete_runtime,
) -> None:
    large = "9" * 64
    recurrence_input = {
        "coefficients": [{"num": large, "den": "1"}],
        "initial_values": [{"num": large, "den": "1"}],
        "coefficient_convention": _RECURRENCE_CONVENTION,
        "scope": "INDICES",
        "term_count": None,
        "indices": [68],
    }
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.recurrence.linear.evaluate",
            input=recurrence_input,
        )
    )
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.recurrence.linear.verify",
            mode=CapabilityMode.VERIFY,
            input={
                "input": recurrence_input,
                "candidate": computed.output["result"],
            },
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
