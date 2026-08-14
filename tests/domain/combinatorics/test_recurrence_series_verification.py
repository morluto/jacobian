from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.rationals import rational_payload as _q
from tests.support.services import DomainTestServices

from jacobian.checker_operations import derive_verification_operation_id
from jacobian.contracts.operations import (
    OperationRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.combinatorics import combinatorics_operations

_RECURRENCE_CONVENTION = "A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"
_P_RECURSIVE_CONVENTION = "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"


@pytest.fixture(scope="module")
def combinatorics_services(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[DomainTestServices]:
    """Install combinatorics and its independent exact checkers only."""

    with open_exact_domain_services(
        tmp_path_factory.mktemp("combinatorics-verification") / "state",
        combinatorics_operations(),
    ) as services:
        yield services


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


def test_recurrence_and_series_results_are_verified_and_forgery_rejected(
    combinatorics_services,
) -> None:
    runtime = combinatorics_services
    for operation_id, payload in _CASES:
        computed = runtime.core.operations.invoke(
            OperationRequest(operation_id=operation_id, input=payload)
        )
        verifier_id = derive_verification_operation_id(operation_id)
        verified = runtime.core.operations.invoke(
            OperationRequest(
                operation_id=verifier_id,
                input={
                    "input": payload,
                    "candidate": computed.output["result"],
                },
            )
        )
        assert verified.execution.status is ExecutionStatus.COMPLETED, operation_id
        assert verified.output["status"] == "VERIFIED", operation_id
        assert verified.output["operation_id"] == operation_id, operation_id
        assert verified.output["verification_record_uri"] in verified.artifact_uris, (
            operation_id
        )
        assert verified.verification_record_uri is not None, operation_id

        forged_candidate = deepcopy(computed.output["result"])
        if operation_id == "combinatorics.recurrence.linear.evaluate":
            forged_candidate["replay_prefix"][7] = _q(14)
            forged_candidate["values"][7]["value"] = _q(14)
        elif operation_id == "combinatorics.recurrence.p_recursive.evaluate":
            forged_candidate["replay_prefix"][7] = _q(5039)
            forged_candidate["values"][7]["value"] = _q(5039)
        else:
            forged_candidate["coefficients"][7] = _q(22)
        rejected = runtime.core.operations.invoke(
            OperationRequest(
                operation_id=verifier_id,
                input={"input": payload, "candidate": forged_candidate},
            )
        )
        assert rejected.execution.status is ExecutionStatus.COMPLETED, operation_id
        assert rejected.output["status"] == "REJECTED", operation_id
        assert rejected.output["conclusion"] == "UNKNOWN", operation_id
        assert rejected.output["verification_record_uri"] is None, operation_id


def test_checker_runtime_binds_only_independent_source(
    combinatorics_services,
) -> None:
    descriptor = next(
        item
        for item in combinatorics_services.core.operations.snapshot().operations
        if item.operation_id == "combinatorics.recurrence.linear.verify"
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
    combinatorics_services,
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
    computed = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.recurrence.linear.evaluate",
            input=recurrence_input,
        )
    )
    verified = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.recurrence.linear.verify",
            input={
                "input": recurrence_input,
                "candidate": computed.output["result"],
            },
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
