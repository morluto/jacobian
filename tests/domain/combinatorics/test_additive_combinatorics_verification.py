from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.contracts.operations import (
    OperationRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.combinatorics import combinatorics_operations


@pytest.fixture
def combinatorics_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install additive-combinatorics operations and exact checkers only."""

    with open_exact_domain_services(
        tmp_path / "state",
        combinatorics_operations(),
    ) as services:
        yield services


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
    combinatorics_services,
    producer_id: str,
    verifier_id: str,
    payload: dict[str, object],
) -> None:
    computed = combinatorics_services.core.operations.invoke(
        OperationRequest(operation_id=producer_id, input=payload)
    )
    verified = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id=verifier_id,
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == producer_id
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert verified.verification_record_uri is not None


def test_fixed_order_negative_result_is_verified_inline(
    combinatorics_services,
) -> None:
    producer_id = "combinatorics.cyclic_difference_set.extension.decide"
    verifier_id = "combinatorics.cyclic_difference_set.extension.verify"
    payload = {"base_elements": _BASE, "target_order": 7}
    computed = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id=producer_id,
            input=payload,
        )
    )
    verified = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id=verifier_id,
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == producer_id
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert verified.verification_record_uri is not None
    assert computed.artifact_uris == ()


def test_fixed_order_positive_witness_is_independently_verified(
    combinatorics_services,
) -> None:
    payload = {"base_elements": ["0", "1"], "target_order": 3}
    computed = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.cyclic_difference_set.extension.decide",
            input=payload,
        )
    )
    verified = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.cyclic_difference_set.extension.verify",
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )

    assert computed.output["result"]["extension"] == [0, 1, 3]
    assert verified.output["status"] == "VERIFIED"
    assert verified.verification_record_uri is not None


def test_inline_checker_rejects_a_contract_valid_false_sidon_result(
    combinatorics_services,
) -> None:
    payload = {"elements": _BASE}
    different = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.integer_set.sidon.decide",
            input={"elements": ["0", "1", "2"]},
        )
    )

    rejected = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.integer_set.sidon.verify",
            input={"input": payload, "candidate": different.output["result"]},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_additive_checker_runtime_binds_both_independent_sources(
    combinatorics_services,
) -> None:
    descriptor = next(
        item
        for item in combinatorics_services.core.operations.snapshot().operations
        if item.operation_id == "combinatorics.integer_set.sidon.verify"
    )
    assert descriptor.provider_runtime is not None
    assert {
        component["provider"]
        for component in descriptor.provider_runtime.configuration["components"]
    } == {
        "jacobian.additive-combinatorics-checker-source",
        "jacobian.combinatorics-exact-checker-source",
    }
