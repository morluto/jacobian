from __future__ import annotations

from copy import deepcopy

import pytest

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus


def _modular_residue_payload(*, coefficient: str = "4") -> dict[str, object]:
    return {
        "modulus": 7,
        "variables": [
            {"name": "x", "residues": [0, 1, 2, 3, 4, 5, 6]},
        ],
        "terms": [{"coefficient": coefficient, "exponents": [3]}],
    }


@pytest.mark.parametrize("value", ("360", "-360", "1", "-1", "101"))
def test_prime_factorization_result_uses_independent_python_flint_replay(
    authorized_complete_runtime,
    value: str,
) -> None:
    producer_id = "integer.compute.prime_factorization"
    verifier_id = "integer.prime_factorization.verify"
    producer_payload = {"value": value}
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=producer_id,
            input=producer_payload,
        )
    )

    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=verifier_id,
            mode=CapabilityMode.VERIFY,
            input={"input": producer_payload, "candidate": computed.output["result"]},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == producer_id
    assert verified.output["verification_record_uri"] is not None
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    provider_runtime = next(
        descriptor.provider_runtime
        for descriptor in authorized_complete_runtime.core.capabilities.catalog().capabilities
        if descriptor.capability_id == verifier_id
    )
    assert provider_runtime is not None
    assert provider_runtime.provider == "jacobian.exact-domain-checkers"
    assert {
        component["provider"]
        for component in provider_runtime.configuration["components"]
    } == {"jacobian.exact-domain-checker-source", "python-flint"}


def test_prime_factorization_verifier_rejects_incomplete_factor_list(
    authorized_complete_runtime,
) -> None:
    producer_payload = {"value": "360"}
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.compute.prime_factorization",
            input=producer_payload,
        )
    )
    forged_candidate = deepcopy(computed.output["result"])
    forged_candidate["factors"] = forged_candidate["factors"][:-1]

    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.prime_factorization.verify",
            mode=CapabilityMode.VERIFY,
            input={"input": producer_payload, "candidate": forged_candidate},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


@pytest.mark.parametrize("value", ("1", "72", "12", "30"))
def test_powerful_number_result_uses_independent_python_flint_replay(
    authorized_complete_runtime,
    value: str,
) -> None:
    producer_id = "integer.decide.powerful"
    verifier_id = "integer.powerful.verify"
    producer_payload = {"value": value}
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=producer_id,
            input=producer_payload,
        )
    )

    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=verifier_id,
            mode=CapabilityMode.VERIFY,
            input={"input": producer_payload, "candidate": computed.output["result"]},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == producer_id
    assert verified.output["verification_record_uri"] is not None
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["verification_record_uri"] in verified.artifact_uris


def test_powerful_number_verifier_rejects_schema_valid_wrong_factor_product(
    authorized_complete_runtime,
) -> None:
    producer_payload = {"value": "72"}
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.decide.powerful",
            input=producer_payload,
        )
    )
    forged_candidate = deepcopy(computed.output["result"])
    forged_candidate.update(
        {
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": True,
            "factors": [
                {"prime": "2", "power": 2},
                {"prime": "3", "power": 2},
            ],
            "violating_primes": [],
        }
    )

    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.powerful.verify",
            mode=CapabilityMode.VERIFY,
            input={"input": producer_payload, "candidate": forged_candidate},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_modular_residue_image_uses_independent_python_flint_replay(
    authorized_complete_runtime,
) -> None:
    producer_id = "modular.polynomial_residue_image.compute"
    verifier_id = "modular.polynomial_residue_image.verify"
    producer_payload = _modular_residue_payload()
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=producer_id,
            input=producer_payload,
        )
    )

    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=verifier_id,
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == producer_id
    assert verified.output["verification_record_uri"] is not None
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    provider_runtime = next(
        descriptor.provider_runtime
        for descriptor in authorized_complete_runtime.core.capabilities.catalog().capabilities
        if descriptor.capability_id == verifier_id
    )
    assert provider_runtime is not None
    assert provider_runtime.provider == "jacobian.exact-domain-checkers"
    assert {
        component["provider"]
        for component in provider_runtime.configuration["components"]
    } == {"jacobian.exact-domain-checker-source", "python-flint"}


def test_modular_residue_verifier_replays_its_materialized_lineage(
    authorized_complete_runtime,
) -> None:
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_residue_image.compute",
            input=_modular_residue_payload(coefficient="3"),
        )
    )
    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_residue_image.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "VERIFIED"
    assert rejected.output["conclusion"] == "TRUE"
    assert rejected.output["verification_record_uri"] is not None
    assert rejected.assurance.level is CapabilityAssuranceLevel.VERIFIED
