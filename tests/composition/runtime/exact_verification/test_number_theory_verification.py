from __future__ import annotations

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
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=producer_id,
            input={"value": value},
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


def test_prime_factorization_verifier_rejects_incomplete_factor_list(
    authorized_complete_runtime,
) -> None:
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.compute.prime_factorization",
            input={"value": "360"},
        )
    )
    result_artifact = authorized_complete_runtime.core.store.get(
        computed.output["result_uri"]
    )
    false_result = authorized_complete_runtime.core.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload={"factors": result_artifact.payload["factors"][:-1]},
        summary="adversarial incomplete prime factorization result",
    )

    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.prime_factorization.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
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
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=producer_id,
            input={"value": value},
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


def test_powerful_number_verifier_rejects_schema_valid_wrong_factor_product(
    authorized_complete_runtime,
) -> None:
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.decide.powerful",
            input={"value": "72"},
        )
    )
    result_artifact = authorized_complete_runtime.core.store.get(
        computed.output["result_uri"]
    )
    false_result = authorized_complete_runtime.core.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload={
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": True,
            "factors": [
                {"prime": "2", "power": 2},
                {"prime": "3", "power": 2},
            ],
            "violating_primes": [],
        },
        summary="adversarial wrong powerful-number factor product",
    )

    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.powerful.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
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
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=producer_id,
            input=_modular_residue_payload(),
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


def test_modular_residue_verifier_rejects_source_result_substitution(
    authorized_complete_runtime,
) -> None:
    source_result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_residue_image.compute",
            input=_modular_residue_payload(),
        )
    )
    substituted_result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_residue_image.compute",
            input=_modular_residue_payload(coefficient="3"),
        )
    )
    source_artifact = authorized_complete_runtime.core.store.get(
        source_result.output["result_uri"]
    )
    candidate_artifact = authorized_complete_runtime.core.store.get(
        substituted_result.output["result_uri"]
    )
    rebound_candidate = authorized_complete_runtime.core.artifacts.put(
        schema_uri=candidate_artifact.manifest.schema_uri,
        semantics_uri=candidate_artifact.manifest.semantics_uri,
        parents=source_artifact.manifest.parents,
        payload=candidate_artifact.payload,
        summary="adversarial modular residue source-result substitution",
    )

    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_residue_image.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": rebound_candidate.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED
