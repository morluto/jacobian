from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.number_theory import build_number_theory_bundle


@pytest.fixture
def number_theory_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install number theory and its exact checkers without a portfolio."""

    with open_exact_domain_services(
        tmp_path / "state",
        build_number_theory_bundle(),
    ) as services:
        yield services


def _modular_residue_payload(*, coefficient: str = "4") -> dict[str, object]:
    return {
        "modulus": 7,
        "variables": [
            {"name": "x", "residues": [0, 1, 2, 3, 4, 5, 6]},
        ],
        "terms": [{"coefficient": coefficient, "exponents": [3]}],
    }


def _modular_identity_payload() -> dict[str, object]:
    return {
        "modulus": 4,
        "variables": ["z"],
        "left": [
            {"coefficient": "9", "exponents": [6]},
            {"coefficient": "27", "exponents": [4]},
            {"coefficient": "30", "exponents": [2]},
            {"coefficient": "9", "exponents": [0]},
        ],
        "right": [
            {"coefficient": "-7", "exponents": [6]},
            {"coefficient": "15", "exponents": [4]},
            {"coefficient": "-10", "exponents": [2]},
            {"coefficient": "1", "exponents": [0]},
        ],
    }


def test_modular_polynomial_identity_uses_independent_python_flint_replay(
    number_theory_services,
) -> None:
    producer_id = "modular.polynomial_identity.compute"
    verifier_id = "modular.polynomial_identity.verify"
    payload = _modular_identity_payload()
    computed = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(capability_id=producer_id, input=payload)
    )
    verified = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=verifier_id,
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["conclusion"] == "TRUE"
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["verification_record_uri"] in verified.artifact_uris


def test_modular_polynomial_identity_verifier_rejects_perturbed_coefficient(
    number_theory_services,
) -> None:
    payload = _modular_identity_payload()
    computed = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_identity.compute",
            input=payload,
        )
    )
    forged = deepcopy(computed.output["result"])
    next(term for term in forged["normalized_left"] if term["exponents"] == [0])[
        "coefficient"
    ] = 2
    forged["identical"] = False
    forged["residual"] = [{"coefficient": 1, "exponents": [0]}]
    rejected = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_identity.verify",
            input={"input": payload, "candidate": forged},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_modular_polynomial_identity_replays_both_declared_term_budgets(
    number_theory_services,
) -> None:
    payload = {
        "modulus": 101,
        "variables": ["x"],
        "left": [{"coefficient": "1", "exponents": [index]} for index in range(33)],
        "right": [{"coefficient": "1", "exponents": [index]} for index in range(32)],
    }
    computed = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_identity.compute", input=payload
        )
    )
    verified = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_identity.verify",
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"


def test_prime_factorization_result_uses_independent_python_flint_replay(
    number_theory_services,
) -> None:
    producer_id = "integer.compute.prime_factorization"
    verifier_id = "integer.prime_factorization.verify"
    for value in ("360", "-360", "1", "-1", "101"):
        producer_payload = {"value": value}
        computed = number_theory_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=producer_id,
                input=producer_payload,
            )
        )

        verified = number_theory_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=verifier_id,
                input={
                    "input": producer_payload,
                    "candidate": computed.output["result"],
                },
            )
        )

        assert verified.execution.status is ExecutionStatus.COMPLETED, value
        assert verified.output["status"] == "VERIFIED", value
        assert verified.output["operation_id"] == producer_id, value
        assert verified.output["verification_record_uri"] is not None, value
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED, value
        assert verified.output["verification_record_uri"] in verified.artifact_uris, (
            value
        )
    provider_runtime = next(
        descriptor.provider_runtime
        for descriptor in number_theory_services.core.capabilities.catalog().capabilities
        if descriptor.capability_id == verifier_id
    )
    assert provider_runtime is not None
    assert provider_runtime.provider == "jacobian.exact-domain-checkers"
    assert {
        component["provider"]
        for component in provider_runtime.configuration["components"]
    } == {"jacobian.exact-domain-checker-source", "python-flint"}


def test_prime_factorization_verifier_rejects_incomplete_factor_list(
    number_theory_services,
) -> None:
    producer_payload = {"value": "360"}
    computed = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.compute.prime_factorization",
            input=producer_payload,
        )
    )
    forged_candidate = deepcopy(computed.output["result"])
    forged_candidate["factors"] = forged_candidate["factors"][:-1]

    rejected = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.prime_factorization.verify",
            input={"input": producer_payload, "candidate": forged_candidate},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_powerful_number_result_uses_independent_python_flint_replay(
    number_theory_services,
) -> None:
    producer_id = "integer.decide.powerful"
    verifier_id = "integer.powerful.verify"
    for value in ("1", "72", "12", "30"):
        producer_payload = {"value": value}
        computed = number_theory_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=producer_id,
                input=producer_payload,
            )
        )

        verified = number_theory_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=verifier_id,
                input={
                    "input": producer_payload,
                    "candidate": computed.output["result"],
                },
            )
        )

        assert verified.execution.status is ExecutionStatus.COMPLETED, value
        assert verified.output["status"] == "VERIFIED", value
        assert verified.output["operation_id"] == producer_id, value
        assert verified.output["verification_record_uri"] is not None, value
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED, value
        assert verified.output["verification_record_uri"] in verified.artifact_uris, (
            value
        )


def test_powerful_number_verifier_rejects_schema_valid_wrong_factor_product(
    number_theory_services,
) -> None:
    producer_payload = {"value": "72"}
    computed = number_theory_services.core.capabilities.invoke(
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

    rejected = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.powerful.verify",
            input={"input": producer_payload, "candidate": forged_candidate},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_modular_residue_image_uses_independent_python_flint_replay(
    number_theory_services,
) -> None:
    producer_id = "modular.polynomial_residue_image.compute"
    verifier_id = "modular.polynomial_residue_image.verify"
    producer_payload = _modular_residue_payload()
    computed = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=producer_id,
            input=producer_payload,
        )
    )

    verified = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=verifier_id,
            input={
                "input": producer_payload,
                "candidate": computed.output["result"],
            },
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
        for descriptor in number_theory_services.core.capabilities.catalog().capabilities
        if descriptor.capability_id == verifier_id
    )
    assert provider_runtime is not None
    assert provider_runtime.provider == "jacobian.exact-domain-checkers"
    assert {
        component["provider"]
        for component in provider_runtime.configuration["components"]
    } == {"jacobian.exact-domain-checker-source", "python-flint"}


def test_modular_residue_verifier_replays_its_materialized_lineage(
    number_theory_services,
) -> None:
    computed = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_residue_image.compute",
            input=_modular_residue_payload(coefficient="3"),
        )
    )
    rejected = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_residue_image.verify",
            input={
                "input": _modular_residue_payload(coefficient="3"),
                "candidate": computed.output["result"],
            },
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "VERIFIED"
    assert rejected.output["conclusion"] == "TRUE"
    assert rejected.output["verification_record_uri"] is not None
    assert rejected.assurance.level is CapabilityAssuranceLevel.VERIFIED
