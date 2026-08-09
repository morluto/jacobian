from __future__ import annotations

from copy import deepcopy
from typing import Any

from jacobian.checker_operations import derive_verification_capability_id
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.exact_domain_verification import InlineExactVerificationRecord
from jacobian.contracts.results import ExecutionStatus

_PRESENTATION = {
    "vertices": ["a", "b", "c"],
    "facets": [["a", "b"], ["b", "c"], ["a", "c"]],
}


def _result_payload(computed) -> dict:
    return computed.output["result"]


def _computed_cases(topology_services) -> list[tuple[str, dict, Any]]:
    canonicalized = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.canonicalize",
            input=_PRESENTATION,
        )
    )
    complex_ = _result_payload(canonicalized)["complex"]
    chain_input = {
        "complex": complex_,
        "coefficient_ring": "PRIME_FIELD",
        "prime": 2,
        "convention": "UNREDUCED",
    }
    chain = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.chain_complex.compute",
            input=chain_input,
        )
    )
    homology_input = {
        "complex": complex_,
        "prime": 2,
        "convention": "UNREDUCED",
    }
    homology = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_homology.compute",
            input=homology_input,
        )
    )
    return [
        ("topology.simplicial_complex.canonicalize", _PRESENTATION, canonicalized),
        ("topology.simplicial_complex.chain_complex.compute", chain_input, chain),
        ("topology.simplicial_homology.compute", homology_input, homology),
    ]


def test_topology_results_are_independently_verified(
    topology_services,
) -> None:
    for producer_id, producer_input, computed in _computed_cases(topology_services):
        verified = topology_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=derive_verification_capability_id(producer_id),
                mode=CapabilityMode.VERIFY,
                input={
                    "input": producer_input,
                    "candidate": _result_payload(computed),
                },
            )
        )

        assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
        assert verified.execution.status is ExecutionStatus.COMPLETED
        assert verified.output["status"] == "VERIFIED"
        assert verified.output["verification_record_uri"] in verified.artifact_uris
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
        record = topology_services.core.store.get(
            verified.output["verification_record_uri"]
        )
        parsed = InlineExactVerificationRecord.model_validate(record.payload)
        assert verified.artifact_uris == (
            verified.output["verification_record_uri"],
            parsed.semantics_uri,
        )


def test_topology_checker_rejects_forged_cycle_evidence(
    topology_services,
) -> None:
    producer_id, _producer_input, homology = _computed_cases(topology_services)[2]
    forged_candidate = deepcopy(_result_payload(homology))
    forged_groups = [dict(group) for group in forged_candidate["groups"]]
    forged_group = dict(forged_groups[1])
    forged_cycles = [dict(vector) for vector in forged_group["cycle_basis"]]
    forged_cycles[0]["coefficients"] = [1, 0, 0]
    forged_group["cycle_basis"] = forged_cycles
    forged_groups[1] = forged_group
    forged_candidate["groups"] = forged_groups

    rejected = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=derive_verification_capability_id(producer_id),
            mode=CapabilityMode.VERIFY,
            input={"input": _producer_input, "candidate": forged_candidate},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_integral_homology_has_a_dedicated_independent_verifier(
    topology_services,
) -> None:
    _producer_id, _canonicalization_input, canonicalized = _computed_cases(
        topology_services
    )[0]
    integral_input = {
        "complex": _result_payload(canonicalized)["complex"],
        "convention": "UNREDUCED",
    }
    computed = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_homology.integral.compute",
            input=integral_input,
        )
    )
    verified = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_homology.integral.verify",
            mode=CapabilityMode.VERIFY,
            input={"input": integral_input, "candidate": _result_payload(computed)},
        )
    )

    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["verification_record_uri"] in verified.artifact_uris


def test_integral_homology_checker_rejects_a_forged_free_generator(
    topology_services,
) -> None:
    _producer_id, _canonicalization_input, canonicalized = _computed_cases(
        topology_services
    )[0]
    integral_input = {
        "complex": _result_payload(canonicalized)["complex"],
        "convention": "UNREDUCED",
    }
    computed = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_homology.integral.compute",
            input=integral_input,
        )
    )
    forged_candidate = deepcopy(_result_payload(computed))
    groups = [dict(group) for group in forged_candidate["groups"]]
    group = dict(groups[1])
    generators = [dict(item) for item in group["free_generators"]]
    cycle = dict(generators[0]["cycle"])
    coefficients = list(cycle["coefficients"])
    coefficients[0] = str(int(coefficients[0]) + 1)
    cycle["coefficients"] = coefficients
    generators[0]["cycle"] = cycle
    group["free_generators"] = generators
    groups[1] = group
    forged_candidate["groups"] = groups
    rejected = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_homology.integral.verify",
            mode=CapabilityMode.VERIFY,
            input={"input": integral_input, "candidate": forged_candidate},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_topology_checker_runtime_binds_only_independent_source(
    topology_services,
) -> None:
    descriptor = next(
        item
        for item in topology_services.core.capabilities.catalog().capabilities
        if item.capability_id == "topology.simplicial_homology.verify"
    )
    assert descriptor.provider_runtime is not None
    assert {
        component["provider"]
        for component in descriptor.provider_runtime.configuration["components"]
    } == {"jacobian.topology-exact-checker-source"}
