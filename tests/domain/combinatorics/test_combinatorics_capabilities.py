from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityDiscoveryRequest,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.combinatorics import build_combinatorics_bundle


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state", build_combinatorics_bundle()
    ) as services:
        yield services


def test_bernoulli_number_has_exact_rational_value(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.compute.bernoulli",
            input={"n": 4},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"value": {"num": "-1", "den": "30"}}


def test_binomial_is_discoverable_from_number_theory_language(
    domain_services: DomainTestServices,
) -> None:
    discovered = domain_services.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="compute exact binomial coefficients for large integers",
            domain="number_theory",
            limit=5,
        )
    )

    assert discovered.matches[0].capability_id == "combinatorics.compute.binomial"
    assert discovered.matches[0].lexical_fit == "STRONG_CANDIDATE"

    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.compute.binomial",
            input={"n": 1912, "k": 16},
        )
    )
    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["value"] == (
        "1431712059377249479518540967853195958045"
    )


def test_combinatorics_resource_atomics_are_exact_computed(
    domain_services: DomainTestServices,
) -> None:
    cases = (
        (
            "combinatorics.compute.fibonacci_pair",
            {"n": 10},
            {"n": 10, "f_n": "55", "f_n_plus_one": "89"},
        ),
        (
            "combinatorics.compute.multinomial",
            {"values": ["2", "1", "1"]},
            {"value": "12"},
        ),
    )
    for capability_id, payload, expected in cases:
        result = domain_services.core.capabilities.invoke(
            CapabilityRequest(capability_id=capability_id, input=payload)
        )
        assert result.execution.status is ExecutionStatus.COMPLETED
        assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
        assert result.output["result"] == expected
