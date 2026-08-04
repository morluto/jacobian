from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import CapabilityAssuranceLevel, CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.combinatorics import build_combinatorics_bundle

_BASE = ["1", "2", "4", "8", "13"]


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state", build_combinatorics_bundle()
    ) as services:
        yield services


def test_integer_sidon_materializes_every_ordered_difference(
    domain_services: DomainTestServices,
) -> None:
    computed = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.integer_set.sidon.decide",
            input={"elements": _BASE},
        )
    )

    result = computed.output["result"]
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result["is_sidon"] is True
    assert len(result["ordered_differences"]) == 20


def test_perfect_difference_set_reports_complete_residue_profile(
    domain_services: DomainTestServices,
) -> None:
    computed = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.cyclic_difference_set.perfect.decide",
            input={"modulus": 7, "residues": [0, 1, 3]},
        )
    )

    result = computed.output["result"]
    assert result["is_perfect"] is True
    assert result["missing_residues"] == []
    assert result["repeated_residues"] == []
    assert len(result["difference_multiplicities"]) == 6


@pytest.mark.parametrize(
    ("order", "candidate_count"),
    ((5, 1), (6, 26), (7, 703)),
)
def test_fixed_order_extension_materializes_complete_negative_decisions(
    domain_services: DomainTestServices,
    order: int,
    candidate_count: int,
) -> None:
    computed = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.cyclic_difference_set.extension.decide",
            input={"base_elements": _BASE, "target_order": order},
        )
    )

    stored = domain_services.core.store.get(computed.output["result_uri"])
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert computed.output["result_uri"] in computed.artifact_uris
    assert computed.output["preview_complete"] is True
    assert stored.payload == computed.output["preview"]
    assert stored.payload["decision"] == "DOES_NOT_EXTEND"
    assert stored.payload["coverage"] == "ALL_CANDIDATES"
    assert stored.payload["candidate_space_size"] == candidate_count


def test_fixed_order_extension_returns_a_complete_positive_witness(
    domain_services: DomainTestServices,
) -> None:
    computed = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.cyclic_difference_set.extension.decide",
            input={"base_elements": ["0", "1"], "target_order": 3},
        )
    )

    stored = domain_services.core.store.get(computed.output["result_uri"])
    assert stored.payload["decision"] == "EXTENDS"
    assert stored.payload["coverage"] == "WITNESS"
    assert stored.payload["extension"] == [0, 1, 3]
