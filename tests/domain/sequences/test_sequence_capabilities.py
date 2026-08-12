from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.sequences import build_sequence_bundle


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(tmp_path / "state", build_sequence_bundle()) as services:
        yield services


def test_prefix_gcds_return_each_prefix_result(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="sequence.compute.prefix_gcds",
            input={"values": ["18", "24", "15"]},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"values": ["18", "6", "3"]}


def test_geometric_sequence_handles_zero_terms_exactly(
    domain_services: DomainTestServices,
) -> None:
    cases = (
        (["0", "0", "1"], False),
        (["1", "0", "0"], True),
        (["0", "0", "0"], True),
        (["2", "4", "8", "16"], True),
        (["8", "-4", "2", "-1"], True),
        (["2", "4", "9"], False),
    )
    for values, expected in cases:
        result = domain_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="sequence.decide.geometric",
                input={"values": values},
            )
        )

        assert result.execution.status is ExecutionStatus.COMPLETED, values
        assert result.output["result"] == {"holds": expected}, values


def test_sequence_products_format_results_beyond_python_digit_limit(
    domain_services: DomainTestServices,
) -> None:
    factor = "1" + ("0" * 2_500)
    expected_product = "1" + ("0" * 5_000)

    product = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="sequence.compute.product",
            input={"values": [factor, factor]},
        )
    )
    prefixes = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="sequence.compute.prefix_products",
            input={"values": [factor, factor]},
        )
    )

    assert product.execution.status is ExecutionStatus.COMPLETED
    assert product.output["result"] == {"value": expected_product}
    assert prefixes.execution.status is ExecutionStatus.COMPLETED
    assert prefixes.output["result"] == {"values": [factor, expected_product]}
