from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.arithmetic import build_arithmetic_bundle

_LARGE_CANONICAL_INTEGER = "1" + ("0" * 4_999) + "1"


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state", build_arithmetic_bundle()
    ) as services:
        yield services


@pytest.mark.parametrize(
    ("capability_id", "payload", "expected"),
    (
        (
            "integer.compute.nth_root",
            {"value": 65, "degree": 3},
            {"root": "4", "exact": False},
        ),
        (
            "rational.compute.continued_fraction",
            {"value": {"num": "-7", "den": "5"}},
            {"terms": ["-2", "1", "1", "2"]},
        ),
        (
            "integer.transform.base_digits",
            {"value": "-10", "base": 2},
            {"sign": -1, "base": 2, "digits": ["1", "0", "1", "0"]},
        ),
    ),
)
def test_arithmetic_capabilities_return_exact_results(
    domain_services: DomainTestServices,
    capability_id: str,
    payload: dict[str, object],
    expected: dict[str, object],
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(capability_id=capability_id, input=payload)
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == expected


def test_rational_product_formats_results_above_python_digit_limit(
    domain_services: DomainTestServices,
) -> None:
    factor = "1" + "0" * 2500

    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="rational.compute.product",
            input={
                "left": {"num": factor, "den": "1"},
                "right": {"num": factor, "den": "1"},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"value": {"num": "1" + "0" * 5000, "den": "1"}}


def test_rational_difference_accepts_contract_sized_components(
    domain_services: DomainTestServices,
) -> None:
    value = {"num": _LARGE_CANONICAL_INTEGER, "den": "1"}

    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="rational.compute.difference",
            input={"left": value, "right": value},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"value": {"num": "0", "den": "1"}}


def test_integer_and_rational_operations_cross_the_large_integer_boundary(
    domain_services: DomainTestServices,
) -> None:
    value = "1" + ("0" * 5_000)

    absolute_value = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.compute.absolute_value",
            input={"value": value},
        )
    )
    digit_count = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.compute.decimal_digit_count",
            input={"value": value},
        )
    )
    digit_sum = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.compute.decimal_digit_sum",
            input={"value": value},
        )
    )
    floor = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="rational.compute.floor",
            input={"value": {"num": value, "den": "1"}},
        )
    )
    ceiling = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="rational.compute.ceiling",
            input={"value": {"num": value, "den": "1"}},
        )
    )
    continued_fraction = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="rational.compute.continued_fraction",
            input={"value": {"num": value, "den": "1"}},
        )
    )

    assert absolute_value.execution.status is ExecutionStatus.COMPLETED
    assert absolute_value.output["result"] == {"value": value}
    assert digit_count.execution.status is ExecutionStatus.COMPLETED
    assert digit_count.output["result"] == {"value": "5001"}
    assert digit_sum.execution.status is ExecutionStatus.COMPLETED
    assert digit_sum.output["result"] == {"value": "1"}
    assert floor.execution.status is ExecutionStatus.COMPLETED
    assert floor.output["result"] == {"value": value}
    assert ceiling.execution.status is ExecutionStatus.COMPLETED
    assert ceiling.output["result"] == {"value": value}
    assert continued_fraction.execution.status is ExecutionStatus.COMPLETED
    assert continued_fraction.output["result"] == {"terms": [value]}
