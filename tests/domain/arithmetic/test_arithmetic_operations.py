from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.arithmetic import arithmetic_operations

_LARGE_CANONICAL_INTEGER = "1" + ("0" * 4_999) + "1"


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(tmp_path / "state", arithmetic_operations()) as services:
        yield services


def test_arithmetic_operations_return_exact_results(
    domain_services: DomainTestServices,
) -> None:
    cases = (
        (
            "integer.compute.nth_root",
            {"value": "65", "degree": 3},
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
    )
    for operation_id, payload, expected in cases:
        result = domain_services.core.operations.invoke(
            OperationRequest(operation_id=operation_id, input=payload)
        )

        assert result.execution.status is ExecutionStatus.COMPLETED, operation_id
        assert result.output["result"] == expected, operation_id


def test_rational_operation_rejects_unreduced_input_before_execution(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="rational.compute.reciprocal",
            input={"value": {"num": "2", "den": "4"}},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_ARITHMETIC_REQUEST"
    assert result.artifact_uris == ()


def test_rational_product_formats_results_above_python_digit_limit(
    domain_services: DomainTestServices,
) -> None:
    factor = "1" + "0" * 2500

    result = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="rational.compute.product",
            input={
                "left": {"num": factor, "den": "1"},
                "right": {"num": factor, "den": "1"},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"value": {"num": "1" + "0" * 5000, "den": "1"}}


@pytest.mark.parametrize(
    ("value", "degree", "expected"),
    (
        ("729000000", 3, {"root": "900", "exact": True}),
        ("729000001", 3, {"root": "900", "exact": False}),
        ("-729000000", 3, {"root": "-900", "exact": True}),
        ("-9", 3, {"root": "-3", "exact": False}),
    ),
)
def test_integer_nth_root_accepts_canonical_integers_above_small_scalar_bound(
    domain_services: DomainTestServices,
    value: str,
    degree: int,
    expected: dict[str, object],
) -> None:
    result = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="integer.compute.nth_root",
            input={"value": value, "degree": degree},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == expected


def test_rational_difference_accepts_contract_sized_components(
    domain_services: DomainTestServices,
) -> None:
    value = {"num": _LARGE_CANONICAL_INTEGER, "den": "1"}

    result = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="rational.compute.difference",
            input={"left": value, "right": value},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"value": {"num": "0", "den": "1"}}


def test_integer_and_rational_operations_cross_the_large_integer_boundary(
    domain_services: DomainTestServices,
) -> None:
    value = "1" + ("0" * 5_000)

    absolute_value = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="integer.compute.absolute_value",
            input={"value": value},
        )
    )
    digit_count = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="integer.compute.decimal_digit_count",
            input={"value": value},
        )
    )
    digit_sum = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="integer.compute.decimal_digit_sum",
            input={"value": value},
        )
    )
    floor = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="rational.compute.floor",
            input={"value": {"num": value, "den": "1"}},
        )
    )
    ceiling = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="rational.compute.ceiling",
            input={"value": {"num": value, "den": "1"}},
        )
    )
    continued_fraction = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="rational.compute.continued_fraction",
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
