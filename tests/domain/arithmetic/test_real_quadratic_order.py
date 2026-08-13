from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.arithmetic import build_arithmetic_bundle


@pytest.fixture
def arithmetic_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_exact_domain_services(
        tmp_path / "state", build_arithmetic_bundle()
    ) as services:
        yield services


def _payload() -> dict[str, object]:
    return {
        "left": {
            "rational_part": {"num": "0", "den": "1"},
            "radical_coefficient": {"num": "3", "den": "8"},
            "radicand": 3,
        },
        "right": {
            "rational_part": {"num": "1", "den": "2"},
            "radical_coefficient": {"num": "1", "den": "20"},
            "radicand": 3,
        },
    }


def test_real_quadratic_order_computes_and_verifies(
    arithmetic_services: DomainTestServices,
) -> None:
    payload = _payload()
    computed = arithmetic_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="arithmetic.real_quadratic.order.compute", input=payload
        )
    )
    result = computed.output["result"]
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert result["order"] == "GT"
    assert result["sign_certificate"]["magnitude_order"] == "LT"
    verified = arithmetic_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="arithmetic.real_quadratic.order.verify",
            input={"input": payload, "candidate": result},
        )
    )
    assert verified.output["status"] == "VERIFIED"
    assert verified.verification_record_uri is not None


def test_real_quadratic_checker_rejects_forged_order(
    arithmetic_services: DomainTestServices,
) -> None:
    payload = _payload()
    computed = arithmetic_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="arithmetic.real_quadratic.order.compute", input=payload
        )
    )
    forged = deepcopy(computed.output["result"])
    forged["order"] = "LT"
    rejected = arithmetic_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="arithmetic.real_quadratic.order.verify",
            input={"input": payload, "candidate": forged},
        )
    )
    assert rejected.execution.status is ExecutionStatus.ERROR
    assert rejected.output["error"]["code"] == "INVALID_EXACT_DOMAIN_INPUT"
    assert rejected.verification_record_uri is None
