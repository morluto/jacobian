from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.number_theory import build_number_theory_bundle


@pytest.fixture
def number_theory_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_exact_domain_services(
        tmp_path / "state", build_number_theory_bundle()
    ) as services:
        yield services


def _payload() -> dict[str, object]:
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


def test_modular_polynomial_identity_computes_and_verifies(
    number_theory_services: DomainTestServices,
) -> None:
    payload = _payload()
    computed = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_identity.compute", input=payload
        )
    )
    comparison = computed.output["result"]
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert comparison["identical"] is True
    assert comparison["residual"] == []
    assert comparison["comparison_scope"] == "FORMAL_COEFFICIENTWISE_IDENTITY"

    verified = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_identity.verify",
            input={"input": payload, "candidate": comparison},
        )
    )
    assert verified.output["status"] == "VERIFIED"
    assert verified.verification_record_uri is not None


def test_modular_polynomial_identity_rejects_forged_coefficient(
    number_theory_services: DomainTestServices,
) -> None:
    payload = _payload()
    computed = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_identity.compute", input=payload
        )
    )
    forged = deepcopy(computed.output["result"])
    forged["normalized_left"][0]["coefficient"] = 2
    forged["identical"] = False
    forged["residual"] = [{"coefficient": 1, "exponents": [0]}]
    rejected = number_theory_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="modular.polynomial_identity.verify",
            input={"input": payload, "candidate": forged},
        )
    )
    assert rejected.output["status"] == "REJECTED"
    assert rejected.verification_record_uri is None
