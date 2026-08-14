from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from jacobian.contracts.exact_domain_verification import InlineExactVerificationRecord
from jacobian.contracts.operations import (
    OperationRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.polynomial import polynomial_operations
from tests.support.exact_domain import open_exact_domain_services
from tests.support.rationals import rational_payload as _q
from tests.support.services import DomainTestServices


def _poly(*coefficients_ascending: int) -> dict[str, object]:
    return {
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": _q(coefficient), "exponents": [exponent]}
                for exponent, coefficient in reversed(
                    tuple(enumerate(coefficients_ascending))
                )
                if coefficient
            ]
        },
    }


def _poly_xy(*terms: tuple[tuple[int, int], int]) -> dict[str, object]:
    return {
        "variables": ["x", "y"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": _q(coefficient),
                    "exponents": list(exponents),
                }
                for exponents, coefficient in terms
            ]
        },
    }


@pytest.fixture
def polynomial_verification_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install polynomial operations and their exact checkers without a portfolio."""

    with open_exact_domain_services(
        tmp_path / "state",
        polynomial_operations(),
    ) as services:
        yield services


def _gcd_input() -> dict[str, object]:
    return {
        "left": _poly(-1, 0, 1),
        "right": _poly(0, 1, 1),
    }


def _computed_gcd(polynomial_verification_services: DomainTestServices):
    return polynomial_verification_services.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.compute.gcd",
            input=_gcd_input(),
        )
    )


def test_public_seam_verifies_exact_producer_result(
    polynomial_verification_services: DomainTestServices,
) -> None:
    provider_runtime = next(
        descriptor.provider_runtime
        for descriptor in polynomial_verification_services.core.operations.snapshot().operations
        if descriptor.operation_id == "polynomial.gcd.verify"
    )
    assert provider_runtime is not None
    assert {
        component["provider"]
        for component in provider_runtime.configuration["components"]
    } == {"jacobian.exact-domain-checker-source", "python-flint"}
    computed = _computed_gcd(polynomial_verification_services)

    verified = polynomial_verification_services.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.gcd.verify",
            input={
                "input": _gcd_input(),
                "candidate": computed.output["result"],
            },
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == "polynomial.compute.gcd"
    assert verified.output["verification_record_uri"] is not None
    record = polynomial_verification_services.core.store.get(
        verified.output["verification_record_uri"]
    )
    parsed = InlineExactVerificationRecord.model_validate(record.payload)
    assert verified.artifact_uris == (
        verified.output["verification_record_uri"],
        parsed.semantics_uri,
    )


def test_public_seam_rejects_validly_shaped_false_result(
    polynomial_verification_services: DomainTestServices,
) -> None:
    _computed_gcd(polynomial_verification_services)

    false_candidate = {
        "gcd": _poly(1),
        "bezout": {
            "left_multiplier": _poly(),
            "right_multiplier": _poly(),
        },
        "normalization": "MONIC",
    }

    rejected = polynomial_verification_services.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.gcd.verify",
            input={
                "input": _gcd_input(),
                "candidate": false_candidate,
            },
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_public_seam_reports_valid_multivariate_result_as_unsupported(
    polynomial_verification_services: DomainTestServices,
) -> None:
    resultant_input = {
        "left": _poly_xy(((1, 0), 1), ((0, 1), 1)),
        "right": _poly_xy(((1, 0), 1), ((0, 0), 1)),
        "elimination_variable": "x",
    }
    computed = polynomial_verification_services.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.compute.resultant",
            input=resultant_input,
        )
    )

    checked = polynomial_verification_services.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.resultant.verify",
            input={
                "input": resultant_input,
                "candidate": computed.output["result"],
            },
        )
    )

    assert checked.execution.status is ExecutionStatus.COMPLETED
    assert checked.output["status"] == "UNSUPPORTED"
    assert checked.output["conclusion"] == "UNKNOWN"
    assert checked.output["witness_uri"] is None
    assert checked.output["verification_record_uri"] is None
