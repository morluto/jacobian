from __future__ import annotations

from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus, InputStatus


def _rational(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


def _request(image: int) -> CapabilityRequest:
    return CapabilityRequest(
        capability_id="polynomial.map.collision.verify",
        input={
            "map": {
                "variables": ["x"],
                "coordinates": [
                    {
                        "terms": [
                            {
                                "coefficient": _rational(1),
                                "exponents": [2],
                            }
                        ]
                    }
                ],
            },
            "first_point": [_rational(-1)],
            "second_point": [_rational(1)],
            "claimed_image": [_rational(image)],
        },
    )


def test_direct_collision_verifier_promotes_only_independent_replay(
    authorized_polynomial_services,
) -> None:

    result = authorized_polynomial_services.core.capabilities.invoke(_request(1))

    assert result.output["collision_verified"] is True
    assert result.output["conclusion"] == "FALSE"
    assert result.output["verification_input"] == {
        "status": InputStatus.ACCEPTED.value,
        "errors": [],
    }
    assert result.verification_record_uri is not None
    record_uri = result.output["verification_record_uri"]
    assert record_uri in result.artifact_uris
    assert result.execution.status is ExecutionStatus.COMPLETED


def test_direct_collision_verifier_fails_closed_for_wrong_image(
    authorized_polynomial_services,
) -> None:

    result = authorized_polynomial_services.core.capabilities.invoke(_request(2))

    assert result.output["collision_verified"] is False
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["verification_input"] == {
        "status": InputStatus.REJECTED.value,
        "errors": ["declared collision does not replay exactly"],
    }
    assert result.output["verification_record_uri"] is None
    assert result.execution.status is ExecutionStatus.COMPLETED


def test_direct_collision_verifier_requires_authorized_reference_checker(
    polynomial_services,
) -> None:
    runtime = polynomial_services

    assert "polynomial.map.collision.verify" not in {
        item.capability_id for item in runtime.core.capabilities.catalog().capabilities
    }
