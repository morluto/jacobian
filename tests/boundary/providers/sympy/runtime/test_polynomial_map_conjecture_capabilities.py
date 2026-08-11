from __future__ import annotations

import json
from pathlib import Path

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus, InputStatus


def _rational(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


def _identity_map() -> dict[str, object]:
    return {
        "variables": ["x"],
        "coordinates": [
            {
                "terms": [
                    {"coefficient": _rational(1), "exponents": [1]},
                ]
            }
        ],
    }


def _square_map() -> dict[str, object]:
    return {
        "variables": ["x"],
        "coordinates": [
            {
                "terms": [
                    {"coefficient": _rational(1), "exponents": [2]},
                ]
            }
        ],
    }


def test_public_jacobian_counterexample_fixture_replays_both_claim_bindings(
    authorized_complete_runtime,
) -> None:
    task_root = (
        Path(__file__).resolve().parents[5]
        / "benchmarks"
        / "datasets"
        / "public-reproductions-v1"
    )
    keller = json.loads(
        (task_root / "jacobian-keller" / "environment" / "input.json").read_text()
    )
    obstruction = json.loads(
        (
            task_root / "jacobian-inverse-obstruction" / "environment" / "input.json"
        ).read_text()
    )
    runtime = authorized_complete_runtime

    keller_result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.keller_condition.verify",
            mode=CapabilityMode.VERIFY,
            input={"map": keller["map"]},
        )
    )
    obstruction_result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.inverse.refute_by_collision",
            mode=CapabilityMode.VERIFY,
            input={
                "map": obstruction["map"],
                "first_point": obstruction["first_point"],
                "second_point": obstruction["second_point"],
                "claimed_image": obstruction["claimed_image"],
            },
        )
    )

    assert keller_result.output["keller_condition_verified"] is True
    assert keller_result.output["determinant"]["terms"] == [
        {"coefficient": {"num": "-2", "den": "1"}, "exponents": [0, 0, 0]}
    ]
    assert obstruction_result.output["noninvertibility_verified"] is True
    assert obstruction_result.output["conclusion"] == "TRUE"


def test_keller_condition_verifies_the_published_style_exact_map(
    authorized_complete_runtime,
) -> None:
    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.keller_condition.verify",
            mode=CapabilityMode.VERIFY,
            input={"map": _identity_map()},
        )
    )

    assert result.output["keller_condition_verified"] is True
    assert result.output["conclusion"] == "TRUE"
    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert result.output["verification_record_uri"] in result.artifact_uris
    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.relationships[0].relation_id == (
        "polynomial.relation.keller-condition"
    )


def test_keller_condition_verifies_a_false_conclusion_for_nonconstant_determinant(
    authorized_complete_runtime,
) -> None:
    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.keller_condition.verify",
            mode=CapabilityMode.VERIFY,
            input={"map": _square_map()},
        )
    )

    assert result.output["keller_condition_verified"] is False
    assert result.output["conclusion"] == "FALSE"
    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert result.output["verification_record_uri"] in result.artifact_uris
    assert result.relationships == ()


def test_collision_refutes_two_sided_inverse(
    authorized_complete_runtime,
) -> None:
    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.inverse.refute_by_collision",
            mode=CapabilityMode.VERIFY,
            input={
                "map": _square_map(),
                "first_point": [_rational(-1)],
                "second_point": [_rational(1)],
                "claimed_image": [_rational(1)],
            },
        )
    )

    assert result.output["noninvertibility_verified"] is True
    assert result.output["conclusion"] == "TRUE"
    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert result.output["verification_record_uri"] in result.artifact_uris
    assert result.relationships[0].relation_id == (
        "polynomial.relation.collision-refutes-two-sided-inverse"
    )


def test_collision_inverse_obstruction_fails_closed_for_wrong_image(
    authorized_complete_runtime,
) -> None:
    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.inverse.refute_by_collision",
            mode=CapabilityMode.VERIFY,
            input={
                "map": _square_map(),
                "first_point": [_rational(-1)],
                "second_point": [_rational(1)],
                "claimed_image": [_rational(2)],
            },
        )
    )

    assert result.output["noninvertibility_verified"] is None
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["verification_input"] == {
        "status": InputStatus.REJECTED.value,
        "errors": ["declared collision does not replay exactly"],
        "warnings": [],
    }
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert result.output["verification_record_uri"] is None


def test_new_checker_capabilities_are_omitted_without_authorization(
    attached_complete_runtime,
) -> None:
    capability_ids = {
        item.capability_id
        for item in attached_complete_runtime.core.capabilities.catalog().capabilities
    }

    assert "polynomial.map.keller_condition.verify" not in capability_ids
    assert "polynomial.map.inverse.refute_by_collision" not in capability_ids
