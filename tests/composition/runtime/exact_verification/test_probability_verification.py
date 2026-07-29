from __future__ import annotations

from typing import Any

import pytest
from tests.support.rationals import rational_payload as _q

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus

_FAIR_BIT = {
    "atoms": [
        {"value": _q(0), "probability": _q(1, 2)},
        {"value": _q(1), "probability": _q(1, 2)},
    ]
}
_GAUSSIAN_POLYNOMIAL = {
    "variable_count": 1,
    "terms": [
        {
            "coefficient": {"real": _q(1), "imaginary": _q(0)},
            "exponents": [0],
        },
        {
            "coefficient": {"real": _q(0), "imaginary": _q(1)},
            "exponents": [1],
        },
    ],
}


@pytest.mark.parametrize(
    ("capability_id", "payload"),
    (
        (
            "probability.finite_distribution.raw_moment.compute",
            {"atoms": _FAIR_BIT["atoms"], "order": 2},
        ),
        (
            "probability.finite_distribution.event_probability.compute",
            {"distribution": _FAIR_BIT, "event_values": [_q(1)]},
        ),
        (
            "probability.finite_distribution.condition.compute",
            {"distribution": _FAIR_BIT, "event_values": [_q(1)]},
        ),
        (
            "probability.finite_distribution.pushforward.compute",
            {
                "distribution": _FAIR_BIT,
                "mapping": [
                    {"source": _q(0), "target": _q(0)},
                    {"source": _q(1), "target": _q(0)},
                ],
            },
        ),
        (
            "probability.finite_distribution.convolution.compute",
            {"left": _FAIR_BIT, "right": _FAIR_BIT},
        ),
        (
            "probability.gaussian_polynomial.moment.compute",
            {"polynomial": _GAUSSIAN_POLYNOMIAL, "order": 2},
        ),
    ),
)
def test_probability_results_are_independently_replayed(
    authorized_complete_runtime,
    capability_id: str,
    payload: dict[str, Any],
) -> None:
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(capability_id=capability_id, input=payload)
    )

    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == capability_id
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


def test_probability_checker_rejects_forged_event_mass(
    authorized_complete_runtime,
) -> None:
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("probability.finite_distribution.event_probability.compute"),
            input={"distribution": _FAIR_BIT, "event_values": [_q(1)]},
        )
    )
    result_artifact = authorized_complete_runtime.core.store.get(
        computed.output["result_uri"]
    )
    false_payload = dict(result_artifact.payload)
    false_payload["event_probability"] = _q(1)
    false_payload["selected_atoms"] = _FAIR_BIT["atoms"]
    false_result = authorized_complete_runtime.core.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload=false_payload,
        summary="adversarial false finite-event probability",
    )

    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
        )
    )

    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED
