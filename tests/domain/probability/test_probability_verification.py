from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from tests.support.rationals import rational_payload as _q

from jacobian.checker_operations import derive_verification_capability_id
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
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
        (
            "probability.graph_reliability.connection_probability.compute",
            {
                "graph": {
                    "vertices": ["a", "b"],
                    "edges": [["a", "b"]],
                },
                "edge_probabilities": [
                    {
                        "edge": ["a", "b"],
                        "open_probability": _q(1, 3),
                    }
                ],
                "terminals": ["a", "b"],
            },
        ),
        (
            "probability.graph_reliability.connection_probability.compute",
            {
                "graph": {
                    "vertices": ["", "a"],
                    "edges": [["", "a"]],
                },
                "edge_probabilities": [
                    {
                        "edge": ["", "a"],
                        "open_probability": _q(1, 3),
                    }
                ],
                "terminals": ["", "a"],
            },
        ),
    ),
)
def test_probability_results_are_independently_replayed(
    probability_services,
    capability_id: str,
    payload: dict[str, Any],
) -> None:
    computed = probability_services.core.capabilities.invoke(
        CapabilityRequest(capability_id=capability_id, input=payload)
    )

    verified = probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=derive_verification_capability_id(capability_id),
            input={
                "input": payload,
                "candidate": computed.output["result"],
            },
        )
    )

    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == capability_id
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


def test_probability_checker_rejects_forged_event_mass(
    probability_services,
) -> None:
    payload = {"distribution": _FAIR_BIT, "event_values": [_q(1)]}
    computed = probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.finite_distribution.event_probability.compute",
            input=payload,
        )
    )
    forged_candidate = deepcopy(computed.output["result"])
    forged_candidate["event_probability"] = _q(1)
    forged_candidate["selected_atoms"] = _FAIR_BIT["atoms"]

    rejected = probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("probability.finite_distribution.event_probability.verify"),
            input={"input": payload, "candidate": forged_candidate},
        )
    )

    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_probability_checker_rejects_forged_graph_reliability(
    probability_services,
) -> None:
    payload = {
        "graph": {"vertices": ["a", "b"], "edges": [["a", "b"]]},
        "edge_probabilities": [{"edge": ["a", "b"], "open_probability": _q(1, 3)}],
        "terminals": ["a", "b"],
    }
    computed = probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=(
                "probability.graph_reliability.connection_probability.compute"
            ),
            input=payload,
        )
    )
    forged_candidate = deepcopy(computed.output["result"])
    false_states = [dict(state) for state in forged_candidate["states"]]
    false_states[0]["terminals_connected"] = True
    forged_candidate["states"] = false_states
    forged_candidate["connection_probability"] = _q(1)

    rejected = probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=(
                "probability.graph_reliability.connection_probability.verify"
            ),
            input={"input": payload, "candidate": forged_candidate},
        )
    )

    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
