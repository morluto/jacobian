from __future__ import annotations

from tests.support.rationals import rational_payload

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)


def _matrix() -> dict[str, object]:
    return {
        "matrix_schema_version": "1",
        "domain": "QQ",
        "entries": [
            [rational_payload(1), rational_payload(2), rational_payload(3)],
            [rational_payload(2), rational_payload(4), rational_payload(6)],
            [rational_payload(0), rational_payload(1), rational_payload(1)],
        ],
    }


def test_matrix_rank_verify_independently_recomputes_rank(
    matrix_services,
) -> None:
    matrix = _matrix()
    computed = matrix_services.core.capabilities.invoke(
        CapabilityRequest(capability_id="matrix.rank.compute", input={"matrix": matrix})
    )
    verified = matrix_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.rank.verify",
            mode=CapabilityMode.VERIFY,
            input={
                "input": {"matrix": matrix},
                "candidate": computed.output["result"],
            },
        )
    )
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["conclusion"] == "TRUE"
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


def test_matrix_rank_verify_rejects_wrong_rank(matrix_services) -> None:
    matrix = _matrix()
    rejected = matrix_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.rank.verify",
            mode=CapabilityMode.VERIFY,
            input={
                "input": {"matrix": matrix},
                "candidate": {
                    "rank": 3,
                    "pivot_columns": [0, 1, 2],
                    "method": "EXACT_RATIONAL_ROW_REDUCTION",
                },
            },
        )
    )
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
