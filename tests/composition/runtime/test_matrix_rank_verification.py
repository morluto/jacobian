from __future__ import annotations

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)


def _matrix() -> dict[str, object]:
    def q(value: int) -> dict[str, str]:
        return {"num": str(value), "den": "1"}

    return {
        "matrix_schema_version": "1",
        "domain": "QQ",
        "entries": [
            [q(1), q(2), q(3)],
            [q(2), q(4), q(6)],
            [q(0), q(1), q(1)],
        ],
    }


def test_matrix_rank_verify_independently_recomputes_rank(
    authorized_complete_runtime,
) -> None:
    matrix = _matrix()
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(capability_id="matrix.rank.compute", input={"matrix": matrix})
    )
    verified = authorized_complete_runtime.core.capabilities.invoke(
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


def test_matrix_rank_verify_rejects_wrong_rank(authorized_complete_runtime) -> None:
    matrix = _matrix()
    rejected = authorized_complete_runtime.core.capabilities.invoke(
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
