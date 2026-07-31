from __future__ import annotations

from tests.support.rationals import rational_payload as _q

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.runtime.model import JacobianRuntime


def _qq(entries: list[list[int]]) -> dict[str, object]:
    return {
        "domain": "QQ",
        "entries": [[_q(value) for value in row] for row in entries],
    }


def _compute_square(
    runtime: JacobianRuntime,
) -> CapabilityResult:
    matrix = _qq([[0, 1], [0, 0]])
    return runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.multiply.compute",
            input={"left": matrix, "right": matrix},
        )
    )


def test_square_zero_product_is_computed_then_independently_verified(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    computed = _compute_square(authorized_complete_runtime)

    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert computed.output["result"] == {
        "product": _qq([[0, 0], [0, 0]]),
        "left_rows": 2,
        "inner_dimension": 2,
        "right_columns": 2,
        "convention": "STANDARD_ROW_BY_COLUMN_PRODUCT_OVER_QQ",
    }

    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == "matrix.multiply.compute"
    assert verified.output["result_uri"] == computed.output["result_uri"]
    assert verified.output["verification_record_uri"] is not None
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


def test_matrix_product_verifier_rejects_a_false_product_without_a_record(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    computed = _compute_square(authorized_complete_runtime)
    installed = authorized_complete_runtime.portfolio.domain_bundles["matrix"]
    false_result = authorized_complete_runtime.core.artifacts.put(
        schema_uri=installed.result_schema_uris["matrix.multiply.compute"],
        semantics_uri=installed.semantics_uri,
        parents=(computed.output["input_uri"],),
        payload={
            "product": _qq([[1, 0], [0, 0]]),
            "left_rows": 2,
            "inner_dimension": 2,
            "right_columns": 2,
            "convention": "STANDARD_ROW_BY_COLUMN_PRODUCT_OVER_QQ",
        },
        summary="adversarial false matrix product",
    )

    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED
