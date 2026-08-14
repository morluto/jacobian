from __future__ import annotations

from pathlib import Path

from tests.support.exact_domain import open_exact_domain_services
from tests.support.operations import invoke_operation

from jacobian.contracts.operations import (
    OperationRequest,
)
from jacobian.domains.rational_linear import rational_linear_operations


def _system() -> dict[str, object]:
    return {
        "system": {
            "variables": ["x", "y"],
            "coefficients": {
                "entries": [
                    [{"num": "2", "den": "1"}, {"num": "1", "den": "1"}],
                    [{"num": "1", "den": "1"}, {"num": "-1", "den": "1"}],
                ]
            },
            "rhs": [{"num": "5", "den": "1"}, {"num": "1", "den": "1"}],
        }
    }


def test_solution_candidate_is_inline_and_replayable(tmp_path: Path) -> None:
    with open_exact_domain_services(
        tmp_path,
        rational_linear_operations(),
    ) as services:
        computed = invoke_operation(
            services, "linear.rational_solution.compute", _system()
        )
        assert computed.output["result"]["values"] == [
            {"num": "2", "den": "1"},
            {"num": "1", "den": "1"},
        ]
        assert computed.artifact_uris == ()
        verified = services.core.operations.invoke(
            OperationRequest(
                operation_id="linear.rational_solution.verify",
                input={"input": _system(), "candidate": computed.output["result"]},
            )
        )
        assert verified.output["status"] == "VERIFIED"
        assert verified.verification_record_uri is not None
