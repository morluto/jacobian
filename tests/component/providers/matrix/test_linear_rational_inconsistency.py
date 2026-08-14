from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.operations import invoke_operation

from jacobian.contracts.linear import (
    LinearRationalInconsistencyFindRequest,
    LinearRationalSolutionFindRequest,
)
from jacobian.contracts.operations import (
    OperationRequest,
)
from jacobian.domains.rational_linear import rational_linear_operations
from jacobian.domains.rational_linear.protocol import (
    RationalLinearCertificateProduced,
    RationalLinearSolutionProduced,
    parse_inconsistency_worker_response,
    parse_solution_worker_response,
)


def _system() -> dict[str, object]:
    return {
        "system": {
            "variables": ["x", "y"],
            "coefficients": {
                "entries": [
                    [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
                    [{"num": "2", "den": "1"}, {"num": "2", "den": "1"}],
                ]
            },
            "rhs": [{"num": "1", "den": "1"}, {"num": "3", "den": "1"}],
        }
    }


def test_rational_linear_worker_payloads_bind_status_and_source_dimensions() -> None:
    solution_request = LinearRationalSolutionFindRequest.model_validate(
        {
            **_system(),
            "system": {
                **_system()["system"],
                "rhs": [{"num": "3", "den": "1"}, {"num": "7", "den": "1"}],
            },
        }
    )
    solution = {
        "protocol": "jacobian.rational-linear-solution-worker/v1",
        "status": "SOLUTION_PRODUCED",
        "values": [
            {"num": "2", "den": "1"},
            {"num": "1", "den": "1"},
        ],
    }
    parsed_solution = parse_solution_worker_response(
        solution,
        expected_value_count=len(solution_request.system.variables),
    )
    assert isinstance(parsed_solution, RationalLinearSolutionProduced)

    inconsistency_request = LinearRationalInconsistencyFindRequest.model_validate(
        _system()
    )
    inconsistency = {
        "protocol": "jacobian.rational-linear-inconsistency-worker/v1",
        "status": "CERTIFICATE_PRODUCED",
        "left_witness": [
            {"num": "-2", "den": "1"},
            {"num": "1", "den": "1"},
        ],
        "rhs_pairing": {"num": "1", "den": "1"},
    }
    parsed_inconsistency = parse_inconsistency_worker_response(
        inconsistency,
        expected_witness_count=len(inconsistency_request.system.rhs),
    )
    assert isinstance(parsed_inconsistency, RationalLinearCertificateProduced)

    for invalid in (
        {**solution, "values": solution["values"][:1]},
        {key: value for key, value in solution.items() if key != "values"},
    ):
        with pytest.raises((TypeError, ValueError)):
            parse_solution_worker_response(
                invalid,
                expected_value_count=len(solution_request.system.variables),
            )


def test_inconsistency_candidate_is_inline_and_replayable(tmp_path: Path) -> None:
    with open_exact_domain_services(
        tmp_path,
        rational_linear_operations(),
    ) as services:
        computed = invoke_operation(
            services,
            "linear.rational_inconsistency.compute",
            _system(),
        )
        assert computed.output["result"]["left_witness"] == [
            {"num": "-2", "den": "1"},
            {"num": "1", "den": "1"},
        ]
        assert computed.artifact_uris == ()
        verified = services.core.operations.invoke(
            OperationRequest(
                operation_id="linear.rational_inconsistency.verify",
                input={"input": _system(), "candidate": computed.output["result"]},
            )
        )
        assert verified.output["status"] == "VERIFIED"
        assert verified.verification_record_uri is not None
