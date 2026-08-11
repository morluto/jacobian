from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.capabilities import invoke_capability
from tests.support.services import open_domain_services

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.linear import (
    LinearRationalInconsistencyFindRequest,
    LinearRationalSolutionFindRequest,
)
from jacobian.domains.rational_linear import build_rational_linear_bundle
from jacobian.domains.rational_linear.protocol import (
    RationalLinearCertificateProduced,
    RationalLinearSolutionProduced,
    parse_inconsistency_worker_response,
    parse_solution_worker_response,
)
from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.operation_installation import OperationInstaller
from jacobian.runtime import CheckerAuthorityMode


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
    bundle = build_rational_linear_bundle()
    with open_domain_services(
        tmp_path, checker_authority=CheckerAuthorityMode.NONE
    ) as services:
        installed = OperationInstaller(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
        ).install(bundle)
        for adapter in installed.adapters:
            services.installation.register_capability(adapter)
        adapters, _ = install_exact_domain_verification(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
            services.installation.verification,
            services.core.checkers,
            bundles={"rational_linear": (bundle, installed)},
            authorize=True,
        )
        for adapter in adapters:
            services.installation.register_capability(adapter)

        computed = invoke_capability(
            services,
            "linear.rational_inconsistency.compute",
            _system(),
        )
        assert computed.output["result"]["left_witness"] == [
            {"num": "-2", "den": "1"},
            {"num": "1", "den": "1"},
        ]
        assert computed.artifact_uris == ()
        verified = services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="linear.rational_inconsistency.verify",
                mode=CapabilityMode.VERIFY,
                input={"input": _system(), "candidate": computed.output["result"]},
            )
        )
        assert verified.output["status"] == "VERIFIED"
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
