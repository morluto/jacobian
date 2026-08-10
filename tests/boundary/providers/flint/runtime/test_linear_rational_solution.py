from __future__ import annotations

from pathlib import Path

from tests.support.capabilities import invoke_capability
from tests.support.services import open_domain_services

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.domains.rational_linear import build_rational_linear_bundle
from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.operation_installation import OperationInstaller
from jacobian.runtime import CheckerAuthorityMode


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
            services, "linear.rational_solution.compute", _system()
        )
        assert computed.output["result"]["values"] == [
            {"num": "2", "den": "1"},
            {"num": "1", "den": "1"},
        ]
        assert computed.artifact_uris == ()
        verified = services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="linear.rational_solution.verify",
                input={"input": _system(), "candidate": computed.output["result"]},
            )
        )
        assert verified.output["status"] == "VERIFIED"
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
