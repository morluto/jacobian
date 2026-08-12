from __future__ import annotations

from pathlib import Path

from tests.support.capabilities import invoke_capability
from tests.support.exact_domain import open_exact_domain_services

from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.domains.rational_linear import build_rational_linear_bundle


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
        build_rational_linear_bundle(),
    ) as services:
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
        assert verified.verification_record_uri is not None
